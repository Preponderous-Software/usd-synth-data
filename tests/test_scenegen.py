import os
from pathlib import Path

import numpy as np
import pytest
import yaml
from pxr import Usd, UsdGeom, UsdLux, UsdShade

from usd_synth.cli import main
from usd_synth.config import load_config, parse_config
from usd_synth.scenegen import generate_scene, scene_filename
from usd_synth.scenegen.tabletop import color_temperature_rgb

ROOT = Path(__file__).resolve().parents[1]
BUNDLED = ROOT / "configs" / "tabletop.yaml"
SMALL = ROOT / "tests" / "data" / "small.yaml"
GOLDEN = ROOT / "tests" / "golden"
NS = "userProperties:usd_synth:"


@pytest.fixture(scope="module")
def cfg():
    return load_config(BUNDLED)


def _stage(text: str) -> Usd.Stage:
    layer_stage = Usd.Stage.CreateInMemory()
    layer_stage.GetRootLayer().ImportFromString(text)
    return layer_stage


def test_same_seed_and_index_is_byte_identical(cfg):
    assert generate_scene(cfg, 42, 7)[1] == generate_scene(cfg, 42, 7)[1]


def test_different_seed_or_index_changes_the_file(cfg):
    base = generate_scene(cfg, 42, 7)[1]
    assert generate_scene(cfg, 43, 7)[1] != base
    assert generate_scene(cfg, 42, 8)[1] != base


def test_scene_alone_equals_scene_inside_a_range(cfg):
    in_range = {i: generate_scene(cfg, 5, i)[1] for i in range(6)}
    reversed_range = {i: generate_scene(cfg, 5, i)[1] for i in reversed(range(3, 9))}
    alone = generate_scene(cfg, 5, 4)[1]
    assert in_range[4] == alone == reversed_range[4]


@pytest.mark.parametrize("name", ["small_seed7_idx0", "small_seed7_idx3"])
def test_golden_files(name):
    """Any change to generated output must be accepted explicitly (USD_SYNTH_UPDATE_GOLDEN=1)."""
    _, seed, idx = name.split("_")
    text = generate_scene(load_config(SMALL), int(seed[4:]), int(idx[3:]), SMALL.parent)[1]
    path = GOLDEN / f"{name}.usda"
    if os.environ.get("USD_SYNTH_UPDATE_GOLDEN") == "1":
        path.write_text(text, encoding="utf-8")
    assert text == path.read_text(encoding="utf-8")


@pytest.mark.parametrize("index", range(25))
def test_stage_structure(cfg, index):
    scene, text = generate_scene(cfg, 11, index)
    stage = _stage(text)
    assert UsdGeom.GetStageUpAxis(stage) == "Z"
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert stage.GetDefaultPrim().GetPath() == "/World"
    prims = list(stage.Traverse())
    assert sum(p.IsA(UsdGeom.Camera) for p in prims) == 1
    assert sum(p.HasAPI(UsdLux.LightAPI) for p in prims) >= 1
    assert "references" not in text  # flattened, no composition arcs

    labelled = [p for p in stage.GetPrimAtPath("/World/Objects").GetChildren()]
    assert len(labelled) == sum(i.kind == "object" for i in scene.instances)
    assert len(labelled) >= 1
    categories = list(stage.GetPrimAtPath("/World").GetAttribute(NS + "categories").Get())
    ids = set()
    for prim in labelled:
        iid = prim.GetAttribute(NS + "instance_id").Get()
        cid = prim.GetAttribute(NS + "category_id").Get()
        assert prim.GetName() == f"inst_{iid:04d}"
        assert categories[cid - 1] == prim.GetAttribute(NS + "category_name").Get()
        ids.add(iid)
        mesh = prim.GetChild(f"inst_{iid:04d}_geom")
        assert mesh.IsA(UsdGeom.Mesh)
        material, _ = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()
        assert material
    for prim in stage.GetPrimAtPath("/World/Distractors").GetChildren():
        assert prim.GetAttribute(NS + "category_id").Get() == 0
        ids.add(prim.GetAttribute(NS + "instance_id").Get())
    assert ids == set(range(1, len(scene.instances) + 1))


def _world_boxes(stage):
    """Axis-aligned world bounds of each instance's transformed mesh points, read from USD."""
    xf = UsdGeom.XformCache()
    boxes = []
    for parent in ("/World/Objects", "/World/Distractors"):
        for prim in stage.GetPrimAtPath(parent).GetChildren():
            pts = []
            for child in Usd.PrimRange(prim):
                if child.IsA(UsdGeom.Mesh):
                    m = np.array(xf.GetLocalToWorldTransform(child))
                    local = np.array(UsdGeom.Mesh(child).GetPointsAttr().Get(), dtype=float)
                    pts.append(local @ m[:3, :3] + m[3, :3])
            world = np.concatenate(pts)
            boxes.append((world.min(axis=0), world.max(axis=0)))
    return boxes


@pytest.mark.parametrize("index", range(40))
def test_objects_rest_on_support_without_overlap(cfg, index):
    scene, text = generate_scene(cfg, 3, index)
    sx, sy = scene.support_size
    margin = 0.02 if scene.support_kind == "bin" else 0.0
    boxes = _world_boxes(_stage(text))
    gap = cfg.objects.placement.min_separation_m
    for lo, hi in boxes:
        assert lo[2] == pytest.approx(0.0, abs=1e-5)  # resting on the support top
        assert lo[0] >= -sx / 2 + margin - 1e-5 and hi[0] <= sx / 2 - margin + 1e-5
        assert lo[1] >= -sy / 2 + margin - 1e-5 and hi[1] <= sy / 2 - margin + 1e-5
    for a in range(len(boxes)):
        for b in range(a + 1, len(boxes)):
            (alo, ahi), (blo, bhi) = boxes[a], boxes[b]
            separated = (
                ahi[0] + gap <= blo[0] + 1e-5
                or bhi[0] + gap <= alo[0] + 1e-5
                or ahi[1] + gap <= blo[1] + 1e-5
                or bhi[1] + gap <= alo[1] + 1e-5
            )
            assert separated, f"instances {a + 1} and {b + 1} overlap"


def test_projected_bbox_matches_camera_projection(cfg):
    scene, text = generate_scene(cfg, 21, 2)
    stage = _stage(text)
    for inst in scene.instances:
        uv, front = scene.camera.project(inst.world_points())
        assert front.all()
        prim = stage.GetPrimAtPath(
            f"/World/{'Objects/inst' if inst.kind == 'object' else 'Distractors/dis'}"
            f"_{inst.instance_id:04d}"
        )
        x, y, w, h = prim.GetAttribute(NS + "projected_bbox").Get()
        assert (x, y) == pytest.approx(tuple(uv.min(axis=0)), abs=1e-3)
        assert (x + w, y + h) == pytest.approx(tuple(uv.max(axis=0)), abs=1e-2)
        area = prim.GetAttribute(NS + "projected_area_px").Get()
        assert 0 < area <= w * h + 1e-3


def test_camera_looks_at_support_centre(cfg):
    for index in range(10):
        scene, _ = generate_scene(cfg, 1, index)
        uv, _ = scene.camera.project([[0.0, 0.0, 0.0]])
        # target jitter is at most 5 cm, so the support centre stays near the image centre
        assert abs(uv[0, 0] - 320) < 200 and abs(uv[0, 1] - 240) < 200


def test_user_asset_is_referenced_then_flattened(tmp_path):
    asset = tmp_path / "assets" / "block.usda"
    asset.parent.mkdir()
    asset.write_text(
        '#usda 1.0\n(\n    defaultPrim = "Block"\n)\n\ndef Mesh "Block"\n{\n'
        "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
        "    point3f[] points = [(-1, -1, -1), (1, -1, -1), (1, 1, 1), (-1, 1, 1)]\n}\n"
    )
    raw = yaml.safe_load(SMALL.read_text())
    raw["objects"]["library"] = [
        {"name": "block", "source": "assets/block.usda", "category": "block", "size_m": [0.1, 0.1]}
    ]
    raw["objects"]["placement"] = {"yaw_deg": [0, 0], "allow_tipped": 0}  # axis-aligned bounds
    raw["distractors"] = {"count": [0, 0]}
    cfg = parse_config(raw)
    scene, text = generate_scene(cfg, 1, 0, tmp_path)
    assert "references" not in text and "block.usda" not in text
    assert 'def Mesh "asset"' in text  # the asset's default prim, flattened in place
    stage = _stage(text)
    for lo, hi in _world_boxes(stage):
        assert (hi - lo).max() == pytest.approx(0.1, abs=1e-4)
        assert lo[2] == pytest.approx(0.0, abs=1e-5)


def test_textures_are_assigned_with_uvs(tmp_path):
    raw = yaml.safe_load(SMALL.read_text())
    raw["materials"] = {"textures": ["wood.png"]}
    _, text = generate_scene(parse_config(raw), 2, 0, tmp_path)
    assert "UsdUVTexture" in text
    assert "texCoord2f[] primvars:st" in text
    assert str(tmp_path / "wood.png") in text


def test_color_temperature():
    warm, daylight = color_temperature_rgb(2700), color_temperature_rgb(6500)
    assert warm[0] == 1.0 and warm[2] < 0.3
    assert min(daylight) > 0.85


def test_cli_scenes_writes_files(tmp_path, capsys):
    rc = main(
        [
            "scenes",
            "--config",
            str(SMALL),
            "--seed",
            "7",
            "--count",
            "2",
            "--start",
            "3",
            "--out",
            str(tmp_path),
        ]
    )
    assert rc == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == [scene_filename(3), scene_filename(4)]
    golden = (GOLDEN / "small_seed7_idx3.usda").read_text(encoding="utf-8")
    assert (tmp_path / scene_filename(3)).read_text(encoding="utf-8") == golden


def _asset_cfg(source, size=None):
    raw = yaml.safe_load(SMALL.read_text())
    entry = {"name": "a", "source": source, "category": "a"}
    if size is not None:
        entry["size_m"] = [size, size]
    raw["objects"]["library"] = [entry]
    raw["objects"]["count"] = [1, 1]
    raw["objects"]["placement"] = {"yaw_deg": [0, 0], "allow_tipped": 0}
    raw["distractors"] = {"count": [0, 0]}
    return parse_config(raw)


def _asset(path, body, up="Z", mpu=1.0, default='    defaultPrim = "Thing"\n'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'#usda 1.0\n(\n{default}    metersPerUnit = {mpu}\n    upAxis = "{up}"\n)\n\n{body}'
    )


def _cube(name, lo, hi, extra=""):
    pts = ", ".join(f"({x}, {y}, {z})" for x in (lo, hi) for y in (lo, hi) for z in (lo, hi))
    return (
        f'def Mesh "{name}"\n{{\n{extra}    int[] faceVertexCounts = [4, 4]\n'
        "    int[] faceVertexIndices = [0, 1, 3, 2, 4, 5, 7, 6]\n"
        f"    point3f[] points = [{pts}]\n}}\n"
    )


def test_asset_units_and_up_axis_are_converted(tmp_path):
    # a 10 cm cube authored in centimetres, Y-up, offset along +Y (its "up")
    _asset(tmp_path / "cm.usda", _cube("Thing", 0, 10), up="Y", mpu=0.01)
    scene, text = generate_scene(_asset_cfg("cm.usda"), 1, 0, tmp_path)
    ((lo, hi),) = _world_boxes(_stage(text))
    np.testing.assert_allclose(hi - lo, [0.1, 0.1, 0.1], atol=1e-6)
    assert lo[2] == pytest.approx(0.0, abs=1e-6)


def test_tiny_asset_scale_stays_grounded(tmp_path):
    # a 1000-unit cube sized to 0.1004 m needs scale 1.004e-4; it must still touch the support
    _asset(tmp_path / "big.usda", _cube("Thing", -500, 500))
    scene, text = generate_scene(_asset_cfg("big.usda", size=0.1004), 1, 0, tmp_path)
    ((lo, hi),) = _world_boxes(_stage(text))
    assert (hi - lo).max() == pytest.approx(0.1004, abs=1e-7)
    assert lo[2] == pytest.approx(0.0, abs=1e-7)


def test_only_the_default_prim_is_bounded(tmp_path):
    body = _cube("Thing", -0.05, 0.05) + _cube("Helper", 5, 6)
    _asset(tmp_path / "two.usda", body)
    scene, text = generate_scene(_asset_cfg("two.usda"), 1, 0, tmp_path)
    ((lo, hi),) = _world_boxes(_stage(text))
    np.testing.assert_allclose(hi - lo, [0.1, 0.1, 0.1], atol=1e-6)


def test_asset_without_default_prim_is_rejected(tmp_path):
    _asset(tmp_path / "nodef.usda", _cube("Thing", 0, 1), default="")
    with pytest.raises(ValueError, match="defaultPrim"):
        generate_scene(_asset_cfg("nodef.usda"), 1, 0, tmp_path)


def test_asset_material_binding_is_stronger_than_descendants(tmp_path):
    _asset(tmp_path / "a.usda", _cube("Thing", 0, 1))
    _, text = generate_scene(_asset_cfg("a.usda", size=0.1), 1, 0, tmp_path)
    stage = _stage(text)  # keep the stage alive while its prims are used
    prim = stage.GetPrimAtPath("/World/Objects/inst_0001/asset")
    rel = UsdShade.MaterialBindingAPI(prim).GetDirectBindingRel()
    assert UsdShade.MaterialBindingAPI.GetMaterialBindingStrength(rel) == "strongerThanDescendants"


def test_textured_support_and_floor_get_uvs(tmp_path):
    raw = yaml.safe_load(SMALL.read_text())
    raw["materials"] = {"textures": ["wood.png"]}
    for index in range(20):
        scene, text = generate_scene(parse_config(raw), 3, index, tmp_path)
        stage = _stage(text)
        for mat, path in [(scene.support_material, "/World/Env/Support/top"),
                          (scene.floor_material, "/World/Env/Background")]:  # fmt: skip
            has_st = UsdGeom.PrimvarsAPI(stage.GetPrimAtPath(path)).HasPrimvar("st")
            assert has_st == (mat.texture is not None)


def test_out_of_range_seed_is_rejected(capsys):
    with pytest.raises(ValueError, match="2\\*\\*63"):
        generate_scene(load_config(SMALL), 2**63, 0)
    rc = main(["scenes", "--config", str(SMALL), "--seed", str(2**63), "--count", "1",
               "--out", "unused"])  # fmt: skip
    assert rc == 1 and "2**63" in capsys.readouterr().err
