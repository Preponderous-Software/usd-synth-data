"""examples/minecraft: Minecraft-USD-Exporter terrain -> labelled usd-synth scenes.

The fake exports here follow the exporter's documented layout (``/World/BlockLibrary`` class
prim with ``mcusd:blockIds``; one ``PointInstancer`` per region with a per-instance
``primvars:mcusd:blockId``), so the tests need neither Minecraft nor the exporter.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, Vt

from usd_synth.coco import build_coco, dump_json, validate_coco
from usd_synth.render import scene_spec

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "minecraft"
sys.path.insert(0, str(EXAMPLE))

import mc_scene  # noqa: E402
import mc_synth  # noqa: E402

THREADS = int(os.environ.get("USD_SYNTH_TEST_THREADS", "4"))
COLOURS = {
    "minecraft:grass_block": (0.36, 0.6, 0.25),
    "minecraft:oak_log": (0.4, 0.3, 0.18),
    "minecraft:oak_leaves": (0.2, 0.5, 0.15),
    "minecraft:water": (0.15, 0.3, 0.95),
    "minecraft:stone": (0.5, 0.5, 0.5),
    "minecraft:sand": (0.86, 0.81, 0.62),
    "minecraft:coal_ore": (0.3, 0.3, 0.3),
    "minecraft:short_grass": (0.4, 0.7, 0.3),
    "minecraft:seagrass": (0.1, 0.5, 0.2),
}


def fake_export(path: Path, blocks: dict[tuple[int, int, int], str], offset=(0, 0, 0)) -> Path:
    """Write a minimal Minecraft-USD-Exporter-shaped stage holding ``blocks`` (MC coords)."""
    ids = ["minecraft:air", *sorted(set(blocks.values()))]
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    lib = stage.CreateClassPrim("/World/BlockLibrary")
    lib.CreateAttribute("mcusd:blockIds", Sdf.ValueTypeNames.StringArray).Set(Vt.StringArray(ids))
    for bid in ids[1:]:
        mesh = UsdGeom.Mesh.Define(stage, f"/World/BlockLibrary/{bid.split(':')[1]}")
        mesh.GetPrim().CreateAttribute("mcusd:namespacedId", Sdf.ValueTypeNames.String).Set(bid)
        mesh.CreateDisplayColorPrimvar().Set(Vt.Vec3fArray([Gf.Vec3f(*COLOURS[bid])]))
    UsdGeom.Xform.Define(stage, "/World/Terrain")
    region = UsdGeom.Xform.Define(stage, "/World/Terrain/r_0_0")
    region.AddTranslateOp().Set(Gf.Vec3d(*offset))
    inst = UsdGeom.PointInstancer.Define(stage, "/World/Terrain/r_0_0/Blocks")
    pos = np.array([np.subtract(p, offset) for p in blocks], dtype=np.float32)
    inst.CreatePositionsAttr(Vt.Vec3fArray.FromNumpy(pos))
    inst.CreateProtoIndicesAttr(Vt.IntArray([0] * len(blocks)))
    pv = UsdGeom.PrimvarsAPI(inst).CreatePrimvar(
        "mcusd:blockId", Sdf.ValueTypeNames.IntArray, UsdGeom.Tokens.vertex
    )
    pv.Set(Vt.IntArray([ids.index(b) for b in blocks.values()]))
    stage.GetRootLayer().Save()
    return path


def island(n=24) -> dict:
    """Grass plane at y=63 with a pond, a beach strip, a stone outcrop and two oak trees."""
    blocks = {}
    for x in range(n):
        for z in range(n):
            blocks[(x, 63, z)] = "minecraft:grass_block"
    for x in range(3, 9):
        for z in range(3, 9):
            blocks[(x, 63, z)] = "minecraft:water"
            blocks[(x, 62, z)] = "minecraft:sand"
    blocks[(5, 63, 5)] = "minecraft:seagrass"  # waterlogged: counts as water
    for z in range(3, 9):
        blocks[(9, 63, z)] = "minecraft:sand"
    for x, z in ((16, 16), (16, 5)):  # two trees, far apart
        for y in range(64, 68):
            blocks[(x, y, z)] = "minecraft:oak_log"
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                blocks[(x + dx, 68, z + dz)] = "minecraft:oak_leaves"
    for y in (64, 65):
        blocks[(4, y, 18)] = "minecraft:stone"
    blocks[(5, 64, 18)] = "minecraft:coal_ore"
    blocks[(12, 64, 12)] = "minecraft:short_grass"  # decoration: dropped
    return blocks


@pytest.fixture
def settings():
    return mc_scene.load_settings(EXAMPLE / "minecraft.yaml")


def test_build_coco_takes_explicit_categories(tmp_path):
    cats = [{"id": 1, "name": "tree", "supercategory": "minecraft"}]
    rec = {
        "scene_index": 0, "file_name": "images/0000.png", "depth_file": "depth/0000.png",
        "scene_file": "views/view_0000.usda", "width": 4, "height": 4,
        "annotations": [{"category_id": 1, "segmentation": mc_synth.encode_rle(np.eye(4) > 0),
                         "area": 4.0, "bbox": [0.0, 0.0, 4.0, 4.0], "iscrowd": 0,
                         "instance_id": 7}],
    }  # fmt: skip
    data = build_coco([rec], None, "test", cats)
    assert data["categories"] == cats and data["annotations"][0]["id"] == 1
    dump_json(data, tmp_path / "a.json")
    assert validate_coco(tmp_path / "a.json") == []
    with pytest.raises(ValueError, match="config or an explicit categories"):
        build_coco([rec], None)


def test_read_export_applies_region_transform(tmp_path):
    blocks = {(-512, 70, -3): "minecraft:stone", (-511, 70, -3): "minecraft:grass_block"}
    terrain = mc_scene.read_export(fake_export(tmp_path / "t.usda", blocks, (-512, 0, 0)))
    pairs = zip(terrain.xyz.tolist(), terrain.block, strict=True)
    got = {tuple(p): terrain.ids[b] for p, b in pairs}
    assert got == blocks
    assert terrain.colors["minecraft:stone"] == pytest.approx(COLOURS["minecraft:stone"])


def test_classify_patterns_drop_and_water(settings):
    ids = ["minecraft:air", "minecraft:oak_log", "minecraft:deepslate_iron_ore",
           "minecraft:seagrass", "minecraft:short_grass", "minecraft:grass_block",
           "minecraft:birch_leaves"]  # fmt: skip
    cls, render = mc_scene.classify(ids, settings)
    names = ["-", *[c.name for c in settings.classes]]
    assert [names[c] if c >= 0 else "drop" for c in cls] == [
        "drop", "tree", "ore", "water", "drop", "-", "tree",
    ]  # fmt: skip
    assert render[3] == "minecraft:water"


def test_components_respect_connectivity():
    diag = np.array([[0, 0, 0], [1, 1, 0], [5, 5, 5]])
    assert mc_scene.components(diag, 26).tolist() == [0, 0, 1]
    assert mc_scene.components(diag, 6).tolist() == [0, 1, 2]


def test_exposed_faces_and_orientation():
    bar = np.array([[0, 0, 0], [1, 0, 0]])
    faces = mc_scene.exposed_faces(bar, mc_scene.Occupancy(bar))
    assert len(faces[0]) == 10  # 12 faces minus the shared pair
    part = mc_scene.quad_mesh(bar, np.zeros(2, dtype=np.int32), faces)
    quads = part.points.reshape(-1, 4, 3)
    # Counter-clockwise from outside: the quad's winding normal matches the face normal.
    winding = np.cross(quads[:, 1] - quads[:, 0], quads[:, 2] - quads[:, 0])
    assert np.allclose(winding / np.linalg.norm(winding, axis=1, keepdims=True), part.normals)
    assert mc_scene.mc_to_zup([[0, 1, 0]]).tolist() == [[0, 0, 1]]  # MC up -> Z up
    assert mc_scene.mc_to_zup([[0, 0, 1]]).tolist() == [[0, -1, 0]]  # MC south -> -Y


def test_build_scene_instances(tmp_path, settings):
    terrain = mc_scene.read_export(fake_export(tmp_path / "t.usda", island()))
    info = mc_scene.build_scene(terrain, settings, tmp_path / "scene.usdc")
    counts = {c: sum(i.category == c for i in info.instances) for c in info.categories}
    # one pond (seagrass merged in), two trees, one outcrop with an ore block; the beach strip
    # and the pond floor touch only along an edge, so 6-connected sand is two components
    assert counts == {"tree": 2, "water": 1, "ore": 1, "sand": 2, "stone": 1}
    assert info.blocks_kept == info.blocks_in - 1  # the short grass is dropped
    stage = Usd.Stage.Open(str(tmp_path / "scene.usdc"))
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    objs = stage.GetPrimAtPath("/World/Objects").GetChildren()
    assert [p.GetAttribute("userProperties:usd_synth:instance_id").Get() for p in objs] == list(
        range(1, 8)
    )
    tree = next(i for i in info.instances if i.category == "tree")
    assert tree.blocks == 4 + 9
    assert len(info.targets["ore"]) == 1 and len(info.targets["tree"]) > 0


def test_views_are_deterministic_and_readable(tmp_path, settings):
    terrain = mc_scene.read_export(fake_export(tmp_path / "t.usda", island(48)))
    small = settings.model_copy(
        update={"views": settings.views.model_copy(update={"margin_blocks": 2,
                                                           "distance": (8.0, 14.0)})}
    )  # fmt: skip
    info = mc_scene.build_scene(terrain, small, tmp_path / "scene.usdc")
    a = mc_scene.sample_view(info, small, 7, 3)
    b = mc_scene.sample_view(info, small, 7, 3)
    assert np.array_equal(a.camera.usd_matrix(), b.camera.usd_matrix())
    assert a.sun_dir == b.sun_dir and a.target_block == b.target_block
    assert a.target_class == info.categories[3 % len(info.categories)]
    (tmp_path / "views").mkdir()
    usd = tmp_path / "views" / "view_0003.usda"
    mc_scene.write_view(a, tmp_path / "scene.usdc", usd, 7, small)
    spec = scene_spec(usd, tmp_path / "out")  # the stock host-side reader
    assert spec["scene_index"] == 3 and spec["dome"]["intensity"] == pytest.approx(a.sky_intensity)
    assert spec["camera"]["focal_length_mm"] == pytest.approx(a.camera.focal_length_mm)
    assert len(Usd.Stage.Open(str(usd)).GetPrimAtPath("/World/Objects").GetChildren()) == 7


def test_label_view_semantic_and_min_pixels():
    from usd_synth.labels import InstanceInfo

    index = np.zeros((10, 10), dtype=np.int32)
    index[0:4, 0:4] = 1  # tree, 16 px
    index[6:10, 6:8] = 2  # water, 8 px
    index[9, 0] = 3  # ore, 1 px: below min_pixels, but still in the semantic mask
    insts = [InstanceInfo(1, "object", "tree", 0), InstanceInfo(2, "object", "water", 0),
             InstanceInfo(3, "object", "ore", 0)]  # fmt: skip
    anns, sem = mc_synth.label_view(index, insts, ["tree", "water", "ore"], min_pixels=4)
    assert [(a["instance_id"], a["category_id"], a["area"]) for a in anns] == [
        (1, 1, 16.0),
        (2, 2, 8.0),
    ]
    assert anns[1]["bbox"] == [6.0, 6.0, 2.0, 4.0]
    assert sem[0, 0] == 1 and sem[9, 7] == 2 and sem[9, 0] == 3 and sem[5, 5] == 0


@pytest.mark.blender
def test_rendered_labels_line_up_with_geometry(tmp_path, blender, settings):
    """Top-down render of the island: class at projected block centres matches the scene."""
    # 1) The driver end to end on one sampled view (steep views fit the small island).
    raw = yaml.safe_load((EXAMPLE / "minecraft.yaml").read_text())
    raw.update(width=160, height=120)
    raw["views"].update(margin_blocks=2, distance=[3, 6], pitch_deg=[-85, -75],
                        focal_length_mm=[30, 32])  # fmt: skip
    (tmp_path / "small.yaml").write_text(yaml.safe_dump(raw))
    big = fake_export(tmp_path / "big.usda", island(64))
    out = tmp_path / "run"
    mc_synth.main([
        str(big), "--config", str(tmp_path / "small.yaml"), "--out", str(out), "--views", "1",
        "--samples", "4", "--threads", str(THREADS), "--blender", blender,
    ])  # fmt: skip
    coco = json.loads((out / "annotations.json").read_text())
    assert validate_coco(out / "annotations.json") == []
    assert coco["annotations"], "a sampled view saw no labelled instance"

    meta = json.loads((out / "meta.json").read_text())
    assert meta["throughput"]["images"] == 1 and meta["render"]["samples"] == 4

    # 2) A known camera straight above the island, through the same writer and worker.
    export = fake_export(tmp_path / "t.usda", island())
    terrain = mc_scene.read_export(export)
    info = mc_scene.build_scene(terrain, settings, tmp_path / "s" / "scene.usdc")
    from usd_synth.camera import PinholeCamera

    eye = mc_scene.mc_to_zup([[12.0, 100.0, 12.0]])[0]
    cam = PinholeCamera.look_at(eye, eye - [0, 0, 1], 35, 36, 160, 120, up=(0, 1, 0))
    view = mc_scene.View(0, cam, "tree", (16, 67, 16), (0.3, 0.2, 0.93), 70.0, 5.0,
                         (1, 1, 1), (0.5, 0.6, 1.0), 1.0, 0.0, -90.0, 0.0)  # fmt: skip
    (tmp_path / "s" / "views").mkdir()
    usd = tmp_path / "s" / "views" / "view_0000.usda"
    mc_scene.write_view(view, tmp_path / "s" / "scene.usdc", usd, 1, settings)
    from types import SimpleNamespace

    from usd_synth.config import ImageConfig, RenderConfig
    from usd_synth.labels import read_instances
    from usd_synth.passes import read_passes
    from usd_synth.render import render_batch

    shim = SimpleNamespace(render=RenderConfig(samples=4), image=ImageConfig(width=160, height=120))
    render_batch(shim, [scene_spec(usd, tmp_path / "r")], blender, threads=THREADS)
    passes = read_passes(tmp_path / "r" / "aux.exr")
    _, sem = mc_synth.label_view(passes.index, read_instances(usd), info.categories, 1)
    names = ["-", *info.categories]

    def pixel(x, top_y, z):
        uv, _ = cam.project(mc_scene.mc_to_zup([[x + 0.5, top_y, z + 0.5]]))
        return int(uv[0][1]), int(uv[0][0])

    def cls_at(x, top_y, z):
        return names[sem[pixel(x, top_y, z)]]

    assert cls_at(6, 64, 6) == "water"
    assert cls_at(9, 64, 5) == "sand"
    assert cls_at(16, 69, 16) == "tree"
    assert cls_at(5, 65, 18) == "ore"
    assert cls_at(4, 66, 18) == "stone"
    assert cls_at(20, 64, 20) == "-"  # plain grass
    rgb = np.asarray(Image.open(tmp_path / "r" / "rgb.png"))
    r, g, b = rgb[pixel(6, 64, 6)][:3].astype(int)
    assert b > r and b > g  # the water mask sits on blue pixels
