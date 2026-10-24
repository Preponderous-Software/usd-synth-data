"""The Strata render backend (``render.engine: strata``, RFC 0026).

The first group needs neither Strata nor Blender: stage translation, the ID packing and the
output contract. Tests marked ``strata`` render with ``libstrata.so`` on llvmpipe and are the
Strata counterparts of the Blender label tests (MVP acceptance criteria 6-8).
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml
from scene_fixtures import top_down_scene, write

from usd_synth import strata_backend as sb
from usd_synth.cli import main
from usd_synth.coco import validate_coco
from usd_synth.config import ConfigError, parse_config
from usd_synth.labels import label_render, read_depth_png
from usd_synth.passes import read_passes
from usd_synth.render import find_renderer, render_batch, scene_spec
from usd_synth.reproduce import reproduce
from usd_synth.scenegen import write_scene
from usd_synth.strata import StrataError, find_strata, linear_to_srgb, srgb_to_linear

ROOT = Path(__file__).resolve().parents[1]
SMALL = ROOT / "tests" / "data" / "small.yaml"
BUNDLED = ROOT / "configs" / "tabletop.yaml"
THREADS = int(os.environ.get("USD_SYNTH_TEST_THREADS", "4"))


def _cfg(width=160, height=120, **render):
    raw = yaml.safe_load(SMALL.read_text())
    raw["image"] = {"width": width, "height": height}
    raw["render"] = {**raw["render"], "engine": "strata", **render}
    return parse_config(raw)


# --------------------------------------------------------------------------- no Strata needed


def test_config_accepts_strata_and_keeps_cycles_default():
    assert _cfg().render.engine == "strata"
    raw = yaml.safe_load(SMALL.read_text())
    assert parse_config(raw).render.engine == "cycles"


def test_strata_with_gpu_is_a_config_error():
    with pytest.raises(ConfigError, match="render: .*gpu is not supported with engine strata"):
        _cfg(device="gpu")


def test_adding_the_engine_value_keeps_existing_config_hashes():
    # The resolved config gained no keys, so datasets made before the backend still reproduce.
    raw = yaml.safe_load(SMALL.read_text())
    assert "engine" in parse_config(raw).resolved()["render"]
    assert set(parse_config(raw).resolved()["render"]) == {
        "engine", "device", "samples", "denoise", "threads", "batch_size",
    }  # fmt: skip


def test_find_strata_explicit_missing_path_is_an_error(tmp_path, monkeypatch):
    monkeypatch.delenv("USD_SYNTH_STRATA", raising=False)
    with pytest.raises(StrataError, match="not found"):
        find_strata(str(tmp_path / "nope.so"))
    monkeypatch.setenv("USD_SYNTH_STRATA", str(tmp_path / "nope.so"))
    with pytest.raises(StrataError, match="not found"):
        find_strata()


def test_find_renderer_picks_the_engine(tmp_path, monkeypatch):
    lib = tmp_path / "libstrata.so"
    lib.write_bytes(b"")
    monkeypatch.setenv("USD_SYNTH_STRATA", str(lib))
    assert find_renderer(_cfg()) == str(lib)


def test_fan_triangulation_of_mixed_polygons():
    corners, face = sb._tri_corners([3, 4, 5])
    assert corners.tolist() == [
        [0, 1, 2], [3, 4, 5], [3, 5, 6], [7, 8, 9], [7, 9, 10], [7, 10, 11],
    ]  # fmt: skip
    assert face.tolist() == [0, 1, 1, 2, 2, 2]


def test_id_packing_round_trips_through_rgba8():
    ids = np.array([0, 1, 255, 256, 65535, 70000, (1 << 24) - 1])
    rgba = np.zeros((1, len(ids), 4), dtype=np.uint8)
    rgba[0, :, :3] = np.rint(sb.id_colors(ids) * 255.0)
    assert sb.decode_ids(rgba)[0].tolist() == ids.tolist()
    with pytest.raises(ValueError, match="2\\*\\*24"):
        sb.id_colors(np.array([1 << 24]))


def test_srgb_transfer_round_trips():
    x = np.linspace(0, 1, 101)
    assert np.allclose(srgb_to_linear(linear_to_srgb(x)), x, atol=1e-9)
    assert np.allclose(linear_to_srgb(np.array([0.0, 1.0])), [0.0, 1.0])


def test_aux_exr_matches_the_blender_contract(tmp_path):
    index = np.array([[0, 3], [7, 7]], dtype=np.int32)
    depth = np.array([[np.inf, 0.9], [1.25, 1.5]], dtype=np.float32)
    sb.write_aux_exr(tmp_path / "aux.exr", depth, index)
    p = read_passes(tmp_path / "aux.exr")
    assert p.index.tolist() == index.tolist() and np.array_equal(p.depth, depth)
    sb.write_aux_exr(tmp_path / "nodepth.exr", None, index)  # ABI 1.8: no depth readback
    assert np.isinf(read_passes(tmp_path / "nodepth.exr").depth).all()


def test_stage_translation_projects_like_the_pinhole_camera(tmp_path):
    """Strata's view and projection put every cube corner where camera.PinholeCamera does."""
    scene = top_down_scene([(-0.15, 0.0, 0.08), (0.12, 0.05, 0.1)], width=640, height=480)
    scene.camera = type(scene.camera).look_at(
        [0.4, -0.6, 0.7], [0.0, 0.0, 0.0], 35.0, 36.0, 640, 480
    )
    usd = write(scene, tmp_path / "scene_000000.usda")
    st = sb.read_stage(usd, 640, 480)
    clip_from_world = st.camera["projection"] @ st.camera["view"]
    for inst in scene.instances:
        pts = inst.world_points()
        expected, _ = scene.camera.project(pts)
        yup = pts @ sb.Z2Y.T
        clip = np.c_[yup, np.ones(len(yup))] @ clip_from_world.T
        ndc = clip[:, :2] / clip[:, 3:]
        got = np.c_[(ndc[:, 0] + 1) * 320, (1 - ndc[:, 1]) * 240]
        assert np.allclose(got, expected, atol=1e-3)


def test_stage_translation_reads_ids_colours_lights(tmp_path):
    scene = top_down_scene([(-0.15, 0.0, 0.08), (0.12, 0.05, 0.1)])
    usd = write(scene, tmp_path / "scene_000000.usda")
    st = sb.read_stage(usd, 160, 120)
    geo = st.geometry
    assert len(geo.positions) % 3 == 0 and geo.positions.shape == geo.normals.shape
    assert sorted(set(geo.ids.tolist())) == [0, 1, 2]
    # every box has 6 quads = 12 triangles = 36 corners
    assert (geo.ids == 1).sum() == 36 and (geo.ids == 2).sum() == 36
    assert np.allclose(np.linalg.norm(geo.normals, axis=1), 1.0, atol=1e-5)
    top = geo.normals[(geo.ids == 1)][:, 1]  # Y-up: the cube's top face normals are +Y
    assert (np.isclose(top, 1.0, atol=1e-6)).sum() == 6
    first = scene.instances[0].material.base_color
    assert np.allclose(geo.colors[geo.ids == 1][0], first, atol=1e-6)
    assert len(st.lights) == 1 and st.dome == {"intensity": 0.5, "color": (1.0, 1.0, 1.0)}
    assert "roughness" in st.ignored and "light_radius" in st.ignored
    (pos, rng, color, intensity, shadows), = sb.point_lights(st)  # fmt: skip
    assert rng > 0 and intensity > 0 and shadows and color == (1.0, 1.0, 1.0)


def test_ids_fall_back_to_prim_names(tmp_path):
    scene = top_down_scene([(0.0, 0.0, 0.08)])
    usd = write(scene, tmp_path / "a" / "scene_000000.usda")
    lines = usd.read_text().splitlines()
    bare = tmp_path / "b" / "scene_000000.usda"
    bare.parent.mkdir()
    bare.write_text("\n".join(x for x in lines if "usd_synth:instance_id" not in x) + "\n")
    assert sorted(set(sb.read_stage(bare, 160, 120).geometry.ids.tolist())) == [0, 1]


# ------------------------------------------------------------------------- Strata renders


@pytest.mark.strata
def test_single_cube_head_on_matches_analytic_projection(tmp_path, strata):
    """MVP criterion 6 on the Strata path: box within 2 px, mask area within 2 %."""
    scene = top_down_scene(
        [(0.0, 0.0, 0.1)], camera_height=1.0, focal_length_mm=50.0, width=640, height=480
    )
    usd = write(scene, tmp_path / "scene_000000.usda")
    cfg = _cfg(640, 480)
    result = render_batch(cfg, [scene_spec(usd, tmp_path / "r")], strata, threads=THREADS)
    (ann,), depth = label_render(usd, tmp_path / "r", cfg)
    uv, _ = scene.camera.project(scene.instances[0].world_points())
    x0, y0 = uv.min(axis=0)
    x1, y1 = uv.max(axis=0)
    bx, by, bw, bh = ann["bbox"]
    assert abs(bx - x0) <= 2 and abs(by - y0) <= 2
    assert abs((bx + bw) - x1) <= 2 and abs((by + bh) - y1) <= 2
    assert ann["area"] == pytest.approx((x1 - x0) * (y1 - y0), rel=0.02)
    info = result.scenes[0]
    assert info["engine"] == "strata" and info["blender"] is None
    assert info["instance_ids"] == [1]
    if info["depth"]:  # Strata ABI 1.9+: planar depth of the top face is 0.9 m
        cx, cy = int((x0 + x1) / 2), int((y0 + y1) / 2)
        assert depth[cy, cx] == pytest.approx(0.9, abs=1e-3)
    else:
        assert np.isinf(depth).all()


@pytest.mark.strata
def test_depth_is_planar(tmp_path, strata):
    """A flat table seen straight down reads the same depth at the centre and the corners."""
    scene = top_down_scene([], camera_height=1.0, width=160, height=120)
    usd = write(scene, tmp_path / "scene_000000.usda")
    cfg = _cfg()
    info = render_batch(cfg, [scene_spec(usd, tmp_path / "r")], strata).scenes[0]
    if not info["depth"]:
        if os.environ.get("USD_SYNTH_REQUIRE_STRATA") == "1":  # CI pins Strata 0.6.0 (ABI 1.12)
            pytest.fail(f"Strata {info['strata']} (ABI {info['strata_abi']}) has no depth readback")
        pytest.skip(f"Strata {info['strata']} (ABI {info['strata_abi']}) has no depth readback")
    depth = read_passes(tmp_path / "r" / "aux.exr").depth
    assert np.allclose(depth[[0, 0, -1, -1, 60], [0, -1, 0, -1, 80]], 1.0, atol=2e-3)


@pytest.mark.strata
def test_ids_seen_are_exactly_the_scene_instances(tmp_path, strata):
    """No blended or stray IDs anywhere: every decoded value is 0 or a real instance."""
    cfg = _cfg(320, 240)
    usds = [write_scene(cfg, 2026, i, tmp_path / "s", SMALL.parent) for i in range(4)]
    specs = [scene_spec(u, tmp_path / "r" / u.stem) for u in usds]
    render_batch(cfg, specs, strata, threads=THREADS)
    for usd in usds:
        st = sb.read_stage(usd, 320, 240)
        index = read_passes(tmp_path / "r" / usd.stem / "aux.exr").index
        assert set(np.unique(index).tolist()) <= {0, *st.geometry.ids.tolist()}


@pytest.mark.strata
def test_bundled_config_640x480_renders(tmp_path, strata):
    from usd_synth.config import load_config

    raw = load_config(BUNDLED).resolved()
    raw["render"]["engine"] = "strata"
    cfg = parse_config(raw)
    usd = write_scene(cfg, 42, 0, tmp_path / "s", BUNDLED.parent)
    info = render_batch(cfg, [scene_spec(usd, tmp_path / "r")], strata).scenes[0]
    from PIL import Image

    rgb = np.asarray(Image.open(tmp_path / "r" / "rgb.png"))
    assert rgb.shape == (480, 640, 3) and rgb.std() > 5  # a lit image, not a flat clear
    assert info["timings"]["render"] < 30 and info["gl"]


@pytest.fixture(scope="module")
def strata_dataset(tmp_path_factory, strata):
    tmp = tmp_path_factory.mktemp("strata-ds")
    raw = yaml.safe_load(SMALL.read_text())
    raw["render"].update(engine="strata", threads=THREADS, batch_size=3)
    cfg_path = tmp / "small.yaml"
    cfg_path.write_text(yaml.safe_dump(raw))
    out = tmp / "ds"
    rc = main(["generate", "--config", str(cfg_path), "--seed", "42", "--count", "4",
               "--out", str(out), "--strata", strata])  # fmt: skip
    assert rc == 0
    return cfg_path, out


@pytest.mark.strata
def test_generate_with_strata_writes_a_valid_dataset(strata_dataset):
    _, out = strata_dataset
    assert validate_coco(out / "annotations.json") == []
    coco = json.loads((out / "annotations.json").read_text())
    assert len(coco["images"]) == 4 and len(coco["annotations"]) > 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["engine"] == "strata" and manifest["blender"] is None
    assert manifest["strata"] and manifest["gl"] and "roughness" in manifest["ignored"]
    assert manifest["samples"] is None and manifest["denoise"] is None
    t = json.loads((out / "throughput.json").read_text())
    assert t["environment"]["engine"] == "strata" and t["environment"]["samples"] is None
    assert t["images_per_hour"] > 0
    d = read_depth_png(out / "depth" / "scene_000000.png")
    assert d.shape == (120, 160)
    if manifest["depth"]:
        assert np.nanmin(d) > 0.3 and np.nanmax(d) < 3.0
    else:
        assert np.isnan(d).all()  # no depth readback: every pixel is "no hit"


@pytest.mark.strata
def test_reproduce_with_strata_is_exact(strata_dataset, tmp_path, strata):
    _, out = strata_dataset
    report = reproduce(out, 2, tmp_path / "repro", strata, threads=THREADS)
    assert report["ok"] and report["usda_identical"] and report["annotations_identical"]
    assert report["rgb_mean_abs_diff"] == 0.0  # rasterization draws no random samples


def test_bundled_strata_config_differs_from_tabletop_only_in_engine():
    from usd_synth.config import load_config

    cycles = load_config(BUNDLED).resolved()
    strata = load_config(ROOT / "configs" / "tabletop-strata.yaml").resolved()
    assert cycles["render"].pop("engine") == "cycles"
    assert strata["render"].pop("engine") == "strata"
    assert cycles == strata
