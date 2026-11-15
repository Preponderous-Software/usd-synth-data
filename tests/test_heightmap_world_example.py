"""examples/heightmap_world: heightmap world packages -> labelled aerial usd-synth scenes.

Most packages here are drawn in the test (a few pixels with known heights and classes), so
every region, height and face count is known exactly. The bundled ``sample/`` package is
checked for shape and size and sampled for views.
"""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from PIL import Image
from pxr import Usd, UsdGeom

from usd_synth.coco import validate_coco
from usd_synth.render import scene_spec

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "heightmap_world"
SAMPLE = EXAMPLE / "sample"
sys.path.insert(0, str(EXAMPLE))

import hm_scene  # noqa: E402
import hm_synth  # noqa: E402
import worldviews  # noqa: E402

THREADS = int(os.environ.get("USD_SYNTH_TEST_THREADS", "4"))
C = {name: k for k, name in enumerate(hm_scene.DEFAULT_CLASSES)}


def write_package(path: Path, height: np.ndarray, ground: np.ndarray, *, bpp=4, origin=(-64, 32),
                  y_offset=64, vectors=None) -> Path:  # fmt: skip
    """A package from block-y heights (NaN = unsurveyed) and class indices."""
    path.mkdir(parents=True, exist_ok=True)
    raw = np.where(np.isfinite(height), np.nan_to_num(height) + y_offset, 0).astype(np.uint16)
    Image.fromarray(raw).save(path / "height.png")
    Image.fromarray(ground.astype(np.uint8)).save(path / "ground.png")
    world = {
        "format": "heightmap-world/1",
        "grid": {"origin_x": origin[0], "origin_z": origin[1], "blocks_per_px": bpp,
                 "width": height.shape[1], "height": height.shape[0], "y_offset": y_offset},
        "classes": hm_scene.DEFAULT_CLASSES,
        "files": {"height": "height.png", "ground": "ground.png"},
    }  # fmt: skip
    (path / "world.json").write_text(json.dumps(world))
    if vectors is not None:
        (path / "vectors.json").write_text(json.dumps(vectors))
    return path


def island(n=24):
    """Water around a plain island with a forest, a lake, a snowy peak and two stone spots."""
    i, j = np.mgrid[0:n, 0:n]
    d = np.hypot(i - n / 2 + 0.5, j - n / 2 + 0.5)
    height = np.where(d < n * 0.38, 70 + (n * 0.38 - d) * 3, 62.0)
    ground = np.where(d < n * 0.38, C["plain"], C["water"])
    ground[4:9, 9:15] = C["forest"]
    ground[13:16, 6:9] = C["water"]  # inland lake
    height[13:16, 6:9] = 66.0
    ground[11:13, 11:13] = C["snow"]
    height[11:13, 11:13] = 110.0
    ground[17, 14] = ground[18, 16] = C["stone"]  # two separate one-pixel outcrops
    return height, ground


@pytest.fixture
def settings():
    return hm_scene.load_settings(EXAMPLE / "heightmap_world.yaml")


@pytest.fixture
def pkg(tmp_path):
    return write_package(tmp_path / "pkg", *island())


def without_trees(settings):
    return settings.model_copy(
        update={"trees": settings.trees.model_copy(update={"enabled": False})}
    )


def count(info, c):
    return sum(i.category == c for i in info.instances)


def test_read_package(pkg):
    world = hm_scene.read_package(pkg)
    assert world.shape == (24, 24) and world.blocks_per_px == 4
    assert (world.origin_x, world.origin_z) == (-64, 32)
    assert world.height[12, 12] == pytest.approx(110.0)  # y_offset removed
    assert world.ground[0, 0] == C["water"] and world.vectors is None
    half = hm_scene.read_package(pkg, stride=2)
    assert half.shape == (12, 12) and half.blocks_per_px == 8
    assert half.height[6, 6] == world.height[12, 12]


def test_unsurveyed_and_void_have_no_height(tmp_path):
    height, ground = island()
    height[0, 0] = np.nan
    ground[0, 1] = C["void"]
    world = hm_scene.read_package(write_package(tmp_path / "p", height, ground))
    assert np.isnan(world.height[0, 0]) and np.isnan(world.height[0, 1])
    assert np.isfinite(world.height[1:, 2:]).all()


def test_read_package_rejects_mismatched_rasters(pkg):
    meta = json.loads((pkg / "world.json").read_text())
    meta["grid"]["width"] = 30
    (pkg / "world.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="world.json says"):
        hm_scene.read_package(pkg)


def test_components_in_raster_order():
    key = np.array(
        [
            [1, 1, -1, 2],
            [-1, 1, -1, 2],
            [3, -1, 1, 1],
            [3, 3, -1, 1],
        ]
    )
    comp = hm_scene.components(key, connectivity=4)
    assert comp.tolist() == [
        [0, 0, -1, 1],
        [-1, 0, -1, 1],
        [2, -1, 3, 3],  # the 1s at (2, 2) touch (1, 1) only diagonally: a new region
        [2, 2, -1, 3],
    ]
    # a U shape joins through its bottom row even though its arms start apart
    u = np.array([[5, -1, 5], [5, -1, 5], [5, 5, 5]])
    assert (hm_scene.components(u)[u >= 0] == 0).all()
    # 8-connected (the default): pixels touching at a corner join
    assert hm_scene.components(key)[2, 2] == 0
    stairs = np.where(np.eye(5, dtype=bool), 7, -1)
    assert hm_scene.components(stairs)[np.eye(5, dtype=bool)].tolist() == [0] * 5
    assert hm_scene.components(stairs, 4)[np.eye(5, dtype=bool)].tolist() == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError):
        hm_scene.components(stairs, 6)


def test_corner_heights_average_surveyed_neighbours():
    h = np.array([[1.0, 3.0], [5.0, np.nan]])
    c = hm_scene.corner_heights(h)
    assert c.shape == (3, 3)
    assert c[1, 1] == pytest.approx(3.0)  # (1 + 3 + 5) / 3: the unsurveyed pixel is skipped
    assert c[0, 0] == 1.0 and c[2, 2] != c[2, 2]  # NaN: no surveyed pixel touches it


def test_build_scene_regions(pkg, settings, tmp_path):
    world = hm_scene.read_package(pkg)
    no_trees = without_trees(settings)
    info = hm_scene.build_scene(world, no_trees, tmp_path / "scene.usdc")
    counts = {c: count(info, c) for c in info.categories}
    assert counts["water"] == 2  # the sea and the lake
    assert counts["forest"] == 1 and counts["snow"] == 1 and counts["stone"] == 2
    assert sum(counts.values()) == len(info.instances) == 6
    assert info.faces == 24 * 24  # one quad per surveyed pixel, labelled or not
    assert info.pixels_by_category["forest"] == 30
    stage = Usd.Stage.Open(str(tmp_path / "scene.usdc"))
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    objs = stage.GetPrimAtPath("/World/Objects").GetChildren()
    ids = [p.GetAttribute("userProperties:usd_synth:instance_id").Get() for p in objs]
    assert ids == list(range(1, 7))
    names = [p.GetAttribute("userProperties:usd_synth:category_name").Get() for p in objs]
    assert names == ["water", "water", "forest", "snow", "stone", "stone"]  # labelled order
    assert stage.GetPrimAtPath("/World/Env").GetChildren(), "plain ground is unlabelled Env"


def test_terrain_quads_face_up_and_sit_on_the_grid(pkg, settings, tmp_path):
    world = hm_scene.read_package(pkg)
    no_trees = without_trees(settings)
    info = hm_scene.build_scene(world, no_trees, tmp_path / "scene.usdc")
    snow = next(i for i in info.instances if i.category == "snow")
    stage = Usd.Stage.Open(str(tmp_path / "scene.usdc"))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(f"/World/Objects/inst_{snow.instance_id:05d}"))
    pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 4)
    assert len(idx) == 4 and len(pts) == 9  # a 2 x 2 patch shares its corners
    # Z-up: block x -> x, block z -> -y; pixels (11..12, 11..12) span x -20..-12, z 76..84
    assert pts[:, 0].min() == -20 and pts[:, 0].max() == -12
    assert pts[:, 1].min() == -84 and pts[:, 1].max() == -76
    assert pts[:, 2].max() == pytest.approx(110.0)  # the middle corner: all four are snow
    for q in idx:
        a, b, c = pts[q[0]], pts[q[1]], pts[q[2]]
        assert np.cross(b - a, c - b)[2] > 0, "terrain quads must face up"


def test_trees_join_their_forest(pkg, settings, tmp_path):
    world = hm_scene.read_package(pkg)
    info = hm_scene.build_scene(world, settings, tmp_path / "scene.usdc", seed=4)
    forest = next(i for i in info.instances if i.category == "forest")
    assert info.trees == forest.trees > 0  # every tree stands in the one forest
    assert forest.faces == 30 + forest.trees * (4 + 6)  # ground quads + trunk sides + cone
    again = hm_scene.build_scene(world, settings, tmp_path / "again.usdc", seed=4)
    assert again.trees == info.trees and again.faces == info.faces
    stage = Usd.Stage.Open(str(tmp_path / "scene.usdc"))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(f"/World/Objects/inst_{forest.instance_id:05d}"))
    z = np.asarray(mesh.GetPointsAttr().Get())[:, 2]
    lo, hi = settings.trees.height_m
    assert z.max() <= world.height[4:9, 9:15].max() + hi + 1e-3
    assert z.max() >= world.height[4:9, 9:15].min() + lo - 1e-3


def test_vectors_split_regions(tmp_path, settings):
    height, ground = island()
    # a lake polygon over the left half of the inland lake (pixel cols 6..8 = x -40..-28)
    lake = {"polygon": [[-41, 80], [-34, 80], [-34, 98], [-41, 98]]}
    vectors = {"format": "heightmap-vectors/1", "lakes": [lake]}
    pkg = write_package(tmp_path / "v", height, ground, vectors=vectors)
    world = hm_scene.read_package(pkg)
    assert world.vectors["lakes"] == [lake]
    features, names = hm_scene.feature_raster(world)
    assert names == ["lake_0"] and features[13:16, 6:8].min() == 1 and features[14, 9] == 0
    no_trees = without_trees(settings)
    info = hm_scene.build_scene(world, no_trees, tmp_path / "scene.usdc")
    waters = [i for i in info.instances if i.category == "water"]
    assert len(waters) == 3  # the sea, the lake inside the polygon, the rest of the lake
    assert [w.feature for w in waters].count("lake_0") == 1


def test_unknown_labelled_class_is_an_error(pkg, settings, tmp_path):
    bad = settings.model_copy(update={"labelled": ["water", "volcano"]})
    with pytest.raises(ValueError, match="volcano"):
        hm_scene.build_scene(hm_scene.read_package(pkg), bad, tmp_path / "scene.usdc")


def test_sample_package_is_small_and_readable():
    total = sum(p.stat().st_size for p in SAMPLE.iterdir())
    assert total < 1_000_000
    meta = json.loads((SAMPLE / "world.json").read_text())
    assert meta["format"] == "heightmap-world/1"
    world = hm_scene.read_package(SAMPLE)
    assert world.vectors is not None and world.vectors["lakes"]
    present = {world.classes[k] for k in np.unique(world.ground)}
    assert {"water", "plain", "sand", "stone", "snow"} <= present


def test_views_are_deterministic_and_aerial(settings, tmp_path):
    world = hm_scene.read_package(SAMPLE)
    info = hm_scene.build_scene(world, settings, tmp_path / "scene.usdc", seed=11)
    (tmp_path / "views").mkdir()
    seen = set()
    for index in range(4):
        a = hm_scene.sample_view(info, settings, 11, index)
        b = hm_scene.sample_view(info, settings, 11, index)
        assert np.array_equal(a.camera.usd_matrix(), b.camera.usd_matrix()) and a.light == b.light
        seen.add(a.target_class)
        assert a.camera.position[2] - a.target[2] == pytest.approx(a.altitude, abs=0.01)
        lo, hi = settings.views.pitch_deg
        assert lo <= a.pitch_deg <= hi
        x0, y0, x1, y1 = info.grid.bounds()
        assert x0 < a.camera.position[0] < x1 and y0 < a.camera.position[1] < y1
        floor = float(np.nanmin(info.grid.corners))
        assert hm_scene.far_range(a.camera, floor) <= settings.views.max_range_m
        usd = tmp_path / "views" / f"view_{index:04d}.usda"
        worldviews.write_view(a.camera, a.light, tmp_path / "scene.usdc", usd, 11, index)
        assert scene_spec(usd, tmp_path / "out")["scene_index"] == index
    assert len(seen) == 4  # classes take turns


@pytest.mark.blender
def test_rendered_labels_line_up_with_geometry(tmp_path, blender, settings):
    """A top-down render of a drawn package: the class at each region's pixels matches."""
    # 1) The driver end to end on the bundled sample.
    raw = yaml.safe_load((EXAMPLE / "heightmap_world.yaml").read_text())
    raw.update(width=160, height=120)
    (tmp_path / "small.yaml").write_text(yaml.safe_dump(raw))
    out = tmp_path / "run"
    hm_synth.main([str(SAMPLE), "--config", str(tmp_path / "small.yaml"), "--out", str(out),
                   "--views", "1", "--samples", "4", "--threads", str(THREADS),
                   "--blender", blender])  # fmt: skip
    assert validate_coco(out / "annotations.json") == []
    coco = json.loads((out / "annotations.json").read_text())
    assert coco["annotations"], "a sampled view saw no labelled region"

    # 2) A known camera straight above the island, through the same writer and worker.
    pkg = write_package(tmp_path / "pkg", *island())
    world = hm_scene.read_package(pkg)
    no_trees = without_trees(settings)
    info = hm_scene.build_scene(world, no_trees, tmp_path / "s" / "scene.usdc")
    from usd_synth.camera import PinholeCamera
    from usd_synth.config import ImageConfig, RenderConfig
    from usd_synth.labels import read_instances
    from usd_synth.passes import read_passes
    from usd_synth.render import render_batch

    x0, y0, x1, y1 = info.grid.bounds()
    eye = np.array([(x0 + x1) / 2, (y0 + y1) / 2, 260.0])
    cam = PinholeCamera.look_at(eye, eye - [0, 0, 1], 35, 36, 320, 240, up=(0, 1, 0))
    light = worldviews.sample_lighting(1, 0, (60, 60), (30, 30), (5, 5), (1, 1))
    (tmp_path / "s" / "views").mkdir()
    usd = tmp_path / "s" / "views" / "view_0000.usda"
    worldviews.write_view(cam, light, tmp_path / "s" / "scene.usdc", usd, 1, 0)
    shim = SimpleNamespace(render=RenderConfig(samples=4), image=ImageConfig(width=320, height=240))
    render_batch(shim, [scene_spec(usd, tmp_path / "r")], blender, threads=THREADS)
    passes = read_passes(tmp_path / "r" / "aux.exr")
    _, sem = worldviews.label_view(passes.index, read_instances(usd), info.categories, 1)
    names = ["-", *info.categories]

    def cls_at(i, j):
        p = info.grid.pixel_centres(np.array([i]), np.array([j]))
        uv, _ = cam.project(p)
        return names[sem[int(uv[0][1]), int(uv[0][0])]]

    assert cls_at(1, 1) == "water"
    assert cls_at(14, 7) == "water"  # the inland lake
    assert cls_at(6, 11) == "forest"
    assert cls_at(11, 11) == "snow"
    assert cls_at(17, 14) == "stone"
    assert cls_at(8, 4) == "-"  # plain ground is unlabelled
