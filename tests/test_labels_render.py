"""Label correctness against rendered fixtures (MVP acceptance criteria 6 and 7)."""

import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml
from scene_fixtures import top_down_scene, write

from usd_synth.coco import build_coco, dump_json, validate_coco
from usd_synth.config import parse_config
from usd_synth.labels import label_render, read_depth_png, write_depth_png
from usd_synth.render import render_batch, scene_spec
from usd_synth.scenegen import write_scene

pytestmark = pytest.mark.blender
SMALL = Path(__file__).resolve().parent / "data" / "small.yaml"
THREADS = 4


def _cfg(width, height, **extra):
    raw = yaml.safe_load(SMALL.read_text())
    raw["image"] = {"width": width, "height": height}
    raw.update(extra)
    return parse_config(raw)


def test_single_cube_head_on_matches_analytic_projection(tmp_path, blender):
    """A 10 cm cube 0.9 m below the camera (top face), 50 mm lens, 640x480."""
    scene = top_down_scene(
        [(0.0, 0.0, 0.1)], camera_height=1.0, focal_length_mm=50.0, width=640, height=480
    )
    usd = write(scene, tmp_path / "scene_000000.usda")
    cfg = _cfg(640, 480)
    render_batch(cfg, [scene_spec(usd, tmp_path / "r")], blender, threads=THREADS)
    (ann,), depth = label_render(usd, tmp_path / "r", cfg)

    # analytic: project the cube's 8 corners with the same pinhole model
    uv, _ = scene.camera.project(scene.instances[0].world_points())
    x0, y0 = uv.min(axis=0)
    x1, y1 = uv.max(axis=0)
    bx, by, bw, bh = ann["bbox"]
    assert abs(bx - x0) <= 2 and abs(by - y0) <= 2
    assert abs((bx + bw) - x1) <= 2 and abs((by + bh) - y1) <= 2
    analytic_area = (x1 - x0) * (y1 - y0)  # seen head-on, the silhouette is the top face
    assert ann["area"] == pytest.approx(analytic_area, rel=0.02)
    # depth of the top face: 0.9 m
    cx, cy = int((x0 + x1) / 2), int((y0 + y1) / 2)
    assert depth[cy, cx] == pytest.approx(0.9, abs=1e-3)


def test_rendered_dataset_is_valid_coco(tmp_path, blender):
    cfg = _cfg(160, 120)
    specs, usds = [], []
    for i in range(3):
        usd = write_scene(cfg, 9, i, tmp_path / "scenes", SMALL.parent)
        usds.append(usd)
        specs.append(scene_spec(usd, tmp_path / "r" / usd.stem))
    render_batch(cfg, specs, blender, threads=THREADS)
    records = []
    for i, usd in enumerate(usds):
        anns, depth = label_render(usd, tmp_path / "r" / usd.stem, cfg)
        (tmp_path / "images").mkdir(exist_ok=True)
        (tmp_path / "depth").mkdir(exist_ok=True)
        shutil.copy(tmp_path / "r" / usd.stem / "rgb.png", tmp_path / "images" / f"{usd.stem}.png")
        write_depth_png(depth, cfg.labels.depth.max_m, tmp_path / "depth" / f"{usd.stem}.png")
        records.append(
            {
                "scene_index": i,
                "file_name": f"images/{usd.stem}.png",
                "depth_file": f"depth/{usd.stem}.png",
                "scene_file": f"scenes/{usd.stem}.usda",
                "width": 160,
                "height": 120,
                "annotations": anns,
            }
        )
    data = build_coco(records, cfg)
    assert len(data["annotations"]) >= 3
    dump_json(data, tmp_path / "annotations.json")
    assert validate_coco(tmp_path / "annotations.json") == []
    d = read_depth_png(tmp_path / "depth" / "scene_000000.png")
    assert d.shape == (120, 160) and np.nanmin(d) > 0.3 and np.nanmax(d) < 3.0
