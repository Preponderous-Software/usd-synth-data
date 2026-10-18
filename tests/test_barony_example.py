"""examples/barony: Barony game states -> labelled usd-synth scenes.

The states here are written by hand with the fields the example reads from Barony's
``GET /api/session/state`` JSON (``grid[x][y].type`` / ``ownerId``; ``armies[]`` with ``x``,
``y``, ``soldiers``, ``playerId``, ``moving``, ``destinationX/Y``), so the tests need no Barony
backend.
"""

import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from pxr import Usd, UsdGeom

from usd_synth.coco import validate_coco
from usd_synth.render import scene_spec

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "barony"
sys.path.insert(0, str(EXAMPLE))

import barony_scene  # noqa: E402
import barony_synth  # noqa: E402
import worldviews  # noqa: E402

THREADS = int(os.environ.get("USD_SYNTH_TEST_THREADS", "4"))


def fake_state(path: Path, width=6, height=5, specials=None, armies=None) -> Path:
    specials = specials if specials is not None else {
        (0, 0): ("CASTLE", 1), (5, 4): ("CASTLE", 2), (2, 1): ("VILLAGE", 0),
        (4, 1): ("VILLAGE", 2),
    }  # fmt: skip
    grid = [[{"type": "EMPTY", "ownerId": 0} for _ in range(height)] for _ in range(width)]
    for (x, y), (kind, owner) in specials.items():
        grid[x][y] = {"type": kind, "ownerId": owner}
    if armies is None:  # one figure each, marching east, so their place is known exactly
        armies = [(1, 3, 3, 1), (4, 3, 2, 2)]
    data = {
        "grid": grid,
        "armies": [{"x": x, "y": y, "soldiers": n, "playerId": p, "moving": True,
                    "destinationX": x + 1, "destinationY": y} for x, y, n, p in armies],
        "tickCount": 12,
    }  # fmt: skip
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def settings():
    return barony_scene.load_settings(EXAMPLE / "barony.yaml")


@pytest.fixture
def board(tmp_path):
    return barony_scene.read_state(fake_state(tmp_path / "game_00_tick_012.json"))


def test_read_state(board):
    assert (board.width, board.height, board.tick) == (6, 5, 12)
    assert board.types[0][0] == "CASTLE" and board.owners[5][4] == 2
    assert [(a.x, a.y, a.soldiers, a.owner, a.destination) for a in board.armies] == [
        (1, 3, 3, 1, (2, 3)),
        (4, 3, 2, 2, (5, 3)),
    ]


def test_build_scene_instances(board, settings, tmp_path):
    info = barony_scene.build_scene(board, settings, tmp_path / "scene.usdc")
    counts = {c: sum(i.category == c for i in info.instances) for c in info.categories}
    assert counts == {"castle": 2, "village": 2, "army": 2}
    assert info.bounds == (0.0, 0.0, 24.0, 20.0)
    stage = Usd.Stage.Open(str(tmp_path / "scene.usdc"))
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    objs = stage.GetPrimAtPath("/World/Objects").GetChildren()
    ids = [p.GetAttribute("userProperties:usd_synth:instance_id").Get() for p in objs]
    assert ids == list(range(1, 7))
    owners = [p.GetAttribute("userProperties:usd_synth:owner").Get() for p in objs]
    assert owners == [1, 0, 2, 2, 1, 2]  # castle, village, village, castle, army, army (x, y order)
    # the same state always gives the same board
    again = barony_scene.build_scene(board, settings, tmp_path / "again.usdc")
    assert again.faces == info.faces


def test_army_size_and_placement(tmp_path, settings):
    armies = [(0, 0, 10, 1), (3, 3, 100, 2)]
    board = barony_scene.read_state(fake_state(tmp_path / "s.json", armies=armies))
    info = barony_scene.build_scene(board, settings, tmp_path / "scene.usdc")
    small, big = [i for i in info.instances if i.category == "army"]
    per_figure = 5 * 3  # body, head and spear: five faces each (no bottoms)
    standard = 5 * 2
    assert small.faces == math.ceil(10 / 3) * per_figure + standard
    assert big.faces == settings.max_figures * per_figure + standard
    spots = barony_scene.army_spots(board, settings.tile_m)
    assert spots[0][1] == pytest.approx(5.0)  # the castle fills its tile: stands in front of it
    assert np.allclose(spots[1], [14.0, 14.0])  # on an empty tile: its centre


def test_views_are_deterministic_and_readable(board, settings, tmp_path):
    info = barony_scene.build_scene(board, settings, tmp_path / "scene.usdc")
    vs = settings.views.model_copy(update={"distance": (4.0, 8.0)})
    small = settings.model_copy(update={"views": vs})
    a = barony_scene.sample_view([info], small, 5, 2)
    b = barony_scene.sample_view([info], small, 5, 2)
    assert np.array_equal(a.camera.usd_matrix(), b.camera.usd_matrix()) and a.light == b.light
    assert a.target_class == "army"
    (tmp_path / "views").mkdir()
    usd = tmp_path / "views" / "view_0002.usda"
    worldviews.write_view(a.camera, a.light, tmp_path / "scene.usdc", usd, 5, 2)
    spec = scene_spec(usd, tmp_path / "out")
    assert spec["scene_index"] == 2
    assert len(Usd.Stage.Open(str(usd)).GetPrimAtPath("/World/Objects").GetChildren()) == 6


@pytest.mark.blender
def test_rendered_labels_line_up_with_geometry(tmp_path, blender, settings):
    """A top-down render of a board: the class at each object's projected position matches."""
    path = fake_state(tmp_path / "game_00_tick_012.json")

    # 1) The driver end to end on one sampled view.
    raw = yaml.safe_load((EXAMPLE / "barony.yaml").read_text())
    raw.update(width=160, height=120)
    raw["views"].update(distance=[4, 8])
    (tmp_path / "small.yaml").write_text(yaml.safe_dump(raw))
    out = tmp_path / "run"
    barony_synth.main([str(path), "--config", str(tmp_path / "small.yaml"), "--out", str(out),
                       "--views", "1", "--samples", "4", "--threads", str(THREADS),
                       "--blender", blender])  # fmt: skip
    assert validate_coco(out / "annotations.json") == []
    coco = json.loads((out / "annotations.json").read_text())
    assert coco["annotations"], "a sampled view saw no labelled instance"

    # 2) A known camera straight above the board, through the same writer and worker.
    board = barony_scene.read_state(path)
    info = barony_scene.build_scene(board, settings, tmp_path / "s" / "scene.usdc")
    from usd_synth.camera import PinholeCamera
    from usd_synth.config import ImageConfig, RenderConfig
    from usd_synth.labels import read_instances
    from usd_synth.passes import read_passes
    from usd_synth.render import render_batch

    eye = np.array([12.0, 10.0, 40.0])
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

    def cls_at(x, y, z):
        uv, _ = cam.project(np.array([[x, y, z]]))
        return names[sem[int(uv[0][1]), int(uv[0][0])]]

    t = settings.tile_m
    assert cls_at(0.5 * t, 0.5 * t, 3.0) == "castle"  # the keep's roof
    assert cls_at(5.5 * t, 4.5 * t, 3.0) == "castle"
    stage = Usd.Stage.Open(str(tmp_path / "s" / "scene.usdc"))
    for inst in info.instances:
        if inst.category != "village":
            continue
        # the centre of the highest roof ridge: the top z shared by exactly two points (a
        # banner pole's top has four)
        pts = np.asarray(
            UsdGeom.Mesh(stage.GetPrimAtPath(f"/World/Objects/inst_{inst.instance_id:05d}"))
            .GetPointsAttr()
            .Get()
        )
        zs, n = np.unique(np.round(pts[:, 2], 4), return_counts=True)
        top = zs[n == 2].max()
        ridge = pts[np.round(pts[:, 2], 4) == top]
        assert cls_at(*ridge.mean(axis=0)) == "village"
    for spot in barony_scene.army_spots(board, t):
        assert cls_at(spot[0], spot[1], 0.9) == "army"  # the single soldier's head
    assert cls_at(3.5 * t, 0.5 * t, 0.25) == "-"  # an empty tile (or its unlabelled scenery)
