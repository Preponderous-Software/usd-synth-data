"""examples/roam: roam-usd world stages -> labelled usd-synth scenes.

The stages here are written by hand in roam-usd's documented layout (Z-up; ``/World/Surface``
rooms with a ``Ground`` quad, a ``Tiles`` ``PointInstancer`` whose prototypes reference
``/World/Prototypes`` meshes and carry ``customData["roam:entityClass"]``, and individual
``Creatures`` cards and ``Structures``), with this file's own sizes and colours, so the tests
need neither Roam nor roam-usd.
"""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from pxr import Gf, Usd, UsdGeom, Vt

from usd_synth.coco import validate_coco
from usd_synth.render import scene_spec

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "roam"
sys.path.insert(0, str(EXAMPLE))

import roam_scene  # noqa: E402
import roam_synth  # noqa: E402
import worldviews  # noqa: E402

THREADS = int(os.environ.get("USD_SYNTH_TEST_THREADS", "4"))

# class: (width, height, lift, colour, individual?)
PROTOS = {
    "Grass": (1.0, 0.1, 0.0, (0.3, 0.6, 0.25)),
    "OakWood": (0.4, 1.5, 0.0, (0.45, 0.3, 0.15)),
    "Leaves": (1.0, 0.4, 1.1, (0.15, 0.45, 0.15)),
    "Stone": (1.0, 1.0, 0.0, (0.5, 0.5, 0.52)),
    "CoalOre": (1.0, 1.0, 0.0, (0.2, 0.2, 0.22)),
    "WoodFloor": (1.0, 0.1, 0.0, (0.6, 0.45, 0.27)),
    "Chest": (0.7, 0.6, 0.0, (0.6, 0.42, 0.2)),
    "Apple": (0.3, 0.3, 0.0, (0.85, 0.1, 0.1)),
}
CHICKEN = (0.95, 0.95, 0.9)


def _box(stage, path, width, height, lift, colour):
    h = width / 2
    xy = [(-h, -h), (h, -h), (h, h), (-h, h)]
    pts = [Gf.Vec3f(x, y, lift) for x, y in xy] + [Gf.Vec3f(x, y, lift + height) for x, y in xy]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray(pts))
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4] * 6))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([i for f in faces for i in f]))
    mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*colour)]))
    return mesh


def fake_world(path: Path, tiles: dict, creatures=(), structures=(), size=24) -> Path:
    """A one-room roam-usd-shaped stage; ``tiles`` maps (tx, ty) -> stacked entity classes."""
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
    stage.CreateClassPrim("/World/Prototypes")
    for cls, (w, h, lift, colour) in PROTOS.items():
        _box(stage, f"/World/Prototypes/{cls}", w, h, lift, colour)
    level = UsdGeom.Xform.Define(stage, "/World/Surface")
    level.GetPrim().SetCustomDataByKey("roam:z", 0)
    room = UsdGeom.Xform.Define(stage, "/World/Surface/Room_0_0")
    room.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
    ground = UsdGeom.Mesh.Define(stage, "/World/Surface/Room_0_0/Ground")
    ground.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(0, -size, 0), Gf.Vec3f(size, -size, 0),
                                           Gf.Vec3f(size, 0, 0), Gf.Vec3f(0, 0, 0)]))  # fmt: skip
    ground.CreateFaceVertexCountsAttr(Vt.IntArray([4]))
    ground.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 3]))
    ground.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.1, 0.1, 0.1)]))

    def centre(tx, ty):
        return tx + 0.5, -(ty + 0.5)

    inst = UsdGeom.PointInstancer.Define(stage, "/World/Surface/Room_0_0/Tiles")
    classes = sorted({c for stack in tiles.values() for c in stack})
    targets = []
    for cls in classes:
        proto = stage.DefinePrim(f"/World/Surface/Room_0_0/Tiles/Prototypes/{cls}")
        proto.GetReferences().AddInternalReference(f"/World/Prototypes/{cls}")
        proto.SetCustomDataByKey("roam:entityClass", cls)
        targets.append(proto.GetPath())
    inst.CreatePrototypesRel().SetTargets(targets)
    pos, idx = [], []
    for (tx, ty), stack in sorted(tiles.items()):
        z = 0.0
        for cls in stack:
            pos.append(Gf.Vec3f(*centre(tx, ty), z))
            idx.append(classes.index(cls))
            w, h, lift, _ = PROTOS[cls]
            z += lift + h
    inst.CreatePositionsAttr(Vt.Vec3fArray(pos))
    inst.CreateProtoIndicesAttr(Vt.IntArray(idx))
    for tx, ty, z in creatures:
        card = UsdGeom.Mesh.Define(stage, f"/World/Surface/Room_0_0/Creatures/Chicken_{tx}_{ty}")
        corners = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
        card.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(x, y, 0) for x, y in corners]))
        card.CreateFaceVertexCountsAttr(Vt.IntArray([4]))
        card.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 3]))
        card.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*CHICKEN)]))
        card.AddTranslateOp().Set(Gf.Vec3d(*centre(tx, ty), z))
        card.GetPrim().SetCustomDataByKey("roam:entityClass", "Chicken")
    for cls, tx, ty, z in structures:
        prim = stage.DefinePrim(f"/World/Surface/Room_0_0/Structures/{cls}_{tx}_{ty}")
        prim.GetReferences().AddInternalReference(f"/World/Prototypes/{cls}")
        UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*centre(tx, ty), z))
        prim.SetCustomDataByKey("roam:entityClass", cls)
    stage.GetRootLayer().Save()
    return path


def glade(size=24):
    """Grass everywhere; one oak with its four leaves (one with a duplicate), a stone outcrop
    with a coal ore block beside it, loose leaf cover, a floor with a chest, an apple and a
    chicken."""
    tiles = {(x, y): ["Grass"] for x in range(size) for y in range(size)}
    tiles[(5, 5)] = ["Grass", "OakWood"]
    for t in ((4, 5), (6, 5), (5, 4), (5, 6)):
        tiles[t] = ["Grass", "Leaves"]
    tiles[(5, 6)] = ["Grass", "Leaves", "Leaves", "Apple"]  # stacked twice, with an apple
    for t in ((14, 4), (15, 4), (14, 5), (15, 5)):
        tiles[t] = ["Grass", "Stone"]
    tiles[(17, 9)] = ["Grass", "CoalOre"]
    tiles[(3, 16)] = ["Grass", "Leaves"]  # leaf cover, far from any trunk
    tiles[(10, 15)] = ["Grass", "WoodFloor"]
    tiles[(11, 15)] = ["Grass", "WoodFloor"]
    tiles[(18, 18)] = ["Grass", "Apple"]
    return tiles, [(8, 10, 0.11)], [("Chest", 11, 15, 0.2)]


@pytest.fixture
def settings():
    return roam_scene.load_settings(EXAMPLE / "roam.yaml")


@pytest.fixture
def world(tmp_path):
    tiles, creatures, structures = glade()
    return roam_scene.read_world(fake_world(tmp_path / "w.usda", tiles, creatures, structures))


def test_read_world(world):
    assert world.bounds == (0.0, -24.0, 24.0, 0.0)
    by_class = {}
    for e in world.entities:
        by_class.setdefault(e.entity_class, []).append(e)
    assert len(by_class["Grass"]) == 24 * 24
    assert len(by_class["Leaves"]) == 6
    oak = by_class["OakWood"][0]
    assert oak.tile == (5, -6)  # Roam (tx, ty) -> world tile (tx, -ty - 1)
    assert oak.points[:, 2].min() == pytest.approx(0.1, abs=1e-6) and oak.points[
        :, 2
    ].max() == pytest.approx(1.6)
    chicken = by_class["Chicken"][0]
    assert chicken.creature and chicken.tile == (8, -11)
    assert by_class["Chest"][0].base.tolist() == [11.5, -15.5, 0.2]


def test_tidy_canopy_cover_and_restack(world, settings):
    kept = roam_scene.tidy(world.entities, settings)
    leaves = [e for e in kept if e.entity_class == "Leaves"]
    assert len(leaves) == 5  # the duplicate on (5, 6) is dropped
    cover = [e for e in leaves if e.cover]
    assert [e.tile for e in cover] == [(3, -17)]
    assert cover[0].points[:, 2].max() == pytest.approx(0.1 + roam_scene.LITTER_HEIGHT)
    canopy = next(e for e in leaves if e.tile == (5, -7))
    assert canopy.points[:, 2].min() == pytest.approx(1.2)  # floats at its lift over grass
    apple = next(e for e in kept if e.entity_class == "Apple" and e.tile == (5, -7))
    assert apple.points[:, 2].min() == pytest.approx(0.1)  # on the ground, under the canopy
    chicken = next(e for e in kept if e.creature)
    assert chicken.points[:, 2].max() - chicken.points[:, 2].min() == pytest.approx(0.55)
    assert roam_scene.classify(cover[0], settings) == 0
    assert roam_scene.classify(canopy, settings) == 1  # leaves count as tree


def test_components():
    tiles = [(0, 0), (1, 1), (1, 1), (5, 5)]
    assert roam_scene.components(tiles, 8).tolist() == [0, 0, 0, 1]
    assert roam_scene.components(tiles, 4).tolist() == [0, 1, 1, 2]


def test_build_scene_instances(world, settings, tmp_path):
    info = roam_scene.build_scene(world, settings, tmp_path / "scene.usdc")
    counts = {c: sum(i.category == c for i in info.instances) for c in info.categories}
    assert counts == {"tree": 1, "stone": 1, "ore": 1, "creature": 1, "structure": 1, "food": 2}
    tree = next(i for i in info.instances if i.category == "tree")
    assert tree.entities == 5 and tree.tiles == 5  # trunk + four canopy tiles
    stone = next(i for i in info.instances if i.category == "stone")
    assert stone.tiles == 4
    structure = next(i for i in info.instances if i.category == "structure")
    assert structure.entities == 3  # two floor tiles and the chest
    stage = Usd.Stage.Open(str(tmp_path / "scene.usdc"))
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    objs = stage.GetPrimAtPath("/World/Objects").GetChildren()
    ids = [p.GetAttribute("userProperties:usd_synth:instance_id").Get() for p in objs]
    assert ids == list(range(1, 8))
    assert not stage.GetPrimAtPath("/World/Looks").GetChildren()[0].GetName().startswith("Sprite")


def test_views_are_deterministic_and_readable(world, settings, tmp_path):
    vs = settings.views.model_copy(update={"margin_tiles": 1, "distance": (3.0, 6.0),
                                           "pitch_deg": (-80.0, -60.0)})  # fmt: skip
    small = settings.model_copy(update={"views": vs})
    info = roam_scene.build_scene(world, small, tmp_path / "scene.usdc")
    a = roam_scene.sample_view([info], small, 7, 3)
    b = roam_scene.sample_view([info], small, 7, 3)
    assert np.array_equal(a.camera.usd_matrix(), b.camera.usd_matrix()) and a.light == b.light
    assert a.target_class == "creature"  # classes take turns: view 3 -> the fourth class
    assert worldviews.ground_hits_inside(a.camera, info.bounds)
    (tmp_path / "views").mkdir()
    usd = tmp_path / "views" / "view_0003.usda"
    worldviews.write_view(a.camera, a.light, tmp_path / "scene.usdc", usd, 7, 3)
    spec = scene_spec(usd, tmp_path / "out")  # the stock host-side reader
    assert spec["scene_index"] == 3 and spec["camera"]["focal_length_mm"] == pytest.approx(
        a.camera.focal_length_mm
    )


@pytest.mark.blender
def test_rendered_labels_line_up_with_geometry(tmp_path, blender, settings):
    """A top-down render of the glade: the class at each object's projected position matches."""
    tiles, creatures, structures = glade()
    path = fake_world(tmp_path / "w.usda", tiles, creatures, structures)

    # 1) The driver end to end on one sampled view.
    raw = yaml.safe_load((EXAMPLE / "roam.yaml").read_text())
    raw.update(width=160, height=120)
    raw["views"].update(margin_tiles=1, distance=[3, 6], pitch_deg=[-80, -60])
    (tmp_path / "small.yaml").write_text(yaml.safe_dump(raw))
    out = tmp_path / "run"
    roam_synth.main([str(path), "--config", str(tmp_path / "small.yaml"), "--out", str(out),
                     "--views", "1", "--samples", "4", "--threads", str(THREADS),
                     "--blender", blender])  # fmt: skip
    assert validate_coco(out / "annotations.json") == []
    meta = json.loads((out / "meta.json").read_text())
    assert meta["throughput"]["images"] == 1 and meta["render"]["samples"] == 4

    # 2) A known camera straight above the glade, through the same writer and worker.
    world = roam_scene.read_world(path)
    info = roam_scene.build_scene(world, settings, tmp_path / "s" / "scene.usdc")
    from usd_synth.camera import PinholeCamera
    from usd_synth.config import ImageConfig, RenderConfig
    from usd_synth.labels import read_instances
    from usd_synth.passes import read_passes
    from usd_synth.render import render_batch

    eye = np.array([12.0, -12.0, 30.0])
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

    def cls_at(tx, ty, z):  # Roam tile (tx, ty), height z
        uv, _ = cam.project(np.array([[tx + 0.5, -(ty + 0.5), z]]))
        return names[sem[int(uv[0][1]), int(uv[0][0])]]

    assert cls_at(5, 5, 1.6) == "tree"  # trunk
    assert cls_at(4, 5, 1.6) == "tree"  # canopy
    assert cls_at(14, 4, 1.1) == "stone"
    assert cls_at(17, 9, 1.1) == "ore"
    assert cls_at(8, 10, 0.66) == "creature"
    assert cls_at(11, 15, 0.8) == "structure"  # the chest on the floor
    assert cls_at(10, 15, 0.2) == "structure"  # the bare floor tile
    assert cls_at(18, 18, 0.4) == "food"
    assert cls_at(3, 16, 0.14) == "-"  # leaf cover is not a tree
    assert cls_at(20, 2, 0.1) == "-"  # plain grass


def test_leaves_without_canopy_class_are_unlabelled(world, settings, tmp_path):
    """With no ground cover configured, a stray leaf stays a (dropped) canopy entity."""
    bare = settings.model_copy(update={"ground_cover": []})
    kept = roam_scene.tidy(world.entities, bare)
    assert not any(e.tile == (3, -17) and e.entity_class == "Leaves" for e in kept)
