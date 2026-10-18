"""Turn a roam-usd world stage into usd-synth-data render scenes.

Host-side half of the Roam example (``roam_synth.py`` is the driver). It reads a stage written
by `roam-usd`_ (``roam-usd export`` of a save, or ``roam-usd generate`` from a seed): Z-up,
one tile = 1 × 1 unit, ``/World/Surface/Room_<x>_<y>`` rooms, each with a ``Ground`` quad, a
``Tiles`` ``PointInstancer`` whose prototypes carry ``customData["roam:entityClass"]``, and
individual prims under ``Creatures`` (flat sprite cards) and ``Structures`` (chests, crops).
Only the geometry and the per-class ``displayColor`` (roam-usd's catalog colours) are used; the
sprite materials ``roam-usd generate`` binds to creature cards are ignored, so no Roam art
reaches the render.

It writes:

* ``scene_<k>.usdc`` — the surface level as plain meshes in the usd-synth stage layout. Every
  connected group of tiles of a labelled class (a tree or a copse, a stone outcrop, an ore
  vein, the home) becomes one ``/World/Objects/inst_NNNNN`` mesh with the usual
  ``userProperties:usd_synth:*`` attributes; classes marked ``per_entity`` (creatures, items)
  get one instance per entity. Everything else (grass, the room grounds) is unlabelled
  ``/World/Env`` geometry.
* ``views/view_NNNN.usda`` — through ``worldviews.write_view``.

Creature cards have no thickness in roam-usd; here each becomes a small box in the card's
catalog colour (``creature_box``), so a creature has a body to see from an oblique camera.
"""

from __future__ import annotations

import fnmatch
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from pxr import Sdf, Tf, Usd, UsdGeom, Vt
from pydantic import BaseModel, ConfigDict, Field

from usd_synth.camera import PinholeCamera
from usd_synth.seeds import stage_rng

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import worldviews  # noqa: E402

ENTITY_CLASS = "roam:entityClass"

# ---------------------------------------------------------------------------------------------
# Settings


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EntityClass(_Model):
    name: str
    entities: list[str] = Field(min_length=1, description="fnmatch patterns on entityClass")
    connectivity: int = Field(8, description="4 (edges) or 8 (edges and corners) on the tiles")
    per_entity: bool = False  # one instance per entity instead of per connected group
    # entities that form the canopy of one ``entities`` trunk on the four tiles around it;
    # with a canopy, every trunk is one instance together with its canopy
    canopy: list[str] = []


class CreatureBox(_Model):
    width: float = 0.7
    height: float = 0.55


class ViewSettings(_Model):
    margin_tiles: float = 3.0  # cameras stay this far inside the world's edge
    distance: tuple[float, float] = (6.0, 22.0)  # horizontal camera-to-target distance
    yaw_deg: tuple[float, float] = (0.0, 360.0)
    pitch_deg: tuple[float, float] = (-75.0, -35.0)  # negative = looking down
    aim_jitter_tiles: float = 2.0
    min_height: float = 3.0  # camera height above the ground plane
    focal_length_mm: tuple[float, float] = (20.0, 35.0)
    sensor_width_mm: float = 36.0
    sun_elevation_deg: tuple[float, float] = (12.0, 75.0)
    sun_azimuth_deg: tuple[float, float] = (0.0, 360.0)
    sun_intensity: tuple[float, float] = (3.5, 6.0)
    sky_intensity: tuple[float, float] = (0.8, 1.3)


class Settings(_Model):
    width: int = 640
    height: int = 480
    samples: int = 64
    denoise: bool = True
    min_visible_pixels: int = 20
    ground_color: tuple[float, float, float] = (0.42, 0.34, 0.24)  # sRGB, our own
    creature_box: CreatureBox = CreatureBox()
    ground_cover: list[str] = []  # canopy entities away from any trunk: flat, unlabelled litter
    classes: list[EntityClass]
    views: ViewSettings = ViewSettings()


def load_settings(path: str | Path) -> Settings:
    return Settings.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------------------------
# Reading the stage


@dataclass
class Entity:
    entity_class: str
    tile: tuple[int, int]  # global tile (floor of world X, floor of world Y)
    room: str  # room prim name
    local: np.ndarray  # (P, 3) points relative to ``base``
    base: np.ndarray  # (3,) world position of the entity's origin (tile centre, stack height)
    counts: np.ndarray  # face vertex counts
    indices: np.ndarray  # face vertex indices
    color: tuple[float, float, float]  # catalog display colour (sRGB)
    creature: bool = False
    cover: bool = False  # ground cover: rendered, never labelled

    @property
    def points(self) -> np.ndarray:
        return self.local + self.base


@dataclass
class World:
    path: Path
    entities: list[Entity]
    rooms: list[tuple[str, tuple[float, float, float, float]]]  # name, (x0, y0, x1, y1)
    layer_data: dict = field(default_factory=dict)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        r = np.array([b for _, b in self.rooms])
        lo, hi = r[:, :2].min(axis=0), r[:, 2:].max(axis=0)
        return float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])


def _apply(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ m[:3, :3] + m[3, :3]


def _mesh_data(prim: Usd.Prim):
    mesh = UsdGeom.Mesh(prim)
    pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
    idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
    dc = mesh.GetDisplayColorAttr().Get()
    color = tuple(float(c) for c in dc[0]) if dc else (1.0, 0.0, 1.0)
    return pts, counts, idx, color


def _tile_of(m: np.ndarray, local=(0.0, 0.0, 0.0)) -> tuple[int, int]:
    p = _apply(m, np.asarray([local], dtype=np.float64))[0]
    return int(math.floor(p[0])), int(math.floor(p[1]))


def _surface(stage: Usd.Stage) -> Usd.Prim:
    world = stage.GetPrimAtPath("/World")
    for child in world.GetChildren() if world else []:
        if child.GetCustomDataByKey("roam:z") == 0 or child.GetName() == "Surface":
            return child
    raise ValueError("no surface level (/World/Surface) in this stage (not a roam-usd export?)")


def read_world(path: str | Path) -> World:
    """Every surface entity of a roam-usd stage, as world-space geometry."""
    stage = Usd.Stage.Open(str(path))
    cache = UsdGeom.XformCache()
    entities: list[Entity] = []
    rooms = []
    for room in _surface(stage).GetChildren():
        if not room.GetName().startswith("Room_"):
            continue
        name = room.GetName()
        ground = room.GetChild("Ground")
        if ground:
            m = np.asarray(cache.GetLocalToWorldTransform(ground), dtype=np.float64)
            pts = _apply(m, _mesh_data(ground)[0])
            rooms.append((name, (*pts[:, :2].min(axis=0), *pts[:, :2].max(axis=0))))
        tiles = room.GetChild("Tiles")
        if tiles and tiles.IsA(UsdGeom.PointInstancer):
            inst = UsdGeom.PointInstancer(tiles)
            m = np.asarray(cache.GetLocalToWorldTransform(tiles), dtype=np.float64)
            protos = []
            for target in inst.GetPrototypesRel().GetTargets():
                proto = stage.GetPrimAtPath(target)
                protos.append((str(proto.GetCustomDataByKey(ENTITY_CLASS)), *_mesh_data(proto)))
            positions = np.asarray(inst.GetPositionsAttr().Get(), dtype=np.float64)
            for pi, pos in zip(inst.GetProtoIndicesAttr().Get(), positions, strict=True):
                cls, pts, counts, idx, color = protos[pi]
                base = _apply(m, pos[None, :])[0]
                local = pts @ m[:3, :3]  # roam-usd rooms and levels are pure translations
                tile = _tile_of(m, pos)
                entities.append(Entity(cls, tile, name, local, base, counts, idx, color))
        for scope, creature in (("Creatures", True), ("Structures", False)):
            parent = room.GetChild(scope)
            for prim in parent.GetChildren() if parent else []:
                if not prim.IsA(UsdGeom.Mesh):
                    continue
                m = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=np.float64)
                pts, counts, idx, color = _mesh_data(prim)
                cls = str(prim.GetCustomDataByKey(ENTITY_CLASS))
                base = m[3, :3].copy()
                entities.append(
                    Entity(
                        cls, _tile_of(m), name, pts @ m[:3, :3], base, counts, idx, color, creature
                    )  # fmt: skip
                )
    if not rooms:
        raise ValueError(f"{path}: no rooms on the surface level")
    return World(Path(path), entities, rooms, dict(stage.GetRootLayer().customLayerData))


def _matches(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


def classify(entity: Entity, settings: Settings) -> int:
    """1-based index of the first class whose patterns match, or 0 (unlabelled).

    Canopy entities (``Leaves``) count as their class's; ground cover is never labelled.
    """
    if entity.cover:
        return 0
    for k, c in enumerate(settings.classes, start=1):
        if _matches(entity.entity_class, c.entities):
            return k
        if c.canopy and _matches(entity.entity_class, c.canopy):
            return k
    return 0


# ---------------------------------------------------------------------------------------------
# Geometry

_BOX_FACES = np.array(
    [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)],
    dtype=np.int32,
)


def creature_box(card: Entity, box: CreatureBox) -> Entity:
    """A box standing where a flat creature card lies, in the card's colour."""
    return Entity(card.entity_class, card.tile, card.room, _box(box.width, box.height),
                  card.base.copy(), np.full(6, 4, dtype=np.int32), _BOX_FACES.ravel(),
                  card.color, creature=True)  # fmt: skip


def _box(width: float, height: float) -> np.ndarray:
    h = width / 2
    xy = [(-h, -h), (h, -h), (h, h), (-h, h)]
    return np.array([(x, y, 0.0) for x, y in xy] + [(x, y, height) for x, y in xy])


LITTER_HEIGHT = 0.04


def tidy(entities: list[Entity], settings: Settings) -> list[Entity]:
    """The entities as rendered: creature cards as boxes, canopies resolved, tiles re-stacked.

    roam-usd stacks every entity of a tile on top of the previous one, so a forest tile with
    leaves from the room's leaf cover and from two neighbouring trees becomes a tall column.
    Here, per tile: the first canopy entity (``Leaves``) next to a trunk of its class is that
    tree's canopy and floats at its catalog lift above the ground stack; further canopy
    entities on the tile are dropped; canopy entities with no trunk next to them are ground
    cover (``ground_cover``), drawn as flat, unlabelled litter. Everything else is re-stacked
    from the ground up in its original order.
    """
    out = [creature_box(e, settings.creature_box) if e.creature else e for e in entities]
    canopy_of = {}  # canopy pattern list -> trunk tiles
    for c in settings.classes:
        if c.canopy:
            trunks = {e.tile for e in out if _matches(e.entity_class, c.entities)}
            canopy_of[tuple(c.canopy)] = trunks
    by_tile: dict[tuple[int, int], list[Entity]] = {}
    for e in out:
        by_tile.setdefault(e.tile, []).append(e)
    kept: list[Entity] = []
    for (tx, ty), stack in by_tile.items():
        stack.sort(key=lambda e: float(e.base[2]))  # stable: original order on ties
        z0 = min(float(e.base[2]) for e in stack)
        z = z0
        has_canopy = has_litter = False
        for e in stack:
            canopy = next((t for p, t in canopy_of.items() if _matches(e.entity_class, list(p))),
                          None)  # fmt: skip
            if canopy is not None:
                near = any((tx + dx, ty + dy) in canopy
                           for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))  # fmt: skip
                if near and not has_canopy:
                    has_canopy = True
                    e.base = np.array([e.base[0], e.base[1], z0 + 0.1])
                    kept.append(e)
                elif not near and not has_litter and _matches(e.entity_class,
                                                              settings.ground_cover):  # fmt: skip
                    has_litter = True
                    e.local = _box(1.0, LITTER_HEIGHT)
                    e.counts, e.indices = np.full(6, 4, dtype=np.int32), _BOX_FACES.ravel()
                    e.base = np.array([e.base[0], e.base[1], z])
                    e.cover = True
                    z += LITTER_HEIGHT
                    kept.append(e)
                continue
            e.base = np.array([e.base[0], e.base[1], z])
            z += max(float(e.local[:, 2].max()), 0.0)
            kept.append(e)
    return kept


def components(tiles: list[tuple[int, int]], connectivity: int = 8) -> np.ndarray:
    """Connected-component label of each tile (duplicates share a label), numbered in order of
    first appearance."""
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if connectivity == 8:
        offsets += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    first: dict[tuple[int, int], int] = {}
    for i, t in enumerate(tiles):
        first.setdefault(t, i)
    comp_of: dict[tuple[int, int], int] = {}
    k = 0
    for t in tiles:
        if t in comp_of:
            continue
        comp_of[t] = k
        stack = [t]
        while stack:
            x, y = stack.pop()
            for dx, dy in offsets:
                n = (x + dx, y + dy)
                if n in first and n not in comp_of:
                    comp_of[n] = k
                    stack.append(n)
        k += 1
    return np.array([comp_of[t] for t in tiles], dtype=np.int64)


def _merge(parts: list[Entity], drop_down_faces: bool = True):
    """Points, counts, indices and per-face look key of several entities in one mesh."""
    pts, counts, idx, look = [], [], [], []
    base = 0
    for e in parts:
        off = 0
        for c in e.counts.tolist():
            face = e.indices[off : off + c]
            off += c
            if drop_down_faces and c >= 3:
                a, b, d = e.points[face[0]], e.points[face[1]], e.points[face[2]]
                n = np.cross(b - a, d - a)
                if n[2] < -0.9 * np.linalg.norm(n):  # faces down: never seen from above
                    continue
            counts.append(c)
            idx.extend((face + base).tolist())
            look.append((e.entity_class, e.color))
        pts.append(e.points)
        base += len(e.points)
    return np.concatenate(pts), np.array(counts, np.int32), np.array(idx, np.int32), look


def _write_mesh(stage, path, parts: list[Entity], looks) -> tuple[UsdGeom.Mesh, int]:
    pts, counts, idx, look = _merge(parts)
    return worldviews.write_mesh(stage, path, pts, counts, idx, look, looks), len(counts)


@dataclass
class Instance:
    instance_id: int
    category: str
    entities: int
    tiles: int
    faces: int


@dataclass
class SceneInfo:
    path: Path
    world: Path
    categories: list[str]
    instances: list[Instance]
    entities: int
    faces: int
    bounds: tuple[float, float, float, float]  # x0, y0, x1, y1 (world units)
    entities_by_class: dict[str, int] = field(default_factory=dict)
    # per class: tile centres (x, y, top z) of its instances' entities, the view targets
    targets: dict[str, np.ndarray] = field(default_factory=dict)
    max_height: float = 0.0


def build_scene(world: World, settings: Settings, out_path: str | Path) -> SceneInfo:
    """Write ``scene.usdc`` (meshes, labelled instances, materials) and return its summary."""
    entities = tidy(world.entities, settings)
    cls = np.array([classify(e, settings) for e in entities], dtype=np.int32)

    categories = [c.name for c in settings.classes]
    stage = worldviews.new_scene(Path(out_path), categories)
    looks = worldviews.Looks(stage)

    instances: list[Instance] = []
    targets: dict[str, list] = {c: [] for c in categories}
    faces_total = 0
    for k, ec in enumerate(settings.classes, start=1):
        sel = np.flatnonzero(cls == k)
        if len(sel) == 0:
            continue
        if ec.canopy:  # one instance per trunk, with the canopy tiles nearest to it
            trunks = [i for i in sel.tolist() if _matches(entities[i].entity_class, ec.entities)]
            tile_trunk = {}
            for i in trunks:
                tile_trunk.setdefault(entities[i].tile, i)
            groups = {i: [i] for i in trunks}
            for i in sel.tolist():
                if i in groups:
                    continue
                tx, ty = entities[i].tile
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    t = tile_trunk.get((tx + dx, ty + dy))
                    if t is not None:
                        groups[t].append(i)
                        break
            groups = list(groups.values())
        elif ec.per_entity:
            groups = [[i] for i in sel.tolist()]
        else:
            comp = components([entities[i].tile for i in sel], ec.connectivity)
            groups = [sel[comp == c].tolist() for c in range(int(comp.max()) + 1)]
        for members in groups:
            parts = [entities[i] for i in members]
            iid = len(instances) + 1
            mesh, nf = _write_mesh(stage, f"/World/Objects/inst_{iid:05d}", parts, looks)
            prim = mesh.GetPrim()
            worldviews.label_instance(prim, iid, k, ec.name)
            tiles = sorted({p.tile for p in parts})
            names = Vt.StringArray(sorted({p.entity_class for p in parts}))
            worldviews.set_attr(prim, "entity_classes", Sdf.ValueTypeNames.StringArray, names)
            instances.append(Instance(iid, ec.name, len(parts), len(tiles), nf))
            faces_total += nf
            for p in parts:
                targets[ec.name].append(
                    (p.tile[0] + 0.5, p.tile[1] + 0.5, float(p.points[:, 2].max()))
                )

    by_room: dict[str, list[Entity]] = {}
    for i in np.flatnonzero(cls == 0).tolist():
        by_room.setdefault(entities[i].room, []).append(entities[i])
    for name, (x0, y0, x1, y1) in world.rooms:
        ground = Entity("Ground", (0, 0), name,
                        np.array([(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0)], float),
                        np.zeros(3), np.array([4], np.int32), np.arange(4, dtype=np.int32),
                        settings.ground_color)  # fmt: skip
        _, nf = _write_mesh(stage, f"/World/Env/{Tf.MakeValidIdentifier(name)}",
                            [ground, *by_room.get(name, [])], looks)  # fmt: skip
        faces_total += nf
    stage.GetRootLayer().Save()

    heights = [float(e.points[:, 2].max()) for e in entities]
    return SceneInfo(
        Path(out_path), world.path, categories, instances, len(entities), faces_total,
        world.bounds,
        {c: int((cls == k).sum()) for k, c in enumerate(categories, start=1)},
        {c: np.array(v).reshape(-1, 3) for c, v in targets.items()},
        max(heights, default=0.0),
    )  # fmt: skip


# ---------------------------------------------------------------------------------------------
# Views


@dataclass(frozen=True)
class View:
    index: int
    world: int  # index of the scene this view looks at
    camera: PinholeCamera
    light: worldviews.Lighting
    target_class: str
    target: tuple[float, float, float]
    yaw_deg: float
    pitch_deg: float
    distance: float


def view_plan(scenes: list[SceneInfo], index: int) -> tuple[int, str]:
    """(scene, class) of view ``index``: classes take turns, and so do the scenes that have
    that class, so rare classes get as many views as common ones."""
    present = [c for c in scenes[0].categories if any(len(s.targets[c]) for s in scenes)]
    if not present:
        raise RuntimeError("no labelled entity in any world")
    cls = present[index % len(present)]
    having = [k for k, s in enumerate(scenes) if len(s.targets[cls])]
    return having[(index // len(present)) % len(having)], cls


def sample_view(
    scenes: list[SceneInfo], settings: Settings, seed: int, index: int, attempts=4000
) -> View:
    """Random camera, sun and sky for view ``index``, aimed near an entity of its class."""
    vs = settings.views
    k, cls = view_plan(scenes, index)
    info = scenes[k]
    rng = stage_rng(seed, index, "camera")
    x0, y0, x1, y1 = info.bounds
    m = vs.margin_tiles
    pool = info.targets[cls]
    for _ in range(attempts):
        tx, ty, tz = pool[int(rng.integers(len(pool)))]
        dist = float(rng.uniform(*vs.distance))
        yaw = float(rng.uniform(*vs.yaw_deg))
        pitch = float(rng.uniform(*vs.pitch_deg))
        focal = float(rng.uniform(*vs.focal_length_mm))
        jitter = rng.uniform(-1.0, 1.0, size=2) * vs.aim_jitter_tiles
        yr = math.radians(yaw)
        bearing = np.array([math.sin(yr), math.cos(yr)])  # yaw 0 = north (+Y)
        ex, ey = np.array([tx, ty]) - bearing * dist
        ez = tz + dist * math.tan(math.radians(-pitch))
        if not (x0 + m <= ex <= x1 - m and y0 + m <= ey <= y1 - m):
            continue
        if ez < max(vs.min_height, info.max_height + 1.0):
            continue
        aim = np.array([tx + jitter[0], ty + jitter[1], tz])
        camera = PinholeCamera.look_at(
            np.array([ex, ey, ez]), aim, focal, vs.sensor_width_mm, settings.width, settings.height
        )
        if not worldviews.ground_hits_inside(camera, info.bounds):
            continue
        break
    else:
        raise RuntimeError(f"view {index}: no camera pose found in {attempts} tries")
    light = worldviews.sample_lighting(
        seed, index, vs.sun_elevation_deg, vs.sun_azimuth_deg, vs.sun_intensity, vs.sky_intensity
    )
    target = (round(float(tx), 3), round(float(ty), 3), round(float(tz), 3))
    return View(
        index, k, camera, light, cls, target, round(yaw, 3), round(pitch, 3), round(dist, 3)
    )
