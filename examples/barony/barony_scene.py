"""Turn Barony game states into usd-synth-data render scenes.

Host-side half of the Barony example (``barony_synth.py`` is the driver). Barony is a
turn-based strategy game on a small grid: each tile is a ``CASTLE``, a ``VILLAGE`` or
``EMPTY`` and has an owner (0 = nobody, 1 = the player, 2 = the AI), and armies of some number
of soldiers stand on tiles. Its backend serves exactly that as JSON (``capture_states.py``
saves it); Barony itself draws the board in 2D and has no 3D models.

The 3D board here is this example's own: every tile is a slab, a castle is walls, corner
towers, a keep and a banner, a village is a few houses with gable roofs, and an army is a
formation of soldier figures behind a standard, sized by its soldier count. Owners are told
apart by colour (banners, soldiers, tile tint). Empty tiles get unlabelled scenery (trees,
rocks, fields) so the background is not uniform. All geometry and colours are generated from
the state and a per-board random stream; nothing is taken from Barony.

Every castle, village and army is one labelled ``/World/Objects/inst_NNNNN`` mesh; the board
and the scenery are unlabelled ``/World/Env`` meshes.
"""

from __future__ import annotations

import json
import math
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from pxr import Sdf
from pydantic import BaseModel, ConfigDict

from usd_synth.camera import PinholeCamera
from usd_synth.seeds import stage_rng

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import worldviews  # noqa: E402

CATEGORIES = ["castle", "village", "army"]

# ---------------------------------------------------------------------------------------------
# Settings


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ViewSettings(_Model):
    margin_tiles: float = 0.5  # cameras stay this far inside the board's edge
    distance: tuple[float, float] = (7.0, 24.0)  # horizontal camera-to-target distance (m)
    yaw_deg: tuple[float, float] = (0.0, 360.0)
    pitch_deg: tuple[float, float] = (-65.0, -28.0)  # negative = looking down
    aim_jitter_m: float = 3.0
    min_height: float = 3.0
    focal_length_mm: tuple[float, float] = (24.0, 45.0)
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
    min_visible_pixels: int = 30
    tile_m: float = 4.0  # edge of one board tile, metres
    soldiers_per_figure: int = 3  # an army of n soldiers shows ceil(n / this) figures
    max_figures: int = 12
    scenery: float = 0.45  # chance that an empty tile without an army gets scenery
    surround_m: float = 30.0  # meadow around the board
    owner_colors: dict[int, tuple[float, float, float]] = {
        0: (0.85, 0.82, 0.70),
        1: (0.20, 0.42, 0.90),
        2: (0.86, 0.20, 0.18),
    }
    views: ViewSettings = ViewSettings()


def load_settings(path: str | Path) -> Settings:
    return Settings.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------------------------
# The state


@dataclass
class Army:
    x: int
    y: int
    soldiers: int
    owner: int
    destination: tuple[int, int] | None


@dataclass
class Board:
    path: Path
    width: int
    height: int
    types: list[list[str]]  # [x][y]: CASTLE | VILLAGE | EMPTY
    owners: list[list[int]]  # [x][y]
    armies: list[Army]
    tick: int = 0

    @property
    def key(self) -> int:
        """A stable number for this board, which seeds its scenery."""
        return zlib.crc32(self.path.name.encode())


def read_state(path: str | Path) -> Board:
    """A Barony game state as served by ``GET /api/session/state``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    grid = data["grid"]
    if not grid or not grid[0]:
        raise ValueError(f"{path}: empty grid (not a Barony game state?)")
    types = [[str(t["type"]) for t in col] for col in grid]
    owners = [[int(t.get("ownerId", 0)) for t in col] for col in grid]
    armies = []
    for a in data.get("armies", []):
        dest = None
        if a.get("moving") and a.get("destinationX") is not None:
            dest = (int(a["destinationX"]), int(a["destinationY"]))
        armies.append(Army(int(a["x"]), int(a["y"]), int(a["soldiers"]), int(a["playerId"]), dest))
    return Board(Path(path), len(grid), len(grid[0]), types, owners, armies,
                 int(data.get("tickCount", 0)))  # fmt: skip


# ---------------------------------------------------------------------------------------------
# Geometry: a list of faces with one look each


class Parts:
    """Accumulates faces (quads and triangles) with a (name, sRGB colour) look per face."""

    def __init__(self):
        self.points: list[np.ndarray] = []
        self.counts: list[int] = []
        self.indices: list[int] = []
        self.looks: list[tuple[str, tuple[float, float, float]]] = []
        self.n = 0

    def add(self, pts: np.ndarray, faces: list[tuple[int, ...]], look: str, rgb) -> None:
        rgb = tuple(round(float(c), 4) for c in rgb)
        self.points.append(np.asarray(pts, dtype=np.float64))
        for f in faces:
            self.counts.append(len(f))
            self.indices.extend(i + self.n for i in f)
            self.looks.append((look, rgb))
        self.n += len(pts)

    def box(self, c, size, rot: float, look: str, rgb, z0: float, bottom=False) -> None:
        """Axis-aligned box (before rotation by ``rot`` about Z) standing at height ``z0``."""
        sx, sy, sz = size[0] / 2, size[1] / 2, size[2]
        xy = _rot(np.array([(-sx, -sy), (sx, -sy), (sx, sy), (-sx, sy)]), rot) + c[:2]
        pts = np.concatenate([np.c_[xy, np.full(4, z0)], np.c_[xy, np.full(4, z0 + sz)]])
        faces = [(4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        if bottom:
            faces.append((0, 3, 2, 1))
        self.add(pts, faces, look, rgb)

    def roof(self, c, size, height: float, rot: float, look: str, rgb, z0: float) -> None:
        """Gable roof over a ``size`` (x, y) footprint, ridge along local X."""
        sx, sy = size[0] / 2, size[1] / 2
        xy = _rot(np.array([(-sx, -sy), (sx, -sy), (sx, sy), (-sx, sy), (-sx, 0), (sx, 0)]), rot)
        xy = xy + c[:2]
        z = np.array([z0] * 4 + [z0 + height] * 2)
        pts = np.c_[xy, z]
        self.add(pts, [(0, 1, 5, 4), (2, 3, 4, 5), (1, 2, 5), (3, 0, 4)], look, rgb)

    def arrays(self):
        return (np.concatenate(self.points), np.array(self.counts), np.array(self.indices),
                self.looks)  # fmt: skip

    @property
    def faces(self) -> int:
        return len(self.counts)


def _rot(xy: np.ndarray, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return xy @ np.array([[c, s], [-s, c]])


STONE = (0.64, 0.62, 0.58)
STONE_DARK = (0.50, 0.48, 0.45)
PLASTER = (0.88, 0.83, 0.70)
ROOFS = [(0.62, 0.26, 0.17), (0.48, 0.32, 0.20), (0.40, 0.36, 0.33), (0.70, 0.55, 0.30)]
WOOD = (0.42, 0.28, 0.15)
SKIN = (0.92, 0.76, 0.60)
GRASS = (0.36, 0.56, 0.25)
MEADOW = (0.30, 0.47, 0.22)
SLAB = 0.25  # tile slab height


def castle(p: Parts, c: np.ndarray, rot: float, banner, z0: float) -> None:
    half = 1.45
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        corner = c + _rot(np.array([[sx * half, sy * half]]), rot)[0]
        p.box(corner, (0.85, 0.85, 2.1), rot, "stone", STONE, z0)
        p.box(corner, (1.0, 1.0, 0.18), rot, "stone_dark", STONE_DARK, z0 + 2.1)
    for side in range(4):
        a = rot + side * math.pi / 2
        mid = c + _rot(np.array([[0.0, -half]]), a)[0]
        width = 2 * half - 0.85
        if side == 0:  # the gate side: two wall pieces and a dark gate
            for d in (-1, 1):
                piece = mid + _rot(np.array([[d * (width / 4 + 0.25), 0.0]]), a)[0]
                p.box(piece, (width / 2 - 0.5, 0.35, 1.35), a, "stone", STONE, z0)
            p.box(mid, (1.0, 0.36, 0.25), a, "stone", STONE, z0 + 1.1)
            p.box(mid, (0.95, 0.30, 1.1), a, "gate", WOOD, z0)
        else:
            p.box(mid, (width, 0.35, 1.35), a, "stone", STONE, z0)
    p.box(c, (1.4, 1.4, 2.8), rot, "stone", STONE, z0)
    p.box(c, (1.6, 1.6, 0.2), rot, "stone_dark", STONE_DARK, z0 + 2.8)
    pole = c + _rot(np.array([[0.4, 0.4]]), rot)[0]
    p.box(pole, (0.07, 0.07, 1.3), rot, "pole", WOOD, z0 + 3.0)
    flag = pole + _rot(np.array([[0.33, 0.0]]), rot)[0]
    p.box(flag, (0.6, 0.04, 0.4), rot, "banner", banner, z0 + 3.85)


def village(p: Parts, c: np.ndarray, rng: np.random.Generator, tile: float, banner, z0: float):
    cells = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    order = rng.permutation(4)[: int(rng.integers(3, 5))]
    for k in order:
        cx, cy = cells[k]
        pos = c + np.array([cx, cy]) * tile * 0.22 + rng.uniform(-0.25, 0.25, 2)
        w, d, h = rng.uniform(0.9, 1.25), rng.uniform(0.65, 0.85), rng.uniform(0.55, 0.8)
        rot = float(rng.uniform(0, math.pi))
        p.box(pos, (w, d, h), rot, "plaster", PLASTER, z0)
        roof = ROOFS[int(rng.integers(len(ROOFS)))]
        p.roof(pos, (w + 0.16, d + 0.2), float(rng.uniform(0.4, 0.6)), rot, "roof", roof, z0 + h)
    if banner is not None:  # an owned village flies its owner's banner
        p.box(c, (0.06, 0.06, 1.6), 0.0, "pole", WOOD, z0)
        p.box(c + np.array([0.28, 0.0]), (0.5, 0.04, 0.32), 0.0, "banner", banner, z0 + 1.25)


def army(p: Parts, c: np.ndarray, figures: int, facing: float, colour, z0: float) -> None:
    cols = min(figures, 4)
    rows = math.ceil(figures / cols)
    spacing = 0.5
    for i in range(figures):
        r, k = divmod(i, cols)
        off = np.array([(k - (cols - 1) / 2) * spacing, (r - (rows - 1) / 2) * spacing])
        pos = c + _rot(off[None, :], facing)[0]
        p.box(pos, (0.26, 0.18, 0.5), facing, "tunic", colour, z0)
        p.box(pos, (0.16, 0.16, 0.16), facing, "skin", SKIN, z0 + 0.5)
        spear = pos + _rot(np.array([[0.17, 0.0]]), facing)[0]
        p.box(spear, (0.04, 0.04, 0.95), facing, "spear", WOOD, z0)
    front = c + _rot(np.array([[0.0, -(rows / 2 + 0.3) * spacing]]), facing)[0]
    p.box(front, (0.06, 0.06, 1.5), facing, "pole", WOOD, z0)
    p.box(front + _rot(np.array([[0.25, 0.0]]), facing)[0], (0.45, 0.04, 0.32), facing,
          "standard", colour, z0 + 1.15)  # fmt: skip


def scenery(p: Parts, c: np.ndarray, rng: np.random.Generator, tile: float, z0: float) -> None:
    kind = rng.choice(["trees", "trees", "rocks", "field"])
    if kind == "trees":
        for _ in range(int(rng.integers(1, 4))):
            pos = c + rng.uniform(-0.35, 0.35, 2) * tile
            h = float(rng.uniform(0.5, 0.9))
            p.box(pos, (0.2, 0.2, h), 0.0, "trunk", WOOD, z0)
            green = (0.16, float(rng.uniform(0.36, 0.48)), 0.14)
            size = float(rng.uniform(0.9, 1.3))
            for j, s in enumerate((1.0, 0.7, 0.4)):
                p.box(pos, (size * s, size * s, 0.45), j * 0.4, "leaves", green, z0 + h + j * 0.4)
    elif kind == "rocks":
        for _ in range(int(rng.integers(1, 4))):
            pos = c + rng.uniform(-0.35, 0.35, 2) * tile
            s = float(rng.uniform(0.3, 0.7))
            p.box(pos, (s, s * 0.8, s * 0.6), float(rng.uniform(0, math.pi)), "rock",
                  (0.55, 0.54, 0.52), z0)  # fmt: skip
    else:
        rot = float(rng.choice([0.0, math.pi / 2]))
        crop = [(0.78, 0.68, 0.32), (0.50, 0.38, 0.22)][int(rng.integers(2))]
        for k in range(5):
            row = c + _rot(np.array([[0.0, (k - 2) * 0.55]]), rot)[0]
            p.box(row, (tile * 0.75, 0.35, 0.08), rot, "crop", crop, z0)


# ---------------------------------------------------------------------------------------------
# Scene


@dataclass
class Instance:
    instance_id: int
    category: str
    owner: int
    tile: tuple[int, int]
    faces: int
    soldiers: int = 0


@dataclass
class SceneInfo:
    path: Path
    board: Path
    categories: list[str]
    instances: list[Instance]
    faces: int
    bounds: tuple[float, float, float, float]  # the board (x0, y0, x1, y1), metres
    targets: dict[str, np.ndarray] = field(default_factory=dict)  # per class: (x, y, z) rows
    size: tuple[int, int] = (0, 0)


def tile_centre(x: int, y: int, tile: float) -> np.ndarray:
    return np.array([(x + 0.5) * tile, (y + 0.5) * tile])


def army_spots(board: Board, tile: float) -> list[np.ndarray]:
    """Where each army stands: the middle of an empty tile, or on the next tile towards the
    board's middle when a castle or village fills its own; several armies on one tile stand
    side by side."""
    spots = []
    on_tile: dict[tuple[int, int], int] = {}
    for a in board.armies:
        n = on_tile.get((a.x, a.y), 0)
        on_tile[(a.x, a.y)] = n + 1
        c = tile_centre(a.x, a.y, tile)
        if board.types[a.x][a.y] != "EMPTY":
            inward = 1.0 if a.y < (board.height - 1) / 2 else -1.0
            c = c + np.array([0.0, inward * tile * 0.75])
        spots.append(c + np.array([(n % 2) * 1.6 - 0.8 * (n > 0), (n // 2) * 1.4]))
    return spots


def build_scene(board: Board, settings: Settings, out_path: str | Path) -> SceneInfo:
    """Write ``scene.usdc`` for one board and return its summary."""
    t = settings.tile_m
    rng = np.random.default_rng([board.key, 1])
    stage = worldviews.new_scene(Path(out_path), CATEGORIES)
    looks = worldviews.Looks(stage)
    colors = settings.owner_colors
    instances: list[Instance] = []
    targets: dict[str, list] = {c: [] for c in CATEGORIES}
    faces = 0

    def emit(parts: Parts, category: str, owner: int, tile_xy, soldiers=0) -> None:
        nonlocal faces
        iid = len(instances) + 1
        pts, counts, idx, lk = parts.arrays()
        mesh = worldviews.write_mesh(stage, f"/World/Objects/inst_{iid:05d}", pts, counts, idx,
                                     lk, looks)  # fmt: skip
        prim = mesh.GetPrim()
        worldviews.label_instance(prim, iid, CATEGORIES.index(category) + 1, category)
        worldviews.set_attr(prim, "owner", Sdf.ValueTypeNames.Int, owner)
        instances.append(Instance(iid, category, owner, tuple(tile_xy), parts.faces, soldiers))
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        targets[category].append(((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[2] * 0.5))
        faces += parts.faces

    spots = army_spots(board, t)
    army_tiles = {(a.x, a.y) for a in board.armies}
    army_tiles |= {(int(p[0] // t), int(p[1] // t)) for p in spots}  # no scenery under armies
    env = Parts()
    w, h = board.width * t, board.height * t
    s = settings.surround_m
    env.add(np.array([(-s, -s, 0), (w + s, -s, 0), (w + s, h + s, 0), (-s, h + s, 0)]),
            [(0, 1, 2, 3)], "meadow", MEADOW)  # fmt: skip
    for x in range(board.width):
        for y in range(board.height):
            c = tile_centre(x, y, t)
            kind, owner = board.types[x][y], board.owners[x][y]
            base = np.array(GRASS) * rng.uniform(0.92, 1.08)
            if owner:
                base = base * 0.7 + np.array(colors[owner]) * 0.3
            env.box(c, (t - 0.08, t - 0.08, SLAB), 0.0, "tile", base.clip(0, 1), 0.0)
            if kind == "CASTLE":
                parts = Parts()
                castle(parts, c, float(rng.choice([0, 1, 2, 3])) * math.pi / 2,
                       colors[owner], SLAB)  # fmt: skip
                emit(parts, "castle", owner, (x, y))
            elif kind == "VILLAGE":
                parts = Parts()
                village(parts, c, rng, t, colors[owner] if owner else None, SLAB)
                emit(parts, "village", owner, (x, y))
            elif (x, y) not in army_tiles and rng.random() < settings.scenery:
                scenery(env, c, rng, t, SLAB)

    for a, spot in zip(board.armies, spots, strict=True):
        if a.destination is not None and a.destination != (a.x, a.y):
            d = tile_centre(*a.destination, t) - tile_centre(a.x, a.y, t)
            facing = math.atan2(d[0], -d[1])  # the formation's front faces its destination
        else:
            facing = float(rng.uniform(0, 2 * math.pi))
        figures = min(
            settings.max_figures, max(1, math.ceil(a.soldiers / settings.soldiers_per_figure))
        )
        parts = Parts()
        army(parts, spot, figures, facing, colors.get(a.owner, colors[0]), SLAB)
        emit(parts, "army", a.owner, (a.x, a.y), a.soldiers)

    pts, counts, idx, lk = env.arrays()
    worldviews.write_mesh(stage, "/World/Env/board", pts, counts, idx, lk, looks)
    faces += env.faces
    stage.GetRootLayer().Save()
    return SceneInfo(
        Path(out_path), board.path, CATEGORIES, instances, faces, (0.0, 0.0, w, h),
        {c: np.array(v).reshape(-1, 3) for c, v in targets.items()}, (board.width, board.height),
    )  # fmt: skip


# ---------------------------------------------------------------------------------------------
# Views


@dataclass(frozen=True)
class View:
    index: int
    board: int  # index of the scene this view looks at
    camera: PinholeCamera
    light: worldviews.Lighting
    target_class: str
    target: tuple[float, float, float]
    yaw_deg: float
    pitch_deg: float
    distance: float


def view_plan(scenes: list[SceneInfo], index: int) -> tuple[int, str]:
    """(scene, class) of view ``index``: classes take turns, and so do the boards."""
    present = [c for c in CATEGORIES if any(len(s.targets[c]) for s in scenes)]
    cls = present[index % len(present)]
    having = [k for k, s in enumerate(scenes) if len(s.targets[cls])]
    return having[(index // len(present)) % len(having)], cls


def sample_view(
    scenes: list[SceneInfo], settings: Settings, seed: int, index: int, attempts=4000
) -> View:
    """Random camera, sun and sky for view ``index``, aimed near an object of its class.

    The camera stays over the board, and every ray of a coarse pixel grid has to land on the
    board or the meadow around it, so the sky and the edge of the meadow stay out of view.
    """
    vs = settings.views
    k, cls = view_plan(scenes, index)
    info = scenes[k]
    rng = stage_rng(seed, index, "camera")
    x0, y0, x1, y1 = info.bounds
    m = vs.margin_tiles * settings.tile_m
    s = settings.surround_m * 0.8
    pool = info.targets[cls]
    for _ in range(attempts):
        tx, ty, tz = pool[int(rng.integers(len(pool)))]
        dist = float(rng.uniform(*vs.distance))
        yaw = float(rng.uniform(*vs.yaw_deg))
        pitch = float(rng.uniform(*vs.pitch_deg))
        focal = float(rng.uniform(*vs.focal_length_mm))
        jitter = rng.uniform(-1.0, 1.0, size=2) * vs.aim_jitter_m
        yr = math.radians(yaw)
        ex, ey = np.array([tx, ty]) - np.array([math.sin(yr), math.cos(yr)]) * dist
        ez = tz + dist * math.tan(math.radians(-pitch))
        if not (x0 + m <= ex <= x1 - m and y0 + m <= ey <= y1 - m) or ez < vs.min_height:
            continue
        aim = np.array([tx + jitter[0], ty + jitter[1], tz])
        camera = PinholeCamera.look_at(
            np.array([ex, ey, ez]), aim, focal, vs.sensor_width_mm, settings.width, settings.height
        )
        if not worldviews.ground_hits_inside(camera, (x0 - s, y0 - s, x1 + s, y1 + s)):
            continue
        break
    else:
        raise RuntimeError(f"view {index}: no camera pose found in {attempts} tries")
    light = worldviews.sample_lighting(
        seed, index, vs.sun_elevation_deg, vs.sun_azimuth_deg, vs.sun_intensity, vs.sky_intensity
    )
    target = (round(float(tx), 3), round(float(ty), 3), round(float(tz), 3))
    return View(index, k, camera, light, cls, target, round(yaw, 3), round(pitch, 3),
                round(dist, 3))  # fmt: skip
