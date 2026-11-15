"""Turn a heightmap world package into usd-synth-data render scenes.

Host-side half of the heightmap-world example (``hm_synth.py`` is the driver). A *heightmap
world package* is a folder that describes a large block world from above, one pixel per
``blocks_per_px`` × ``blocks_per_px`` blocks (the format is described in this example's
README):

* ``world.json`` — the grid (origin, pixel size, raster size, ``y_offset``) and the ground
  class names;
* ``height.png`` — 16-bit grey, top surface ``y + y_offset`` (0 = not surveyed);
* ``ground.png`` — 8-bit grey, an index into the class names (0 = void);
* ``vectors.json`` (optional) — polygons (lakes, forests, zones) in block coordinates.

It writes ``scene.usdc`` in the usd-synth stage layout (Z-up, metres, one block = 1 m):

* the terrain as a heightfield mesh, one quad per pixel. Every connected region of one
  labelled ground class (a lake, a forest, a snowfield, a town) becomes one
  ``/World/Objects/inst_NNNNN`` mesh with the usual ``userProperties:usd_synth:*``
  attributes, so the stock render worker gives it its own Object Index. Where a vector
  feature (a lake polygon, a forest, a climate zone) covers part of a region, the part inside
  the feature is its own instance. Unlabelled classes (``plain`` by default) are tiled
  ``/World/Env`` meshes; void and unsurveyed pixels have no geometry;
* optionally, simple tree proxies (a trunk and a cone) scattered over forest pixels. They are
  baked into the mesh of the forest region they stand in, so a forest's mask covers its
  canopy as well as the ground under it.

Views are aerial: a camera high over the terrain, looking down at a pixel of the view's class
(classes take turns by view index). ``views/view_NNNN.usda`` is written by the shared
``worldviews.write_view``. Every random draw comes from ``usd_synth.seeds.stage_rng``.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw
from pxr import Sdf
from pydantic import BaseModel, ConfigDict, Field

from usd_synth.camera import PinholeCamera
from usd_synth.seeds import stage_rng

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import worldviews  # noqa: E402

# The class names a package uses when ``world.json`` does not list its own.
DEFAULT_CLASSES = [
    "void", "water", "plain", "forest", "snow", "sand", "stone", "badland", "ice", "lava",
    "built", "farm",
]  # fmt: skip
VECTOR_LAYERS = ("lakes", "forests", "zones")

# ---------------------------------------------------------------------------------------------
# Settings


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TreeSettings(_Model):
    enabled: bool = True
    per_px: float = Field(0.5, ge=0, description="mean tree proxies per forest pixel")
    height_m: tuple[float, float] = (6.0, 12.0)
    crown_frac: tuple[float, float] = (0.28, 0.4)  # crown radius / tree height
    trunk_rgb: tuple[float, float, float] = (0.36, 0.25, 0.16)
    crown_rgb: tuple[float, float, float] = (0.13, 0.36, 0.16)
    max_trees: int = 60000


class ViewSettings(_Model):
    margin_px: int = 4  # cameras and targets stay this many pixels inside the raster
    altitude_m: tuple[float, float] = (100.0, 300.0)  # camera height above its target
    yaw_deg: tuple[float, float] = (0.0, 360.0)  # compass bearing from camera to target
    pitch_deg: tuple[float, float] = (-80.0, -40.0)  # negative = looking down
    aim_jitter_m: float = 20.0
    min_clearance_m: float = 30.0  # camera height above the terrain under and around it
    max_range_m: float = 640.0  # farthest ground a frame may show (depth PNGs end at 655 m)
    focal_length_mm: tuple[float, float] = (18.0, 35.0)
    sensor_width_mm: float = 36.0
    sun_elevation_deg: tuple[float, float] = (15.0, 70.0)
    sun_azimuth_deg: tuple[float, float] = (0.0, 360.0)
    sun_intensity: tuple[float, float] = (3.5, 6.0)
    sky_intensity: tuple[float, float] = (0.8, 1.3)


class Settings(_Model):
    width: int = 640
    height: int = 480
    samples: int = 64
    denoise: bool = True
    min_visible_pixels: int = 30
    stride: int = Field(1, ge=1, description="use every n-th pixel (wide worlds)")
    vertical_scale: float = 1.0  # terrain height exaggeration
    labelled: list[str] = Field(min_length=1, description="ground classes that get labels")
    colors: dict[str, tuple[float, float, float]] = {}  # sRGB display colour per ground class
    env_tile_px: int = 64  # unlabelled ground is split into meshes of this many pixels square
    connectivity: int = 8  # 4: regions join along pixel edges; 8: also at pixel corners
    trees: TreeSettings = TreeSettings()
    views: ViewSettings = ViewSettings()


def load_settings(path: str | Path) -> Settings:
    return Settings.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------------------------
# Reading a package


@dataclass
class World:
    path: Path
    height: np.ndarray  # (H, W) float top surface y; NaN where unsurveyed or void
    ground: np.ndarray  # (H, W) uint8 index into ``classes``
    classes: list[str]
    origin_x: int  # block x of the left edge of column 0
    origin_z: int  # block z of the top edge of row 0
    blocks_per_px: int
    vectors: dict | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.height.shape


def _grey(path: Path) -> np.ndarray:
    im = Image.open(path)
    if im.mode not in ("L", "I;16", "I;16B", "I"):
        raise ValueError(f"{path}: expected a greyscale PNG, got mode {im.mode}")
    return np.asarray(im)


def read_package(path: str | Path, stride: int = 1) -> World:
    """A package's rasters (every ``stride``-th pixel) and its vectors, if it has them."""
    root = Path(path)
    meta = json.loads((root / "world.json").read_text(encoding="utf-8"))
    grid = meta["grid"]
    files = meta.get("files", {})
    raw_h = _grey(root / files.get("height", "height.png")).astype(np.int64)
    ground = _grey(root / files.get("ground", "ground.png")).astype(np.uint8)
    shape = (int(grid["height"]), int(grid["width"]))
    if raw_h.shape != shape or ground.shape != shape:
        raise ValueError(
            f"{root}: rasters are {raw_h.shape} and {ground.shape}, world.json says {shape}"
        )
    classes = list(meta.get("classes", DEFAULT_CLASSES))
    if int(ground.max()) >= len(classes):
        raise ValueError(f"{root}: ground.png has class {int(ground.max())}, not in classes")
    raw_h, ground = raw_h[::stride, ::stride], ground[::stride, ::stride]
    height = (raw_h - int(grid.get("y_offset", 0))).astype(np.float64)
    void = classes.index("void") if "void" in classes else -1
    height[(raw_h == 0) | (ground == void)] = np.nan
    vec_path = root / files.get("vectors", "vectors.json")
    vectors = json.loads(vec_path.read_text(encoding="utf-8")) if vec_path.exists() else None
    return World(
        root, height, ground, classes, int(grid["origin_x"]), int(grid["origin_z"]),
        int(grid["blocks_per_px"]) * stride, vectors,
    )  # fmt: skip


# ---------------------------------------------------------------------------------------------
# Rasters


def components(key: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """Connected regions of equal non-negative ``key`` (-1 = none, stays -1).

    ``connectivity`` 4 joins pixels that share an edge; 8 also joins pixels that touch at a
    corner, so a beach drawn as a diagonal staircase of pixels stays one region.

    Regions are numbered 0..K-1 in raster order of their first pixel. Works on runs of equal
    keys along each row (a union-find over runs), so wide worlds stay fast.
    """
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    reach = 1 if connectivity == 8 else 0
    h, w = key.shape
    parent: list[int] = []

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    run_rows = []  # per row: (start, end, key, run id)
    prev: list[tuple[int, int, int, int]] = []
    for i in range(h):
        row = key[i]
        cuts = np.flatnonzero(np.diff(row)) + 1
        starts = np.r_[0, cuts]
        ends = np.r_[cuts, w]
        runs = []
        p = 0
        for s, e in zip(starts.tolist(), ends.tolist(), strict=True):
            k = int(row[s])
            if k < 0:
                continue
            rid = len(parent)
            parent.append(rid)
            while p < len(prev) and prev[p][1] + reach <= s:
                p += 1
            q = p
            while q < len(prev) and prev[q][0] < e + reach:
                if prev[q][2] == k:
                    a, b = find(rid), find(prev[q][3])
                    if a != b:
                        parent[max(a, b)] = min(a, b)
                q += 1
            runs.append((s, e, k, rid))
        run_rows.append(runs)
        prev = runs
    out = np.full(key.shape, -1, dtype=np.int64)
    number: dict[int, int] = {}
    for i, runs in enumerate(run_rows):
        for s, e, _, rid in runs:
            r = find(rid)
            if r not in number:
                number[r] = len(number)
            out[i, s:e] = number[r]
    return out


def feature_raster(world: World) -> tuple[np.ndarray, list[str]]:
    """Per pixel: which vector feature covers it (0 = none, k = ``names[k - 1]``).

    Lakes, forests and zones are drawn in that order, so a later feature wins where two
    overlap. Polygons are ``[[x, z], ...]`` in block coordinates; a forest may instead be a
    circle (``at`` [x, z] and ``radius`` in blocks).
    """
    h, w = world.shape
    im = Image.new("I", (w, h), 0)
    draw = ImageDraw.Draw(im)
    names: list[str] = []
    b = world.blocks_per_px

    def px(x, z):  # block coordinates -> pixel coordinates (x = column)
        return ((x - world.origin_x) / b - 0.5, (z - world.origin_z) / b - 0.5)

    for layer in VECTOR_LAYERS:
        for n, feat in enumerate((world.vectors or {}).get(layer, [])):
            names.append(f"{feat.get('kind', layer.rstrip('s'))}_{n}")
            fid = len(names)
            if feat.get("polygon"):
                pts = [px(x, z) for x, z in feat["polygon"]]
                if len(pts) >= 3:
                    draw.polygon(pts, fill=fid)
            elif feat.get("at") is not None and feat.get("radius"):
                cx, cy = px(*feat["at"])
                r = feat["radius"] / b
                draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fid)
    return np.asarray(im, dtype=np.int64), names


def corner_heights(height: np.ndarray) -> np.ndarray:
    """(H+1, W+1) heights at pixel corners: the mean of the surveyed pixels around each."""
    h, w = height.shape
    ok = np.isfinite(height)
    val = np.where(ok, height, 0.0)
    total = np.zeros((h + 1, w + 1))
    count = np.zeros((h + 1, w + 1))
    for di in (0, 1):
        for dj in (0, 1):
            total[di : di + h, dj : dj + w] += val
            count[di : di + h, dj : dj + w] += ok
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)


# ---------------------------------------------------------------------------------------------
# Geometry


class Grid:
    """Pixel <-> Z-up world coordinates (x east, y north, z up; one block = 1 m)."""

    def __init__(self, world: World, vertical_scale: float = 1.0):
        self.world = world
        self.b = world.blocks_per_px
        self.vs = vertical_scale
        self.corners = corner_heights(world.height) * vertical_scale

    def corner_xy(self, i, j):
        w = self.world
        return w.origin_x + np.asarray(j) * self.b, -(w.origin_z + np.asarray(i) * self.b)

    def surface_z(self, i: np.ndarray, j: np.ndarray, u: np.ndarray, v: np.ndarray):
        """Height inside pixel (i, j) at fractions (u across, v down), bilinear in its corners."""
        c = self.corners
        top = c[i, j] * (1 - u) + c[i, j + 1] * u
        bottom = c[i + 1, j] * (1 - u) + c[i + 1, j + 1] * u
        return top * (1 - v) + bottom * v

    def pixel_centres(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        x, y = self.corner_xy(i + 0.5, j + 0.5)
        z = self.surface_z(i, j, np.full(len(i), 0.5), np.full(len(i), 0.5))
        return np.stack([x, y, z], axis=-1).astype(np.float64)

    def bounds(self) -> tuple[float, float, float, float]:
        h, w = self.world.shape
        x0, y1 = self.corner_xy(0, 0)
        x1, y0 = self.corner_xy(h, w)
        return float(x0), float(y0), float(x1), float(y1)


def terrain_part(grid: Grid, i: np.ndarray, j: np.ndarray):
    """Points, counts and indices of one quad per pixel (i, j), sharing corner vertices.

    Quads wind counter-clockwise seen from above, so their normals point up.
    """
    ci = np.stack([i, i + 1, i + 1, i], axis=1)  # corners: NW, SW, SE, NE
    cj = np.stack([j, j, j + 1, j + 1], axis=1)
    w1 = grid.world.shape[1] + 1
    flat = (ci * w1 + cj).ravel()
    uniq, inverse = np.unique(flat, return_inverse=True)
    ui, uj = np.divmod(uniq, w1)
    x, y = grid.corner_xy(ui, uj)
    points = np.stack([x, y, grid.corners[ui, uj]], axis=1)
    return points, np.full(len(i), 4, dtype=np.int32), inverse.astype(np.int32)


_CONE_SIDES = 6


def tree_part(base: np.ndarray, height: np.ndarray, crown: np.ndarray):
    """Proxy trees at ``base`` (N, 3): a square trunk and a six-sided cone, no bottoms.

    Returns points, face counts, indices, and per face 0 (trunk) or 1 (crown).
    """
    pts, counts, idx, kind = [], [], [], []
    n0 = 0
    ang = np.linspace(0, 2 * np.pi, _CONE_SIDES, endpoint=False)
    for (bx, by, bz), th, cr in zip(base.tolist(), height.tolist(), crown.tolist(), strict=True):
        t = max(cr * 0.18, 0.25)  # half trunk width
        trunk_top = bz + th * 0.35
        sq = [(bx - t, by - t), (bx + t, by - t), (bx + t, by + t), (bx - t, by + t)]
        pts += [(x, y, bz) for x, y in sq] + [(x, y, trunk_top) for x, y in sq]
        for k in range(4):  # side quads, counter-clockwise seen from outside
            a, b = k, (k + 1) % 4
            idx += [n0 + a, n0 + b, n0 + 4 + b, n0 + 4 + a]
            counts.append(4)
            kind.append(0)
        ring = n0 + 8
        pts += [(bx + cr * math.cos(a), by + cr * math.sin(a), trunk_top - th * 0.1) for a in ang]
        pts.append((bx, by, bz + th))
        apex = ring + _CONE_SIDES
        for k in range(_CONE_SIDES):
            idx += [ring + k, ring + (k + 1) % _CONE_SIDES, apex]
            counts.append(3)
            kind.append(1)
        n0 = apex + 1
    return (
        np.asarray(pts, dtype=np.float64).reshape(-1, 3),
        np.asarray(counts, dtype=np.int32),
        np.asarray(idx, dtype=np.int32),
        kind,
    )


# ---------------------------------------------------------------------------------------------
# Scene stage


@dataclass
class Instance:
    instance_id: int
    category: str
    pixels: int
    faces: int
    trees: int = 0
    feature: str = ""


@dataclass
class SceneInfo:
    path: Path
    categories: list[str]
    instances: list[Instance]
    faces: int
    trees: int
    grid: Grid
    pixels_by_category: dict[str, int] = field(default_factory=dict)
    targets: dict[str, np.ndarray] = field(default_factory=dict)  # Z-up pixel centres per class
    target_regions: dict[str, np.ndarray] = field(default_factory=dict)  # region of each target


def default_color(name: str) -> tuple[float, float, float]:
    return DEFAULT_COLORS.get(name, (0.8, 0.8, 0.8))


DEFAULT_COLORS = {
    "water": (0.20, 0.38, 0.72),
    "plain": (0.45, 0.62, 0.30),
    "forest": (0.20, 0.42, 0.20),
    "snow": (0.94, 0.95, 0.97),
    "sand": (0.86, 0.80, 0.56),
    "stone": (0.52, 0.52, 0.52),
    "badland": (0.72, 0.42, 0.24),
    "ice": (0.66, 0.80, 0.95),
    "lava": (0.95, 0.40, 0.08),
    "built": (0.62, 0.55, 0.50),
    "farm": (0.70, 0.66, 0.30),
}


def plant_trees(grid: Grid, i: np.ndarray, j: np.ndarray, ts: TreeSettings, rng):
    """Tree bases, heights and crown radii over pixels (i, j); a Poisson count per pixel."""
    n = rng.poisson(ts.per_px, size=len(i))
    pi, pj = np.repeat(i, n), np.repeat(j, n)
    u, v = rng.uniform(0.1, 0.9, size=(2, len(pi)))
    x, y = grid.corner_xy(pi + v, pj + u)
    z = grid.surface_z(pi, pj, u, v)
    height = rng.uniform(*ts.height_m, size=len(pi))
    crown = height * rng.uniform(*ts.crown_frac, size=len(pi))
    return np.stack([x, y, z], axis=1), height, crown, pi, pj


def build_scene(world: World, settings: Settings, out_path: str | Path, seed: int = 0):
    """Write ``scene.usdc`` (terrain, tree proxies, labelled regions) and return its summary.

    ``seed`` only drives where tree proxies stand (``stage_rng(seed, 0, "objects")``).
    """
    unknown = [c for c in settings.labelled if c not in world.classes]
    if unknown:
        raise ValueError(f"labelled classes not in the package: {unknown}")
    categories = list(settings.labelled)
    grid = Grid(world, settings.vertical_scale)
    ok = np.isfinite(world.height)
    names = world.classes
    cat_of_class = np.array(
        [categories.index(c) + 1 if c in categories else 0 for c in names], dtype=np.int64
    )
    cat = np.where(ok, cat_of_class[world.ground], -1)  # -1 none, 0 unlabelled, k category
    features, feature_names = feature_raster(world)
    key = np.where(cat > 0, cat * (len(feature_names) + 1) + features, -1)
    region = components(key, settings.connectivity)

    colors = {c: default_color(c) for c in names}
    colors.update(settings.colors)
    stage = worldviews.new_scene(Path(out_path), categories)
    looks = worldviews.Looks(stage)

    trees = None
    if settings.trees.enabled and "forest" in categories:
        fi, fj = np.nonzero(ok & (world.ground == names.index("forest")))
        rng = stage_rng(seed, 0, "objects")
        trees = plant_trees(grid, fi, fj, settings.trees, rng)
        if len(trees[0]) > settings.trees.max_trees:
            keep = np.sort(rng.choice(len(trees[0]), settings.trees.max_trees, replace=False))
            trees = tuple(a[keep] for a in trees)
    tree_region = region[trees[3], trees[4]] if trees is not None else np.zeros(0, np.int64)
    tree_looks = [("trunk", settings.trees.trunk_rgb), ("crown", settings.trees.crown_rgb)]

    instances: list[Instance] = []
    total_faces = 0
    total_trees = 0
    flat_region = region.ravel()
    order = np.argsort(flat_region, kind="stable")
    sorted_r = flat_region[order]
    cuts = np.flatnonzero(np.r_[True, np.diff(sorted_r) != 0, True])
    # instances in category order, then raster order of the region's first pixel
    groups = []
    for a, b in zip(cuts[:-1], cuts[1:], strict=True):
        r = int(sorted_r[a])
        if r < 0:
            continue
        px = order[a:b]
        k = int(cat.ravel()[px[0]])
        groups.append((k, r, px))
    groups.sort(key=lambda g: (g[0], g[1]))
    for k, r, px in groups:
        i, j = np.divmod(px, world.shape[1])
        cname = categories[k - 1]
        cls_name = names[int(world.ground[i[0], j[0]])]
        points, counts, indices = terrain_part(grid, i, j)
        face_looks = [(cls_name, colors[cls_name])] * len(i)
        n_trees = 0
        if trees is not None:
            sel = np.flatnonzero(tree_region == r)
            if len(sel):
                tp, tc, ti, tk = tree_part(trees[0][sel], trees[1][sel], trees[2][sel])
                indices = np.r_[indices, ti + len(points)]
                points = np.r_[points, tp]
                counts = np.r_[counts, tc]
                face_looks += [tree_looks[t] for t in tk]
                n_trees = len(sel)
        iid = len(instances) + 1
        mesh = worldviews.write_mesh(stage, f"/World/Objects/inst_{iid:05d}", points, counts,
                                     indices, face_looks, looks)  # fmt: skip
        prim = mesh.GetPrim()
        worldviews.label_instance(prim, iid, k, cname)
        worldviews.set_attr(prim, "pixels", Sdf.ValueTypeNames.Int, int(len(i)))
        worldviews.set_attr(prim, "ground_class", Sdf.ValueTypeNames.String, cls_name)
        fid = int(features[i[0], j[0]])
        feature = feature_names[fid - 1] if fid else ""
        if feature:
            worldviews.set_attr(prim, "feature", Sdf.ValueTypeNames.String, feature)
        instances.append(Instance(iid, cname, int(len(i)), int(len(counts)), n_trees, feature))
        total_faces += len(counts)
        total_trees += n_trees

    t = settings.env_tile_px
    ei, ej = np.nonzero(cat == 0)
    tiles = np.stack([ei // t, ej // t], axis=1)
    for ti_, tj_ in np.unique(tiles, axis=0).tolist() if len(ei) else []:
        sel = (tiles[:, 0] == ti_) & (tiles[:, 1] == tj_)
        i, j = ei[sel], ej[sel]
        points, counts, indices = terrain_part(grid, i, j)
        face_looks = [(names[g], colors[names[g]]) for g in world.ground[i, j].tolist()]
        worldviews.write_mesh(stage, f"/World/Env/ground_{ti_:02d}_{tj_:02d}", points, counts,
                              indices, face_looks, looks)  # fmt: skip
        total_faces += len(counts)
    stage.GetRootLayer().Save()

    targets, target_regions = {}, {}
    by_cat = {}
    for k, c in enumerate(categories, start=1):
        i, j = np.nonzero(cat == k)
        by_cat[c] = int(len(i))
        targets[c] = grid.pixel_centres(i, j)
        target_regions[c] = region[i, j]
    return SceneInfo(
        Path(out_path), categories, instances, int(total_faces), total_trees, grid, by_cat,
        targets, target_regions,
    )  # fmt: skip


# ---------------------------------------------------------------------------------------------
# Views


@dataclass(frozen=True)
class View:
    index: int
    camera: PinholeCamera
    light: worldviews.Lighting
    target_class: str
    target: tuple[float, float, float]  # Z-up
    yaw_deg: float
    pitch_deg: float
    altitude: float


def _terrain_z(grid: Grid, x: float, y: float, radius_px: int = 0) -> float | None:
    """Highest corner height within ``radius_px`` pixels of (x, y); None outside the grid."""
    w = grid.world
    j = int(math.floor((x - w.origin_x) / grid.b))
    i = int(math.floor((-y - w.origin_z) / grid.b))
    h, wd = w.shape
    if not (0 <= i < h and 0 <= j < wd):
        return None
    c = grid.corners[
        max(i - radius_px, 0) : i + radius_px + 2, max(j - radius_px, 0) : j + radius_px + 2
    ]
    return float(np.nanmax(c)) if np.isfinite(c).any() else float("-inf")


def line_of_sight(grid: Grid, eye: np.ndarray, target: np.ndarray, stop: float = 5.0) -> bool:
    """True if the segment eye -> target stays above the terrain until ``stop`` m short."""
    d = target - eye
    length = float(np.linalg.norm(d))
    step = max(grid.b * 0.5, 1.0)
    for t in np.arange(step, max(length - stop, step), step):
        p = eye + d * (t / length)
        top = _terrain_z(grid, p[0], p[1])
        if top is None or p[2] < top:
            return False
    return True


def far_range(camera: PinholeCamera, z: float, grid=(9, 7)) -> float:
    """Longest distance from the camera to the plane at height ``z`` over a pixel grid."""
    scale = camera.focal_length_mm / camera.sensor_width_mm * camera.width
    worst = 0.0
    for u in np.linspace(0, camera.width, grid[0]):
        for v in np.linspace(0, camera.height, grid[1]):
            d = camera.rotation @ np.array(
                [(u - camera.width / 2) / scale, -(v - camera.height / 2) / scale, -1.0]
            )
            if d[2] >= -1e-6:
                return math.inf
            t = (z - camera.position[2]) / d[2]
            worst = max(worst, float(np.linalg.norm(d * t)))
    return worst


def sample_view(info: SceneInfo, settings: Settings, seed: int, index: int, attempts=4000):
    """Random aerial camera, sun and sky for view ``index``, aimed near a pixel of its class.

    Classes take turns by view index, so a rare class (lava, a small town) gets as many views
    as a common one; within a class every region is equally likely, so the sea does not take
    every water view from the lakes. Every ray of a coarse pixel grid has to reach the lowest
    terrain inside the raster and within ``max_range_m``, so the world's edge and the sky stay
    out of frame.
    """
    vs = settings.views
    grid = info.grid
    rng = stage_rng(seed, index, "camera")
    classes = [c for c in info.categories if len(info.targets.get(c, ()))]
    if not classes:
        raise RuntimeError("the package has no pixel of any labelled class")
    cls = classes[index % len(classes)]
    pool = info.targets[cls]
    regions = info.target_regions[cls]
    by_region = [np.flatnonzero(regions == r) for r in np.unique(regions)]
    x0, y0, x1, y1 = grid.bounds()
    m = vs.margin_px * grid.b
    inner = (x0 + m, y0 + m, x1 - m, y1 - m)
    floor = float(np.nanmin(grid.corners))
    for _ in range(attempts):
        members = by_region[int(rng.integers(len(by_region)))]  # regions take turns too
        target = pool[members[int(rng.integers(len(members)))]]
        altitude = float(rng.uniform(*vs.altitude_m))
        yaw = float(rng.uniform(*vs.yaw_deg))
        pitch = float(rng.uniform(*vs.pitch_deg))
        focal = float(rng.uniform(*vs.focal_length_mm))
        jitter = rng.uniform(-1.0, 1.0, size=2) * vs.aim_jitter_m
        if not (inner[0] <= target[0] <= inner[2] and inner[1] <= target[1] <= inner[3]):
            continue
        yr = math.radians(yaw)
        dist = altitude / math.tan(math.radians(-pitch))
        eye = target + np.array([-math.sin(yr) * dist, -math.cos(yr) * dist, altitude])
        if not (inner[0] <= eye[0] <= inner[2] and inner[1] <= eye[1] <= inner[3]):
            continue
        ground = _terrain_z(grid, eye[0], eye[1], radius_px=2)
        if ground is None or eye[2] < ground + vs.min_clearance_m:
            continue
        if not line_of_sight(grid, eye, target):
            continue
        aim = target + np.array([jitter[0], jitter[1], 0.0])
        camera = PinholeCamera.look_at(
            eye, aim, focal, vs.sensor_width_mm, settings.width, settings.height
        )
        if not worldviews.ground_hits_inside(camera, inner, z=floor):
            continue
        if far_range(camera, floor) > vs.max_range_m:
            continue
        break
    else:
        raise RuntimeError(f"view {index}: no camera pose found in {attempts} tries")
    light = worldviews.sample_lighting(
        seed, index, vs.sun_elevation_deg, vs.sun_azimuth_deg, vs.sun_intensity, vs.sky_intensity
    )
    return View(
        index, camera, light, cls, tuple(round(float(v), 3) for v in target), round(yaw, 3),
        round(pitch, 3), round(altitude, 3),
    )  # fmt: skip
