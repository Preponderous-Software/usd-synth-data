"""Turn a Minecraft-USD-Exporter terrain export into usd-synth-data render scenes.

Host-side half of the Minecraft example (``mc_synth.py`` is the driver). It reads the
exporter's ``PointInstancer`` terrain (``world.usda`` or ``terrain.usdc``: one instancer per
region, per-instance ``primvars:mcusd:blockId`` indexing ``/World/BlockLibrary.mcusd:blockIds``)
and writes:

* ``scene.usdc`` — the terrain as plain meshes in the usd-synth stage layout (Z-up). Every
  connected component of a labelled block class (a tree, a lake, an ore vein, a beach, a stone
  outcrop) becomes one ``/World/Objects/inst_NNNN`` mesh carrying the usual
  ``userProperties:usd_synth:*`` attributes, so the stock render worker gives it its own
  Object Index; all other blocks are unlabelled ``/World/Env`` meshes (pass index 0).
* ``views/view_NNNN.usda`` — one small layer per view that sublayers ``scene.usdc`` and adds a
  randomized camera, sun and sky; ``usd_synth.render.scene_spec`` reads it unchanged.

Only block faces that touch an empty cell are emitted, and each block type gets a
``UsdPreviewSurface`` bound through a ``GeomSubset``. Non-cube decoration (grass tufts,
flowers, leaf litter) is dropped; waterlogged plants count as water. Every random draw comes
from ``usd_synth.seeds.stage_rng``, so view *i* depends only on ``(seed, i)``.
"""

from __future__ import annotations

import fnmatch
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdLux, UsdShade, Vt
from pydantic import BaseModel, ConfigDict, Field

from usd_synth.camera import PinholeCamera
from usd_synth.seeds import stage_rng

NS = "userProperties:usd_synth:"

# ---------------------------------------------------------------------------------------------
# Settings


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BlockClass(_Model):
    name: str
    blocks: list[str] = Field(min_length=1, description="fnmatch patterns on namespaced IDs")
    connectivity: int = Field(6, description="6 (faces) or 26 (faces, edges and corners)")


class ViewSettings(_Model):
    margin_blocks: int = 8  # keep cameras and targets this far inside the export's X/Z bounds
    distance: tuple[float, float] = (14.0, 60.0)  # horizontal camera-to-target distance
    yaw_deg: tuple[float, float] = (0.0, 360.0)  # compass bearing from camera to target
    pitch_deg: tuple[float, float] = (-50.0, -10.0)  # negative = looking down
    aim_jitter_blocks: float = 6.0  # the look-at point is this far from the target, at most
    min_clearance: float = 2.0  # camera height above the highest block within 3 blocks
    focal_length_mm: tuple[float, float] = (18.0, 32.0)
    sensor_width_mm: float = 36.0
    sun_elevation_deg: tuple[float, float] = (8.0, 72.0)
    sun_azimuth_deg: tuple[float, float] = (0.0, 360.0)
    sun_intensity: tuple[float, float] = (2.5, 4.5)
    sky_intensity: tuple[float, float] = (0.6, 1.1)


class Settings(_Model):
    width: int = 640
    height: int = 480
    samples: int = 64
    denoise: bool = True
    min_visible_pixels: int = 30
    drop: list[str] = []
    as_water: list[str] = []
    classes: list[BlockClass]
    views: ViewSettings = ViewSettings()


def load_settings(path: str | Path) -> Settings:
    return Settings.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------------------------
# Reading the export


@dataclass
class Terrain:
    xyz: np.ndarray  # (N, 3) int32 Minecraft block coordinates (x east, y up, z south)
    block: np.ndarray  # (N,) int32 index into ``ids``
    ids: list[str]  # namespaced block IDs
    colors: dict[str, tuple[float, float, float]]  # display colour (sRGB) per block ID
    bbox: list[int] | None = None  # the export's mcusd:bbox, if the root layer records it


def read_export(path: str | Path) -> Terrain:
    """Every exported block of a Minecraft-USD-Exporter stage, in world block coordinates."""
    stage = Usd.Stage.Open(str(path))
    library = stage.GetPrimAtPath("/World/BlockLibrary")
    if not library:
        raise ValueError(f"{path}: no /World/BlockLibrary (not a Minecraft-USD-Exporter stage?)")
    ids = list(library.GetAttribute("mcusd:blockIds").Get())
    colors = {}
    for proto in library.GetAllChildren():  # a class prim: default traversal skips it
        nsid = proto.GetAttribute("mcusd:namespacedId").Get()
        dc = proto.GetAttribute("primvars:displayColor").Get()
        if nsid and dc:
            colors[nsid] = tuple(float(c) for c in dc[0])
    cache = UsdGeom.XformCache()
    xyz, block = [], []
    terrain = stage.GetPrimAtPath("/World/Terrain")
    for prim in Usd.PrimRange(terrain):
        if not prim.IsA(UsdGeom.PointInstancer):
            continue
        inst = UsdGeom.PointInstancer(prim)
        pos = np.asarray(inst.GetPositionsAttr().Get(), dtype=np.float64)
        bid = UsdGeom.PrimvarsAPI(prim).GetPrimvar("mcusd:blockId").Get()
        if len(pos) == 0:
            continue
        m = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=np.float64)
        world = pos @ m[:3, :3] + m[3, :3]
        xyz.append(np.rint(world).astype(np.int32))
        block.append(np.asarray(bid, dtype=np.int32))
    if not xyz:
        raise ValueError(f"{path}: no terrain PointInstancers under /World/Terrain")
    bbox = stage.GetRootLayer().customLayerData.get("mcusd:bbox")
    return Terrain(
        np.concatenate(xyz), np.concatenate(block), ids, colors, list(bbox) if bbox else None
    )


def _matches(block_id: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(block_id, p) for p in patterns)


def classify(ids: list[str], settings: Settings) -> tuple[np.ndarray, list[str]]:
    """Per block-table entry: class index (0 = unlabelled, -1 = dropped) and render block ID.

    ``as_water`` entries (waterlogged plants, bubble columns) render and label as water.
    """
    cls = np.zeros(len(ids), dtype=np.int32)
    render_ids = list(ids)
    for i, bid in enumerate(ids):
        if _matches(bid, settings.as_water):
            render_ids[i] = bid = "minecraft:water"
        if bid in ("minecraft:air", "minecraft:cave_air", "minecraft:void_air") or _matches(
            bid, settings.drop
        ):
            cls[i] = -1
            continue
        for k, c in enumerate(settings.classes, start=1):
            if _matches(bid, c.blocks):
                cls[i] = k
                break
    return cls, render_ids


# ---------------------------------------------------------------------------------------------
# Connected components and exposed faces

FACE_DIRS = np.array(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.int32
)
_N6 = [tuple(d) for d in FACE_DIRS]
_N26 = [
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
]


def components(xyz: np.ndarray, connectivity: int = 6) -> np.ndarray:
    """Connected-component label (0..K-1) of each block; components are numbered in order of
    their first block in ``xyz``, so the result is deterministic for a given input order."""
    if connectivity not in (6, 26):
        raise ValueError("connectivity must be 6 or 26")
    offsets = _N6 if connectivity == 6 else _N26
    index = {tuple(p): i for i, p in enumerate(xyz.tolist())}
    label = np.full(len(xyz), -1, dtype=np.int64)
    k = 0
    for start in range(len(xyz)):
        if label[start] >= 0:
            continue
        label[start] = k
        queue = deque([start])
        while queue:
            x, y, z = xyz[queue.popleft()].tolist()
            for dx, dy, dz in offsets:
                j = index.get((x + dx, y + dy, z + dz))
                if j is not None and label[j] < 0:
                    label[j] = k
                    queue.append(j)
        k += 1
    return label


class Occupancy:
    """Dense occupancy grid over the blocks' bounds, for exposed-face tests."""

    def __init__(self, xyz: np.ndarray):
        self.lo = xyz.min(axis=0)
        shape = xyz.max(axis=0) - self.lo + 1
        self.grid = np.zeros(tuple(int(s) for s in shape), dtype=bool)
        self.grid[tuple((xyz - self.lo).T)] = True

    def occupied(self, xyz: np.ndarray) -> np.ndarray:
        rel = xyz - self.lo
        inside = np.all((rel >= 0) & (rel < self.grid.shape), axis=1)
        out = np.zeros(len(xyz), dtype=bool)
        out[inside] = self.grid[tuple(rel[inside].T)]
        return out


# Corners of each cube face (Minecraft coordinates, unit cube at the origin), counter-clockwise
# seen from outside, matching FACE_DIRS.
_FACE_CORNERS = np.array(
    [
        [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]],  # +x
        [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],  # -x
        [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],  # +y (top)
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],  # -y
        [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],  # +z
        [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],  # -z
    ],
    dtype=np.float64,
)


def mc_to_zup(points: np.ndarray) -> np.ndarray:
    """Minecraft (x east, y up, z south) to usd-synth's Z-up frame (x east, y north, z up).

    A proper rotation, so face winding (and therefore outward normals) is preserved.
    """
    p = np.asarray(points, dtype=np.float64)
    return np.stack([p[..., 0], -p[..., 2], p[..., 1]], axis=-1)


def exposed_faces(xyz: np.ndarray, occ: Occupancy) -> tuple[np.ndarray, np.ndarray]:
    """(block index, face direction index) of every face whose neighbour cell is empty."""
    blocks, dirs = [], []
    for d, step in enumerate(FACE_DIRS):
        free = ~occ.occupied(xyz + step)
        idx = np.flatnonzero(free)
        blocks.append(idx)
        dirs.append(np.full(len(idx), d, dtype=np.int32))
    return np.concatenate(blocks), np.concatenate(dirs)


@dataclass
class MeshPart:
    points: np.ndarray  # (4F, 3) Z-up
    face_block: np.ndarray  # (F,) render block ID index of each face
    normals: np.ndarray  # (F, 3) Z-up, one per face


def quad_mesh(xyz: np.ndarray, block: np.ndarray, faces: tuple[np.ndarray, np.ndarray]):
    """Unshared quads for the given faces (four points each, so shading stays flat)."""
    bi, di = faces
    corners = xyz[bi][:, None, :] + _FACE_CORNERS[di]  # (F, 4, 3)
    return MeshPart(
        mc_to_zup(corners.reshape(-1, 3)),
        block[bi],
        mc_to_zup(FACE_DIRS[di].astype(np.float64)),
    )


# ---------------------------------------------------------------------------------------------
# Scene stage


@dataclass
class Instance:
    instance_id: int
    category: str
    blocks: int
    faces: int


@dataclass
class SceneInfo:
    path: Path
    categories: list[str]
    instances: list[Instance]
    blocks_in: int
    blocks_kept: int
    faces: int
    heightmap: np.ndarray  # top surface Y+1 per (x - x0, z - z0); -inf where no block
    origin_xz: tuple[int, int]
    blocks_by_category: dict[str, int] = field(default_factory=dict)
    # per class: blocks open to the sky or to a cliff face (candidate view targets), MC coords
    targets: dict[str, np.ndarray] = field(default_factory=dict)


def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _look(stage: Usd.Stage, block_id: str, rgb) -> UsdShade.Material:
    name = block_id.split(":", 1)[-1].replace("/", "_")
    path = f"/World/Looks/{Tf.MakeValidIdentifier(name)}"
    existing = UsdShade.Material.Get(stage, path)
    if existing:
        return existing
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    lin = Gf.Vec3f(*(round(_srgb_to_linear(c), 6) for c in rgb))
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(lin)
    water = block_id == "minecraft:water"
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.08 if water else 0.9)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _write_mesh(stage, path, part: MeshPart, render_ids, colors) -> UsdGeom.Mesh:
    mesh = UsdGeom.Mesh.Define(stage, path)
    n = len(part.face_block)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(part.points.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(n, 4, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.arange(4 * n, dtype=np.int32)))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(part.normals.astype(np.float32)))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.uniform)
    lo, hi = part.points.min(axis=0), part.points.max(axis=0)
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*lo), Gf.Vec3f(*hi)]))
    binding = UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
    kinds = np.unique(part.face_block)
    for b in kinds:
        bid = render_ids[b]
        material = _look(stage, bid, colors.get(bid, (1.0, 0.0, 1.0)))
        if len(kinds) == 1:
            binding.Bind(material)
            break
        subset = binding.CreateMaterialBindSubset(
            Tf.MakeValidIdentifier(bid.split(":", 1)[-1]),
            Vt.IntArray.FromNumpy(np.flatnonzero(part.face_block == b).astype(np.int32)),
        )
        UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(material)
    return mesh


def _set(prim: Usd.Prim, name: str, type_name, value) -> None:
    prim.CreateAttribute(NS + name, type_name, custom=True).Set(value)


def build_scene(terrain: Terrain, settings: Settings, out_path: str | Path, tile: int = 32):
    """Write ``scene.usdc`` (meshes, labelled instances, materials) and return its summary."""
    table_cls, render_ids = classify(terrain.ids, settings)
    keep = table_cls[terrain.block] >= 0
    xyz, block = terrain.xyz[keep], terrain.block[keep]
    cls = table_cls[block]
    occ = Occupancy(xyz)
    colors = dict(terrain.colors)
    colors.setdefault("minecraft:water", (0.25, 0.44, 0.94))

    stage = Usd.Stage.CreateNew(str(out_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    categories = [c.name for c in settings.classes]
    _set(world, "categories", Sdf.ValueTypeNames.StringArray, Vt.StringArray(categories))
    UsdGeom.Scope.Define(stage, "/World/Looks")
    UsdGeom.Xform.Define(stage, "/World/Env")
    UsdGeom.Xform.Define(stage, "/World/Objects")

    instances: list[Instance] = []
    total_faces = 0
    for k, bc in enumerate(settings.classes, start=1):
        sel = np.flatnonzero(cls == k)
        if len(sel) == 0:
            continue
        comp = components(xyz[sel], bc.connectivity)
        order = np.argsort(comp, kind="stable")
        bounds = np.flatnonzero(np.r_[True, np.diff(comp[order]) != 0, True])
        for a, b in zip(bounds[:-1], bounds[1:], strict=True):
            members = sel[order[a:b]]
            faces = exposed_faces(xyz[members], occ)
            if len(faces[0]) == 0:
                continue
            iid = len(instances) + 1
            path = f"/World/Objects/inst_{iid:05d}"
            mesh = _write_mesh(stage, path, quad_mesh(xyz[members], block[members], faces),
                               render_ids, colors)  # fmt: skip
            prim = mesh.GetPrim()
            _set(prim, "instance_id", Sdf.ValueTypeNames.Int, iid)
            _set(prim, "category_id", Sdf.ValueTypeNames.Int, k)
            _set(prim, "category_name", Sdf.ValueTypeNames.String, bc.name)
            _set(prim, "kind", Sdf.ValueTypeNames.String, "object")
            _set(prim, "blocks", Sdf.ValueTypeNames.Int, int(len(members)))
            instances.append(Instance(iid, bc.name, int(len(members)), int(len(faces[0]))))
            total_faces += len(faces[0])

    rest = np.flatnonzero(cls == 0)
    tiles = (xyz[rest][:, [0, 2]] // tile).astype(np.int64)
    keys = np.unique(tiles, axis=0)
    for tx, tz in keys.tolist():
        members = rest[(tiles[:, 0] == tx) & (tiles[:, 1] == tz)]
        faces = exposed_faces(xyz[members], occ)
        if len(faces[0]) == 0:
            continue
        name = f"ground_{tx - keys[:, 0].min():02d}_{tz - keys[:, 1].min():02d}"
        _write_mesh(stage, f"/World/Env/{name}", quad_mesh(xyz[members], block[members], faces),
                    render_ids, colors)  # fmt: skip
        total_faces += len(faces[0])
    stage.GetRootLayer().Save()

    x0, z0 = int(xyz[:, 0].min()), int(xyz[:, 2].min())
    hm = np.full((int(xyz[:, 0].max()) - x0 + 1, int(xyz[:, 2].max()) - z0 + 1), -np.inf)
    np.maximum.at(hm, (xyz[:, 0] - x0, xyz[:, 2] - z0), xyz[:, 1] + 1.0)
    by_cat = {c: int((cls == k).sum()) for k, c in enumerate(categories, start=1)}
    targets = {
        c: xyz[(cls == k) & sky_exposed(xyz, hm, (x0, z0))] for k, c in enumerate(categories, 1)
    }
    return SceneInfo(
        Path(out_path), categories, instances, int(len(terrain.xyz)), int(len(xyz)),
        total_faces, hm, (x0, z0), by_cat, targets,
    )  # fmt: skip


def sky_exposed(xyz: np.ndarray, hm: np.ndarray, origin_xz: tuple[int, int]) -> np.ndarray:
    """Blocks a camera above the terrain can see: the top of their column, or reaching above a
    neighbouring column's surface (a cliff or bank face). Cave walls are not exposed."""
    i, j = xyz[:, 0] - origin_xz[0], xyz[:, 2] - origin_xz[1]
    top = xyz[:, 1] + 1.0
    out = top >= hm[i, j]
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ni, nj = i + di, j + dj
        ok = (ni >= 0) & (ni < hm.shape[0]) & (nj >= 0) & (nj < hm.shape[1])
        neighbour = np.full(len(xyz), np.inf)
        neighbour[ok] = hm[ni[ok], nj[ok]]
        out |= xyz[:, 1] >= neighbour
    return out


# ---------------------------------------------------------------------------------------------
# Views


@dataclass(frozen=True)
class View:
    index: int
    camera: PinholeCamera  # Z-up world
    target_class: str  # the class whose block the camera was aimed at
    target_block: tuple[int, int, int]  # Minecraft coordinates
    sun_dir: tuple[float, float, float]  # direction *towards* the sun, Z-up
    sun_elevation_deg: float
    sun_intensity: float
    sun_color: tuple[float, float, float]
    sky_color: tuple[float, float, float]
    sky_intensity: float
    yaw_deg: float
    pitch_deg: float
    distance: float


def _lerp(a, b, t):
    return tuple(round(float(x + (y - x) * t), 6) for x, y in zip(a, b, strict=True))


def _column_top(hm, origin_xz, x, z, radius=0):
    i, j = int(math.floor(x)) - origin_xz[0], int(math.floor(z)) - origin_xz[1]
    if not (0 <= i < hm.shape[0] and 0 <= j < hm.shape[1]):
        return None
    window = hm[max(i - radius, 0) : i + radius + 1, max(j - radius, 0) : j + radius + 1]
    return float(np.max(window))


def line_of_sight(hm, origin_xz, eye, target, stop=1.5) -> bool:
    """True if the segment eye->target stays above the heightmap until ``stop`` blocks short
    of the target (Minecraft coordinates)."""
    d = target - eye
    length = float(np.linalg.norm(d))
    for t in np.arange(0.5, max(length - stop, 0.5), 0.5):
        p = eye + d * (t / length)
        top = _column_top(hm, origin_xz, p[0], p[2])
        if top is None or p[1] < top:
            return False
    return True


def stays_inside(camera: PinholeCamera, hm, origin_xz, grid=(9, 7)) -> bool:
    """True if every downward-looking ray of a coarse pixel grid lands on the terrain before
    it leaves the export, so the cut edges of the exported box stay out of the picture (rays
    above the horizon only see sky)."""
    scale = camera.focal_length_mm / camera.sensor_width_mm * camera.width
    for u in np.linspace(0, camera.width, grid[0]):
        for v in np.linspace(0, camera.height, grid[1]):
            d_cam = np.array([(u - camera.width / 2) / scale, -(v - camera.height / 2) / scale, -1])
            d = camera.rotation @ d_cam
            d_mc = np.array([d[0], d[2], -d[1]]) / np.linalg.norm(d)
            if d_mc[1] >= -0.02:
                continue
            eye = np.array([camera.position[0], camera.position[2], -camera.position[1]])
            t = 0.0
            while True:
                t += 0.75
                p = eye + d_mc * t
                top = _column_top(hm, origin_xz, p[0], p[2])
                if top is None or p[1] < -128:
                    return False
                if p[1] <= top:
                    break
    return True


def sample_view(info: SceneInfo, settings: Settings, seed: int, index: int, attempts=2000) -> View:
    """Random camera, sun and sky for view ``index``.

    The camera is aimed near a random visible block of one class (classes take turns by view
    index, so rare ones such as ore get as many views as common ones), from a random bearing,
    distance and pitch and with a random focal length. Poses whose line of sight to the target
    is blocked, that leave the export or show its cut edges, or that put the camera inside
    terrain are redrawn.
    """
    vs = settings.views
    rng = stage_rng(seed, index, "camera")
    hm, origin = info.heightmap, info.origin_xz
    x0, z0 = origin
    lo_x, hi_x = x0 + vs.margin_blocks, x0 + hm.shape[0] - vs.margin_blocks
    lo_z, hi_z = z0 + vs.margin_blocks, z0 + hm.shape[1] - vs.margin_blocks
    classes = [c for c in info.categories if len(info.targets.get(c, ())) > 0]
    if not classes:
        raise RuntimeError("no labelled block is visible from above in this export")
    cls = classes[index % len(classes)]  # round-robin, so rare classes get as many views
    pool = info.targets[cls]
    for _ in range(attempts):
        tb = pool[int(rng.integers(len(pool)))]
        target = tb + np.array([0.5, 1.0, 0.5])  # top centre of the block
        if not (lo_x <= target[0] <= hi_x and lo_z <= target[2] <= hi_z):
            continue
        dist = float(rng.uniform(*vs.distance))
        yaw = float(rng.uniform(*vs.yaw_deg))
        pitch = float(rng.uniform(*vs.pitch_deg))
        focal = float(rng.uniform(*vs.focal_length_mm))
        jitter = rng.uniform(-1.0, 1.0, size=3) * vs.aim_jitter_blocks * np.array([1, 0.3, 1])
        yr = math.radians(yaw)
        bearing = np.array([math.sin(yr), 0.0, -math.cos(yr)])  # yaw 0 = north (-z)
        eye = target - bearing * dist
        eye[1] = target[1] + dist * math.tan(math.radians(-pitch))
        if not (lo_x <= eye[0] <= hi_x and lo_z <= eye[2] <= hi_z):
            continue
        ground = _column_top(hm, origin, eye[0], eye[2], radius=3)
        if ground is None or eye[1] < ground + vs.min_clearance:
            continue
        if not line_of_sight(hm, origin, eye, target):
            continue
        aim = target + jitter
        camera = PinholeCamera.look_at(
            mc_to_zup(eye), mc_to_zup(aim), focal, vs.sensor_width_mm,
            settings.width, settings.height,
        )  # fmt: skip
        if not stays_inside(camera, hm, origin):
            continue
        break
    else:
        raise RuntimeError(f"view {index}: no camera pose found in {attempts} tries")

    light_rng = stage_rng(seed, index, "lighting")
    el = float(light_rng.uniform(*vs.sun_elevation_deg))
    az = math.radians(float(light_rng.uniform(*vs.sun_azimuth_deg)))
    day = min(max((el - 8.0) / 40.0, 0.0), 1.0)  # 0 = sunrise/sunset, 1 = full day
    sun_dir = (
        math.cos(math.radians(el)) * math.cos(az),
        math.cos(math.radians(el)) * math.sin(az),
        math.sin(math.radians(el)),
    )
    return View(
        index=index,
        camera=camera,
        target_class=cls,
        target_block=tuple(int(v) for v in tb),
        sun_dir=tuple(round(v, 6) for v in sun_dir),
        sun_elevation_deg=round(el, 3),
        sun_intensity=round(float(light_rng.uniform(*vs.sun_intensity)), 4),
        sun_color=_lerp((1.0, 0.62, 0.38), (1.0, 0.96, 0.9), day),
        sky_color=_lerp((0.85, 0.6, 0.48), (0.45, 0.66, 1.0), day),
        sky_intensity=round(float(light_rng.uniform(*vs.sky_intensity)) * (0.6 + 0.4 * day), 4),
        yaw_deg=round(yaw, 3),
        pitch_deg=round(pitch, 3),
        distance=round(dist, 3),
    )


def write_view(view: View, scene_path: Path, out_path: Path, seed: int, settings: Settings):
    """A small ``.usda`` that sublayers the scene and adds this view's camera, sun and sky."""
    stage = Usd.Stage.CreateNew(str(out_path))
    rel = Path(scene_path).resolve().relative_to(Path(out_path).resolve().parent.parent)
    stage.GetRootLayer().subLayerPaths.append(f"../{rel.as_posix()}")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    _set(world, "scene_index", Sdf.ValueTypeNames.Int, view.index)
    _set(world, "dataset_seed", Sdf.ValueTypeNames.Int64, seed)
    _set(world, "image_width", Sdf.ValueTypeNames.Int, settings.width)
    _set(world, "image_height", Sdf.ValueTypeNames.Int, settings.height)

    UsdGeom.Xform.Define(stage, "/World/Lights")
    sun = UsdLux.DistantLight.Define(stage, "/World/Lights/Sun")
    sun.CreateIntensityAttr(view.sun_intensity)
    sun.CreateColorAttr(Gf.Vec3f(*view.sun_color))
    sun.CreateAngleAttr(1.0)
    # A DistantLight shines along its local -Z: aim -Z away from the sun.
    to_sun = np.asarray(view.sun_dir)
    rot = PinholeCamera.look_at(-to_sun, np.zeros(3), 50, 36, 2, 2).rotation
    m = np.eye(4)
    m[:3, :3] = rot.T
    sun.AddTransformOp().Set(Gf.Matrix4d(m.tolist()))
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr(view.sky_intensity)
    dome.CreateColorAttr(Gf.Vec3f(*view.sky_color))

    cam = view.camera
    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    camera.AddTransformOp().Set(Gf.Matrix4d(cam.usd_matrix()))
    camera.CreateProjectionAttr(UsdGeom.Tokens.perspective)
    camera.CreateFocalLengthAttr(round(cam.focal_length_mm, 6))
    camera.CreateHorizontalApertureAttr(round(cam.sensor_width_mm, 6))
    camera.CreateVerticalApertureAttr(round(cam.vertical_aperture_mm, 6))
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))
    _set(camera.GetPrim(), "focal_length_mm", Sdf.ValueTypeNames.Double, cam.focal_length_mm)
    _set(camera.GetPrim(), "sensor_width_mm", Sdf.ValueTypeNames.Double, cam.sensor_width_mm)
    stage.GetRootLayer().Save()
