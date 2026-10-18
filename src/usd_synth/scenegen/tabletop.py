"""Tabletop / bin-picking scenario: sample one scene description from a config and seed.

All random choices are drawn from the per-stage streams in :mod:`usd_synth.seeds`, so a scene
depends only on ``(config, dataset seed, scene index)``.
"""

from __future__ import annotations

import colorsys
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..camera import PinholeCamera, convex_hull_area
from ..config import Config, LibraryEntry
from ..seeds import stage_rng
from . import primitives
from .primitives import MeshData

TABLE_THICKNESS_M = 0.04
BIN_WALL_HEIGHT_M = 0.12
BIN_WALL_THICKNESS_M = 0.02
FLOOR_Z_M = -0.75
FLOOR_SIZE_M = 20.0
PLACEMENT_ATTEMPTS = 100
DISTRACTOR_SHAPES = ("box", "cylinder", "sphere")


@dataclass
class Material:
    name: str
    base_color: tuple[float, float, float]
    roughness: float
    metallic: float
    texture: str | None = None


@dataclass
class Instance:
    instance_id: int
    kind: str  # "object" | "distractor"
    category_id: int  # 0 for distractors
    category_name: str
    asset: str  # library entry name, or "distractor:<shape>"
    rotation: np.ndarray  # (3, 3) local -> world
    translation: np.ndarray  # (3,)
    local_points: np.ndarray  # mesh points, or the asset's bounding-box corners
    mesh: MeshData | None  # None for referenced assets
    asset_path: str | None
    asset_scale: float
    material: Material
    projected_bbox: tuple[float, float, float, float] | None = None
    projected_area_px: float = 0.0

    def world_points(self) -> np.ndarray:
        return self.local_points @ self.rotation.T + self.translation

    def matrix(self) -> list[list[float]]:
        m = np.eye(4)
        m[:3, :3] = (self.rotation * self.asset_scale).T
        m[3, :3] = self.translation
        return m.tolist()


@dataclass
class Light:
    name: str
    kind: str  # "sphere" | "dome"
    intensity: float
    color: tuple[float, float, float]
    position: np.ndarray | None = None
    radius: float = 0.0
    texture: str | None = None


@dataclass
class Scene:
    index: int
    dataset_seed: int
    width: int
    height: int
    support_kind: str
    support_size: tuple[float, float]
    support_parts: list[tuple[str, MeshData, np.ndarray]]  # (name, mesh, translation)
    support_material: Material
    floor_material: Material
    instances: list[Instance]
    lights: list[Light]
    camera: PinholeCamera
    categories: list[str] = field(default_factory=list)


def color_temperature_rgb(kelvin: float) -> tuple[float, float, float]:
    """Approximate linear RGB of a black body (Tanner Helland's fit), normalized to max 1."""
    t = kelvin / 100.0
    r = 255.0 if t <= 66 else 329.698727446 * (t - 60) ** -0.1332047592
    g = (
        99.4708025861 * math.log(t) - 161.1195681661
        if t <= 66
        else 288.1221695283 * (t - 60) ** -0.0755148492
    )
    if t >= 66:
        b = 255.0
    elif t <= 19:
        b = 0.0
    else:
        b = 138.5177312231 * math.log(t - 10) - 305.0447927307
    rgb = [min(max(c, 0.0), 255.0) / 255.0 for c in (r, g, b)]
    srgb_to_linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    peak = max(srgb_to_linear)
    return tuple(c / peak for c in srgb_to_linear)


def _rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _uniform(rng: np.random.Generator, bounds) -> float:
    lo, hi = bounds
    return float(rng.uniform(lo, hi)) if hi > lo else float(lo)


def _integer(rng: np.random.Generator, bounds) -> int:
    lo, hi = bounds
    return int(rng.integers(lo, hi + 1))


class _Materials:
    def __init__(self, cfg: Config, config_dir: Path, rng: np.random.Generator):
        self.cfg, self.config_dir, self.rng, self.count = cfg.materials, config_dir, rng, 0

    def next(self) -> Material:
        m, rng = self.cfg, self.rng
        self.count += 1
        h, s, v = (_uniform(rng, getattr(m.base_color, k)) for k in "hsv")
        metallic = float(rng.choice(m.metallic.choice, p=m.metallic.p))
        texture = None
        roughness = _uniform(rng, m.roughness)
        use_texture = rng.random() < 0.5  # always drawn, so textures on/off never shifts colours
        tex_pick = int(rng.integers(0, max(len(m.textures), 1)))
        if m.textures and use_texture:
            texture = str((self.config_dir / m.textures[tex_pick]).resolve())
        return Material(
            f"mat_{self.count:04d}", colorsys.hsv_to_rgb(h, s, v), roughness, metallic, texture
        )


def _support(cfg: Config, rng: np.random.Generator):
    kind = str(rng.choice(cfg.support.kind))
    sx, sy = _uniform(rng, cfg.support.size_m.x), _uniform(rng, cfg.support.size_m.y)
    parts = [("top", primitives.box([sx, sy, TABLE_THICKNESS_M]), [0, 0, -TABLE_THICKNESS_M / 2])]
    region = (sx / 2, sy / 2)
    if kind == "bin":
        t, h = BIN_WALL_THICKNESS_M, BIN_WALL_HEIGHT_M
        parts += [
            ("wall_xn", primitives.box([t, sy, h]), [-(sx - t) / 2, 0, h / 2]),
            ("wall_xp", primitives.box([t, sy, h]), [(sx - t) / 2, 0, h / 2]),
            ("wall_yn", primitives.box([sx - 2 * t, t, h]), [0, -(sy - t) / 2, h / 2]),
            ("wall_yp", primitives.box([sx - 2 * t, t, h]), [0, (sy - t) / 2, h / 2]),
        ]
        region = (sx / 2 - t, sy / 2 - t)
    parts = [(n, mesh, np.array(pos, dtype=np.float64)) for n, mesh, pos in parts]
    return kind, (sx, sy), parts, region


def _asset_corners(path: Path) -> np.ndarray:
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(path))
    box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(
        stage.GetPseudoRoot()
    )
    rng = box.ComputeAlignedRange()
    lo, hi = np.array(rng.GetMin()), np.array(rng.GetMax())
    if rng.IsEmpty():
        raise ValueError(f"asset {path} has no geometry to bound")
    return np.array(
        [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    )


def _make_object(entry: LibraryEntry, rng: np.random.Generator, config_dir: Path):
    """(local points, mesh or None, asset path or None, uniform asset scale)."""
    if entry.source.startswith("primitive:"):
        size = _uniform(rng, entry.size_m)
        mesh, _ = primitives.build(entry.source.split(":", 1)[1], size, rng)
        return mesh.points, mesh, None, 1.0
    path = (config_dir / entry.source).resolve()
    corners = _asset_corners(path)
    scale = 1.0
    size = _uniform(rng, entry.size_m) if entry.size_m else None
    if size is not None:
        scale = size / float(np.max(corners.max(axis=0) - corners.min(axis=0)))
    return corners * scale, None, str(path), scale


def _footprint(points: np.ndarray) -> np.ndarray:
    return np.concatenate([points[:, :2].min(axis=0), points[:, :2].max(axis=0)])


def _overlaps(a: np.ndarray, b: np.ndarray, gap: float) -> bool:
    return not (
        a[2] + gap <= b[0] or b[2] + gap <= a[0] or a[3] + gap <= b[1] or b[3] + gap <= a[1]
    )


def _place(points, rng, yaw_deg, tipped, region, placed, gap):
    """Pose and position an object on the support top (z = 0) without overlapping others.

    Returns ``(rotation, translation, footprint)`` or None if no free spot was found.
    """
    rot = _rot_z(math.radians(_uniform(rng, yaw_deg)))
    if tipped:
        rot = rot @ _rot_x(math.pi / 2)
    rotated = points @ rot.T
    lo, hi = rotated.min(axis=0), rotated.max(axis=0)
    half = (hi[:2] - lo[:2]) / 2
    centre = (hi[:2] + lo[:2]) / 2
    free = np.array(region) - half
    if np.any(free < 0):  # larger than the support region
        return None
    for _ in range(PLACEMENT_ATTEMPTS):
        xy = rng.uniform(-1.0, 1.0, size=2) * free - centre
        t = np.array([xy[0], xy[1], -lo[2]])
        fp = _footprint(rotated + t)
        if not any(_overlaps(fp, other, gap) for other in placed):
            return rot, t, fp
    return None


def _project(inst: Instance, cam: PinholeCamera) -> None:
    uv, in_front = cam.project(inst.world_points())
    if not np.all(in_front):
        inst.projected_bbox, inst.projected_area_px = None, 0.0
        return
    lo, hi = uv.min(axis=0), uv.max(axis=0)
    inst.projected_bbox = (float(lo[0]), float(lo[1]), float(hi[0] - lo[0]), float(hi[1] - lo[1]))
    inst.projected_area_px = convex_hull_area(uv)


def sample_scene(cfg: Config, dataset_seed: int, index: int, config_dir: Path | str = ".") -> Scene:
    config_dir = Path(config_dir)
    categories = cfg.categories()
    support_rng = stage_rng(dataset_seed, index, "support")
    kind, size, parts, region = _support(cfg, support_rng)

    mats = _Materials(cfg, config_dir, stage_rng(dataset_seed, index, "materials"))
    support_mat, floor_mat = mats.next(), mats.next()

    obj_rng = stage_rng(dataset_seed, index, "objects")
    pose_rng = stage_rng(dataset_seed, index, "poses")
    place = cfg.objects.placement
    placed: list[np.ndarray] = []
    instances: list[Instance] = []
    for _ in range(_integer(obj_rng, cfg.objects.count)):
        entry = cfg.objects.library[int(obj_rng.integers(0, len(cfg.objects.library)))]
        points, mesh, asset_path, scale = _make_object(entry, obj_rng, config_dir)
        tipped = bool(pose_rng.random() < place.allow_tipped)
        pose = _place(
            points, pose_rng, place.yaw_deg, tipped, region, placed, place.min_separation_m
        )
        material = mats.next()
        if pose is None:
            continue
        rot, t, fp = pose
        placed.append(fp)
        instances.append(
            Instance(
                len(instances) + 1,
                "object",
                categories.index(entry.category) + 1,
                entry.category,
                entry.name,
                rot,
                t,
                points,
                mesh,
                asset_path,
                scale,
                material,
            )
        )

    dis_rng = stage_rng(dataset_seed, index, "distractors")
    for _ in range(_integer(dis_rng, cfg.distractors.count)):
        shape = DISTRACTOR_SHAPES[int(dis_rng.integers(0, len(DISTRACTOR_SHAPES)))]
        mesh, _ = primitives.build(shape, _uniform(dis_rng, cfg.distractors.size_m), dis_rng)
        tipped = bool(dis_rng.random() < place.allow_tipped)
        pose = _place(
            mesh.points, dis_rng, (0.0, 360.0), tipped, region, placed, place.min_separation_m
        )
        material = mats.next()
        if pose is None:
            continue
        rot, t, fp = pose
        placed.append(fp)
        instances.append(
            Instance(
                len(instances) + 1,
                "distractor",
                0,
                "distractor",
                f"distractor:{shape}",
                rot,
                t,
                mesh.points,
                mesh,
                None,
                1.0,
                material,
            )
        )

    light_rng = stage_rng(dataset_seed, index, "lighting")
    lc = cfg.lighting
    lights = []
    for k in range(_integer(light_rng, lc.key_lights)):
        dist = float(light_rng.uniform(1.0, 2.5))
        elev = math.radians(float(light_rng.uniform(30.0, 80.0)))
        azim = float(light_rng.uniform(0.0, 2.0 * math.pi))
        pos = dist * np.array(
            [math.cos(elev) * math.cos(azim), math.cos(elev) * math.sin(azim), math.sin(elev)]
        )
        lights.append(
            Light(
                f"Key_{k + 1}",
                "sphere",
                _uniform(light_rng, lc.intensity),
                color_temperature_rgb(_uniform(light_rng, lc.color_temp_k)),
                pos,
                float(light_rng.uniform(0.05, 0.2)),
            )
        )
    dome_intensity = _uniform(light_rng, lc.dome.intensity)
    hdri_pick = int(light_rng.integers(0, max(len(lc.dome.hdris), 1)))
    if lc.dome.enabled:
        hdri = str((config_dir / lc.dome.hdris[hdri_pick]).resolve()) if lc.dome.hdris else None
        lights.append(Light("Dome", "dome", dome_intensity, (1.0, 1.0, 1.0), texture=hdri))

    cam_rng = stage_rng(dataset_seed, index, "camera")
    cc = cfg.camera
    dist = _uniform(cam_rng, cc.distance_m)
    elev = math.radians(_uniform(cam_rng, cc.elevation_deg))
    azim = math.radians(_uniform(cam_rng, cc.azimuth_deg))
    focal = _uniform(cam_rng, cc.focal_length_mm)
    jitter = cam_rng.uniform(-1.0, 1.0, size=3) * cc.jitter_target_m
    target = np.array([0.0, 0.0, 0.0]) + jitter
    eye = target + dist * np.array(
        [math.cos(elev) * math.cos(azim), math.cos(elev) * math.sin(azim), math.sin(elev)]
    )
    camera = PinholeCamera.look_at(
        eye, target, focal, cc.sensor_width_mm, cfg.image.width, cfg.image.height
    )
    for inst in instances:
        _project(inst, camera)

    return Scene(
        index,
        dataset_seed,
        cfg.image.width,
        cfg.image.height,
        kind,
        size,
        parts,
        support_mat,
        floor_mat,
        instances,
        lights,
        camera,
        categories,
    )
