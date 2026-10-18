"""Shared half of the world examples (``examples/minecraft``, ``examples/roam``, ...).

Every world example builds its own ``scene.usdc`` in the usd-synth stage layout (labelled
instances as ``/World/Objects/inst_NNNNN`` meshes, everything else under ``/World/Env``) and
samples camera poses over it. What happens after that is the same for every world, and lives
here:

* :func:`sample_lighting` and :func:`write_view` — a sun, a sky colour and a camera, written as
  a small ``views/view_NNNN.usda`` that sublayers the scene;
* :func:`render_views` — render the views with the stock Blender worker and turn each Object
  Index pass into COCO annotations, a semantic mask and a depth PNG;
* :func:`write_dataset` — the validated ``annotations.json`` and the ``render`` and
  ``throughput`` sections of ``meta.json``.

Labels are read from the render: each instance mesh gets its own Object Index, so occlusion,
the image border and the camera's view are exact.
"""

from __future__ import annotations

import math
import shutil
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdLux, UsdShade, Vt

from usd_synth.camera import PinholeCamera
from usd_synth.coco import build_coco, dump_json, validate_coco
from usd_synth.config import ImageConfig, RenderConfig
from usd_synth.labels import encode_rle, read_instances, tight_bbox
from usd_synth.passes import read_passes
from usd_synth.pipeline import cpu_model
from usd_synth.render import render_batch, scene_spec
from usd_synth.seeds import stage_rng

NS = "userProperties:usd_synth:"
DEPTH_SCALE_M = 0.01  # depth PNG unit: centimetres (outdoor ranges exceed 16-bit millimetres)

# ---------------------------------------------------------------------------------------------
# Lighting and the per-view layer


@dataclass(frozen=True)
class Lighting:
    sun_dir: tuple[float, float, float]  # direction *towards* the sun, Z-up
    sun_elevation_deg: float
    sun_intensity: float
    sun_color: tuple[float, float, float]
    sky_color: tuple[float, float, float]
    sky_intensity: float


def _lerp(a, b, t):
    return tuple(round(float(x + (y - x) * t), 6) for x, y in zip(a, b, strict=True))


def sample_lighting(
    seed: int,
    index: int,
    elevation_deg: tuple[float, float],
    azimuth_deg: tuple[float, float],
    sun_intensity: tuple[float, float],
    sky_intensity: tuple[float, float],
) -> Lighting:
    """Sun and sky of view ``index``, from ``stage_rng(seed, index, "lighting")``.

    A low sun gives a warm sun colour and a dusk-coloured sky; a high sun gives white light
    under a blue sky.
    """
    rng = stage_rng(seed, index, "lighting")
    el = float(rng.uniform(*elevation_deg))
    az = math.radians(float(rng.uniform(*azimuth_deg)))
    day = min(max((el - 8.0) / 40.0, 0.0), 1.0)  # 0 = sunrise/sunset, 1 = full day
    sun_dir = (
        math.cos(math.radians(el)) * math.cos(az),
        math.cos(math.radians(el)) * math.sin(az),
        math.sin(math.radians(el)),
    )
    sun = round(float(rng.uniform(*sun_intensity)), 4)
    sky = round(float(rng.uniform(*sky_intensity)) * (0.6 + 0.4 * day), 4)
    return Lighting(
        sun_dir=tuple(round(v, 6) for v in sun_dir),
        sun_elevation_deg=round(el, 3),
        sun_intensity=sun,
        sun_color=_lerp((1.0, 0.62, 0.38), (1.0, 0.96, 0.9), day),
        sky_color=_lerp((0.85, 0.6, 0.48), (0.45, 0.66, 1.0), day),
        sky_intensity=sky,
    )


def write_view(
    camera: PinholeCamera,
    light: Lighting,
    scene_path: Path,
    out_path: Path,
    seed: int,
    index: int,
) -> None:
    """A small ``.usda`` that sublayers the scene and adds this view's camera, sun and sky.

    ``out_path`` must sit one directory below the scene's directory (``views/view_NNNN.usda``
    next to ``scene.usdc``), so the sublayer path stays relative.
    """
    stage = Usd.Stage.CreateNew(str(out_path))
    rel = Path(scene_path).resolve().relative_to(Path(out_path).resolve().parent.parent)
    stage.GetRootLayer().subLayerPaths.append(f"../{rel.as_posix()}")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    set_attr(world, "scene_index", Sdf.ValueTypeNames.Int, index)
    set_attr(world, "dataset_seed", Sdf.ValueTypeNames.Int64, seed)
    set_attr(world, "image_width", Sdf.ValueTypeNames.Int, camera.width)
    set_attr(world, "image_height", Sdf.ValueTypeNames.Int, camera.height)

    UsdGeom.Xform.Define(stage, "/World/Lights")
    sun = UsdLux.DistantLight.Define(stage, "/World/Lights/Sun")
    sun.CreateIntensityAttr(light.sun_intensity)
    sun.CreateColorAttr(Gf.Vec3f(*light.sun_color))
    sun.CreateAngleAttr(1.0)
    # A DistantLight shines along its local -Z: aim -Z away from the sun.
    to_sun = np.asarray(light.sun_dir)
    rot = PinholeCamera.look_at(-to_sun, np.zeros(3), 50, 36, 2, 2).rotation
    m = np.eye(4)
    m[:3, :3] = rot.T
    sun.AddTransformOp().Set(Gf.Matrix4d(m.tolist()))
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr(light.sky_intensity)
    dome.CreateColorAttr(Gf.Vec3f(*light.sky_color))

    usd_cam = UsdGeom.Camera.Define(stage, "/World/Camera")
    usd_cam.AddTransformOp().Set(Gf.Matrix4d(camera.usd_matrix()))
    usd_cam.CreateProjectionAttr(UsdGeom.Tokens.perspective)
    usd_cam.CreateFocalLengthAttr(round(camera.focal_length_mm, 6))
    usd_cam.CreateHorizontalApertureAttr(round(camera.sensor_width_mm, 6))
    usd_cam.CreateVerticalApertureAttr(round(camera.vertical_aperture_mm, 6))
    usd_cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))
    cam_prim = usd_cam.GetPrim()
    set_attr(cam_prim, "focal_length_mm", Sdf.ValueTypeNames.Double, camera.focal_length_mm)
    set_attr(cam_prim, "sensor_width_mm", Sdf.ValueTypeNames.Double, camera.sensor_width_mm)
    stage.GetRootLayer().Save()


# ---------------------------------------------------------------------------------------------
# Scene geometry


def srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


class Looks:
    """One ``UsdPreviewSurface`` material per (name, sRGB colour), under ``/World/Looks``."""

    def __init__(self, stage: Usd.Stage, roughness: float = 0.85):
        self.stage = stage
        self.roughness = roughness
        self.cache: dict[tuple, UsdShade.Material] = {}

    def get(self, name: str, rgb) -> UsdShade.Material:
        key = (name, tuple(round(float(c), 4) for c in rgb))
        if key in self.cache:
            return self.cache[key]
        ident = Tf.MakeValidIdentifier(name)
        path = f"/World/Looks/{ident}"
        n = 1
        while self.stage.GetPrimAtPath(path):
            n += 1
            path = f"/World/Looks/{ident}_{n}"
        material = UsdShade.Material.Define(self.stage, path)
        shader = UsdShade.Shader.Define(self.stage, f"{path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        lin = Gf.Vec3f(*(round(srgb_to_linear(c), 6) for c in rgb))
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(lin)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(self.roughness)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        self.cache[key] = material
        return material


def write_mesh(
    stage: Usd.Stage,
    path: str,
    points: np.ndarray,
    counts: np.ndarray,
    indices: np.ndarray,
    face_looks: list[tuple[str, tuple[float, float, float]]],
    looks: Looks,
) -> UsdGeom.Mesh:
    """A flat-shaded mesh whose faces are bound to ``looks`` by (name, colour), one
    ``GeomSubset`` per look when there is more than one."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(points, dtype=np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.asarray(counts, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(indices, dtype=np.int32)))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    lo, hi = np.min(points, axis=0), np.max(points, axis=0)
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*lo), Gf.Vec3f(*hi)]))
    binding = UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
    kinds = sorted(set(face_looks))
    for n, key in enumerate(kinds):
        material = looks.get(*key)
        if len(kinds) == 1:
            binding.Bind(material)
            break
        faces = [i for i, k in enumerate(face_looks) if k == key]
        subset = binding.CreateMaterialBindSubset(
            Tf.MakeValidIdentifier(f"{key[0]}_{n}"), Vt.IntArray(faces)
        )
        UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(material)
    return mesh


def set_attr(prim: Usd.Prim, name: str, type_name, value) -> None:
    """``userProperties:usd_synth:<name>``, the namespace the render worker reads."""
    prim.CreateAttribute(NS + name, type_name, custom=True).Set(value)


def label_instance(prim: Usd.Prim, instance_id: int, category_id: int, category: str) -> None:
    """Mark ``prim`` as labelled instance ``instance_id`` of ``category``."""
    set_attr(prim, "instance_id", Sdf.ValueTypeNames.Int, instance_id)
    set_attr(prim, "category_id", Sdf.ValueTypeNames.Int, category_id)
    set_attr(prim, "category_name", Sdf.ValueTypeNames.String, category)
    set_attr(prim, "kind", Sdf.ValueTypeNames.String, "object")


def new_scene(path: Path, categories: list[str]) -> Usd.Stage:
    """An empty stage in the usd-synth layout: Z-up, metres, ``/World/{Looks,Env,Objects}``."""
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(root)
    set_attr(root, "categories", Sdf.ValueTypeNames.StringArray, Vt.StringArray(categories))
    UsdGeom.Scope.Define(stage, "/World/Looks")
    UsdGeom.Xform.Define(stage, "/World/Env")
    UsdGeom.Xform.Define(stage, "/World/Objects")
    return stage


def ground_hits_inside(camera: PinholeCamera, bounds, z: float = 0.0, grid=(9, 7)) -> bool:
    """True if every ray of a coarse pixel grid meets the plane at height ``z`` inside
    ``bounds`` (x0, y0, x1, y1), so the scene's edge and the sky stay out of the picture."""
    x0, y0, x1, y1 = bounds
    scale = camera.focal_length_mm / camera.sensor_width_mm * camera.width
    for u in np.linspace(0, camera.width, grid[0]):
        for v in np.linspace(0, camera.height, grid[1]):
            d_cam = np.array([(u - camera.width / 2) / scale, -(v - camera.height / 2) / scale, -1])
            d = camera.rotation @ d_cam
            if d[2] >= -1e-6:
                return False
            p = camera.position + d * ((z - camera.position[2]) / d[2])
            if not (x0 <= p[0] <= x1 and y0 <= p[1] <= y1):
                return False
    return True


# ---------------------------------------------------------------------------------------------
# Labels


def label_view(index: np.ndarray, instances, categories: list[str], min_pixels: int):
    """COCO annotations (no IDs yet) and the semantic class map for one rendered view."""
    cat_of = {i.instance_id: categories.index(i.category_name) + 1 for i in instances}
    lut = np.zeros(max(cat_of, default=0) + 1, dtype=np.uint8)
    for iid, cat in cat_of.items():
        lut[iid] = cat
    semantic = lut[np.clip(index, 0, len(lut) - 1)]
    semantic[(index < 0) | (index >= len(lut))] = 0
    ids, counts = np.unique(index[index > 0], return_counts=True)
    anns = []
    for iid, n in zip(ids.tolist(), counts.tolist(), strict=True):
        if iid not in cat_of or n < min_pixels:
            continue
        mask = index == iid
        anns.append(
            {
                "category_id": cat_of[iid],
                "segmentation": encode_rle(mask),
                "area": float(n),
                "bbox": tight_bbox(mask),
                "iscrowd": 0,
                "instance_id": int(iid),
            }
        )
    return anns, semantic


def palette_png(classes: np.ndarray, path: Path, colours: Sequence[tuple[int, int, int]]) -> None:
    im = Image.fromarray(classes.astype(np.uint8), mode="P")
    flat = [0, 0, 0] + [c for rgb in colours for c in rgb]
    im.putpalette(flat + [0] * (768 - len(flat)))
    im.save(path, optimize=True)


# ---------------------------------------------------------------------------------------------
# Rendering a run


@dataclass
class RenderRun:
    records: list[dict] = field(default_factory=list)  # per-image COCO input (build_coco)
    per_view: list[dict] = field(default_factory=list)  # instances and pixels per class
    render_s: list[float] = field(default_factory=list)  # Cycles render time per view
    wall_s: float = 0.0  # render + label wall time, Blender start-up included
    blender: dict = field(default_factory=dict)  # Blender version, device, threads


def render_views(
    out: Path,
    count: int,
    categories: list[str],
    colours: Sequence[tuple[int, int, int]],
    *,
    width: int,
    height: int,
    samples: int,
    denoise: bool,
    min_visible_pixels: int,
    blender: str,
    threads: int | None = None,
    batch_size: int = 6,
    log: Callable[[str], None] = print,
) -> RenderRun:
    """Render ``out/views/view_NNNN.usda`` for NNNN < ``count`` and write the labelled images.

    Writes ``images/``, ``semantic/`` (palette PNG, 0 = unlabelled, k = ``categories[k-1]``),
    ``depth/`` (16-bit centimetres, 0 = sky) and the worker's raw output under ``render/``.
    """
    for sub in ("images", "semantic", "depth", "render"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    shim = SimpleNamespace(  # the two Config sections render_batch reads
        render=RenderConfig(samples=samples, denoise=denoise),
        image=ImageConfig(width=width, height=height),
    )
    run = RenderRun()
    t_start = time.perf_counter()
    for b in range(0, count, batch_size):
        batch = range(b, min(b + batch_size, count))
        specs = [scene_spec(out / "views" / f"view_{i:04d}.usda", out / "render" / f"{i:04d}")
                 for i in batch]  # fmt: skip
        result = render_batch(shim, specs, blender, threads=threads, timeout=7200)
        for rj in result.scenes:
            run.blender = {k: rj[k] for k in ("blender", "device", "threads")}
        for i, rj in zip(batch, result.scenes, strict=True):
            run.render_s.append(rj["timings"]["render"])
            rdir = out / "render" / f"{i:04d}"
            usd = out / "views" / f"view_{i:04d}.usda"
            passes = read_passes(rdir / "aux.exr")
            anns, semantic = label_view(
                passes.index, read_instances(usd), categories, min_visible_pixels
            )
            shutil.copyfile(rdir / "rgb.png", out / "images" / f"{i:04d}.png")
            palette_png(semantic, out / "semantic" / f"{i:04d}.png", colours)
            depth = np.asarray(passes.depth, dtype=np.float64)
            ok = np.isfinite(depth) & (depth > 0) & (depth < 655.0)
            cm = np.where(ok, np.rint(depth / DEPTH_SCALE_M), 0).astype(np.uint16)
            Image.fromarray(cm).save(out / "depth" / f"{i:04d}.png")
            run.records.append(
                {
                    "scene_index": i,
                    "file_name": f"images/{i:04d}.png",
                    "depth_file": f"depth/{i:04d}.png",
                    "scene_file": f"views/view_{i:04d}.usda",
                    "width": width,
                    "height": height,
                    "annotations": anns,
                }
            )
            counts = {c: 0 for c in categories}
            for a in anns:
                counts[categories[a["category_id"] - 1]] += 1
            pixels = {c: int((semantic == k).sum()) for k, c in enumerate(categories, start=1)}
            run.per_view.append({"view": i, "instances": counts, "pixels": pixels})
            log(f"view {i:04d}: {rj['timings']['render']:.1f} s render, {len(anns)} instances")
    run.wall_s = time.perf_counter() - t_start
    return run


def write_dataset(
    out: Path,
    run: RenderRun,
    categories: list[str],
    description: str,
    supercategory: str,
    *,
    width: int,
    height: int,
    samples: int,
    denoise: bool,
) -> dict:
    """Write and validate ``annotations.json``; return the shared ``meta.json`` sections."""
    coco_categories = [
        {"id": k, "name": c, "supercategory": supercategory}
        for k, c in enumerate(categories, start=1)
    ]
    coco = build_coco(run.records, None, description, coco_categories)
    coco["info"]["depth_scale_m"] = DEPTH_SCALE_M
    dump_json(coco, out / "annotations.json")
    errors = validate_coco(out / "annotations.json")
    if errors:
        raise SystemExit("invalid COCO:\n" + "\n".join(errors[:20]))
    images = len(run.records)
    totals = {c: sum(v["instances"][c] for v in run.per_view) for c in categories}
    return {
        "annotations": {"total": len(coco["annotations"]), "per_class": totals},
        "render": {
            "engine": "cycles",
            **run.blender,
            "samples": samples,
            "denoise": denoise,
            "resolution": [width, height],
            "render_s_mean": round(float(np.mean(run.render_s)), 2),
            "render_s_p95": round(float(np.percentile(run.render_s, 95)), 2),
            "cpu": cpu_model(),
        },
        "throughput": {
            "images": images,
            "wall_seconds": round(run.wall_s, 1),
            "images_per_hour": round(images / run.wall_s * 3600, 1),
            "note": "render + label wall time (Blender start-up, USD import, render, labels); "
            "scene build excluded",
        },
    }
