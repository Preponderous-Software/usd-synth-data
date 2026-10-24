"""Strata render backend (``render.engine: strata``): a fast rasterized tier beside Cycles.

Runs in the host process through :mod:`usd_synth.strata` (ctypes over ``libstrata.so``),
headless on Mesa llvmpipe through EGL surfaceless. Reads the same flattened stage the Blender
worker imports and writes the same per-scene files (``rgb.png``, ``aux.exr`` with ``Depth`` and
``IndexOB`` channels, ``render.json``), so labelling and everything downstream are unchanged.
Design: Preponderous RFC 0026.

Each scene is drawn twice:

1. **Lit**: every mesh with its ``UsdPreviewSurface`` diffuse colour (or texture), sphere lights
   as shadowed point lights over a flat ambient from the dome intensity; the linear readback is
   sRGB-encoded on the host.
2. **ID**: every mesh unlit with its instance ID packed into the vertex colour
   (``r | g << 8 | b << 16``). The unlit shader writes the colour unchanged to an RGBA8 target
   with no multisampling, so the decoded IDs are exact: an index pass without edge blending.

Depth comes from ``strata_read_depth_f32`` (Strata ABI 1.9) when the library has it; with an
older library the depth channel is ``+inf`` everywhere (written as 0 = no hit in ``depth.png``)
and ``render.json`` records ``"depth": false``.

What a rasterizer cannot show is ignored and recorded per scene in ``render.json`` under
``ignored``: roughness, metallic, the dome colour and HDRI, and light radius (shadows are hard).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .strata import Engine, find_strata, linear_to_srgb

NS = "userProperties:usd_synth:"
NAME_RE = re.compile(r"^(?:inst|dis)_(\d+)")
# USD stages here are Z-up; Strata's world is Y-up: (x, y, z) -> (x, z, -y), a proper rotation.
Z2Y = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
GREY = (0.8, 0.8, 0.8)

# Light calibration (RFC 0026 open question 4). Blender imports a SphereLight of intensity I as
# a point light of I * pi W, whose diffuse response at distance d is albedo * I / (4 pi d^2).
# Strata's point light falls off as window^2 / (1 + 16 x^2), x = d / range. Each light gets
# range = RANGE_FACTOR * d_ref (d_ref = distance to the scene's centre) and the intensity that
# matches Cycles at d_ref, times EXPOSURE, which was set by comparing mean image brightness
# with Cycles renders of the same scenes (ARCHITECTURE.md, "Strata backend").
RANGE_FACTOR = 3.0
EXPOSURE = 0.03
AMBIENT_SCALE = 1.0


@dataclass
class Geometry:
    """Triangle soup for one stage, already in Strata's Y-up world (three vertices a triangle)."""

    positions: np.ndarray  # (V, 3) float32
    normals: np.ndarray  # (V, 3) float32
    colors: np.ndarray  # (V, 3) linear RGB
    ids: np.ndarray  # (V,) instance id, 0 = unlabelled
    uvs: np.ndarray  # (V, 2), 0 where untextured
    textures: list = field(default_factory=list)  # (V,) texture path or None, as an object array


@dataclass
class StageScene:
    geometry: Geometry
    lights: list  # (position_yup, intensity, color, radius)
    dome: dict | None
    camera: dict  # view, projection, eye, near, far
    ignored: list


def _tri_corners(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fan triangulation: (T, 3) face-vertex (corner) indices and each triangle's face."""
    counts = np.asarray(counts, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    ntri = np.maximum(counts - 2, 0)
    face = np.repeat(np.arange(len(counts)), ntri)
    first = np.cumsum(ntri) - ntri
    j = np.arange(int(ntri.sum())) - np.repeat(first, ntri) + 1
    s = starts[face]
    return np.stack([s, s + j, s + j + 1], axis=1), face


def _instance_id(prim) -> int:
    """From the nearest ancestor's ``instance_id`` property, else its name (``inst_0007``),
    as the Blender worker does; 0 for everything that is not an instance."""
    p = prim
    while p and not p.IsPseudoRoot():
        attr = p.GetAttribute(NS + "instance_id")
        if attr and attr.HasValue():
            return int(attr.Get())
        match = NAME_RE.match(p.GetName())
        if match:
            return int(match.group(1))
        p = p.GetParent()
    return 0


def _surface(prim):
    """(linear base colour, texture path or None, metallic, roughness) of the bound material."""
    from pxr import UsdShade

    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    if not material:
        return GREY, None, 0.0, 0.5
    shader = material.ComputeSurfaceSource()[0]
    if not shader:
        return GREY, None, 0.0, 0.5
    color, texture = GREY, None
    diffuse = shader.GetInput("diffuseColor")
    if diffuse:
        if diffuse.HasConnectedSource():
            source = diffuse.GetConnectedSource()[0]
            file_input = UsdShade.Shader(source.GetPrim()).GetInput("file")
            asset = file_input.Get() if file_input else None
            if asset:
                texture = asset.resolvedPath or asset.path
        elif diffuse.Get() is not None:
            color = tuple(float(c) for c in diffuse.Get())

    def scalar(name, default):
        inp = shader.GetInput(name)
        value = inp.Get() if inp else None
        return float(value) if value is not None else default

    return color, texture, scalar("metallic", 0.0), scalar("roughness", 0.5)


def read_stage(usd_path: Path | str, width: int, height: int) -> StageScene:
    """Meshes, lights and camera of a flattened stage, translated for Strata."""
    from pxr import Usd, UsdGeom, UsdLux

    stage = Usd.Stage.Open(str(usd_path))
    cache = UsdGeom.XformCache()
    parts = {k: [] for k in ("p", "n", "c", "i", "uv", "tex")}
    ignored = set()
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        if mesh.ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        fvi = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        if len(points) == 0 or len(counts) == 0:
            continue
        xf = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=np.float64)  # row vectors
        m3 = xf[:3, :3]
        corners, face = _tri_corners(counts)
        left = mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded
        if left != (np.linalg.det(m3) < 0):  # keep counter-clockwise winding seen from outside
            corners = corners[:, [0, 2, 1]]
        flat = corners.reshape(-1)
        world = (points @ m3 + xf[3, :3]) @ Z2Y.T
        pos = world[fvi[flat]]

        normals, interp = mesh.GetNormalsAttr().Get(), mesh.GetNormalsInterpolation()
        nrm = None
        if normals is not None and len(normals):
            normals = np.asarray(normals, dtype=np.float64)
            if interp == UsdGeom.Tokens.faceVarying and len(normals) == len(fvi):
                nrm = normals[flat]
            elif interp in (UsdGeom.Tokens.vertex, UsdGeom.Tokens.varying):
                nrm = normals[fvi[flat]]
            elif interp == UsdGeom.Tokens.uniform and len(normals) == len(counts):
                nrm = np.repeat(normals[face], 3, axis=0)
        if nrm is not None:
            nrm = nrm @ np.linalg.inv(m3).T @ Z2Y.T
            if left:
                nrm = -nrm
        else:  # flat normals from the (already oriented) triangles
            a, b, c = pos[0::3], pos[1::3], pos[2::3]
            nrm = np.repeat(np.cross(b - a, c - a), 3, axis=0)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)

        color, texture, metallic, _ = _surface(prim)
        ignored.add("roughness")
        if metallic > 0:
            ignored.add("metallic")
        uv = np.zeros((len(flat), 2))
        if texture:
            st = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
            if st and st.HasValue():
                values = np.asarray(st.ComputeFlattened(), dtype=np.float64)
                if st.GetInterpolation() == UsdGeom.Tokens.faceVarying and len(values) == len(fvi):
                    uv = values[flat]
                elif len(values) == len(points):
                    uv = values[fvi[flat]]
            uv = np.stack([uv[:, 0], 1.0 - uv[:, 1]], axis=1)  # Strata: v = 0 is the first row
        parts["p"].append(pos)
        parts["n"].append(nrm)
        parts["c"].append(np.broadcast_to(np.asarray(color, dtype=np.float64), pos.shape))
        parts["i"].append(np.full(len(pos), _instance_id(prim), dtype=np.int64))
        parts["uv"].append(uv)
        parts["tex"].append(np.full(len(pos), texture, dtype=object))
    if not parts["p"]:
        raise ValueError(f"{usd_path}: no visible meshes")
    geo = Geometry(
        positions=np.concatenate(parts["p"]).astype(np.float32),
        normals=np.concatenate(parts["n"]).astype(np.float32),
        colors=np.concatenate(parts["c"]).astype(np.float32),
        ids=np.concatenate(parts["i"]),
        uvs=np.concatenate(parts["uv"]).astype(np.float32),
        textures=np.concatenate(parts["tex"]),
    )

    lights, dome = [], None
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        if prim.IsA(UsdLux.SphereLight):
            light = UsdLux.SphereLight(prim)
            xf = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=np.float64)
            radius = light.GetRadiusAttr().Get() or 0.0
            if radius > 0:
                ignored.add("light_radius")
            lights.append(
                (Z2Y @ xf[3, :3], float(light.GetIntensityAttr().Get()),
                 tuple(float(c) for c in light.GetColorAttr().Get()), float(radius))
            )  # fmt: skip
        elif prim.IsA(UsdLux.DomeLight):
            light = UsdLux.DomeLight(prim)
            color = tuple(float(c) for c in light.GetColorAttr().Get())
            texture = light.GetTextureFileAttr().Get()
            dome = {"intensity": float(light.GetIntensityAttr().Get()), "color": color}
            if any(abs(c - color[0]) > 1e-6 for c in color):
                ignored.add("dome_color")
            if texture and texture.path:
                ignored.add("dome_texture")
    return StageScene(geo, lights, dome, _camera(stage, cache, width, height), sorted(ignored))


def _camera(stage, cache, width: int, height: int) -> dict:
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath("/World/Camera")
    if not prim or not prim.IsA(UsdGeom.Camera):
        prim = next(
            (p for p in Usd.PrimRange(stage.GetPseudoRoot()) if p.IsA(UsdGeom.Camera)), None
        )
    if prim is None:
        raise ValueError("stage has no camera")
    cam = UsdGeom.Camera(prim)
    focal = prim.GetAttribute(NS + "focal_length_mm").Get() or cam.GetFocalLengthAttr().Get()
    sensor = (
        prim.GetAttribute(NS + "sensor_width_mm").Get() or cam.GetHorizontalApertureAttr().Get()
    )
    near, far = (float(v) for v in (cam.GetClippingRangeAttr().Get() or (0.01, 100.0)))
    m = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=np.float64).T  # column vectors
    to_usd = np.eye(4)
    to_usd[:3, :3] = Z2Y.T  # Strata world -> USD world
    view = np.linalg.inv(m) @ to_usd
    fx = 2.0 * float(focal) / float(sensor)  # horizontal sensor fit, as camera.PinholeCamera
    fy = fx * width / height
    proj = np.zeros((4, 4))
    proj[0, 0], proj[1, 1] = fx, fy
    proj[2, 2] = -(far + near) / (far - near)
    proj[2, 3] = -2.0 * far * near / (far - near)
    proj[3, 2] = -1.0
    return {"view": view, "projection": proj, "eye": Z2Y @ m[:3, 3], "near": near, "far": far}


def point_lights(scene: StageScene) -> list[tuple]:
    """Sphere lights as Strata point lights, calibrated at the scene centre (see RANGE_FACTOR)."""
    geo = scene.geometry
    pts = geo.positions[geo.ids > 0] if (geo.ids > 0).any() else geo.positions
    centre = (pts.min(axis=0) + pts.max(axis=0)) / 2.0
    x2 = (1.0 / RANGE_FACTOR) ** 2
    falloff = (1.0 - x2 * x2) ** 2 / (1.0 + 16.0 * x2)
    out = []
    for pos, intensity, color, _radius in scene.lights:
        d = max(float(np.linalg.norm(pos - centre)), 1e-3)
        cycles = intensity / (4.0 * np.pi * d * d)  # diffuse response per unit albedo
        out.append((pos, RANGE_FACTOR * d, color, EXPOSURE * cycles / falloff, True))
    return out


def id_colors(ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.int64)
    if ids.max(initial=0) >= 1 << 24:
        raise ValueError("instance ids must be < 2**24 for the Strata ID pass")
    return np.stack([ids & 255, (ids >> 8) & 255, (ids >> 16) & 255], axis=1) / 255.0


def decode_ids(rgba: np.ndarray) -> np.ndarray:
    c = rgba.astype(np.int32)
    return c[..., 0] | (c[..., 1] << 8) | (c[..., 2] << 16)


def draw(engine: Engine, scene: StageScene, *, with_depth: bool = True):
    """Lit RGB (uint8, sRGB), instance index (int32) and planar depth (float32 or None)."""
    geo = scene.geometry
    engine.set_camera(**scene.camera)
    engine.clear_meshes()
    keys = [t for t in dict.fromkeys(geo.textures.tolist())]
    for key in keys:
        sel = geo.textures == key
        if key is None:
            engine.add_mesh(geo.positions[sel], geo.normals[sel], geo.colors[sel])
        else:
            white = np.ones((int(sel.sum()), 3), dtype=np.float32)
            engine.add_mesh(
                geo.positions[sel], geo.normals[sel], white, geo.uvs[sel], engine.texture(key)
            )
    engine.set_point_lights(point_lights(scene))
    ambient = scene.dome["intensity"] * scene.dome["color"][0] if scene.dome else 0.0
    engine.set_space_lighting(AMBIENT_SCALE * ambient)
    lit = engine.frame()
    rgb = np.rint(linear_to_srgb(lit[..., :3] / 255.0) * 255.0).astype(np.uint8)

    engine.clear_meshes()
    engine.add_mesh(geo.positions, geo.normals, id_colors(geo.ids), unlit=True)
    index = decode_ids(engine.frame())
    depth = engine.read_depth() if with_depth and engine.has_depth else None
    return rgb, index, depth


def write_aux_exr(path: Path | str, depth: np.ndarray | None, index: np.ndarray) -> None:
    """The channels :func:`usd_synth.passes.read_passes` reads from the Blender worker's EXR."""
    import OpenEXR

    if depth is None:
        depth = np.full(index.shape, np.inf, dtype=np.float32)
    header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
    channels = {
        "ViewLayer.Depth.Z": np.ascontiguousarray(depth, dtype=np.float32),
        "ViewLayer.IndexOB.X": np.ascontiguousarray(index, dtype=np.float32),
    }
    OpenEXR.File(header, channels).write(str(path))


_ENGINES: dict[str, Engine] = {}
_WARNED: set[str] = set()


def engine_for(lib_path: str, width: int, height: int, threads: int | None) -> tuple[Engine, float]:
    """The process's engine (created once) and the seconds spent creating it now (0 if reused)."""
    if lib_path in _ENGINES:
        engine = _ENGINES[lib_path]
        engine.resize(width, height)
        return engine, 0.0
    # CPU rendering on llvmpipe; LP_NUM_THREADS is read when the context is created.
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    os.environ.setdefault("GALLIUM_DRIVER", "llvmpipe")
    if threads:
        os.environ["LP_NUM_THREADS"] = str(threads)
    t0 = time.perf_counter()
    engine = Engine(lib_path, width, height)
    _ENGINES[lib_path] = engine
    return engine, time.perf_counter() - t0


def render_one(engine: Engine, spec: dict, width: int, height: int) -> dict:
    from PIL import Image

    timings = {}
    t0 = time.perf_counter()
    scene = read_stage(spec["usd"], width, height)
    timings["import"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    rgb, index, depth = draw(engine, scene)
    timings["render"] = time.perf_counter() - t1
    t2 = time.perf_counter()
    out = Path(spec["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(out / "rgb.png")
    write_aux_exr(out / "aux.exr", depth, index)
    timings["write"] = time.perf_counter() - t2
    for item in scene.ignored:
        if item not in _WARNED:
            _WARNED.add(item)
            print(f"usd-synth: strata backend ignores {item} (recorded in render.json)",
                  file=sys.stderr)  # fmt: skip
    info = {
        "usd": spec["usd"],
        "engine": "strata",
        "blender": None,
        "strata": engine.version,
        "strata_abi": engine.abi,
        "gl": engine.gpu,
        "device": "cpu",
        "threads": int(os.environ.get("LP_NUM_THREADS", "0") or 0),
        "samples": None,
        "seed": spec["seed"],
        "resolution": [width, height],
        "depth": depth is not None,
        "ignored": scene.ignored,
        "instance_ids": sorted(int(i) for i in np.unique(index) if i),
        "timings": timings,
    }
    (out / "render.json").write_text(json.dumps(info, indent=2, sort_keys=True), encoding="utf-8")
    return info


def render_batch_strata(cfg, specs: list[dict], lib_path: str | None, threads: int | None = None):
    """Render ``specs`` in this process; same result shape as :func:`render.render_batch`."""
    from .render import BatchResult, RenderError

    try:
        engine, start_s = engine_for(
            find_strata(lib_path), cfg.image.width, cfg.image.height, threads
        )
    except Exception as err:  # library missing, ABI mismatch, no EGL/llvmpipe
        raise RenderError(f"Strata backend could not start: {err}") from None
    result = BatchResult(blender_start_s=start_s)
    for spec in specs:
        try:
            result.scenes.append(render_one(engine, spec, cfg.image.width, cfg.image.height))
        except Exception as err:
            result.errors.append(
                {"out_dir": spec["out_dir"], "message": f"{type(err).__name__}: {err}"}
            )
    if result.errors:
        detail = "; ".join(f"{e['out_dir']}: {e['message']}" for e in result.errors)
        raise RenderError(f"Strata backend failed: {detail}")
    return result
