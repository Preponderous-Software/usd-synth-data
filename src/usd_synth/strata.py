"""The Strata engine as the render backend (:mod:`usd_synth.strata_backend`) uses it.

The C ABI (``include/strata/strata.h``) is reached through Strata's official Python binding,
``strata-engine`` (``bindings/python`` in the Strata repository, pinned in ``pyproject.toml``),
which checks the ABI major version and every struct layout against the loaded library. This
module adds only what the backend needs on top: finding the library, NumPy in and out, and
the minimum ABI.

The library is found as ``--strata`` > ``$USD_SYNTH_STRATA`` > ``/opt/strata/lib/libstrata.so``
(the worker image) > ``libstrata.so`` on the loader path, mirroring :func:`render.find_blender`.

Depth readback (``strata_read_depth_f32``) arrives in ABI 1.9 (Strata 0.5.0). It is used when the
loaded library has it and skipped otherwise, so ABI 1.8 (Strata 0.4.0) renders RGB and labels
with no depth pass; :attr:`Engine.has_depth` says which.
"""

from __future__ import annotations

import ctypes as C
import ctypes.util
import os
from pathlib import Path

import numpy as np
import strata as _strata

MIN_ABI_MINOR = 8  # ABI 1.7 textured/unlit meshes, 1.5 SPACE lighting, 1.8 as released in 0.4.0
IMAGE_PATH = "/opt/strata/lib/libstrata.so"

_FP, _U32P = C.POINTER(C.c_float), C.POINTER(C.c_uint32)


class StrataError(RuntimeError):
    pass


def find_strata(explicit: str | None = None) -> str:
    """Path of ``libstrata.so``: ``--strata`` > ``$USD_SYNTH_STRATA`` > the image path > the
    loader path."""
    for candidate in (explicit, os.environ.get("USD_SYNTH_STRATA"), IMAGE_PATH):
        if candidate:
            if Path(candidate).exists():
                return str(candidate)
            if candidate is not IMAGE_PATH:
                raise StrataError(f"Strata library not found at {candidate}")
    found = ctypes.util.find_library("strata")
    if found:
        return found
    raise StrataError(
        "Strata library not found: pass --strata, set USD_SYNTH_STRATA, or put libstrata.so "
        "on the loader path (release tarballs: github.com/Preponderous-Software/strata/releases)"
    )


_LIBS: dict[str, _strata.Library] = {}


def load(path: str) -> _strata.Library:
    """Load ``libstrata`` once per process through the official binding (ABI and layouts
    checked) and require ABI 1.8 or newer."""
    if path in _LIBS:
        return _LIBS[path]
    try:
        lib = _strata.Library(C.CDLL(path), path)
    except OSError as err:
        raise StrataError(f"cannot load Strata library {path}: {err}") from None
    except _strata.StrataError as err:
        raise StrataError(f"{path}: {err}") from None
    if lib.abi_minor < MIN_ABI_MINOR:
        raise StrataError(
            f"{path} has Strata ABI {lib.abi_version}.{lib.abi_minor}; "
            f"usd-synth needs {_strata.ABI_VERSION}.{MIN_ABI_MINOR}+"
        )
    _LIBS[path] = lib
    return lib


def _fptr(a: np.ndarray):
    return a.ctypes.data_as(_FP)


class Engine:
    """One headless Strata engine (one EGL context), reused for every scene of a process."""

    def __init__(self, lib_path: str, width: int, height: int):
        self.lib_path = lib_path
        self.lib = lib = load(lib_path)
        try:
            self._engine = _strata.Engine(lib, width, height, surface=_strata.SURFACE_HEADLESS)
        except _strata.StrataError as err:
            raise StrataError(str(err)) from None
        self.width, self.height = width, height
        self.version = lib.version
        self.abi = f"{lib.abi_version}.{lib.abi_minor}"
        self.gpu = self._engine.gpu_description()
        self.has_depth = lib.has("strata_read_depth_f32")
        self._textures: dict[str, int] = {}

    def close(self) -> None:
        self._engine.close()

    def resize(self, width: int, height: int) -> None:
        if (width, height) != (self.width, self.height):
            self._engine.resize(width, height)
            self.width, self.height = width, height

    def set_camera(self, view: np.ndarray, projection: np.ndarray, eye, near: float, far: float):
        """``view`` and ``projection`` are 4x4 column-vector matrices (``clip = P @ V @ p``)."""
        cam = _strata.Camera()
        cam.view[:] = np.asarray(view, dtype=np.float32).T.reshape(-1).tolist()
        cam.projection[:] = np.asarray(projection, dtype=np.float32).T.reshape(-1).tolist()
        cam.eye[:] = [float(v) for v in eye]
        cam.near_plane, cam.far_plane = near, far
        self._engine.set_camera(cam)

    def set_space_lighting(self, ambient: float) -> None:
        light = self.lib.lighting_defaults(_strata.LIGHTING_SPACE)
        light.legacy_ambient = float(np.clip(ambient, 0.0, 1.0))
        self._engine.set_lighting(light)

    def set_point_lights(self, lights: list[tuple]) -> None:
        """``lights``: ``(position, range, color, intensity, cast_shadows)`` tuples."""
        out, shadowed = [], 0
        for pos, rng, color, intensity, shadows in lights[: _strata.MAX_POINT_LIGHTS]:
            light = _strata.PointLight()
            light.position[:] = [float(v) for v in pos]
            light.range = rng
            light.color[:] = [float(v) for v in color]
            light.intensity = intensity
            if shadows and shadowed < _strata.MAX_SHADOWED_POINT_LIGHTS:
                light.flags, shadowed = _strata.POINT_LIGHT_CAST_SHADOWS, shadowed + 1
            out.append(light)
        self._engine.set_point_lights(out)

    def clear_meshes(self) -> None:
        self._engine.clear_meshes()

    def texture(self, path: str) -> int:
        """Upload an image file once per engine (decoded from sRGB to Strata's linear space)."""
        if path not in self._textures:
            from PIL import Image

            img = np.asarray(Image.open(path).convert("RGBA"), dtype=np.float64) / 255.0
            img[..., :3] = srgb_to_linear(img[..., :3])
            rgba = np.ascontiguousarray(np.rint(img * 255.0), dtype=np.uint8)
            h, w = rgba.shape[:2]
            self._textures[path] = self._engine.create_texture_rgba8(
                w, h, rgba.tobytes(), _strata.TEXTURE_MIPMAPS
            )
        return self._textures[path]

    def add_mesh(self, positions, normals, colors, uvs=None, texture=0, unlit=False) -> None:
        """Unindexed triangle soup (every three vertices one triangle), split under the cap.

        Uses the binding's structs with NumPy-owned buffers (no per-vertex Python copies)."""
        n = len(positions)
        cap = _strata.MAX_MESH_VERTICES
        step = cap - cap % 3
        for start in range(0, n, step):
            sl = slice(start, min(start + step, n))
            p = np.ascontiguousarray(positions[sl], dtype=np.float32)
            nr = None if normals is None else np.ascontiguousarray(normals[sl], dtype=np.float32)
            c = np.ascontiguousarray(colors[sl], dtype=np.float32)
            uv = None if uvs is None else np.ascontiguousarray(uvs[sl], dtype=np.float32)
            idx = np.arange(len(p), dtype=np.uint32)
            tm = _strata.TexturedMesh()
            m = tm.mesh
            m.positions, m.colors = _fptr(p), _fptr(c)
            m.normals = None if nr is None else _fptr(nr)
            m.indices = idx.ctypes.data_as(_U32P)
            m.vertex_count = m.index_count = len(p)
            m.color[:] = [1.0, 1.0, 1.0, 1.0]
            tm.uvs = None if uv is None else _fptr(uv)
            tm.texture = texture
            tm.flags = _strata.MESH_UNLIT if unlit else 0
            out = C.c_uint32()
            fn = self.lib.require("strata_add_textured_mesh")
            self.lib.check(fn(self._engine.handle, C.byref(tm), C.byref(out)), "add_mesh")

    def frame(self, clear_color=(0.0, 0.0, 0.0, 1.0)) -> np.ndarray:
        """Draw one frame and read it back: ``(H, W, 4)`` uint8, top row first."""
        self._engine.begin_frame(clear_color=clear_color)
        self._engine.end_frame()
        rgba, w, h = self._engine.read_pixels()
        return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4).copy()

    def read_depth(self) -> np.ndarray:
        """Planar depth of the last frame in world units, ``+inf`` where nothing was drawn
        (ABI 1.9). Raises :class:`StrataError` on older libraries."""
        if not self.has_depth:
            raise StrataError(f"Strata {self.version} (ABI {self.abi}) has no depth readback")
        depth, w, h = self._engine.read_depth()
        return np.frombuffer(depth, dtype=np.float32).reshape(h, w).copy()


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)
