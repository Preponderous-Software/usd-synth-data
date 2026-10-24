"""A small ctypes binding to Strata's versioned C ABI (``include/strata/strata.h``).

Only what the Strata render backend (:mod:`usd_synth.strata_backend`) uses is declared. Every
struct is checked against ``strata_struct_size()`` when the library is loaded, so a library
built with a different layout fails loudly instead of corrupting memory.

The library is found as ``--strata`` > ``$USD_SYNTH_STRATA`` > ``/opt/strata/lib/libstrata.so``
(the worker image) > ``libstrata.so`` on the loader path, mirroring :func:`render.find_blender`.

Depth readback (``strata_read_depth_f32``) arrives in ABI 1.9. It is used when the loaded
library has it and skipped otherwise, so ABI 1.8 (Strata 0.4.0) renders RGB and labels with no
depth pass; :attr:`Engine.has_depth` says which.
"""

from __future__ import annotations

import ctypes as C
import ctypes.util
import os
from pathlib import Path

import numpy as np

ABI_MAJOR = 1
MIN_ABI_MINOR = 8  # ABI 1.7 textured/unlit meshes, 1.5 SPACE lighting, 1.8 as released in 0.4.0
DEPTH_ABI_MINOR = 9  # strata_read_depth_f32
IMAGE_PATH = "/opt/strata/lib/libstrata.so"

SURFACE_HEADLESS = 1
LIGHTING_SPACE = 2
CLEAR_COLOR, CLEAR_DEPTH = 1, 2
MESH_UNLIT = 0x1
TEXTURE_MIPMAPS = 0x1
POINT_LIGHT_CAST_SHADOWS = 0x1
MAX_MESHES = 256
MAX_MESH_VERTICES = 16_777_216
MAX_POINT_LIGHTS = 32
MAX_SHADOWED_POINT_LIGHTS = 4

F3, F4, F16 = C.c_float * 3, C.c_float * 4, C.c_float * 16
_FP, _U32P = C.POINTER(C.c_float), C.POINTER(C.c_uint32)


class StrataError(RuntimeError):
    pass


class EngineDesc(C.Structure):
    _fields_ = [
        ("struct_size", C.c_uint32), ("abi_version", C.c_uint32), ("surface", C.c_int32),
        ("width", C.c_int32), ("height", C.c_int32), ("flags", C.c_uint32),
        ("title", C.c_char_p), ("get_proc_address", C.c_void_p),
        ("get_proc_address_user", C.c_void_p),
    ]  # fmt: skip


class Camera(C.Structure):
    _fields_ = [
        ("view", F16), ("projection", F16), ("eye", F3),
        ("near_plane", C.c_float), ("far_plane", C.c_float),
    ]  # fmt: skip


class Lighting(C.Structure):
    _fields_ = [
        ("mode", C.c_int32), ("legacy_direction", F3), ("legacy_ambient", C.c_float),
        ("time_of_day", C.c_float), ("latitude_degrees", C.c_float),
        ("cascade_count", C.c_int32), ("shadow_map_size", C.c_int32),
        ("shadow_distance", C.c_float), ("split_lambda", C.c_float), ("fog", C.c_int32),
    ]  # fmt: skip


class FrameDesc(C.Structure):
    _fields_ = [("clear_flags", C.c_uint32), ("clear_color", F4)]


class PointLight(C.Structure):
    _fields_ = [
        ("position", F3), ("range", C.c_float), ("color", F3),
        ("intensity", C.c_float), ("flags", C.c_uint32),
    ]  # fmt: skip


class Mesh(C.Structure):
    _fields_ = [
        ("positions", _FP), ("normals", _FP), ("colors", _FP), ("indices", _U32P),
        ("vertex_count", C.c_int32), ("index_count", C.c_int32), ("color", F4),
    ]  # fmt: skip


class TexturedMesh(C.Structure):
    _fields_ = [("mesh", Mesh), ("uvs", _FP), ("texture", C.c_uint32), ("flags", C.c_uint32)]


# strata.h STRATA_STRUCT_* ids
_STRUCTS = {
    1: EngineDesc, 2: Camera, 3: Lighting, 4: FrameDesc, 7: PointLight, 9: Mesh, 14: TexturedMesh,
}  # fmt: skip


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


_LIBS: dict[str, C.CDLL] = {}


def load(path: str) -> C.CDLL:
    """Load ``libstrata`` once per process, check the ABI and every struct size."""
    if path in _LIBS:
        return _LIBS[path]
    try:
        lib = C.CDLL(path)
    except OSError as err:
        raise StrataError(f"cannot load Strata library {path}: {err}") from None
    lib.strata_abi_version.restype = C.c_uint32
    lib.strata_abi_minor.restype = C.c_uint32
    lib.strata_version_string.restype = C.c_char_p
    lib.strata_last_error.restype = C.c_char_p
    lib.strata_gpu_description.restype = C.c_char_p
    lib.strata_gpu_description.argtypes = [C.c_void_p]
    major, minor = lib.strata_abi_version(), lib.strata_abi_minor()
    if major != ABI_MAJOR or minor < MIN_ABI_MINOR:
        raise StrataError(
            f"{path} has Strata ABI {major}.{minor}; usd-synth needs {ABI_MAJOR}.{MIN_ABI_MINOR}+"
        )
    for sid, struct in _STRUCTS.items():
        size = lib.strata_struct_size(sid)
        if size != C.sizeof(struct):
            raise StrataError(
                f"{path}: {struct.__name__} is {size} bytes, the binding expects "
                f"{C.sizeof(struct)}; the library does not match strata.h ABI {ABI_MAJOR}.x"
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
        self._handle = C.c_void_p()
        desc = EngineDesc(C.sizeof(EngineDesc), ABI_MAJOR, SURFACE_HEADLESS, width, height, 0)
        self._ok(lib.strata_create(C.byref(desc), C.byref(self._handle)))
        self.width, self.height = width, height
        self.version = lib.strata_version_string().decode()
        self.abi = f"{lib.strata_abi_version()}.{lib.strata_abi_minor()}"
        self.gpu = lib.strata_gpu_description(self._handle).decode()
        self.has_depth = lib.strata_abi_minor() >= DEPTH_ABI_MINOR and hasattr(
            lib, "strata_read_depth_f32"
        )
        self._textures: dict[str, int] = {}

    def _ok(self, status: int) -> None:
        if status != 0:
            raise StrataError(f"Strata status {status}: {self.lib.strata_last_error().decode()}")

    def close(self) -> None:
        if self._handle:
            self.lib.strata_destroy(self._handle)
            self._handle = C.c_void_p()

    def resize(self, width: int, height: int) -> None:
        if (width, height) != (self.width, self.height):
            self._ok(self.lib.strata_resize(self._handle, width, height))
            self.width, self.height = width, height

    def set_camera(self, view: np.ndarray, projection: np.ndarray, eye, near: float, far: float):
        """``view`` and ``projection`` are 4x4 column-vector matrices (``clip = P @ V @ p``)."""

        def colmajor(m):
            return F16(*np.asarray(m, dtype=np.float32).T.reshape(-1))

        cam = Camera(colmajor(view), colmajor(projection), F3(*eye), near, far)
        self._ok(self.lib.strata_set_camera(self._handle, C.byref(cam)))

    def set_space_lighting(self, ambient: float) -> None:
        light = Lighting()
        self._ok(self.lib.strata_lighting_defaults(LIGHTING_SPACE, C.byref(light)))
        light.legacy_ambient = float(np.clip(ambient, 0.0, 1.0))
        self._ok(self.lib.strata_set_lighting(self._handle, C.byref(light)))

    def set_point_lights(self, lights: list[tuple]) -> None:
        """``lights``: ``(position, range, color, intensity, cast_shadows)`` tuples."""
        lights = lights[:MAX_POINT_LIGHTS]
        arr = (PointLight * max(len(lights), 1))()
        shadowed = 0
        for k, (pos, rng, color, intensity, shadows) in enumerate(lights):
            flag = 0
            if shadows and shadowed < MAX_SHADOWED_POINT_LIGHTS:
                flag, shadowed = POINT_LIGHT_CAST_SHADOWS, shadowed + 1
            arr[k] = PointLight(F3(*pos), rng, F3(*color), intensity, flag)
        self._ok(
            self.lib.strata_set_point_lights(self._handle, arr if lights else None, len(lights))
        )

    def clear_meshes(self) -> None:
        self._ok(self.lib.strata_clear_meshes(self._handle))

    def texture(self, path: str) -> int:
        """Upload an image file once per engine (decoded from sRGB to Strata's linear space)."""
        if path not in self._textures:
            from PIL import Image

            img = np.asarray(Image.open(path).convert("RGBA"), dtype=np.float64) / 255.0
            img[..., :3] = srgb_to_linear(img[..., :3])
            rgba = np.ascontiguousarray(np.rint(img * 255.0), dtype=np.uint8)
            h, w = rgba.shape[:2]
            out = C.c_uint32()
            data = rgba.ctypes.data_as(C.POINTER(C.c_uint8))
            self._ok(
                self.lib.strata_create_texture_rgba8(
                    self._handle, w, h, data, TEXTURE_MIPMAPS, C.byref(out)
                )
            )
            self._textures[path] = out.value
        return self._textures[path]

    def add_mesh(self, positions, normals, colors, uvs=None, texture=0, unlit=False) -> None:
        """Unindexed triangle soup (every three vertices one triangle), split under the cap."""
        n = len(positions)
        step = MAX_MESH_VERTICES - MAX_MESH_VERTICES % 3
        for start in range(0, n, step):
            sl = slice(start, min(start + step, n))
            p = np.ascontiguousarray(positions[sl], dtype=np.float32)
            nr = None if normals is None else np.ascontiguousarray(normals[sl], dtype=np.float32)
            c = np.ascontiguousarray(colors[sl], dtype=np.float32)
            uv = None if uvs is None else np.ascontiguousarray(uvs[sl], dtype=np.float32)
            idx = np.arange(len(p), dtype=np.uint32)
            mesh = Mesh(
                _fptr(p), None if nr is None else _fptr(nr), _fptr(c),
                idx.ctypes.data_as(_U32P), len(p), len(idx), F4(1.0, 1.0, 1.0, 1.0),
            )  # fmt: skip
            tm = TexturedMesh(
                mesh, None if uv is None else _fptr(uv), texture, MESH_UNLIT if unlit else 0
            )
            out = C.c_uint32()
            self._ok(self.lib.strata_add_textured_mesh(self._handle, C.byref(tm), C.byref(out)))

    def frame(self, clear_color=(0.0, 0.0, 0.0, 1.0)) -> np.ndarray:
        """Draw one frame and read it back: ``(H, W, 4)`` uint8, top row first."""
        desc = FrameDesc(CLEAR_COLOR | CLEAR_DEPTH, F4(*clear_color))
        self._ok(self.lib.strata_begin_frame(self._handle, C.byref(desc)))
        self._ok(self.lib.strata_end_frame(self._handle))
        buf = np.empty((self.height, self.width, 4), dtype=np.uint8)
        w, h = C.c_int32(), C.c_int32()
        data = buf.ctypes.data_as(C.POINTER(C.c_uint8))
        self._ok(
            self.lib.strata_read_pixels_rgba8(
                self._handle, data, buf.nbytes, C.byref(w), C.byref(h)
            )
        )
        return buf

    def read_depth(self) -> np.ndarray:
        """Planar depth of the last frame in world units, ``+inf`` where nothing was drawn
        (ABI 1.9). Raises :class:`StrataError` on older libraries."""
        if not self.has_depth:
            raise StrataError(f"Strata {self.version} (ABI {self.abi}) has no depth readback")
        fn = self.lib.strata_read_depth_f32
        buf = np.empty((self.height, self.width), dtype=np.float32)
        w, h = C.c_int32(), C.c_int32()
        self._ok(fn(self._handle, _fptr(buf), buf.nbytes, C.byref(w), C.byref(h)))
        return buf


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)
