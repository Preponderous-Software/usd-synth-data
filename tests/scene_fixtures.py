"""Hand-built scenes with known geometry, for renderer and label-correctness tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from usd_synth.camera import PinholeCamera, convex_hull_area
from usd_synth.scenegen import primitives
from usd_synth.scenegen.tabletop import Instance, Light, Material, Scene
from usd_synth.scenegen.usd_writer import scene_to_usda

GREY = Material("mat_0001", (0.5, 0.5, 0.5), 0.6, 0.0)


def _material(n: int, rgb) -> Material:
    return Material(f"mat_{n:04d}", rgb, 0.5, 0.0)


def top_down_scene(
    cubes: list[tuple[float, float, float]],
    *,
    camera_height: float = 1.0,
    focal_length_mm: float = 35.0,
    width: int = 160,
    height: int = 120,
    index: int = 0,
    dataset_seed: int = 0,
) -> Scene:
    """A large table (top at z = 0) seen straight down, with axis-aligned cubes ``(x, y, size)``."""
    support = [("top", primitives.box([4.0, 4.0, 0.04]), np.array([0.0, 0.0, -0.02]))]
    camera = PinholeCamera.look_at(
        [0.0, 0.0, camera_height], [0.0, 0.0, 0.0], focal_length_mm, 36.0, width, height
    )
    instances = []
    for k, (x, y, size) in enumerate(cubes, start=1):
        mesh = primitives.box([size, size, size])
        inst = Instance(
            k, "object", 1, "box", "box", np.eye(3), np.array([x, y, size / 2]),
            mesh.points, mesh, None, np.eye(3), _material(k + 2, (0.8, 0.2 * k % 1, 0.1)),
        )  # fmt: skip
        uv, _ = camera.project(inst.world_points())
        lo, hi = uv.min(axis=0), uv.max(axis=0)
        inst.projected_bbox = (lo[0], lo[1], hi[0] - lo[0], hi[1] - lo[1])
        inst.projected_area_px = convex_hull_area(uv)
        instances.append(inst)
    lights = [
        Light("Key_1", "sphere", 60.0, (1.0, 1.0, 1.0), np.array([0.5, 0.5, 2.0]), 0.1),
        Light("Dome", "dome", 0.5, (1.0, 1.0, 1.0)),
    ]
    return Scene(
        index, dataset_seed, width, height, "table", (4.0, 4.0), support, GREY,
        _material(2, (0.3, 0.3, 0.3)), instances, lights, camera, ["box"],
    )  # fmt: skip


def write(scene: Scene, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(scene_to_usda(scene), encoding="utf-8")
    return path
