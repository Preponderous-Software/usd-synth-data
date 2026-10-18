"""Pinhole camera model shared by the scene generator and the label checks.

Conventions match USD and Blender: the camera looks down its local -Z axis with +Y up, the
sensor is fitted horizontally (``sensor_width_mm`` spans the image width), and pixel
coordinates have their origin at the top-left corner of the top-left pixel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PinholeCamera:
    rotation: np.ndarray  # (3, 3) columns = camera x, y, z axes in world space
    position: np.ndarray  # (3,)
    focal_length_mm: float
    sensor_width_mm: float
    width: int
    height: int

    @classmethod
    def look_at(cls, eye, target, focal_length_mm, sensor_width_mm, width, height, up=(0, 0, 1)):
        eye = np.asarray(eye, dtype=np.float64)
        forward = np.asarray(target, dtype=np.float64) - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.asarray(up, dtype=np.float64))
        if np.linalg.norm(right) < 1e-9:  # looking straight down/up: pick a stable right
            right = np.array([1.0, 0.0, 0.0])
        right /= np.linalg.norm(right)
        cam_up = np.cross(right, forward)
        rot = np.stack([right, cam_up, -forward], axis=1)
        return cls(rot, eye, float(focal_length_mm), float(sensor_width_mm), width, height)

    @property
    def vertical_aperture_mm(self) -> float:
        return self.sensor_width_mm * self.height / self.width

    def to_camera(self, points_world) -> np.ndarray:
        pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        return (pts - self.position) @ self.rotation

    def project(self, points_world) -> tuple[np.ndarray, np.ndarray]:
        """Pixel coordinates (N, 2) and a mask of points in front of the camera."""
        cam = self.to_camera(points_world)
        depth = -cam[:, 2]
        in_front = depth > 1e-6
        safe = np.where(in_front, depth, 1.0)
        scale = self.focal_length_mm / self.sensor_width_mm * self.width
        u = self.width / 2.0 + scale * cam[:, 0] / safe
        v = self.height / 2.0 - scale * cam[:, 1] / safe
        return np.stack([u, v], axis=1), in_front

    def usd_matrix(self) -> list[list[float]]:
        """Row-vector 4x4 transform (USD convention) of the camera prim."""
        m = np.eye(4)
        m[:3, :3] = self.rotation.T
        m[3, :3] = self.position
        return m.tolist()


def convex_hull_area(points2d: np.ndarray) -> float:
    """Area of the convex hull of 2-D points (Andrew's monotone chain)."""
    pts = sorted({(float(x), float(y)) for x, y in points2d})
    if len(pts) < 3:
        return 0.0

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1])
    x, y = hull[:, 0], hull[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
