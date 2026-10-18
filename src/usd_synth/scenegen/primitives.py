"""Procedural primitive meshes, centred on the origin, sized in metres.

Each builder returns a :class:`MeshData` with polygon faces wound counter-clockwise seen from
outside (USD's default ``rightHanded`` orientation). Tessellation is fixed so the same
dimensions always give the same mesh.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SEGMENTS = 24
RINGS = 12


@dataclass(frozen=True)
class MeshData:
    points: np.ndarray  # (N, 3) float64
    face_counts: list[int]
    face_indices: list[int]


def _quads_grid(rows: int, cols: int, offset: int = 0, wrap: bool = True):
    """Quad faces over a (rows x cols) vertex grid, wrapping around the columns."""
    counts, idx = [], []
    ncols = cols if wrap else cols - 1
    for r in range(rows - 1):
        for c in range(ncols):
            c2 = (c + 1) % cols
            a = offset + r * cols + c
            b = offset + r * cols + c2
            d = offset + (r + 1) * cols + c
            e = offset + (r + 1) * cols + c2
            counts.append(4)
            idx += [a, b, e, d]
    return counts, idx


def box(dims) -> MeshData:
    hx, hy, hz = (float(d) / 2.0 for d in dims)
    pts = np.array(
        [[sx * hx, sy * hy, sz * hz] for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)],
        dtype=np.float64,
    )
    faces = [
        [0, 2, 3, 1],  # -z
        [4, 5, 7, 6],  # +z
        [0, 1, 5, 4],  # -y
        [2, 6, 7, 3],  # +y
        [0, 4, 6, 2],  # -x
        [1, 3, 7, 5],  # +x
    ]
    return MeshData(pts, [4] * 6, [i for f in faces for i in f])


def _lathe(profile: list[tuple[float, float]]) -> MeshData:
    """Revolve a (radius, z) profile, bottom to top, around the z axis.

    Profile ends with radius 0 become single pole vertices with triangle fans.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, SEGMENTS, endpoint=False)
    rings = [(r, z) for r, z in profile]
    bottom_pole = rings[0][0] == 0.0
    top_pole = rings[-1][0] == 0.0
    body = rings[1 if bottom_pole else 0 : len(rings) - (1 if top_pole else 0)]
    pts = []
    for r, z in body:
        pts += [[r * np.cos(a), r * np.sin(a), z] for a in angles]
    counts, idx = _quads_grid(len(body), SEGMENTS)
    n = len(pts)
    if bottom_pole:
        pts.append([0.0, 0.0, rings[0][1]])
        for c in range(SEGMENTS):
            counts.append(3)
            idx += [n, (c + 1) % SEGMENTS, c]
        n += 1
    else:
        counts.append(SEGMENTS)
        idx += list(reversed(range(SEGMENTS)))
    top_start = (len(body) - 1) * SEGMENTS
    if top_pole:
        pts.append([0.0, 0.0, rings[-1][1]])
        for c in range(SEGMENTS):
            counts.append(3)
            idx += [top_start + c, top_start + (c + 1) % SEGMENTS, n]
    else:
        counts.append(SEGMENTS)
        idx += [top_start + c for c in range(SEGMENTS)]
    return MeshData(np.array(pts, dtype=np.float64), counts, idx)


def cylinder(radius: float, height: float) -> MeshData:
    h = height / 2.0
    return _lathe([(radius, -h), (radius, h)])


def sphere(radius: float) -> MeshData:
    phis = np.linspace(-np.pi / 2, np.pi / 2, RINGS + 1)
    return _lathe(
        [(0.0, -radius)]
        + [(radius * np.cos(p), radius * np.sin(p)) for p in phis[1:-1]]
        + [(0.0, radius)]
    )


def capsule(radius: float, length: float) -> MeshData:
    """A cylinder with hemispherical caps; ``length`` is the overall tip-to-tip length."""
    half_body = max(length / 2.0 - radius, 0.0)
    half = RINGS // 2
    phis = np.linspace(0.0, np.pi / 2, half + 1)
    lower = [(radius * np.cos(p), -half_body - radius * np.sin(p)) for p in phis[::-1]]
    upper = [(radius * np.cos(p), half_body + radius * np.sin(p)) for p in phis]
    profile = [(0.0, lower[0][1])] + lower[1:] + upper[:-1] + [(0.0, upper[-1][1])]
    return _lathe(profile)


def torus(major: float, minor: float) -> MeshData:
    u = np.linspace(0.0, 2.0 * np.pi, SEGMENTS, endpoint=False)
    v = np.linspace(0.0, 2.0 * np.pi, RINGS, endpoint=False)
    pts = []
    for vv in v:  # rows: around the tube
        for uu in u:  # columns: around the main ring
            r = major + minor * np.cos(vv)
            pts.append([r * np.cos(uu), r * np.sin(uu), minor * np.sin(vv)])
    counts, idx = [], []
    for row in range(RINGS):
        nrow = (row + 1) % RINGS
        for c in range(SEGMENTS):
            c2 = (c + 1) % SEGMENTS
            counts.append(4)
            idx += [
                row * SEGMENTS + c,
                row * SEGMENTS + c2,
                nrow * SEGMENTS + c2,
                nrow * SEGMENTS + c,
            ]
    return MeshData(np.array(pts, dtype=np.float64), counts, idx)


def build(shape: str, size: float, rng: np.random.Generator) -> tuple[MeshData, dict]:
    """Build a primitive whose largest extent is ``size``; aspect ratios come from ``rng``."""
    if shape == "box":
        dims = [size, size * rng.uniform(0.5, 1.0), size * rng.uniform(0.4, 1.0)]
        return box(dims), {"dims": dims}
    if shape == "cylinder":
        radius = size * rng.uniform(0.25, 0.5)
        return cylinder(radius, size), {"radius": radius, "height": size}
    if shape == "sphere":
        return sphere(size / 2.0), {"radius": size / 2.0}
    if shape == "capsule":
        radius = size * rng.uniform(0.15, 0.3)
        return capsule(radius, size), {"radius": radius, "length": size}
    if shape == "torus":
        minor = size * rng.uniform(0.08, 0.15)
        major = size / 2.0 - minor
        return torus(major, minor), {"major": major, "minor": minor}
    raise ValueError(f"unknown primitive {shape!r}")
