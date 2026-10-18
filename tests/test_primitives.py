from collections import Counter

import numpy as np
import pytest

from usd_synth.scenegen import primitives


def _edges(mesh):
    edges = Counter()
    pos = 0
    for n in mesh.face_counts:
        face = mesh.face_indices[pos : pos + n]
        pos += n
        for a, b in zip(face, face[1:] + face[:1], strict=True):
            edges[(min(a, b), max(a, b))] += 1
    return edges


@pytest.mark.parametrize("shape", ["box", "cylinder", "sphere", "capsule", "torus"])
def test_primitive_is_closed_and_sized(shape):
    mesh, _ = primitives.build(shape, 0.1, np.random.default_rng(0))
    assert sum(mesh.face_counts) == len(mesh.face_indices)
    assert max(mesh.face_indices) == len(mesh.points) - 1
    # closed 2-manifold: every edge is shared by exactly two faces
    assert set(_edges(mesh).values()) == {2}
    extent = mesh.points.max(axis=0) - mesh.points.min(axis=0)
    assert extent.max() == pytest.approx(0.1, rel=1e-6)
    np.testing.assert_allclose(
        (mesh.points.max(axis=0) + mesh.points.min(axis=0)) / 2, 0, atol=1e-9
    )


def test_unknown_primitive_rejected():
    with pytest.raises(ValueError):
        primitives.build("cone", 0.1, np.random.default_rng(0))


def test_box_faces_point_outward():
    mesh = primitives.box([1, 1, 1])
    pos = 0
    for n in mesh.face_counts:
        p = mesh.points[mesh.face_indices[pos : pos + n]]
        pos += n
        normal = np.cross(p[1] - p[0], p[2] - p[0])
        assert np.dot(normal, p.mean(axis=0)) > 0
