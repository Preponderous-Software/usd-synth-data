import numpy as np
import pytest

from usd_synth.camera import PinholeCamera, convex_hull_area


def _camera(**kw):
    # 50 mm lens on a 36 mm sensor at 360 px wide: 500 px per unit of x/depth
    args = dict(eye=(0, 0, 0), target=(0, 1, 0), focal_length_mm=50, sensor_width_mm=36)
    args.update(kw)
    return PinholeCamera.look_at(width=360, height=240, **args)


def test_look_at_rotation_is_right_handed_and_looks_down_minus_z():
    eye, target = np.array([1.0, -2.0, 0.5]), np.array([0.2, 0.3, -0.1])
    cam = _camera(eye=eye, target=target)
    np.testing.assert_allclose(cam.rotation.T @ cam.rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(cam.rotation) == pytest.approx(1.0)
    target_cam = cam.to_camera(target)[0]
    np.testing.assert_allclose(target_cam[:2], 0, atol=1e-12)
    assert target_cam[2] == pytest.approx(-np.linalg.norm(target - eye))
    # camera +Y stays on the world-up side of the view
    assert cam.rotation[:, 1] @ np.array([0, 0, 1]) > 0


def test_look_at_straight_down_falls_back_to_world_x_right():
    cam = _camera(eye=(0, 0, 5), target=(0, 0, 0))
    np.testing.assert_allclose(cam.rotation, np.eye(3), atol=1e-12)


def test_look_at_stores_intrinsics_as_floats():
    cam = _camera()
    assert isinstance(cam.focal_length_mm, float) and cam.focal_length_mm == 50.0
    assert isinstance(cam.sensor_width_mm, float) and cam.sensor_width_mm == 36.0
    assert (cam.width, cam.height) == (360, 240)


def test_vertical_aperture_follows_horizontal_sensor_fit():
    assert _camera().vertical_aperture_mm == pytest.approx(24.0)


def test_project_pixel_coordinates():
    cam = _camera()
    uv, front = cam.project([[0, 3, 0], [0.1, 2, 0.05]])
    assert front.tolist() == [True, True]
    np.testing.assert_allclose(uv[0], [180, 120])  # on the optical axis: image centre
    # x right / up in camera space -> u grows, v shrinks (top-left pixel origin)
    np.testing.assert_allclose(uv[1], [180 + 500 * 0.1 / 2, 120 - 500 * 0.05 / 2])


def test_project_flags_points_behind_or_at_the_camera():
    cam = _camera()
    uv, front = cam.project([[0, -1, 0], [0, 0, 0], [0, 1e-3, 0]])
    assert front.tolist() == [False, False, True]
    assert np.isfinite(uv).all()


def test_usd_matrix_is_row_vector_camera_to_world():
    cam = _camera(eye=(1.0, -2.0, 0.5), target=(0.2, 0.3, -0.1))
    m = np.array(cam.usd_matrix())
    assert m.shape == (4, 4)
    np.testing.assert_allclose(m[:, 3], [0, 0, 0, 1])
    p_cam = np.array([0.3, -0.2, -1.5])
    p_world = (np.append(p_cam, 1.0) @ m)[:3]
    np.testing.assert_allclose(p_world, cam.position + cam.rotation @ p_cam)
    np.testing.assert_allclose(cam.to_camera(p_world)[0], p_cam, atol=1e-12)


def test_convex_hull_area_of_known_shapes():
    assert convex_hull_area(np.array([[0, 0], [4, 0], [0, 3]])) == pytest.approx(6.0)
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    assert convex_hull_area(square) == pytest.approx(1.0)
    # interior, edge and duplicate points do not change the hull
    extra = np.array([[0.5, 0.5], [0.5, 0], [1, 1], [0, 0]])
    assert convex_hull_area(np.vstack([square, extra])) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "points",
    [[], [[1, 2]], [[0, 0], [1, 1]], [[0, 0], [1, 1], [2, 2], [3, 3]], [[1, 1]] * 5],
)
def test_convex_hull_area_degenerate_is_zero(points):
    assert convex_hull_area(np.array(points, dtype=float).reshape(-1, 2)) == 0.0
