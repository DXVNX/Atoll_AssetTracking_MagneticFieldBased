import numpy as np
import pytest

from src.adapters.uji_orientation import android_orientation_matrix, uji_body_to_world


@pytest.mark.parametrize("angles", [(0, 0, 0), (90, 0, 0), (20, -30, 40), (359, 10, -15)])
def test_android_rotation_is_proper_and_preserves_norm(angles):
    matrix = android_orientation_matrix(*angles)
    assert matrix.T @ matrix == pytest.approx(np.eye(3), abs=1e-12)
    assert np.linalg.det(matrix) == pytest.approx(1.0, abs=1e-12)
    vector = np.array([3.0, -4.0, 12.0])
    assert np.linalg.norm(matrix @ vector) == pytest.approx(np.linalg.norm(vector), abs=1e-12)


def test_identity_rotation():
    assert android_orientation_matrix(0, 0, 0) == pytest.approx(np.eye(3), abs=1e-12)


def test_ninety_degree_azimuth_maps_device_y_to_world_east():
    world = uji_body_to_world(np.array([0.0, 1.0, 0.0]), 90, 0, 0)
    assert world == pytest.approx([1.0, 0.0, 0.0], abs=1e-12)


def test_round_trip():
    matrix = android_orientation_matrix(123, -20, 35)
    body = np.array([11.0, -7.0, 2.0])
    assert matrix.T @ (matrix @ body) == pytest.approx(body, abs=1e-12)
