from typing import Tuple

import numpy as np

import pytest
from hybrid_routing.utils.spherical import DEG2RAD, angle_p0_to_p1, dist_p0_to_p1


@pytest.mark.parametrize(
    ("p0", "p1", "d"),
    [
        ((0, 0), (0, 10), 1113),
        ((0, 0), (10, 0), 1113),
        ((-10, -10), (10, 10), 3140),
        ((180, 0), (-180, 0), 0),
    ],
)
def test_dist_p0_to_p1(p0: Tuple[float], p1: Tuple[float], d: float):
    # To radians
    p0 = [x * DEG2RAD for x in p0]
    p1 = [x * DEG2RAD for x in p1]
    dist = dist_p0_to_p1(p0, p1)
    np.testing.assert_allclose(dist, d, rtol=0.1)


@pytest.mark.parametrize(
    ("p0", "p1", "a"),
    [
        ((0, 0), (0, 10), np.pi / 2),
        ((0, 0), (10, 0), 0),
        ((-10, -10), (10, 10), np.pi / 4),
        ((180, 0), (-180, 0), 0),
    ],
)
def test_angle_p0_to_p1(p0: Tuple[float], p1: Tuple[float], a: float):
    # To radians
    p0 = [x * DEG2RAD for x in p0]
    p1 = [x * DEG2RAD for x in p1]
    ang = angle_p0_to_p1(p0, p1)
    np.testing.assert_allclose(ang, a, rtol=0.1)
