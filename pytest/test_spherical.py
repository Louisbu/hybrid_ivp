from typing import Tuple

import numpy as np

import pytest
from hybrid_routing.utils.spherical import DEG2RAD, dist_to_dest


@pytest.mark.parametrize(
    ("p0", "p1", "d"),
    [
        ((0, 0), (0, 10), 1113),
        ((0, 0), (10, 0), 1113),
        ((-10, -10), (10, 10), 3140),
        ((180, 0), (-180, 1), 0),
    ],
)
def test_dist_to_dest(p0: Tuple[float], p1: Tuple[float], d: float):
    # To radians
    p0 = [x * DEG2RAD for x in p0]
    p1 = [x * DEG2RAD for x in p1]
    dist = dist_to_dest(p0, p1)
    np.testing.assert_allclose(dist, d, rtol=0.1)
