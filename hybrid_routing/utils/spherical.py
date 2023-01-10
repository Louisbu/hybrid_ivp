from typing import Tuple

import numpy as np

RADIUS = 6367.449  # meters
RAD2M = RADIUS / (2 * np.pi)  # Radians to meters conversion
DEG2RAD = np.pi / 180


def lonlatunitvector(p: Tuple[float]) -> np.array:
    lon, lat = p[0], p[1]
    return np.array([np.cos(lon) * np.cos(lat), np.sin(lon) * np.cos(lat), np.sin(lat)])


def dist_p0_to_p1(p0: Tuple[float], p1: Tuple[float]) -> float:
    """Compute the distance between two points, defined in radians. Returns meters."""
    return RADIUS * np.arccos(np.dot(lonlatunitvector(p0), lonlatunitvector(p1)))


def angle_p0_to_p1(p0: Tuple[float], p1: Tuple[float]) -> float:
    """Return angle (in radians) between two points, w.r.t. X-axis. Returns radians."""
    a1, b1, c1 = lonlatunitvector(p0)
    a2, b2, c2 = lonlatunitvector(p1)
    gvec = np.array(
        [-a2 * b1 + a1 * b2, -(a1 * a2 + b1 * b2) * c1 + (a1**2 + b1**2) * c2]
    )
    gd = dist_p0_to_p1(p0, p1)
    vector = np.nan_to_num(gvec * gd / np.sqrt(gvec**2), 0)
    return np.arctan2(vector[1], vector[0])
