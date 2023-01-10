from typing import Tuple

import numpy as np

RADIUS = 6367.449  # meters
RAD2M = RADIUS / (2 * np.pi)  # Radians to meters conversion
DEG2RAD = np.pi / 180


def lonlatunitvector(p: Tuple[float]) -> np.array:
    lon, lat = p[0], p[1]
    return np.array([np.cos(lon) * np.cos(lat), np.sin(lon) * np.cos(lat), np.sin(lat)])


def dist_to_dest(p0: Tuple[float], p1: Tuple[float]) -> float:
    """Compute the distance between two points, defined in radians. Returns meters."""
    return RADIUS * np.arccos(np.dot(lonlatunitvector(p0), lonlatunitvector(p1)))
