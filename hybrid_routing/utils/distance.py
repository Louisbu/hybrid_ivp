from typing import Tuple
import numpy as np


def dist_to_dest(p0: Tuple[float], p1: Tuple[float]) -> float:
    """Compute the distance between two points."""
    return np.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)
