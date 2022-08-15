from typing import Iterable, Tuple
from hybrid_routing.jax_utils.route import RouteJax
import numpy as np


def dist_to_dest(p0: Tuple[float], p1: Tuple[float]) -> float:
    """Compute the distance between two points."""
    return np.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)


def min_dist_to_dest(list_routes: Iterable[RouteJax], pt_goal: Tuple) -> int:
    """Out of a list of routes, returns the index of the route the ends
    at the minimum distance to the goal.

    Parameters
    ----------
    list_routes : Iterable[np.array]
        List of routes, defined by (x, y, theta)
    pt_goal : _type_
        Goal point, defined by (x, y)

    Returns
    -------
    int
        Index of the route that ends at the minimum distance to the goal.
    """
    min_dist = np.inf
    for idx, route in enumerate(list_routes):
        dist = dist_to_dest((route.x[-1], route.y[-1]), pt_goal)
        if dist < min_dist:
            min_dist = dist
            idx_best_point = idx
    return idx_best_point
