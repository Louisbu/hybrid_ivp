from typing import List, Tuple

import jax.numpy as jnp
import numpy as np

from hybrid_routing.jax_utils.dnj import DNJ
from hybrid_routing.jax_utils.route import RouteJax
from hybrid_routing.utils.distance import dist_to_dest


def run_dnj(
    dnj: DNJ,
    q0: Tuple[float, float],
    q1: Tuple[float, float],
    vel: float = 2.0,
    angle_amplitude: float = np.pi,
    num_points: int = 80,
    num_routes: int = 3,
    num_segments: int = 3,
    num_iter: int = 500,
) -> List[RouteJax]:
    x_start, y_start = q0
    x_end, y_end = q1
    dist = dist_to_dest(q0, q1)
    t_end = dist / vel
    list_routes: List[RouteJax] = [None] * num_routes
    for idx in range(num_routes):
        x_pts = np.array([x_start])
        y_pts = np.array([y_start])
        for j in range(num_segments):
            dx = x_end - x_pts[-1]
            dy = y_end - y_pts[-1]
            ang = np.arctan2(dy, dx)
            ang += np.random.uniform(-0.5, 0.5, 1) * angle_amplitude
            x_pts = np.concatenate(
                [x_pts, x_pts[-1] + np.cos(ang) * dist / (num_segments + 1)]
            )
            y_pts = np.concatenate(
                [y_pts, y_pts[-1] + np.sin(ang) * dist / (num_segments + 1)]
            )
        x_pts = np.concatenate([x_pts, [x_end]])
        y_pts = np.concatenate([y_pts, [y_end]])
        x = np.linspace(x_pts[:-1], x_pts[1:], int(num_points / num_segments)).flatten()
        y = np.linspace(y_pts[:-1], y_pts[1:], int(num_points / num_segments)).flatten()
        list_routes[idx] = RouteJax(
            x=jnp.array(x),
            y=jnp.array(y),
            t=jnp.linspace(0, t_end, len(x)),
        )
    while True:
        for route in list_routes:
            route.optimize_distance(dnj, num_iter=num_iter)
        yield list_routes
