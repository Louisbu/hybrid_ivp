from typing import List, Optional, Tuple

import numpy as np
from hybrid_routing.jax_utils.route import RouteJax
from hybrid_routing.jax_utils.zivp import (
    solve_discretized_zermelo,
    solve_ode_zermelo,
    solve_rk_zermelo,
)
from hybrid_routing.utils.distance import dist_to_dest
from hybrid_routing.vectorfields.base import Vectorfield


def min_dist_to_dest(list_routes: List[RouteJax], pt_goal: Tuple) -> int:
    """Out of a list of routes, returns the index of the route the ends
    at the minimum distance to the goal.

    Parameters
    ----------
    list_routes : List[np.array]
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


def optimize_route(
    vectorfield: Vectorfield,
    x_start: float,
    y_start: float,
    x_end: float,
    y_end: float,
    time_iter: float = 2,
    time_step: float = 0.1,
    angle_amplitude: float = np.pi,
    num_angles: int = 5,
    vel: float = 5,
    dist_min: Optional[float] = None,
    use_rk: bool = False,
) -> List[RouteJax]:

    """
    System of ODE is from Zermelo's Navigation Problem https://en.wikipedia.org/wiki/Zermelo%27s_navigation_problem#General_solution)
    1) This function first computes the locally optimized paths with Scipy's ODE solver.
    Given the starting coordinates (x_start, y_start), time (t_max), speed of the ship (vel),
    and the direction the ship points in (angle_amplitude / num_angles), the ODE solver returns
    a list of points on the locally optimized path.
    2) We then use a loop to compute all locally optimal paths with given angles in the
    angle amplitude and store them in a list.
    3) We next finds the list of paths with an end point (x1, y1) that has the smallest
    Euclidean distance to the destination (x_end, y_end).
    4) We then use the end point (x1, y1) on that path to compute the next set of paths
    by repeating the above algorithm.
    5) This function terminates till the last end point is within a neighbourhood of the
    destination (defaults vel * time_end).

    Parameters
    ----------
    vectorfield : Vectorfield
        Background vectorfield for the ship to set sail on
    x_start : float
        x-coordinate of the starting position
    y_start : float
        y-coordinate of the starting position
    x_end : float
        x-coordinate of the destinating position
    y_end : float
        y-coordinate of the destinating position
    time_iter : float, optional
        The total amount of time the ship is allowed to travel by at each iteration,
        by default 2
    time_step : float, optional
        Number of steps to reach from 0 to time_iter (equivalently, how "smooth" each path is),
        by default 0.1
    angle_amplitude : float, optional
        The search cone range in radians, by default pi
    num_angles : int, optional
        Number of initial search angles, by default 5
    vel : float, optional
        Speed of the ship (unit unknown), by default 5
    dist_min : float, optional
        Minimum terminating distance around the destination (x_end, y_end), by default None
    use_rk : bool, optional
        Use Runge-Kutta solver instead of odeint solver


    Yields
    ------
    Iterator[List[RouteJax]]
        Returns a list with all paths generated within the search cone.
        The path that terminates closest to destination is on top.
    """
    # Compute angle between first and last point
    dx = x_end - x_start
    dy = y_end - y_start
    cone_center = np.arctan2(dy, dx)

    # Position now
    x = x_start
    y = y_start
    # Time now
    t = 0

    # Compute minimum distance as the average distance
    # transversed during one loop
    dist_min = vel * time_iter if dist_min is None else dist_min

    # Choose solving method depends on whether the vectorfield is discrete
    if use_rk:
        fun = solve_rk_zermelo
    elif vectorfield.is_discrete:
        fun = solve_discretized_zermelo
    else:
        fun = solve_ode_zermelo

    while dist_to_dest((x, y), (x_end, y_end)) > dist_min:
        # Compute time at the end of this step
        t_end = t + time_iter

        list_routes = fun(
            vectorfield,
            x,
            y,
            time_start=t,
            time_end=t_end,
            time_step=time_step,
            cone_center=cone_center,
            angle_amplitude=angle_amplitude,
            num_angles=num_angles,
            vel=vel,
        )

        x_old, y_old = x, y
        idx_best = min_dist_to_dest(list_routes, (x_end, y_end))
        route_best = list_routes[idx_best]
        x, y = route_best.x[-1], route_best.y[-1]
        t = t_end

        # Recompute the cone center
        dx = x_end - x
        dy = y_end - y
        cone_center = np.arctan2(dy, dx)

        # Move best route to first position
        list_routes.insert(0, list_routes.pop(idx_best))
        yield list_routes

        if x == x_old and y == y_old:
            break
