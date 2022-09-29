from typing import List

import numpy as np
from hybrid_routing.jax_utils.route import RouteJax
from hybrid_routing.vectorfields.base import Vectorfield


def solve_rk_zermelo(
    vectorfield: Vectorfield,
    x: float,
    y: float,
    time_start: float = 0,
    time_end: float = 2,
    time_step: float = 0.1,
    cone_center: float = 0,
    angle_amplitude: float = np.pi,
    num_angles: int = 5,
    vel: float = 2.0,
) -> List[RouteJax]:
    """This function first computes the locally optimized paths with Runge-Kutta 4 solver method.
    Given the starting coordinates (x_start, y_start), time (t_max), speed of the ship (vel),
    and the direction the ship points in (angle_amplitude / num_angles), the solver returns
    a list of points on the locally optimized path.

    Parameters
    ----------
    vectorfield : Vectorfield
        Background vectorfield for the ship to set sail on
    x : float
        x-coordinate of the starting position
    y : float
        y-coordinate of the starting position
    time_start : float, optional
        Start time of the iteration, by default 0
    time_end : float, optional
        End time of the iteration, by default 2
    time_step : float, optional
        Number of steps to reach from 0 to time_iter (equivalently, how "smooth" each path is), by default 0.1
    cone_center : float, optional
        Center of the cone of search in radians, by default 0
    angle_amplitude : float, optional
        The search cone range in radians, by default pi
    num_angles : int, optional
        Number of initial search angles, by default 5
    vel : float, optional
        Speed of the ship (unit unknown), by default 2

    Returns
    -------
    List[RouteJax]
        Returns a list with all paths generated within the search cone.
    """
    # Define the time steps
    arr_t = np.arange(time_start, time_end, time_step)

    # Define the search cone
    delta = 1e-4 if angle_amplitude <= 1e-4 else angle_amplitude / 2
    if num_angles > 1:
        thetas = np.linspace(
            cone_center - delta,
            cone_center + delta,
            num_angles,
        )
    else:
        thetas = [cone_center]

    # Initializes the arrays containing the coordinates
    arr_q = [
        np.stack((np.repeat(x, num_angles), np.repeat(y, num_angles), thetas))
    ] * len(arr_t)
    # Update the coordinates following the RK algorithm
    for idx, t0 in enumerate(arr_t[:-1]):
        q0 = arr_q[idx]
        k1 = np.asarray(vectorfield.ode_zermelo(q0, t0, vel=vel))
        k2 = np.asarray(
            vectorfield.ode_zermelo(
                q0 + k1 * time_step / 2, t0 + time_step / 2, vel=vel
            )
        )
        k3 = np.asarray(
            vectorfield.ode_zermelo(
                q0 + k2 * time_step / 2, t0 + time_step / 2, vel=vel
            )
        )
        k4 = np.asarray(
            vectorfield.ode_zermelo(q0 + k3 * time_step, t0 + time_step, vel=vel)
        )
        q1 = q0 + time_step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        arr_q[idx + 1] = q1

    # Shape is (num_time_steps, 3, num_angles) where 3 = (x, y, theta)
    arr_q = np.asarray(arr_q)
    # Initialize list of routes and store one route per theta
    list_routes: List[RouteJax] = [None] * num_angles
    for idx in range(num_angles):
        list_routes[idx] = RouteJax(
            x=[v[idx] for v in arr_q[:, 0, :]],
            y=[v[idx] for v in arr_q[:, 1, :]],
            t=arr_t,
            theta=[v[idx] for v in arr_q[:, 2, :]],
        )

    return list_routes
