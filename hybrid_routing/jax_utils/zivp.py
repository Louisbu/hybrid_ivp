from typing import List
import numpy as np
from scipy.integrate import odeint
from hybrid_routing.vectorfields.base import Vectorfield
from hybrid_routing.jax_utils.route import RouteJax


def solve_ode_zermelo(
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
    """This function first computes the locally optimized paths with Scipy's ODE solver.
    Given the starting coordinates (x_start, y_start), time (t_max), speed of the ship (vel),
    and the direction the ship points in (angle_amplitude / num_angles), the ODE solver returns
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
    t = np.arange(time_start, time_end, time_step)

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

    list_routes: List[RouteJax] = [None] * len(thetas)
    for idx, theta in enumerate(thetas):
        p = [x, y, theta]
        sol = odeint(vectorfield.ode_zermelo, p, t, args=(vel,))
        list_routes[idx] = RouteJax(sol[:, 0], sol[:, 1], t, theta=sol[:, 2])

    return list_routes
