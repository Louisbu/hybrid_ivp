from typing import List
import numpy as np
from scipy.integrate import odeint
from hybrid_routing.vectorfields.base import Vectorfield
from hybrid_routing.jax_utils.route import RouteJax


def solve_ode_zermelo(
    vectorfield: Vectorfield,
    x: float,
    y: float,
    time_max: float = 2,
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
    time_max : float, optional
        The total amount of time the ship is allowed to travel by at each iteration, by default 2
    time_step : float, optional
        Number of steps to reach from 0 to time_max (equivalently, how "smooth" each path is), by default 0.1
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
    t = np.arange(0, time_max, time_step)

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
        list_routes[idx] = RouteJax(sol[:, 0], sol[:, 1], theta=sol[:, 2])

    return list_routes


def solve_matrix(
    vectorfield: Vectorfield,
    x: float,
    y: float,
    time_max: float = 2,
    time_step: float = 0.1,
    cone_center: float = 1.0,
    angle_amplitude: float = 0.4,
    num_angles: int = 1,
    vel: float = 0.5,
) -> List[List[float]]:
    t = np.arange(0, time_max, time_step)
    list_routes = []
    thetas = np.linspace(
        cone_center - angle_amplitude / 2,
        cone_center + angle_amplitude / 2,
        num_angles,
    )
    for theta in thetas:
        list_pts = []
        x_temp, y_temp, theta_temp = x, y, theta
        for steps in t:
            dx = (
                vel * time_step * np.cos(theta)
                + vectorfield.get_current_from_matrix(x_temp, y_temp)[0]
            )

            dy = (
                vel * time_step * np.sin(theta)
                + vectorfield.get_current_from_matrix(x_temp, y_temp)[1]
            )
            x_temp += dx
            y_temp += dy
            p_temp = [x_temp, y_temp, theta_temp]
            list_pts.append(p_temp)
        list_routes.append(list_pts)
    return list_routes
