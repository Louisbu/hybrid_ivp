from typing import List

import numpy as np
from hybrid_routing.jax_utils.route import RouteJax
from hybrid_routing.vectorfields.base import Vectorfield
from scipy.integrate import odeint


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
        list_routes[idx] = RouteJax(x=sol[:, 0], y=sol[:, 1], t=t, theta=sol[:, 2])

    return list_routes


def solve_discretized_zermelo(
    vectorfield: Vectorfield,
    x: float,
    y: float,
    time_start: float = 0,
    time_end: float = 2,
    time_step: float = 0.1,
    cone_center: float = 1.0,
    angle_amplitude: float = 0.4,
    num_angles: int = 1,
    vel: float = 0.5,
) -> List[RouteJax]:
    """his function instead of using the Scipy's ODE solver, we take advantage of the discretized vectorfield.

    Parameters
    ----------
    vectorfield : Vectorfield
        The vectorfield (background waves) for the ship to sail on
    x : float
        x-coordinate of the starting position
    y : float
        y_coordinate of the starting position
    time_end : float, optional
        The total time for the ship to travel at each iteration, by default 2
    time_step : float, optional
        The "smoothness" of the path at each local iteration, by default 0.1
    cone_center : float, optional
        The direction of where the boat points at initially, by default 1.0
    angle_amplitude : float, optional
        The search angle around the cone_center, by default 0.4
    num_angles : int, optional
        Number of initial search angles, by default 1
    vel : float, optional
        velocity of the vessel, by default 0.5

    Returns
    -------
    List[RouteJax]
        Returns a list of all paths thats generated at each cone search. All points of the paths are of RouteJax object.
    """

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
        # Initialize list of (x, y) coordinates
        list_x, list_y = [x] * len(t), [y] * len(t)
        list_theta = [theta] * len(t)
        # (x, y) points will be updated during the iteration
        x_temp, y_temp = x, y
        # Compute the vessel velocity components
        v_x = vel * np.cos(theta)
        v_y = vel * np.sin(theta)
        # Loop through the time steps
        for idx2, _ in enumerate(t):
            # Compute the displacement, affected by the vectorfield
            vf_x, vf_y = vectorfield.get_current(x_temp, y_temp)
            dx = (v_x + vf_x) * time_step
            dy = (v_y + vf_y) * time_step
            x_temp += dx
            y_temp += dy
            list_x[idx2] = x_temp
            list_y[idx2] = y_temp
        # Include the new route in the list
        list_routes[idx] = RouteJax(list_x, list_y, theta=list_theta)
    return list_routes


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
