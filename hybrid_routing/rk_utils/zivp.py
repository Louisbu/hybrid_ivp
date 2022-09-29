from typing import List

import numpy as np
from hybrid_routing.rk_utils.route import RouteRK
from hybrid_routing.vectorfields.base import Vectorfield


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
) -> List[RouteRK]:
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
    List[RouteRK]
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

    list_routes: List[RouteRK] = [None] * len(thetas)
    for idx, theta in enumerate(thetas):
        arr_x = [x] * len(arr_t)
        arr_y = [y] * len(arr_t)
        arr_theta = [theta] * len(arr_t)
        for j, t0 in enumerate(arr_t[:-1]):
            q0 = np.asarray([arr_x[j], arr_y[j], arr_theta[j]])
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
            arr_x[j + 1] = q1[0]
            arr_y[j + 1] = q1[1]
            arr_theta[j + 1] = q1[2]
        list_routes[idx] = RouteRK(x=arr_x, y=arr_y, t=arr_t, theta=arr_theta)

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
) -> List[RouteRK]:
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
    List[RouteRK]
        Returns a list of all paths thats generated at each cone search. All points of the paths are of RouteRK object.
    """

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
    list_routes: List[RouteRK] = [None] * len(thetas)

    for idx, theta in enumerate(thetas):
        # Initialize list of (x, y) coordinates
        list_x, list_y = [x] * len(arr_t), [y] * len(arr_t)
        list_theta = [theta] * len(arr_t)
        # (x, y) points will be updated during the iteration
        x_temp, y_temp = x, y
        # Compute the vessel velocity components
        v_x = vel * np.cos(theta)
        v_y = vel * np.sin(theta)
        # Loop through the time steps
        for idx2, _ in enumerate(arr_t):
            # Compute the displacement, affected by the vectorfield
            vf_x, vf_y = vectorfield.get_current(x_temp, y_temp)
            dx = (v_x + vf_x) * time_step
            dy = (v_y + vf_y) * time_step
            x_temp += dx
            y_temp += dy
            list_x[idx2] = x_temp
            list_y[idx2] = y_temp
        # Include the new route in the list
        list_routes[idx] = RouteRK(list_x, list_y, theta=list_theta)
    return list_routes
