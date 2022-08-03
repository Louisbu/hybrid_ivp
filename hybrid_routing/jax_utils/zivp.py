from typing import Iterable
import numpy as np
from scipy.integrate import odeint
from hybrid_routing.vectorfields.base import Vectorfield


def solve_wave(
    vectorfield: Vectorfield,
    x: float,
    y: float,
    time_max: float = 2,
    time_step: float = 0.1,
    cone_center: float = 0,
    angle_amplitude: float = np.pi,
    num_angles: int = 5,
    vel: float = 2.0,
) -> Iterable[Iterable[float, float, float]]:
    """This function encapsulates the first step in the docstring in hybrid_routing.utils.optimize.utils.optimize_route.

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
    time_max : float, optional
        The total amount of time the ship is allowed to travel by at each iteration, by default 2
    time_step : float, optional
        Number of steps to reach from 0 to time_max (equivalently, how "smooth" each path is), by default 0.1
    angle_amplitude : float, optional
        The search cone range in radians, by default 0.25
    num_angles : int, optional
        Number of initial search angles, by default 50
    vel : float, optional
        Speed of the ship (unit unknown), by default 5

    Returns
    -------
    Iterable[Iterable[float, float, float]]
        Returns a list with all paths generated within the search cone.
    """
    t = np.arange(0, time_max, time_step)
    list_routes = []
    thetas = np.linspace(
        cone_center - angle_amplitude / 2,
        cone_center + angle_amplitude / 2,
        num_angles,
    )

    for theta in thetas:
        # p = tf.constant([x, y, theta])
        p = [x, y, theta]
        # sol = solver.solve(vectorfield.wave, t_init, p, solution_times)
        sol = odeint(vectorfield.wave, p, t, args=(vel,))
        list_routes.append(sol)

    return list_routes
