import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

from hybrid_routing.jax_utils.dnj import DNJ
from hybrid_routing.tf_utils.zivp import dist_to_dest, min_dist_to_dest
from hybrid_routing.vectorfields.base import Vectorfield


def dnj_optimize(
    pts: jnp.array, t_total: float, dnj: DNJ, num_iter: int = 50
) -> jnp.array:
    pts_smooth = pts
    for iteration in range(num_iter):
        pts_smooth = dnj.optimize_distance(pts_smooth, t_total)
    return pts_smooth


def optimize_route(
    vectorfield: Vectorfield,
    x_start: float,
    y_start: float,
    x_end: float,
    y_end: float,
    dist_min: float = 10,
    time_max: float = 2,
    angle_amplitude: float = 0.25,
    num_angles: int = 50,
    vel: float = 5,
):
    # Compute angle between first and last point
    dx = x_end - x_start
    dy = y_end - y_start
    cone_center = np.arctan2(dy, dx)

    # Position now
    x = x_start
    y = y_start

    # t_init = tf.constant(0)
    # solution_times = tfp.math.ode.ChosenBySolver(tf.constant(step_time))

    t = np.linspace(0, time_max, 20)

    # solver = tfp.math.ode.BDF()

    while dist_to_dest((x, y), (x_end, y_end)) > dist_min:

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

        for pt in list_routes:
            plt.plot(pt[:, 0], pt[:, 1], c="gray")

        x_old, y_old = x, y
        idx_best = min_dist_to_dest(list_routes, (x_end, y_end))
        x, y, theta = list_routes[idx_best][-1]
        cone_center = theta

        # Move best route to first position
        list_routes.insert(0, list_routes.pop(idx_best))
        yield list_routes

        if x == x_old and y == y_old:
            break
