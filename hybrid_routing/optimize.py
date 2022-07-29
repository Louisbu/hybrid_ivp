from math import pi

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

from hybrid_routing.jax_utils.dnj import DNJ
from hybrid_routing.tf_utils.zivp import dist_to_dest, min_dist_to_dest
from hybrid_routing.vectorfields.base import Vectorfield
from hybrid_routing.vectorfields.constant_current import ConstantCurrent


def optimize_route(
    vectorfield: Vectorfield,
    x_start: float,
    y_start: float,
    x_end: float,
    y_end: float,
    dist_min: float = 10,
    time_max: float = 2,
    time_step: float = 0.1,
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
    t = np.arange(0, time_max, time_step)

    # solver = tfp.math.ode.BDF()

    while dist_to_dest((x, y), (x_end, y_end)) > vel / 2:

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


def main():
    vectorfield = ConstantCurrent()
    dnj = DNJ(vectorfield)
    x_start, y_start = 0, 0
    x_end, y_end = 6, 6
    time_max = 2
    angle_amplitude = pi / 2
    num_angles = 10
    vel = 5

    pts = jnp.array([[x_start, y_start]])
    t_total = 0

    for list_routes in optimize_route(
        vectorfield,
        x_start,
        y_start,
        x_end,
        y_end,
        time_max=time_max,
        angle_amplitude=angle_amplitude,
        num_angles=num_angles,
        dist_min=3 * vel / 4,
        vel=vel,
    ):
        print("Scipy done!")
        route = list_routes[0]
        pts = jnp.concatenate([pts, jnp.array(route[:, :2])])

        t_total += time_max
    print("Number of points:", pts.shape[0])
    print("Start iteration...")
    for iteration in range(50):
        pts = dnj.optimize_distance(pts)
        print("Iteration:", iteration)

    print("Number of points:", pts.shape[0])


if __name__ == "__main__":
    main()
