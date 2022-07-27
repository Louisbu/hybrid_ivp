import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from hybrid_routing.tf_utils.zivp import dist_to_dest, min_dist_to_dest, wave


def optimize_route(
    x_start: float,
    y_start: float,
    x_end: float,
    y_end: float,
    step_time: float = 20,
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

    t = tf.constant(np.linspace(0, 2, step_time))
    p = tf.constant([x, y, theta])

    steps = []
    solver = tfp.math.ode.BDF()

    while dist_to_dest((x, y), (x_end, y_end)) > 3:

        candidates = []
        thetas = np.linspace(
            cone_center - angle_amplitude / 2,
            cone_center + angle_amplitude / 2,
            num_angles,
        )

        for theta in thetas:
            sol = solver.solve(wave, t[0], p, t[1:])
            candidates.append(sol[-1])

        for pt in candidates:
            plt.scatter(pt[0], pt[1], s=10, c="gray")

        x_old, y_old = x, y
        x, y, theta = min_dist_to_dest(candidates, (x_end, y_end))
        steps.append((x, y, theta))
        cone_center = theta

        yield (x, y, theta)

        if x == x_old and y == y_old:
            break
