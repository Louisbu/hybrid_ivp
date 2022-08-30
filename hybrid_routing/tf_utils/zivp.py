from typing import Iterable
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from hybrid_routing.tf_utils.benchmark import background_vector_field


def ode_zermelo(
    t: tf.Tensor, p: tf.Tensor, vel: tf.Tensor = tf.constant(3, dtype=tf.float32)
) -> tf.Tensor:
    x, y, theta = p

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        tape.watch(y)
        x0, x1 = background_vector_field(x, y)

    dxdt = tf.math.multiply(tf.cos(theta), vel) + x0
    dydt = tf.math.multiply(tf.sin(theta), vel) + x1

    du = tape.jacobian(x0, [x, y])
    dv = tape.jacobian(x1, [x, y])
    dthetadt = (
        tf.math.multiply(dv[0], tf.math.pow(tf.sin(theta), 2))
        + tf.math.multiply(
            tf.math.multiply(tf.sin(theta), tf.cos(theta)), du[0] - dv[1]
        )
        - tf.math.multiply(du[1], tf.math.pow(tf.cos(theta), 2))
    )
    p_new = tf.convert_to_tensor([dxdt, dydt, dthetadt], dtype=tf.float32)
    return tf.constant(p_new)


def solve_ode_zermelo(
    x: float,
    y: float,
    time_step: float = 0.1,
    cone_center: float = 0,
    angle_amplitude: float = np.pi,
    num_angles: int = 5,
) -> Iterable[Iterable[float]]:
    """This function first computes the locally optimized paths with Scipy's ODE solver.
    Given the starting coordinates (x_start, y_start), time (t_max), speed of the ship (vel),
    and the direction the ship points in (angle_amplitude / num_angles), the ODE solver returns
    a list of points on the locally optimized path.

    Parameters
    ----------
    x : float
        x-coordinate of the starting position
    y : float
        y-coordinate of the starting position
    time_end : float, optional
        The total amount of time the ship is allowed to travel by at each iteration, by default 2
    time_step : float, optional
        Number of steps to reach from 0 to time_end (equivalently, how "smooth" each path is), by default 0.1
    cone_center : float, optional
        Center of the cone of search in radians, by default 0
    angle_amplitude : float, optional
        The search cone range in radians, by default pi
    num_angles : int, optional
        Number of initial search angles, by default 5

    Returns
    -------
    Iterable[Iterable[float]]
        Returns a list with all paths generated within the search cone.
    """
    t_init = tf.constant(0)
    solution_times = tfp.math.ode.ChosenBySolver(tf.constant(time_step))

    solver = tfp.math.ode.BDF()
    list_routes = []
    thetas = np.linspace(
        cone_center - angle_amplitude / 2,
        cone_center + angle_amplitude / 2,
        num_angles,
    )

    for theta in thetas:
        p = tf.constant([x, y, theta])
        sol = solver.solve(ode_zermelo, t_init, p, solution_times)
        list_routes.append(sol)

    return list_routes


if __name__ == "__main__":
    p = tf.constant([1.0, 1.0, 0.2], dtype=tf.float32)
    t = tf.constant(np.linspace(0, 10, 1), dtype=tf.float32)
    print(ode_zermelo(t, p))
