import numpy as np
from jax import vmap, jacfwd, jacrev, jit
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_probability as tfp
from hybrid_routing.tf_benchmark import background_vector_field


def tf_wave(t, p, vel=3):
    x, y, theta = tf.constant(p)
    vector_field = background_vector_field(x, y)
    dxdt = vel * tf.cos(theta) + vector_field[0]
    dydt = vel * tf.sin(theta) + vector_field[1]

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        tape.watch(y)
        x0, x1 = background_vector_field(x, y)
    du = tape.jacobian(x0, [x, y])
    dv = tape.jacobian(x1, [x, y])
    dthetadt = (
        dv[0] * tf.sin(theta) ** 2
        + tf.sin(theta) * tf.cos(theta) * (du[0] - dv[1])
        - du[1] * tf.cos(theta) ** 2
    )
    p = tf.convert_to_tensor([dxdt, dydt, dthetadt], dtype=tf.float32)
    return tf.constant(p)


def dist_to_dest(p0, p1):
    return np.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)


def min_dist_to_dest(candidates, pN):
    dist = []
    min_dist = dist_to_dest(candidates[0], pN)
    best_point = candidates[0]
    for i in range(len(candidates)):
        dist = dist_to_dest(candidates[i], pN)
        if dist < min_dist:
            min_dist = dist
            best_point = candidates[i]
    return best_point


if __name__ == "__main__":
    print(tf_wave(np.linspace(0, 10, 1), [1.0, 1.0, 0.2]))
