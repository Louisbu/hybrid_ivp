import numpy as np
from hybrid_routing.tensorflow.benchmark import background_vector_field

import tensorflow as tf


def tf_wave(
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
    p = tf.constant([1.0, 1.0, 0.2], dtype=tf.float32)
    t = tf.constant(np.linspace(0, 10, 1), dtype=tf.float32)
    print(tf_wave(t, p))
