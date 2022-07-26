import tensorflow as tf


def R(a, b, x, y):
    coeff = 1 / (3 * ((x - a) ** 2 + (y - b) ** 2) + 1)
    R = (coeff * (-y + b), coeff * (x - a))
    return R


@tf.function
def background_vector_field(x: tf.constant, y: tf.constant) -> tf.constant:
    field = 1.7 * (
        tf.math.negative(R(2, 2, x, y))
        + tf.math.negative(R(4, 4, x, y))
        + tf.math.negative(R(2, 5, x, y))
        + R(5, 1, x, y)
    )
    x = field[0]
    y = field[1]
    return (x, y)
