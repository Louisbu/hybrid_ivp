import tensorflow as tf


C1 = tf.constant(1.0, dtype=tf.float32)
C2 = tf.constant(2.0, dtype=tf.float32)
C3 = tf.constant(3.0, dtype=tf.float32)
C4 = tf.constant(4.0, dtype=tf.float32)
C5 = tf.constant(5.0, dtype=tf.float32)


def R(a: tf.Tensor, b: tf.Tensor, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    coeff = tf.math.divide(
        C1,
        tf.math.multiply(
            C3,
            (tf.math.pow(x - a, 2) + tf.math.pow(y - b, 2)),
        )
        + C1,
    )
    return (tf.math.multiply(coeff, -y + b), tf.math.multiply(coeff, x - a))


@tf.function
def background_vector_field(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:

    field = tf.math.multiply(
        tf.constant(1.7, dtype=tf.float32),
        tf.math.negative(R(C2, C2, x, y))
        + tf.math.negative(R(C4, C4, x, y))
        + tf.math.negative(R(C2, C5, x, y))
        + R(C5, C1, x, y),
    )
    return (field[0], field[1])
