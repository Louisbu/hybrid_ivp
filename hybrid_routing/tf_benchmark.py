import jax.numpy as jnp
from matplotlib import backend_bases
import tensorflow as tf

def R(a, b, x, y):
    coeff = 1 / (3 * ((x - a) ** 2 + (y - b) ** 2) + 1)
    R = (coeff * (-y + b), coeff * (x - a))
    return R

@tf.function
def background_vector_field(x: tf.constant, y: tf.constant) -> tf.constant:

    # field = (0.01 * (y + 1), 0.01 * (-x - 3))
    # field = (jnp.cos(2 * x - y - 6), 2 / 3 * np.sin(y) + x - 3)
    field = 1.7 * (
        tf.math.negative(R(2, 2, x, y))
        + tf.math.negative(R(4, 4, x, y))
        + tf.math.negative(R(2, 5, x, y))
        + R(5, 1, x, y)
    )
    x = field[0]
    y = field[1]
    # field = (np.cos(2 * x - y - 6), 1 / 3 * jnp.sin(y) + x - 3)
    # W = (0,0)
    # W = (x+2,-x*3)
    return (x, y)