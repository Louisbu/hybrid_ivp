import jax.numpy as jnp


def R(a, b, x, y):
    coeff = 1 / (3 * ((x - a) ** 2 + (y - b) ** 2) + 1)
    R = (coeff * (-y + b), coeff * (x - a))
    return jnp.asarray(R)


def background_vector_field(x: jnp.array, y: jnp.array) -> jnp.array:
    # field = (0.01 * (y + 1), 0.01 * (-x - 3))
    # field = (jnp.cos(2 * x - y - 6), 2 / 3 * np.sin(y) + x - 3)

    field = 1.7 * (
        jnp.negative(R(2, 2, x, y))
        + jnp.negative(R(4, 4, x, y))
        + jnp.negative(R(2, 5, x, y))
        + R(5, 1, x, y)
    )
    # field = (np.cos(2 * x - y - 6), 1 / 3 * jnp.sin(y) + x - 3)
    # W = (0,0)
    # W = (x+2,-x*3)

    return jnp.asarray(field)
    # return jnp.asarray(W) para que el output ya esté en forma de vector.


def no_current(x: jnp.array, y: jnp.array) -> jnp.array:
    return jnp.array([0.0, 0.0])


def steady_current(x: jnp.array, y: jnp.array) -> jnp.array:
    return jnp.array([1, 1])


def periodic_current(x: jnp.array, y: jnp.array) -> jnp.array:
    return jnp.array([jnp.sin(x), jnp.cos(x)])


