from hybrid_routing.vectorfields.base import Vectorfield
import jax.numpy as jnp


def R(a, b, x, y):
    coeff = 1 / (3 * ((x - a) ** 2 + (y - b) ** 2) + 1)
    R = (coeff * (-y + b), coeff * (x - a))
    return jnp.asarray(R)


class FourVortices(Vectorfield):
    def __init__(self):
        pass

    def get_current(self, x, y):
        field = 1.7 * (
            jnp.negative(R(2, 2, x, y))
            + jnp.negative(R(4, 4, x, y))
            + jnp.negative(R(2, 5, x, y))
            + R(5, 1, x, y)
        )
        return jnp.asarray(field)
