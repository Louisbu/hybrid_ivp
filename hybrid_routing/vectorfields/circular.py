from hybrid_routing.vectorfields.base import Vectorfield
import jax.numpy as jnp


class Circular(Vectorfield):
    def __init__(self):
        pass

    def get_current(self, x, y):
        return jnp.asarray([0.01 * (y + 1), 0.01 * (-x - 3)])
