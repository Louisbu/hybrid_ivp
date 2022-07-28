from hybrid_routing.vectorfields.base import Vectorfield
import jax.numpy as jnp


class Circular(Vectorfield):
    def __init__(self):
        pass

    def dvdx(self, x, y):
        return -1

    def dvdy(self, x, y):
        return 0

    def dudx(self, x, y):
        return 0

    def dudy(self, x, y):
        return 1

    def get_current(self, x, y):
        return jnp.asarray([1 * (y + 1), 1 * (-x - 3)])
