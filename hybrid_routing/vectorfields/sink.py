from hybrid_routing.vectorfields.base import Vectorfield
import jax.numpy as jnp


class Sink(Vectorfield):
    def __init__(self):
        pass

    def dvdx(self, x, y):
        return 0

    def dvdy(self, x, y):
        return -1 / 75

    def dudx(self, x, y):
        return -1 / 75

    def dudy(self, x, y):
        return 0

    def get_current(self, x, y):
        return jnp.asarray([-(x - 10) / 25, -(y - 8) / 25])
