from hybrid_routing.vectorfields.base import Vectorfield
import jax.numpy as jnp


class NoCurrent(Vectorfield):
    def __init__(self):
        pass

    def get_current(self, x, y):
        return jnp.asarray([0., 0.])
