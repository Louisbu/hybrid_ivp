from hybrid_routing.vectorfields.base import Vectorfield
import jax.numpy as jnp


class Source(Vectorfield):
    def __init__(self):
        pass

    def get_current(self, x, y):
        return jnp.asarray([x / 3 + 3, y / 3 + 3])
