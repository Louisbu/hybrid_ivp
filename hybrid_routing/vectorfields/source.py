from hybrid_routing.vectorfields.base import Vectorfield
import jax.numpy as jnp


class Source(Vectorfield):
    def __init__(self):
        pass

    def get_current(self, x, y):
        return jnp.asarray([(x - 5) / 75, (y - 5) / 75])
