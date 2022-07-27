from hybrid_routing.vectorfields.base import Vectorfield
import jax.numpy as jnp


class Sink(Vectorfield):
    def __init__(self):
        pass

    def get_current(self, x, y):
        return jnp.asarray([-x / 3 - 2, -y / 3 - 2])
