from hybrid_routing.vectorfields.base import Vectorfield
import jax.numpy as jnp


class HillBowl(Vectorfield):
    def __init__(self):
        pass

    def get_current(self, x, y):
        return jnp.asarray([1, jnp.sin(x**2 + y**2)])
