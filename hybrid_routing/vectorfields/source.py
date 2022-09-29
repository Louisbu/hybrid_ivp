from typing import Iterable
from hybrid_routing.vectorfields.base import Vectorfield
import jax.numpy as jnp


class Source(Vectorfield):
    def dv(self, x: float, y: float) -> float:
        return (0, 1 / 75)

    def du(self, x: float, y: float) -> float:
        return (1 / 75, 0)

    def get_current(self, x: jnp.array, y: jnp.array) -> Iterable[float]:
        return jnp.asarray([(x - 5) / 25, (y - 5) / 25])
