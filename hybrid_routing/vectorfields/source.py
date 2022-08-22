from typing import Iterable
from hybrid_routing.vectorfields.base import Vectorfield
import jax.numpy as jnp


class Source(Vectorfield):
    def dvdx(self, x: float, y: float) -> float:
        return 0

    def dvdy(self, x: float, y: float) -> float:
        return 1 / 75

    def dudx(self, x: float, y: float) -> float:
        return 1 / 75

    def dudy(self, x: float, y: float) -> float:
        return 0

    def get_current(self, x: float, y: float) -> Iterable[float]:
        return jnp.asarray([(x - 5) / 25, (y - 5) / 25])
