from hybrid_routing.vectorfields.base import Vectorfield
import jax.numpy as jnp
from typing import Iterable


def R(a: float, b: float, x: float, y: float) -> Iterable[float]:
    """Coefficient subroutine for the coefficient used in the paper. Pair (a, b) gives a local vortex.

    Parameters
    ----------
    a : float
        x-coordinate of a local vortex
    b : float
        y-coordinate of a local vortex
    x : float
        x-coordinate
    y : float
        y-coordinate

    Returns
    -------
    Iterable[float]
        outputs a 2-dim float array.
    """
    coeff = 1 / (3 * ((x - a) ** 2 + (y - b) ** 2) + 1)
    R = (coeff * (-y + b), coeff * (x - a))
    return jnp.asarray(R)


class FourVortices(Vectorfield):
    """Vectorfield example demonstrated in Figure 2 in https://arxiv.org/pdf/2109.05559.pdf,
    implements Vectorfield class."""

    def get_current(self, x: jnp.array, y: jnp.array) -> jnp.array:
        field = 1.7 * (
            jnp.negative(R(2, 2, x, y))
            + jnp.negative(R(4, 4, x, y))
            + jnp.negative(R(2, 5, x, y))
            + R(5, 1, x, y)
        )
        return jnp.asarray(field)
