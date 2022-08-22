from hybrid_routing.vectorfields.base import Vectorfield
import jax.numpy as jnp


class Sink(Vectorfield):
    """Sink vector field, implements Vectorfield class.
    Sink coordinates defined by setting u, v to 0 and solve for x, y.
    Vectorfield defined by:
    W: (x, y) -> (u, v), u(x, y) = -1 / 25 * (x - 8), v(x, y) = -1 / 25 * (y - 8)
    with:
        du/dx = -1 / 25,    du/dy = 0
        dv/dx = 0      ,    dv/dy = -1/25
    """

    def dvdx(self, x: float, y: float) -> float:
        return 0

    def dvdy(self, x: float, y: float) -> float:
        return -1 / 25

    def dudx(self, x: float, y: float) -> float:
        return -1 / 25

    def dudy(self, x: float, y: float) -> float:
        return 0

    def get_current(self, x: float, y: float) -> jnp.array:
        return jnp.asarray([-(x - 8) / 25, -(y - 8) / 25])
