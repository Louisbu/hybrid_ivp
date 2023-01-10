import jax.numpy as jnp

from hybrid_routing.vectorfields.base import Vectorfield


class Swirlys(Vectorfield):
    """Implements Vectorfield class.
    Vectorfield defined by:
    W: (x, y) -> (u, v), u(x, y) = cos(2x - y - 6), v(x, y) = 2/3 sin(y) + x - 3
    with:
        du/dx = 2 * sin(-2x + y + 6),   du/dy = - sin(-2x + y + 6)
        dv/dx = 1,                      dv/dy = 2/3 cos(y)
    """

    def get_current(self, x: jnp.array, y: jnp.array) -> jnp.array:
        return jnp.asarray([jnp.cos(2 * x - y - 6), 2 * jnp.sin(y) / 3 + x - 3])

    def _ode_zermelo_euclidean(self, p, t, vel=jnp.float16(1)):
        x, y, theta = p
        vector_field = self.get_current(x, y)
        dxdt = vel * jnp.cos(theta) + vector_field[0]
        dydt = vel * jnp.sin(theta) + vector_field[1]

        dthetadt = (
            jnp.sin(theta) ** 2
            + jnp.sin(theta)
            * jnp.cos(theta)
            * (2 * jnp.sin(-2 * x + y + 6) - 2 / 3 * jnp.cos(y))
            - -jnp.sin(-2 * x + y + 6) * jnp.cos(theta) ** 2
        )

        return [dxdt, dydt, dthetadt]
