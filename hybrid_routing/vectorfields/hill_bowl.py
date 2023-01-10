import jax.numpy as jnp

from hybrid_routing.vectorfields.base import Vectorfield


class HillBowl(Vectorfield):
    """Implements Vectorfield class.
    Vectorfield defined by:
    W: (x, y) -> (u, v), u(x, y) = 1, v(x, y) = sin(x^2 + y^2)
    with:
        du/dx = 0,      du/dy = 0
        dv/dx = 2 * x * cos(x^2 + y^2),  dv/dy = 2 * y * cos(x^2 + y^2)
    """

    def get_current(self, x: jnp.array, y: jnp.array) -> jnp.array:
        return jnp.asarray([jnp.ones(x.shape), jnp.sin(x**2 + y**2)])

    def _ode_zermelo_euclidean(self, p, t, vel=jnp.float16(1)):
        x, y, theta = p
        vector_field = self.get_current(x, y)
        dxdt = vel * jnp.cos(theta) + vector_field[0]
        dydt = vel * jnp.sin(theta) + vector_field[1]
        # dthetadt = 0.01 * (-jnp.sin(theta) ** 2 - jnp.cos(theta) ** 2)
        dthetadt = 2 * x * jnp.cos(x**2 + y**2) * jnp.sin(
            theta
        ) ** 2 + -2 * jnp.sin(theta) * jnp.cos(theta) * y * jnp.cos(x**2 + y**2)

        return [dxdt, dydt, dthetadt]
