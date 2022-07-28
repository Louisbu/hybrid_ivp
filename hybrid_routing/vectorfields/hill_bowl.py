from hybrid_routing.vectorfields.base import Vectorfield
import jax.numpy as jnp


class HillBowl(Vectorfield):
    def __init__(self):
        pass

    def get_current(self, x, y):
        return jnp.asarray([x / x, jnp.sin(x**2 + y**2)])

    def wave(self, p, t, vel=jnp.float16(1)):
        x, y, theta = p
        vector_field = self.get_current(x, y)
        dxdt = vel * jnp.cos(theta) + vector_field[0]
        dydt = vel * jnp.sin(theta) + vector_field[1]
        # dthetadt = 0.01 * (-jnp.sin(theta) ** 2 - jnp.cos(theta) ** 2)
        dthetadt = 4 * x * jnp.sin(x**2 + y**2) * jnp.cos(
            x**2 + y**2
        ) * jnp.sin(theta) ** 2 + -4 * jnp.sin(theta) * jnp.cos(theta) * y * jnp.sin(
            x**2 + y**2
        ) * jnp.cos(
            x**2 + y**2
        )

        return [dxdt, dydt, dthetadt]
