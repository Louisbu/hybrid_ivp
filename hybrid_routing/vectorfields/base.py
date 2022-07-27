from abc import ABC, abstractmethod
import jax.numpy as jnp
from jax import jacfwd, jacrev, jit


class Vectorfield(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_current(self, x, y):
        pass

    def dvdx(self, x, y):
        dvx = jit(jacrev(self.get_current, argnums=1))
        return dvx(x, y)[0]

    def dvdy(self, x, y):
        dvy = jit(jacrev(self.get_current, argnums=1))
        return dvy(x, y)[1]

    def dudx(self, x, y):
        dux = jit(jacfwd(self.get_current, argnums=0))
        return dux(x, y)[0]

    def dudy(self, x, y):
        duy = jit(jacfwd(self.get_current, argnums=0))
        return duy(x, y)[1]

    def wave(self, p, t, vel=jnp.float16(0.5)):
        x, y, theta = p
        vector_field = self.get_current(x, y)
        dxdt = vel * jnp.cos(theta) + vector_field[0]
        dydt = vel * jnp.sin(theta) + vector_field[1]
        # dthetadt = 0.01 * (-jnp.sin(theta) ** 2 - jnp.cos(theta) ** 2)
        dthetadt = (
            self.dvdx(x, y) * jnp.sin(theta) ** 2
            + jnp.sin(theta) * jnp.cos(theta) * (self.dudx(x, y) - self.dvdy(x, y))
            - self.dudy(x, y) * jnp.cos(theta) ** 2
        )

        return [dxdt, dydt, dthetadt]
