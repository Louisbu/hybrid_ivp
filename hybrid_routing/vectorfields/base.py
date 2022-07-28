from abc import ABC, abstractmethod

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
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

    def wave(self, p, t, vel=jnp.float16(0.1)):
        x, y, theta = p
        vector_field = self.get_current(x, y)
        dxdt = vel * jnp.cos(theta) + vector_field[0]
        dydt = vel * jnp.sin(theta) + vector_field[1]
        dthetadt = (
            self.dvdx(x, y) * jnp.sin(theta) ** 2
            + jnp.sin(theta) * jnp.cos(theta) * (self.dudx(x, y) - self.dvdy(x, y))
            - self.dudy(x, y) * jnp.cos(theta) ** 2
        )

        return [dxdt, dydt, dthetadt]

    def plot(
        self,
        x_min: float = 0,
        x_max: float = 125,
        y_min: float = 0,
        y_max: float = 125,
        step: float = 20,
    ):
        """Plots the vector field

        Parameters
        ----------
        x_min : float, optional
            Left limit of X axes, by default 0
        x_max : float, optional
            Right limit of X axes, by default 125
        y_min : float, optional
            Bottom limit of Y axes, by default 0
        y_max : float, optional
            Up limit of Y axes, by default 125
        step : float, optional
            Distance between points to plot, by default 10
        """
        x, y = np.meshgrid(
            np.linspace(x_min, x_max, step), np.linspace(y_min, y_max, step)
        )
        u, v = self.get_current(x, y)
        plt.quiver(x, y, u, v)
