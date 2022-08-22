from abc import ABC, abstractmethod
from typing import Iterable

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import jacfwd, jacrev, jit


class Vectorfield(ABC):
    """The parent class of vector fields.

    Methods
    ----------
    get_current : _type_
        pass upon initialization, returns the current in tuples `(u, v)` given the position of the boat `(x, y)`
    """

    def __init__(
        self,
        x_min: float = 0,
        x_max: float = 10,
        y_min: float = 0,
        y_max: float = 10,
        step: float = 1,
    ):
        """
        Parameters
        ----------
        x_min : float, optional
            Minimum x-value of the grid, by default 0
        x_max : float, optional
            Maximum x-value of the grid, by default 10
        y_min : float, optional
            Minimum y-value of the grid, by default 0
        y_max : float, optional
            Maximum y_value of the grid, by default 10
        step : float, optional
            "Fineness" of the grid, by default 1
        """
        self.arr_x = jnp.arange(x_min, x_max, step)
        self.arr_y = jnp.arange(y_min, y_max, step)
        mat_x, mat_y = jnp.meshgrid(self.arr_x, self.arr_y)
        self.u, self.v = self.get_current(mat_x, mat_y)

    @abstractmethod
    def get_current(self, x: jnp.array, y: jnp.array) -> jnp.array:
        pass

    """
    Takes the Jacobian (a 2x2 matrix) of the background vectorfield (W) using JAX package 
    by Google LLC if it is not specified in the children classes.
    
    Jax docs: https://jax.readthedocs.io/en/latest/_autosummary/jax.jacfwd.html#jax.jacfwd 
            & https://jax.readthedocs.io/en/latest/_autosummary/jax.jacrev.html#jax.jacrev.
    
    `W: R^2 -> R^2, W: (x,y) -> (u,v)`
    Each function below returns a specific linearized partial derivative with respect to the variable.

    Parameters
    ----------
    x : x-coordinate of the boat's current location.
    y : y-coordinate of the boat's current location.

    Returns
    -------
    float
        The value of dv/dx, dv/dy, du/dx, du/dy, with respect to the call.
    """

    def dvdx(self, x: float, y: float) -> float:
        dvx = jit(jacrev(self.get_current, argnums=1))
        return dvx(x, y)[0]

    def dvdy(self, x: float, y: float) -> float:
        dvy = jit(jacrev(self.get_current, argnums=1))
        return dvy(x, y)[1]

    def dudx(self, x: float, y: float) -> float:
        dux = jit(jacfwd(self.get_current, argnums=0))
        return dux(x, y)[0]

    def dudy(self, x: float, y: float) -> float:
        duy = jit(jacfwd(self.get_current, argnums=0))
        return duy(x, y)[1]

    def ode_zermelo(
        self,
        p: Iterable[float],
        t: Iterable[float],
        vel: jnp.float16 = jnp.float16(0.1),
    ) -> Iterable[float]:
        """System of ODE set up for scipy initial value problem method to solve in optimize.py

        Parameters
        ----------
        p : Iterable[float]
            Initial position: `(x, y, theta)`. The pair `(x,y)` is the position of the boat and
            `theta` is heading (in radians) of the boat (with respect to the x-axis).
        t : Iterable[float]
            Array of time steps, evenly spaced inverval from t_start to t_end, of length `n`.
        vel : jnp.float16, optional
            Speed of the boat, by default jnp.float16(0.1)

        Returns
        -------
        Iterable[float]
            A list of coordinates on the locally optimal path of length `n`, same format as `p`: `(x, y, theta)`.
        """
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

    def get_current_discrete(self, x: jnp.array, y: jnp.array) -> jnp.array:
        """Takes the current values (u,v) at a given point (x,y) on the grid.

        Parameters
        ----------
        x : jnp.array
            x-coordinate of the ship
        y : jnp.array
            y-coordinate of the ship

        Returns
        -------
        jnp.array
            The current's velocity in x and y direction (u, v)
        """
        idx = jnp.argmin(jnp.abs(self.arr_x - x))
        idy = jnp.argmin(jnp.abs(self.arr_y - y))
        return jnp.asarray([self.u[idx, idy], self.v[idx, idy]])

    def plot(
        self,
        x_min: float = -4,
        x_max: float = 4,
        y_min: float = -4,
        y_max: float = 4,
        step: float = 1,
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
            Distance between points to plot, by default .5
        """
        x, y = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
        u, v = self.get_current(x, y)
        plt.quiver(x, y, u, v)
