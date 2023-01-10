from abc import ABC, abstractmethod
from typing import Iterable, Tuple

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import jacfwd, jacrev, jit

from hybrid_routing.utils.spherical import RAD2M


class Vectorfield(ABC):
    """The parent class of vector fields.

    Methods
    ----------
    get_current : _type_
        pass upon initialization, returns the current in tuples `(u, v)` given the position of the boat `(x, y)`
    """

    rad2m = jnp.float32(RAD2M)  # Radians to meters conversion

    def __init__(self, spherical: bool = False):
        self._dv = jit(jacrev(self.get_current, argnums=1))
        self._du = jit(jacfwd(self.get_current, argnums=0))
        self.is_discrete = False
        self.spherical = spherical
        if spherical:
            self.ode_zermelo = self._ode_zermelo_spherical
        else:
            self.ode_zermelo = self._ode_zermelo_euclidean

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

    def du(self, x: jnp.array, y: jnp.array) -> Tuple[jnp.array]:
        out = jnp.asarray([self._du(x, y) for x, y in zip(x.ravel(), y.ravel())])
        return out[:, 0].reshape(x.shape), out[:, 1].reshape(x.shape)

    def dv(self, x: jnp.array, y: jnp.array) -> Tuple[jnp.array]:
        out = jnp.asarray([self._dv(x, y) for x, y in zip(x.ravel(), y.ravel())])
        return out[:, 0].reshape(x.shape), out[:, 1].reshape(x.shape)

    def _ode_zermelo_euclidean(
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
            `dxdt`, `dydt` are in m / s, `dthetadt` is in rad / s
        """
        x, y, theta = p
        u, v = self.get_current(x, y)
        st, ct = jnp.sin(theta), jnp.cos(theta)
        dxdt = vel * ct + u
        dydt = vel * st + v
        dvdx, dvdy = self.dv(x, y)
        dudx, dudy = self.du(x, y)
        dthetadt = dvdx * st**2 + st * ct * (dudx - dvdy) - dudy * ct**2

        return [dxdt, dydt, dthetadt]

    def _ode_zermelo_spherical(
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
            `theta` is heading (in radians) of the boat (with respect to the x-axis).jnp.cos(x)
        t : Iterable[float]
            Array of time steps, evenly spaced inverval from t_start to t_end, of length `n`.
        vel : jnp.float16, optional
            Speed of the boat, by default jnp.float16(0.1)

        Returns
        -------
        Iterable[float]
            A list of coordinates on the locally optimal path of length `n`, same format as `p`: `(x, y, theta)`.
            `dxdt`, `dydt`, `dthetadt` are in rad / s
        """
        x, y, theta = p
        u, v = self.get_current(x, y)  # m / s
        st, ct = jnp.sin(theta), jnp.cos(theta)  # Assuming theta in radians
        cy = jnp.cos(y)  # Assuming y is in radians
        dxdt = (vel * ct + u) / cy / self.rad2m  # rad / s
        dydt = (vel * st + v) / self.rad2m  # rad / s
        dvdx, dvdy = self.dv(x, y)  # m / (rad * s)
        dudx, dudy = self.du(x, y)  # m / (rad * s)

        dthetadt = (
            dvdx * (st**2) / cy
            + st * ct * (dudx - dvdy * cy) / cy
            - dudy * ct**2
            - ct * jnp.tan(y) * (vel + u * ct + v * st)
        ) / self.rad2m  # rad / s

        return [dxdt, dydt, dthetadt]

    def discretize(
        self,
        x_min: float = 0,
        x_max: float = 10,
        y_min: float = 0,
        y_max: float = 10,
        step: float = 1,
    ) -> "VectorfieldDiscrete":
        """Discretizes the vectorfield

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

        Returns
        -------
        VectorfieldDiscrete
            Discretized vectorfield
        """
        if self.is_discrete:
            return self
        else:
            return VectorfieldDiscrete(
                self, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, step=step
            )

    def plot(
        self,
        x_min: float = -4,
        x_max: float = 4,
        y_min: float = -4,
        y_max: float = 4,
        step: float = 1,
        **kwargs
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
        plt.quiver(x, y, u, v, **kwargs)


class VectorfieldDiscrete(Vectorfield):
    def __init__(
        self,
        vectorfield: Vectorfield,
        x_min: float = 0,
        x_max: float = 10,
        y_min: float = 0,
        y_max: float = 10,
        step: float = 1,
    ):
        # Copy all atributes of the original vectorfield into this one
        self.__dict__.update(vectorfield.__dict__)
        self.arr_x = jnp.arange(x_min, x_max, step)
        self.arr_y = jnp.arange(y_min, y_max, step)
        mat_x, mat_y = jnp.meshgrid(self.arr_x, self.arr_y)
        u, v = vectorfield.get_current(mat_x, mat_y)
        self.u, self.v = u.T, v.T
        # Define methods to get closest indexes
        self.closest_idx = jnp.vectorize(lambda x: jnp.argmin(jnp.abs(self.arr_x - x)))
        self.closest_idy = jnp.vectorize(lambda y: jnp.argmin(jnp.abs(self.arr_y - y)))
        self.is_discrete = True

    def get_current(self, x: jnp.array, y: jnp.array) -> jnp.array:
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
        idx, idy = self.closest_idx(x), self.closest_idy(y)
        return jnp.asarray([self.u[idx, idy], self.v[idx, idy]])
