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

    is_discrete = False

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
        color: str = "black",
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
        plt.quiver(x, y, u, v, color=color)


class VectorfieldDiscrete(Vectorfield):
    is_discrete = True

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
        self.step = step
        mat_x, mat_y = jnp.meshgrid(self.arr_x, self.arr_y)
        u, v = vectorfield.get_current(mat_x, mat_y)
        self.u, self.v = u.T, v.T
        # Define methods to get closest indexes
        self.closest_idx = jnp.vectorize(lambda x: jnp.argmin(jnp.abs(self.arr_x - x)))
        self.closest_idy = jnp.vectorize(lambda y: jnp.argmin(jnp.abs(self.arr_y - y)))

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

    def get_surrounding_pts_and_vectors(self, x: jnp.array, y: jnp.array) -> jnp.array:
        idx, idy = self.closest_idx(x), self.closest_idy(y)
        dx = self.arr_x[idx] - jnp.atleast_1d(x)
        dy = self.arr_y[idy] - jnp.atleast_1d(y)
        mask_x, mask_y = dx > 0, dy > 0
        if mask_x.any():
            idx0 = idx.at[mask_x].set(idx[mask_x] - 1)
        else:
            idx0 = idx
        if mask_y.any():
            idy0 = idy.at[mask_y].set(idy[mask_y] - 1)
        else:
            idy0 = idy
        idx1, idy1 = idx0 + 1, idy0 + 1
        x0, y0 = self.arr_x[idx0], self.arr_y[idy0]
        x1, y1 = self.arr_x[idx1], self.arr_y[idy1]
        u00, v00 = self.u[idx0, idy0], self.v[idx0, idy0]
        u01, v01 = self.u[idx0, idy1], self.v[idx0, idy1]
        u10, v10 = self.u[idx1, idy0], self.v[idx1, idy0]
        u11, v11 = self.u[idx1, idy1], self.v[idx1, idy1]
        return (
            jnp.asarray([x0, y0, x1, y1]),
            jnp.asarray([[u00, v00], [u01, v01], [u10, v10], [u11, v11]]),
        )

    def interpolate_poly_fit(self, x: jnp.array, y: jnp.array):
        x, y = jnp.atleast_1d(x), jnp.atleast_1d(y)
        pts, w = self.get_surrounding_pts_and_vectors(x.flatten(), y.flatten())
        # https://en.wikipedia.org/wiki/Bilinear_interpolation#Polynomial_fit
        x0, x1, y0, y1 = pts

        A = jnp.array(
            [
                [jnp.ones(x0.shape), x0, y0, x0 * y0],
                [jnp.ones(x0.shape), x0, y1, x0 * y1],
                [jnp.ones(x0.shape), x1, y0, x1 * y0],
                [jnp.ones(x0.shape), x1, y1, x1 * y1],
            ]
        )
        Ainv = jnp.linalg.inv(jnp.rollaxis(A, -1))
        w = jnp.rollaxis(w, -1)
        w_new = []
        for i in range(len(x)):
            a = jnp.dot(Ainv[i], w[i])
            A = jnp.array([1, x[i], y[i], x[i] * y[i]])
            w_new.append(jnp.dot(A, a))
        return jnp.array(w_new).T

    def interpolate_weighted_mean(self, x: jnp.array, y: jnp.array):
        pts, w = self.get_surrounding_pts_and_vectors(x, y)
        x0, x1, y0, y1 = pts
        a = jnp.array([[1, 1], [x, x], [y, y], [x * y, x * y]])
        A = jnp.array(
            [
                [1, 1, 1, 1],
                [x0, x0, x1, x1],
                [y0, y1, y0, y1],
                [x0 * y0, x0 * y1, x1 * y0, x1 * y1],
            ]
        )
        w = jnp.dot(jnp.linalg.inv(A), a).T
        interp_x = w[0][0] + w[0][1] * x + w[0][2] * y + w[0][3] * x * y
        interp_y = w[1][0] + w[1][1] * x + w[1][2] * y + w[1][3] * x * y
        return (interp_x, interp_y)
