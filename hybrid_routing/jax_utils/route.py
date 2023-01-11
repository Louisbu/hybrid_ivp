from typing import Optional

import jax.numpy as jnp
import numpy as np

from hybrid_routing.utils.euclidean import (
    ang_between_coords,
    ang_mod_to_components,
    components_to_ang_mod,
    dist_between_coords,
    dist_p0_to_p1,
)
from hybrid_routing.vectorfields.base import Vectorfield


class RouteJax:
    def __init__(
        self,
        x: jnp.array,
        y: jnp.array,
        t: Optional[jnp.array] = None,
        theta: Optional[jnp.array] = None,
    ):
        self.x = jnp.atleast_1d(x)
        self.y = jnp.atleast_1d(y)
        self.t = jnp.atleast_1d(t) if t is not None else jnp.arange(0, len(self.x), 1)
        assert len(self.x) == len(self.y) == len(self.t), "Array lengths are not equal"
        self.theta = jnp.atleast_1d(theta) if theta is not None else jnp.zeros_like(x)

    def __len__(self) -> int:
        return len(self.x)

    def __repr__(self) -> str:
        return (
            f"RouteJax(x0={self.x[0]:.2f}, y0={self.y[0]:.2f}, "
            f"xN={self.x[-1]:.2f}, yN={self.y[-1]:.2f}, "
            f"length={len(self)})"
        )

    def __str__(self) -> str:
        return (
            f"RouteJax(x0={self.x[0]:.2f}, y0={self.y[0]:.2f}, "
            f"xN={self.x[-1]:.2f}, yN={self.y[-1]:.2f}, "
            f"length={len(self)})"
        )

    @property
    def pts(self):
        return jnp.stack([self.x, self.y], axis=1)

    @property
    def dt(self):
        return -np.diff(self.t)

    @property
    def dx(self):
        return -np.diff(self.x)

    @property
    def dy(self):
        return -np.diff(self.y)

    @property
    def dxdt(self):
        return self.dx / self.dt

    @property
    def dydt(self):
        return self.dy / self.dt

    def append_points(
        self,
        x: jnp.array,
        y: jnp.array,
        t: Optional[jnp.array] = None,
        theta: Optional[jnp.array] = None,
    ):
        """Append new points to the end of the route

        Parameters
        ----------
        x : jnp.array
            Coordinates on X-axis, typically longitudes
        y : jnp.array
            Coordinates on X-axis, typically latitudes
        t : jnp.array
            Timestamp of each point, typically in seconds
        """
        self.x = jnp.concatenate([self.x, jnp.atleast_1d(x)])
        self.y = jnp.concatenate([self.y, jnp.atleast_1d(y)])
        t = t if t is not None else self.t + jnp.arange(0, len(x), 1)
        self.t = jnp.concatenate([self.t, jnp.atleast_1d(t)])
        theta = theta if theta is not None else jnp.full_like(x, self.theta[-1])
        self.theta = jnp.concatenate([self.theta, jnp.atleast_1d(theta)])

    def append_point_end(self, x: float, y: float, vel: float):
        """Append an end point to the route and compute its timestamp.
        It does not take into account the effect of vectorfields.

        Parameters
        ----------
        x : float
            Coordinate on X-axis, typically longitude
        y : float
            Coordinate on X-axis, typically latitude
        vel : float
            Vessel velocity, typically in meters per second
        """
        dist = dist_p0_to_p1((self.x[-1], self.y[-1]), (x, y))
        t = dist / vel + self.t[-1]
        self.append_points(x, y, t)

    def recompute_times(self, vel: float, vf: Vectorfield, interp: bool = True):
        """Given a vessel velocity and a vectorfield, recompute the
        times for each coordinate contained in the route

        Parameters
        ----------
        vel : float
            Vessel velocity
        vf : Vectorfield
            Vectorfield
        interp : bool, optional
            Interpolate route points to x10 more before recomputing times.
            This should improve the accuracy, by default True
        """
        x, y = self.x, self.y
        if interp:
            # Interpolate route to x10 points to improve the precision
            i = np.linspace(0, len(x), num=10 * len(x))
            j = np.linspace(0, len(x), num=len(x))
            x = np.interp(i, j, x)
            y = np.interp(i, j, y)
        # Angle over ground and the distance between points
        a_g = ang_between_coords(x, y)
        d = dist_between_coords(x, y)
        # Componentes of the velocity of vectorfield
        v_cx, v_cy = vf.get_current(x[:-1], y[:-1])
        # Angle and module of the velocity of vectorfield
        a_c, v_c = components_to_ang_mod(v_cx, v_cy)
        # Angle of the vectorfield w.r.t. the direction over ground
        a_cg = a_c - a_g
        # Components of the vectorfield w.r.t. the direction over ground
        v_cg_para, v_cg_perp = ang_mod_to_components(a_cg, v_c)
        # The perpendicular component of the vessel velocity must compensate the vectorfield
        v_vg_perp = -v_cg_perp
        # Component of the vessel velocity parallel w.r.t. the direction over ground
        v_vg_para = np.sqrt(np.power(vel, 2) - np.power(v_vg_perp, 2))
        # Velocity over ground is the sum of vessel and vectorfield parallel components
        v_g = v_vg_para + v_cg_para
        # Time is distance divided by velocity over ground
        t = np.divide(d, v_g)
        assert (
            t >= 0
        ).all(), f"There can't be negative times. Raise vessel velocity over {vel}."
        # Update route times
        t = np.concatenate([[0], np.cumsum(t)])
        if interp:
            # If we interpolated to x10 points, get the original ones
            self.t = np.interp(j, i, t)
        else:
            self.t = t
