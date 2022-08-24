from typing import Optional

import jax.numpy as jnp
from hybrid_routing.jax_utils.dnj import DNJ
from hybrid_routing.utils.distance import dist_to_dest


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

    @property
    def pts(self):
        return jnp.stack([self.x, self.y], axis=1)

    def append_points(self, x: jnp.array, y: jnp.array, t: jnp.array):
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
        self.t = jnp.concatenate([self.t, jnp.atleast_1d(t)])

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
        dist = dist_to_dest((self.x[-1], self.y[-1]), (x, y))
        t = dist / vel + self.t[-1]
        self.append_points(x, y, t)

    def optimize_distance(self, dnj: DNJ, num_iter: int = 10):
        pts = self.pts
        for iteration in range(num_iter):
            pts_old = pts
            pts = dnj.optimize_distance(pts)
            # TODO: Sometimes the DNJ produces NaNs, understand why and fix
            # Temporal Solution: NaNs are replaced with last valid value
            mask_nan = jnp.isnan(pts)
            pts = pts.at[mask_nan].set(pts_old[mask_nan])
        # Update the points of the route
        self.x = pts[:, 0]
        self.y = pts[:, 1]
