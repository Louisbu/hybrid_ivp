from typing import Optional

import jax.numpy as jnp
from hybrid_routing.jax_utils.dnj import DNJ


class RouteJax:
    def __init__(
        self,
        x: jnp.array,
        y: jnp.array,
        t: jnp.array,
        theta: Optional[jnp.array] = None,
    ):
        self.x = jnp.atleast_1d(x)
        self.y = jnp.atleast_1d(y)
        self.t = jnp.atleast_1d(t)
        assert len(self.x) == len(self.y) == len(self.t), "Array lengths are not equal"
        self.theta = jnp.atleast_1d(theta) if theta is not None else jnp.zeros_like(x)

    @property
    def pts(self):
        return jnp.stack([self.x, self.y], axis=1)

    def append_points(self, x: jnp.array, y: jnp.array, t: jnp.array):
        self.x = jnp.concatenate([self.x, jnp.atleast_1d(x)])
        self.y = jnp.concatenate([self.y, jnp.atleast_1d(y)])
        self.t = jnp.concatenate([self.t, jnp.atleast_1d(t)])

    def optimize_distance(self, dnj: DNJ, num_iter: int = 10):
        pts = self.pts
        for iteration in range(num_iter):
            pts = dnj.optimize_distance(pts)
        self.x = pts[:, 0]
        self.y = pts[:, 1]
