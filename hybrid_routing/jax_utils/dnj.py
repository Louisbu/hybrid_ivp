import warnings
from functools import partial
from typing import Callable, List, Tuple

import jax.numpy as jnp
import numpy as np
from jax import grad, jacfwd, jacrev, jit, vmap

from hybrid_routing.jax_utils.route import RouteJax
from hybrid_routing.vectorfields.base import Vectorfield


# defines the hessian of our functions
def hessian(f: Callable, argnums: int = 0):
    return jacfwd(jacrev(f, argnums=argnums), argnums=argnums)


class DNJ:
    def __init__(
        self,
        vectorfield: Vectorfield,
        time_step: float = 0.1,
        optimize_for: str = "fuel",
    ):
        self.vectorfield = vectorfield
        self.time_step = time_step
        h = time_step

        if optimize_for == "fuel":

            def cost_function(x: jnp.array, xp: jnp.array) -> jnp.array:
                w = vectorfield.get_current(x[0], x[1])
                cost = jnp.sqrt((xp[0] - w[0]) ** 2 + (xp[1] - w[1]) ** 2)
                return cost

        elif optimize_for == "time":

            def cost_function(x: jnp.array, xp: jnp.array) -> jnp.array:
                w = vectorfield.get_current(x[0], x[1])
                a = 1 - (w[0] ** 2 + w[1] ** 2)
                cost = (
                    jnp.sqrt(
                        1 / a * (xp[0] ** 2 + xp[1] ** 2)
                        + 1 / (a**2) * (w[0] * xp[0] + w[1] * xp[1]) ** 2
                    )
                    - 1 / a * (w[0] * xp[0] + w[1] * xp[1])
                ) ** 2
                return cost

        else:
            raise ValueError("unrecognized cost function")

        def cost_function_discretized(q0: jnp.array, q1: jnp.array) -> jnp.array:
            l1 = cost_function(q0, (q1 - q0) / h)
            l2 = cost_function(q1, (q1 - q0) / h)
            ld = h / 2 * (l1**2 + l2**2)
            return ld

        d1ld = grad(cost_function_discretized, argnums=0)
        d2ld = grad(cost_function_discretized, argnums=1)
        d11ld = hessian(cost_function_discretized, argnums=0)
        d22ld = hessian(cost_function_discretized, argnums=1)

        def optimize(qkm1: jnp.array, qk: jnp.array, qkp1: jnp.array) -> jnp.array:
            b = -d2ld(qkm1, qk) - d1ld(qk, qkp1)
            a = d22ld(qkm1, qk) + d11ld(qk, qkp1)
            return jnp.linalg.solve(a, b)

        self.cost_function = cost_function
        self.cost_function_discretized = cost_function_discretized
        self.optim_vect = vmap(optimize, in_axes=(0, 0, 0), out_axes=0)

    def __hash__(self):
        return hash(())

    def __eq__(self, other):
        return isinstance(other, DNJ)

    @partial(jit, static_argnums=(0, 2))
    def optimize_distance(self, pts: jnp.array, damping: float = 0.9) -> jnp.array:
        pts_new = jnp.copy(pts)
        q = self.optim_vect(pts[:-2], pts[1:-1], pts[2:])
        return pts_new.at[1:-1].set(damping * q + pts[1:-1])

    def optimize_route(self, route: RouteJax, num_iter: int = 10):
        """Optimizes a route for any number of iterations

        Parameters
        ----------
        route : RouteJax
            Route to optimize
        num_iter : int, optional
            Number of DNJ iterations, by default 10
        """
        pts = route.pts
        for iteration in range(num_iter):
            pts_old = pts
            pts = self.optimize_distance(pts)
            # TODO: Sometimes the DNJ produces NaNs, understand why and fix
            # Temporal Solution: NaNs are replaced with last valid value
            mask_nan = jnp.isnan(pts)
            pts = pts.at[mask_nan].set(pts_old[mask_nan])
        # Warn user if NaNs appeared
        if mask_nan.any():
            warnings.warn(
                "There has been NaNs in the last iteration. "
                "This may prevent optimization."
            )
        # Update the points of the route
        route.x = pts[:, 0]
        route.y = pts[:, 1]


class DNJRandomGuess:
    def __init__(
        self,
        vectorfield: Vectorfield,
        q0: Tuple[float, float],
        q1: Tuple[float, float],
        time_step: float = 0.1,
        optimize_for: str = "fuel",
        angle_amplitude: float = np.pi,
        num_points: int = 80,
        num_routes: int = 3,
        num_iter: int = 500,
    ):
        """Initializes a DNJ with random guesses"""
        x_start, y_start = q0
        x_end, y_end = q1
        list_routes: List[RouteJax] = [None] * num_routes
        # Randomly select number of segments per route
        num_segments = np.random.randint(2, 5, num_routes)
        for idx_route in range(num_routes):
            # We first will choose the bounding points of each segment
            x_pts = [x_start]
            y_pts = [y_start]
            dist = []
            for idx_seg in range(num_segments[idx_route] - 1):
                # The shooting direction is centered on the final destination
                dx = x_end - x_pts[-1]
                dy = y_end - y_pts[-1]
                ang = np.arctan2(dy, dx)
                # Randomly select angle deviation
                ang_dev = np.random.uniform(-0.5, 0.5, 1) * angle_amplitude
                # Randomly select the distance travelled
                d = np.sqrt(dx**2 + dy**2) * np.random.uniform(0.1, 0.9, 1)
                # Get the final point of the segment
                x_pts.append(x_pts[-1] + d * np.cos(ang + ang_dev))
                y_pts.append(y_pts[-1] + d * np.sin(ang + ang_dev))
                dist.append(d)
            # Append final point
            dx = x_end - x_pts[-1]
            dy = y_end - y_pts[-1]
            d = np.sqrt(dx**2 + dy**2)
            x_pts.append(x_end)
            y_pts.append(y_end)
            dist.append(d)
            dist = np.array(dist).flatten()
            # To ensure the points of the route are equi-distant,
            # the number of points per segment will depend on its distance
            # in relation to the total distance travelled
            num_points_seg = (num_points * dist / dist.sum()).astype(int)
            # Start generating the points
            x = np.array([x_start])
            y = np.array([y_start])
            for idx_seg in range(num_segments[idx_route]):
                x_new = np.linspace(
                    x_pts[idx_seg], x_pts[idx_seg + 1], num_points_seg[idx_seg]
                ).flatten()
                x = np.concatenate([x, x_new[1:]])
                y_new = np.linspace(
                    y_pts[idx_seg], y_pts[idx_seg + 1], num_points_seg[idx_seg]
                ).flatten()
                y = np.concatenate([y, y_new[1:]])
            # Add the route to the list
            list_routes[idx_route] = RouteJax(x, y)
        # Store parameters
        self.dnj = DNJ(
            vectorfield=vectorfield, time_step=time_step, optimize_for=optimize_for
        )
        self.list_routes = list_routes
        self.num_iter = num_iter
        self.total_iter: int = 0

    def __next__(self) -> List[RouteJax]:
        for route in self.list_routes:
            self.dnj.optimize_route(route, num_iter=self.num_iter)
        self.total_iter += self.num_iter
        return self.list_routes
