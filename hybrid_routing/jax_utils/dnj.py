from functools import partial
from typing import Callable

import jax.numpy as jnp
from hybrid_routing.vectorfields.base import Vectorfield
from hybrid_routing.vectorfields.constant_current import ConstantCurrent
from jax import grad, jacfwd, jacrev, jit, vmap
from pyparsing import Iterable


def hessian(f: Callable, argnums: int = 0):
    return jacfwd(jacrev(f, argnums=argnums), argnums=argnums)


class DNJ:
    def __init__(
        self,
        vectorfield: Vectorfield,
        time_step: float = 0.1,
        discrete_vectorfield: bool = False,
    ):
        self.vectorfield = vectorfield
        self.time_step = time_step
        self.discrete_vectorfield = discrete_vectorfield
        h = time_step
        if discrete_vectorfield:
            get_current = vectorfield.get_current_discrete
        else:
            get_current = vectorfield.get_current

        def cost_function(x: jnp.array, xp: jnp.array) -> Iterable[float]:
            w = get_current(x[0], x[1])
            cost = jnp.sqrt(((xp[0] - w[0]) ** 2 + (xp[1] - w[1]) ** 2))
            return cost

        def discretized_cost_function(q0: jnp.array, q1: jnp.array) -> jnp.array:
            l1 = cost_function(q0, (q1 - q0) / h)
            l2 = cost_function(q1, (q1 - q0) / h)
            ld = h / 2 * (jnp.power(l1, 2) + jnp.power(l2, 2))
            return ld

        d1ld = grad(discretized_cost_function, argnums=0)
        d2ld = grad(discretized_cost_function, argnums=1)
        d11ld = hessian(discretized_cost_function, argnums=0)
        d22ld = hessian(discretized_cost_function, argnums=1)

        def optimize(qkm1: jnp.array, qk: jnp.array, qkp1: jnp.array) -> jnp.array:
            b = -d2ld(qkm1, qk) - d1ld(qk, qkp1)
            a = d22ld(qkm1, qk) + d11ld(qk, qkp1)
            return jnp.linalg.solve(a, b)

        self.cost_function = cost_function
        self.discretized_cost_function = discretized_cost_function
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
