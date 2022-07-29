from functools import partial
from typing import Callable

import jax.numpy as jnp
from hybrid_routing.vectorfields.base import Vectorfield
from hybrid_routing.vectorfields.constant_current import ConstantCurrent
from jax import grad, jacfwd, jacrev, jit
from pyparsing import Iterable


def hessian(f: Callable, argnums: int = 0):
    return jacfwd(jacrev(f, argnums=argnums), argnums=argnums)


class DNJ:
    def __init__(self, vectorfield: Vectorfield) -> None:
        def cost_function(x: jnp.array, xp: jnp.array) -> Iterable[float]:
            w = vectorfield.get_current(x[0], x[1])
            cost = 0.5 * ((xp[0] - w[0]) ** 2 + (xp[1] - w[1]) ** 2)
            return cost

        def discretized_cost_function(
            q0: jnp.array, q1: jnp.array, h: float
        ) -> Iterable[float]:
            L1 = cost_function(q0, (q1 - q0) / h)
            L2 = cost_function(q1, 1 / h * (q1 - q0))
            L_d = h / 2 * (L1**2 + L2**2)
            return L_d

        self.cost_function = cost_function
        self.discretized_cost_function = discretized_cost_function

        self.D1Ld = jit(grad(discretized_cost_function, argnums=0))
        self.D2Ld = jit(grad(discretized_cost_function, argnums=1))
        self.D11Ld = jit(hessian(discretized_cost_function, argnums=0))
        self.D22Ld = jit(hessian(discretized_cost_function, argnums=1))

    def __hash__(self):
        return hash(())

    def __eq__(self, other):
        return isinstance(other, DNJ)

    @partial(jit, static_argnums=(0, 2, 3, 4))
    def optimize_distance(
        self,
        pts: jnp.array,
        t_total: float,
        damping: float = 0.9,
    ) -> jnp.array:
        num_points = len(pts)
        h = t_total / (num_points - 1)
        # Implement method for the trajectory
        x = jnp.asarray(pts)
        x_new = jnp.copy(x)
        for idx in range(1, num_points - 1):
            qkm1 = x[idx - 1]
            qk = x[idx]
            qkp1 = x[idx + 1]

            b = -self.D2Ld(qkm1, qk, h) - self.D1Ld(qk, qkp1, h)
            a = self.D22Ld(qkm1, qk, h) + self.D11Ld(qk, qkp1, h)

            Q = jnp.linalg.solve(a, b)

            x_new = x_new.at[idx].set(damping * Q + qk)
        return x_new


def main():
    x0 = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    t_total = 30
    num_points = 10
    n_iter = 10
    dnj = DNJ(vectorfield=ConstantCurrent())

    print("Cost\n", dnj.cost_function(x0, x0))
    print("\nDiscretize\n", dnj.discretized_cost_function(x0, x0, 0.5))
    print("\nDerivative\n", dnj.D2Ld(x0[0], x0[0], 0.5))

    x = dnj.optimize_distance(x0, t_total, num_points, n_iter)
    print(x)


if __name__ == "__main__":
    main()
