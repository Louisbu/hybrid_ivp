import jax.numpy as jnp
from hybrid_routing.tf_utils.benchmark import background_vector_field
from jax import grad, jacfwd, jacrev, jit


def cost_function(x, xp):

    w = background_vector_field(x[0], x[1])
    cost = 0.5 * ((xp[0] - w[0]) ** 2 + (xp[1] - w[1]) ** 2)
    return cost


def cost_function_discretized(q0, q1, h):
    L1 = cost_function(q0, (q1 - q0) / h)
    L2 = cost_function(q1, 1 / h * (q1 - q0))
    L_d = h / 2 * (L1 + L2)
    return L_d


def hessian(f, argnums=0):
    return jacfwd(jacrev(f, argnums=argnums), argnums=argnums)


D1Ld = jit(grad(cost_function_discretized, argnums=0))
D2Ld = jit(grad(cost_function_discretized, argnums=1))
D11Ld = jit(hessian(cost_function_discretized, argnums=0))
D22Ld = jit(hessian(cost_function_discretized, argnums=1))


def optimize_distance(pts, T, N, n):
    damping = 0.9  # ~1

    h = T / (N - 1)

    # Implementar el método para la trayectoria.
    x = jnp.array(pts)
    A = []
    A.append(x[0])
    for k in range(n):
        for l in range(1, N - 1):
            qkm1 = jnp.array(x[l - 1])

            qk = jnp.array(x[l])

            qkp1 = jnp.array(x[l + 1])

            b = -D2Ld(qkm1, qk, h) - D1Ld(qk, qkp1, h)
            a = D22Ld(qkm1, qk, h) + D11Ld(qk, qkp1, h)

            Q = jnp.linalg.solve(a, b)

            A.append(damping * Q + x[l])
        x = jnp.array(A)
        A = [x[0]]
    return x


def main():
    x0 = jnp.array([0.0, 0.0])
    xN = jnp.array([6.0, 5.0])
    T = 30
    N = 50
    n = 20
    x = optimize_distance(x0, xN, T, N, n)
    print([x, xN])


if __name__ == "__main__":
    main()
