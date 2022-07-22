import numpy as np
from jax import vmap, jacfwd, jacrev, jit
import jax.numpy as jnp
from hybrid_routing.benchmark import background_vector_field
import tensorflow as tf
import tensorflow_probability as tfp

def dvdx(x, y):
    dvx = jit(jacrev(background_vector_field, argnums=1))
    return dvx(x, y)[0]


def dvdy(x, y):
    dvy = jit(jacrev(background_vector_field, argnums=1))
    return dvy(x, y)[1]


def dudx(x, y):
    dux = jit(jacfwd(background_vector_field, argnums=0))
    return dux(x, y)[0]


def dudy(x, y):
    duy = jit(jacfwd(background_vector_field, argnums=0))
    return duy(x, y)[1]


def wave(p, t, vel=jnp.float16(0.5)):
    x, y, theta = p
    vector_field = background_vector_field(x, y)
    dxdt = vel * jnp.cos(theta) + vector_field[0]
    dydt = vel * jnp.sin(theta) + vector_field[1]
    # dthetadt = 0.01 * (-jnp.sin(theta) ** 2 - jnp.cos(theta) ** 2)
    dthetadt = (
        dvdx(x, y) * jnp.sin(theta) ** 2
        + jnp.sin(theta) * jnp.cos(theta) * (dudx(x, y) - dvdy(x, y))
        - dudy(x, y) * jnp.cos(theta) ** 2
    )

    return [dxdt, dydt, dthetadt]


def dist_to_dest(p0, p1):
    return np.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)


def min_dist_to_dest(candidates, pN):
    dist = []
    min_dist = dist_to_dest(candidates[0], pN)
    best_point = candidates[0]
    for i in range(len(candidates)):
        dist = dist_to_dest(candidates[i], pN)
        if dist < min_dist:
            min_dist = dist
            best_point = candidates[i]
    return best_point

def tf_wave(p, t, vel=tf.constant(0.5)):
    x, y, theta = p
    vector_field = background_vector_field(x, y)
    dxdt = vel * jnp.cos(theta) + vector_field[0]
    dydt = vel * jnp.sin(theta) + vector_field[1]
    # dthetadt = 0.01 * (-jnp.sin(theta) ** 2 - jnp.cos(theta) ** 2)
    dthetadt = (
        dvdx(x, y) * jnp.sin(theta) ** 2
        + jnp.sin(theta) * jnp.cos(theta) * (dudx(x, y) - dvdy(x, y))
        - dudy(x, y) * jnp.cos(theta) ** 2
    )

    return tf.convert_to_tensor[dxdt, dydt, dthetadt]