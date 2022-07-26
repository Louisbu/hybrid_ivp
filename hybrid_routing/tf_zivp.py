import numpy as np
from jax import vmap, jacfwd, jacrev, jit
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_probability as tfp
from hybrid_routing.tf_benchmark import background_vector_field


def tf_dvdx(x, y):
    dvx = jit(jacrev(background_vector_field, argnums=1))
    return dvx(x, y)[0]


def tf_dvdy(x, y):
    dvy = jit(jacrev(background_vector_field, argnums=1))
    return dvy(x, y)[1]


def tf_dudx(x, y):
    dux = jit(jacfwd(background_vector_field, argnums=0))
    return dux(x, y)[0]


def tf_dudy(x, y):
    duy = jit(jacfwd(background_vector_field, argnums=0))
    return duy(x, y)[1]

# jacobian of the vector field must be declared inside the tensorflow "with" environment
def tf_wave(t, p, vel=1):
    x, y, theta = tf.constant(p)
    vector_field = background_vector_field(x, y)
    dxdt = vel * tf.cos(theta) + vector_field[0]
    dydt = vel * tf.sin(theta) + vector_field[1]

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        tape.watch(y)
        x0, x1 = background_vector_field(x,y)
    du = tape.jacobian(x0, [x,y])
    dv = tape.jacobian(x1, [x,y])
    dthetadt = (
        du[1] * tf.sin(theta) ** 2
        + tf.sin(theta) * tf.cos(theta) * (du[0] - dv[1])
        - dv[0] * tf.cos(theta) ** 2
    )
    #print(type(dxdt))
    p = tf.convert_to_tensor([dxdt, dydt, dthetadt], dtype = tf.float32)
    return tf.constant(p)



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

if __name__=="__main__":
    tf_wave([1., 1., 0.2], np.linspace(0,10,1))