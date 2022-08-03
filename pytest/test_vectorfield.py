import jax.numpy as jnp
from hybrid_routing.vectorfields import ConstantCurrent


def test_constant_current_vectorfield():
    vectorfield = ConstantCurrent()
    assert (vectorfield.get_current(0, 0) == jnp.asarray([0.2, -0.2])).all()
