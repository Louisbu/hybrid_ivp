import jax.numpy as jnp
from hybrid_routing.vectorfields import Circular, NoCurrent


def test_no_current_vectorfield():
    vectorfield = NoCurrent()
    assert (vectorfield.get_current(0, 0) == jnp.asarray([0, 0])).all()
    assert vectorfield.dudx(0, 0) == 0
    assert vectorfield.dvdx(0, 0) == 0
    assert vectorfield.dudy(0, 0) == 0
    assert vectorfield.dvdy(0, 0) == 0


def test_circular_vectorfield():
    vectorfield = Circular()
    assert (vectorfield.get_current(0, 0) == jnp.asarray([0.05, -0.15])).all()
    assert vectorfield.dudx(0, 0) == 0
    assert vectorfield.dvdx(0, 0) == -0.05
    assert vectorfield.dudy(0, 0) == 0.05
    assert vectorfield.dvdy(0, 0) == 0
