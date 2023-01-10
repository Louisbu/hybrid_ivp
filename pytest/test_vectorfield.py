import jax.numpy as jnp

import pytest
from hybrid_routing.vectorfields import Circular, NoCurrent


def test_no_current_vectorfield():
    vectorfield = NoCurrent()
    assert (vectorfield.get_current(0, 0) == jnp.asarray([0, 0])).all()
    assert vectorfield.du(0, 0)[0] == 0
    assert vectorfield.dv(0, 0)[0] == 0
    assert vectorfield.du(0, 0)[1] == 0
    assert vectorfield.dv(0, 0)[1] == 0


def test_circular_vectorfield():
    vectorfield = Circular()
    assert (vectorfield.get_current(0, 0) == jnp.asarray([0.05, -0.15])).all()
    assert vectorfield.du(0, 0)[0] == 0
    assert vectorfield.dv(0, 0)[0] == -0.05
    assert vectorfield.du(0, 0)[1] == 0.05
    assert vectorfield.dv(0, 0)[1] == 0


@pytest.mark.parametrize("x", [-2, 0, 2])
@pytest.mark.parametrize("theta", [0, 1])
@pytest.mark.parametrize("vel", [5, 10])
def test_ode_zermelo(x: float, theta: float, vel: float):
    vf_euclidean = Circular(spherical=False)
    vf_spherical = Circular(spherical=True)

    p = (x, 0, theta)
    t = [0, 5, 10]
    dx_euc, dy_euc, dt_euc = vf_euclidean.ode_zermelo(p, t, vel=vel)
    dx_sph, dy_sph, dt_sph = vf_spherical.ode_zermelo(p, t, vel=vel)
    assert dx_euc == dx_sph, "dxdt not equal"
    assert dy_euc == dy_sph, "dydt not equal"
    assert dt_euc == dt_sph, "dthetadt not equal"
