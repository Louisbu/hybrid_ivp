from typing import List

import numpy as np

import pytest
from hybrid_routing.jax_utils.route import RouteJax
from hybrid_routing.jax_utils.zivp import solve_rk_zermelo
from hybrid_routing.utils.spherical import DEG2RAD, RAD2M
from hybrid_routing.vectorfields import NoCurrent

VEL = 10


@pytest.fixture
def list_routes() -> List[RouteJax]:
    vf = NoCurrent(spherical=True)

    thetas = np.linspace(-np.pi, np.pi, 30)[1:-1]
    x = np.zeros(thetas.shape)
    y = np.zeros(thetas.shape)

    return solve_rk_zermelo(
        vf, x, y, thetas, time_start=0, time_end=350, time_step=1, vel=VEL
    )


def test_around_globe(list_routes: List[RouteJax]):
    """Assert the routes turn around the globe and end at the starting point"""

    for route in list_routes:
        # Convert from radians to degrees
        x = route.x / DEG2RAD
        y = route.y / DEG2RAD

        # Locate the zero crossings in the "y" axis
        # When latitude goes back to 0
        zc = np.where(np.diff(np.sign(y)))[0]

        # At those points, longitude should be 0 (180, -180 is the same)
        check = (
            np.isclose(x[zc], 0, rtol=1)
            | np.isclose(x[zc], 180, rtol=1)
            | np.isclose(x[zc], -180, rtol=1)
        )

        assert np.all(check), "Route does not turn around the globe"


def test_speed(list_routes: List[RouteJax]):
    """Assert the routes maintain a constant velocity"""
    max_diff, max_mean = 0, 0
    for route in list_routes:
        lat = route.y
        # Velocity components in m / s
        vlon = route.dxdt * RAD2M
        vlat = route.dydt * RAD2M
        # Velocity module in m / s
        v = np.sqrt((np.cos(lat[:-1]) * vlon) ** 2 + vlat**2)
        np.testing.assert_allclose(v, VEL, atol=0.35)
        # Store maximum deviation
        md = max(np.abs(v) - VEL)
        if md > max_diff:
            max_diff = md
        # Store maximum mean deviation
        md = np.mean(np.abs(v - VEL))
        if md > max_mean:
            max_mean = md
    print(
        "\nVelocity."
        f"\n  Maximum difference: {max_diff}"
        f"\n  Max. avg. difference: {max_mean}"
    )


def test_tangencial_acceleration_x(list_routes: List[RouteJax]):
    """Assert the tangencial acceleration is close to 0"""
    max_diff, max_mean = 0, 0
    for route in list_routes:
        lat = route.y
        # Velocity in rad / s
        vlon = route.dxdt
        # Acceleration in rad / s2
        alat = np.diff(route.dydt) / route.dt[:-1]
        atan = alat + np.cos(lat[:-2]) * np.sin(lat[:-2]) * vlon[:-1] ** 2
        np.testing.assert_allclose(atan, 0, atol=0.004)
        # Store maximum deviation
        md = max(np.abs(atan))
        if md > max_diff:
            max_diff = md
        # Store maximum mean deviation
        md = np.mean(np.abs(atan))
        if md > max_mean:
            max_mean = md
    print(
        "\nTan. acceleration X."
        f"\n  Maximum difference: {max_diff}"
        f"\n  Max. avg. difference: {max_mean}"
    )


def test_tangencial_acceleration_y(list_routes: List[RouteJax]):
    """Assert the tangencial acceleration is close to 0"""
    max_diff, max_mean = 0, 0
    for route in list_routes:
        lat = route.y
        # Velocity in rad / s
        vlon = route.dxdt
        vlat = route.dydt
        # Acceleration in rad / s2
        alon = np.diff(route.dxdt) / route.dt[:-1]
        atan = alon - 2 * np.tan(lat[:-2]) * vlon[:-1] * vlat[:-1]
        np.testing.assert_allclose(atan, 0, atol=0.045)
        # Store maximum deviation
        md = max(np.abs(atan))
        if md > max_diff:
            max_diff = md
        # Store maximum mean deviation
        md = np.mean(np.abs(atan))
        if md > max_mean:
            max_mean = md
    print(
        "\nTan. acceleration Y."
        f"\n  Maximum difference: {max_diff}"
        f"\n  Max. avg. difference: {max_mean}"
    )
