import numpy as np

from hybrid_routing.jax_utils.zivp import solve_rk_zermelo
from hybrid_routing.vectorfields import NoCurrent


def test_runge_kutta_spherical():
    vf = NoCurrent(spherical=True)

    thetas = np.linspace(-np.pi, np.pi, 50)[1:-1]
    x = np.zeros(thetas.shape)
    y = np.zeros(thetas.shape)

    list_routes = solve_rk_zermelo(
        vf, x, y, thetas, time_start=0, time_end=350, time_step=2, vel=10
    )

    for route in list_routes:
        # Convert from radians to degrees
        x = route.x * 180 / np.pi
        y = route.y * 180 / np.pi

        # Locate the zero crossings in the "y" axis
        # When latitude goes back to 0
        zc = np.where(np.diff(np.sign(y)))[0]

        # At those points, longitude should be 0 (180, -180 is the same)
        check = (
            np.isclose(x[zc], 0, rtol=1)
            | np.isclose(x[zc], 180, rtol=1)
            | np.isclose(x[zc], -180, rtol=1)
        )

        assert np.all(check)
