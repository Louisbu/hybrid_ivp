import numpy as np
from hybrid_routing.jax_utils.dnj import DNJ
from hybrid_routing.jax_utils.optimize import optimize_route
from hybrid_routing.jax_utils.route import RouteJax
from hybrid_routing.vectorfields.constant_current import ConstantCurrent


def test_optimize():
    vectorfield = ConstantCurrent()

    x_start, y_start = 0, 0
    x_end, y_end = 10, 10
    time_max = 2
    angle_amplitude = np.pi / 2
    num_angles = 10
    vel = 1

    route_opt = RouteJax(x=x_start, y=y_start)

    iter_optim = optimize_route(
        vectorfield,
        x_start,
        y_start,
        x_end,
        y_end,
        time_max=time_max,
        angle_amplitude=angle_amplitude,
        num_angles=num_angles,
        vel=vel,
    )

    for list_routes in iter_optim:
        route = list_routes[0]
        route_opt.append_points(route.x, route.y)

    assert len(route_opt.x) == len(route_opt.y)
    assert len(route_opt.x) == 141

    dnj = DNJ(vectorfield)
    route_opt.optimize_distance(dnj, num_iter=100)
    assert len(route_opt.x) == len(route_opt.y)
    assert len(route_opt.x) == 141
