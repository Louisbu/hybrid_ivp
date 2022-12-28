import numpy as np

from hybrid_routing.jax_utils.dnj import DNJ
from hybrid_routing.jax_utils.optimize import Optimizer
from hybrid_routing.jax_utils.route import RouteJax
from hybrid_routing.vectorfields.constant_current import ConstantCurrent


def test_optimize():
    vectorfield = ConstantCurrent()

    x_start, y_start = 0, 0
    x_end, y_end = 10, 10
    time_iter = 2
    angle_amplitude = np.pi / 2
    num_angles = 10
    vel = 1

    route_opt = RouteJax(x=x_start, y=y_start, t=0)
    optimizer = Optimizer(
        vectorfield,
        time_iter=time_iter,
        angle_amplitude=angle_amplitude,
        num_angles=num_angles,
        vel=vel,
        use_rk=False,
    )

    iter_optim = optimizer.optimize_route(x_start, y_start, x_end, y_end)

    for list_routes in iter_optim:
        route = list_routes[0]
        route_opt.append_points(route.x, route.y, route.t)

    assert len(route_opt.x) == len(route_opt.y)
    assert len(route_opt.x) == 141

    dnj = DNJ(vectorfield)
    dnj.optimize_route(route_opt, num_iter=100)
    assert len(route_opt.x) == len(route_opt.y)
    assert len(route_opt.x) == 141


def test_optimize_rk():
    vectorfield = ConstantCurrent()

    x_start, y_start = 0, 0
    x_end, y_end = 10, 10
    time_iter = 2
    angle_amplitude = np.pi / 2
    num_angles = 10
    vel = 1

    route_opt = RouteJax(x=x_start, y=y_start, t=0)
    optimizer = Optimizer(
        vectorfield,
        time_iter=time_iter,
        angle_amplitude=angle_amplitude,
        num_angles=num_angles,
        vel=vel,
        use_rk=False,
    )

    iter_optim = optimizer.optimize_route(x_start, y_start, x_end, y_end)

    for list_routes in iter_optim:
        route = list_routes[0]
        route_opt.append_points(route.x, route.y, route.t)

    assert len(route_opt.x) == len(route_opt.y)
    assert len(route_opt.x) == 141

    dnj = DNJ(vectorfield)
    dnj.optimize_route(route_opt, num_iter=100)
    assert len(route_opt.x) == len(route_opt.y)
    assert len(route_opt.x) == 141
