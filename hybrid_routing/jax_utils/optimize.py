from copy import deepcopy
from typing import List, Optional, Tuple
from xml.dom.expatbuilder import parseString

import numpy as np
from hybrid_routing.jax_utils.route import RouteJax
from hybrid_routing.jax_utils.zivp import (
    solve_discretized_zermelo,
    solve_ode_zermelo,
    solve_rk_zermelo,
)
from hybrid_routing.utils.distance import dist_to_dest
from hybrid_routing.vectorfields.base import Vectorfield


def min_dist_to_dest(list_routes: List[RouteJax], pt_goal: Tuple) -> int:
    """Out of a list of routes, returns the index of the route the ends
    at the minimum distance to the goal.

    Parameters
    ----------
    list_routes : List[np.array]
        List of routes, defined by (x, y, theta)
    pt_goal : _type_
        Goal point, defined by (x, y)

    Returns
    -------
    int
        Index of the route that ends at the minimum distance to the goal.
    """
    min_dist = np.inf
    for idx, route in enumerate(list_routes):
        dist = dist_to_dest((route.x[-1], route.y[-1]), pt_goal)
        if dist < min_dist:
            min_dist = dist
            idx_best_point = idx
    return idx_best_point


class Optimizer:
    def __init__(
        self,
        vectorfield: Vectorfield,
        time_iter: float = 2,
        time_step: float = 0.1,
        angle_amplitude: float = np.pi,
        num_angles: int = 5,
        vel: float = 5,
        dist_min: Optional[float] = None,
        use_rk: bool = False,
    ):
        """Optimizer class

        Parameters
        ----------
        vectorfield : Vectorfield
            Background vectorfield for the ship to set sail on
        time_iter : float, optional
            The total amount of time the ship is allowed to travel by at each iteration,
            by default 2
        time_step : float, optional
            Number of steps to reach from 0 to time_iter (equivalently, how "smooth" each path is),
            by default 0.1
        angle_amplitude : float, optional
            The search cone range in radians, by default pi
        num_angles : int, optional
            Number of initial search angles, by default 5
        vel : float, optional
            Speed of the ship (unit unknown), by default 5
        dist_min : float, optional
            Minimum terminating distance around the destination (x_end, y_end), by default None
        use_rk : bool, optional
            Use Runge-Kutta solver instead of odeint solver
        """
        self.vectorfield = vectorfield

        # Choose solving method depends on whether the vectorfield is discrete
        if use_rk:
            self.solver = solve_rk_zermelo
        elif vectorfield.is_discrete:
            self.solver = solve_discretized_zermelo
        else:
            self.solver = solve_ode_zermelo

        # Compute minimum distance as the average distance
        # transversed during one loop
        self.dist_min = vel * time_iter if dist_min is None else dist_min

        # Define the search cone
        self.angle_delta = 1e-4 if angle_amplitude <= 1e-4 else angle_amplitude / 2

        # Store the other parameters
        self.time_iter = time_iter
        self.time_step = time_step
        self.angle_amplitude = angle_amplitude
        self.num_angles = num_angles
        self.vel = vel

    def _optimize_by_best(
        self, x_start: float, y_start: float, x_end: float, y_end: float
    ) -> List[RouteJax]:
        """
        System of ODE is from Zermelo's Navigation Problem https://en.wikipedia.org/wiki/Zermelo%27s_navigation_problem#General_solution)
        1) This function first computes the locally optimized paths with Scipy's ODE solver.
        Given the starting coordinates (x_start, y_start), time (t_max), speed of the ship (vel),
        and the direction the ship points in (angle_amplitude / num_angles), the ODE solver returns
        a list of points on the locally optimized path.
        2) We then use a loop to compute all locally optimal paths with given angles in the
        angle amplitude and store them in a list.
        3) We next finds the list of paths with an end point (x1, y1) that has the smallest
        Euclidean distance to the destination (x_end, y_end).
        4) We then use the end point (x1, y1) on that path to compute the next set of paths
        by repeating the above algorithm.
        5) This function terminates till the last end point is within a neighbourhood of the
        destination (defaults vel * time_end).

        Parameters
        ----------
        x_start : float
            x-coordinate of the starting position
        y_start : float
            y-coordinate of the starting position
        x_end : float
            x-coordinate of the destinating position
        y_end : float
            y-coordinate of the destinating position

        Yields
        ------
        Iterator[List[RouteJax]]
            Returns a list with all paths generated within the search cone.
            The path that terminates closest to destination is on top.
        """
        # Compute angle between first and last point
        dx = x_end - x_start
        dy = y_end - y_start
        cone_center = np.arctan2(dy, dx)

        # Position now
        x = x_start
        y = y_start
        # Time now
        t = 0

        while dist_to_dest((x, y), (x_end, y_end)) > self.dist_min:
            # Compute time at the end of this step
            t_end = t + self.time_iter

            list_routes = self.solver(
                self.vectorfield,
                x,
                y,
                time_start=t,
                time_end=t_end,
                time_step=self.time_step,
                cone_center=cone_center,
                angle_amplitude=self.angle_amplitude,
                num_angles=self.num_angles,
                vel=self.vel,
            )

            x_old, y_old = x, y
            idx_best = min_dist_to_dest(list_routes, (x_end, y_end))
            route_best = list_routes[idx_best]
            x, y = route_best.x[-1], route_best.y[-1]
            t = t_end

            # Recompute the cone center
            dx = x_end - x
            dy = y_end - y
            cone_center = np.arctan2(dy, dx)

            # Move best route to first position
            list_routes.insert(0, list_routes.pop(idx_best))
            yield list_routes

            if x == x_old and y == y_old:
                break

    def _optimize_by_direction(
        self, x_start: float, y_start: float, x_end: float, y_end: float
    ) -> List[RouteJax]:
        # Compute angle between first and last point
        dx = x_end - x_start
        dy = y_end - y_start
        cone_center = np.arctan2(dy, dx)

        # Position now
        x = x_start
        y = y_start
        # Time now
        t = 0

        # Initialize the routes
        list_routes: List[RouteJax] = [
            RouteJax(x_start, y_start, t, theta)
            for theta in np.linspace(
                cone_center - self.angle_delta,
                cone_center + self.angle_delta,
                self.num_angles,
            )
        ]

        while dist_to_dest((x, y), (x_end, y_end)) > self.dist_min:
            # Compute time at the end of this step
            t_end = t + self.time_iter

            list_routes_new: List[RouteJax] = []

            for route in list_routes:
                route_new = self.solver(
                    self.vectorfield,
                    route.x[-1],
                    route.y[-1],
                    time_start=t,
                    time_end=t_end,
                    time_step=self.time_step,
                    cone_center=route.theta[-1],
                    angle_amplitude=0,
                    num_angles=1,
                    vel=self.vel,
                )[0]
                route.append_points(
                    route_new.x, route_new.y, t=route_new.t, theta=route_new.theta
                )

                # Compute angle between first and last point
                dx = x_end - route_new.x[-1]
                dy = y_end - route_new.y[-1]
                # Drop routes which heading is not inside search cone
                angle_min, angle_max = (
                    np.arctan2(dy, dx) + np.array([-1, 1]) * self.angle_delta / 2
                )
                if route_new.theta[-1] > angle_max or route_new.theta[-1] < angle_min:
                    pass
                else:
                    list_routes_new.append(route)

            list_routes = list_routes_new

            x_old, y_old = x, y
            idx_best = min_dist_to_dest(list_routes, (x_end, y_end))
            route_best = list_routes[idx_best]
            x, y = route_best.x[-1], route_best.y[-1]
            t = t_end

            # Recompute the cone center
            dx = x_end - x
            dy = y_end - y
            cone_center = np.arctan2(dy, dx)

            # Move best route to first position
            list_routes.insert(0, list_routes.pop(idx_best))
            yield list_routes

            # If routes were dropped, recompute new ones
            num_missing = self.num_angles - len(list_routes)
            if num_missing > 0:
                thetas = np.linspace(
                    cone_center - self.angle_delta,
                    cone_center + self.angle_delta,
                    self.num_angles,
                )
                route_new = deepcopy(route_best)
                for theta in thetas:
                    route_new.theta = route_new.theta.at[-1].set(theta)
                    list_routes.append(deepcopy(route_new))

            if x == x_old and y == y_old:
                break

    def optimize_route(
        self, x_start: float, y_start: float, x_end: float, y_end: float
    ) -> List[RouteJax]:
        return self._optimize_by_direction(x_start, y_start, x_end, y_end)
