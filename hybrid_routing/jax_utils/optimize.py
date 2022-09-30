from copy import deepcopy
from typing import List, Optional, Tuple

import numpy as np
from hybrid_routing.jax_utils.route import RouteJax
from hybrid_routing.jax_utils.zivp import (
    solve_discretized_zermelo,
    solve_ode_zermelo,
    solve_rk_zermelo,
)
from hybrid_routing.utils.distance import dist_to_dest
from hybrid_routing.vectorfields.base import Vectorfield


def compute_cone_center(
    x_start: float, y_start: float, x_end: float, y_end: float
) -> float:
    """Compute the angle between two points in radians"""
    dx = x_end - x_start
    dy = y_end - y_start
    return np.arctan2(dy, dx)


def compute_thetas_in_cone(
    cone_center: float, angle_amplitude: float, num_angles: int
) -> np.array:
    # Define the search cone
    delta = 1e-4 if angle_amplitude <= 1e-4 else angle_amplitude / 2
    if num_angles > 1:
        thetas = np.linspace(
            cone_center - delta,
            cone_center + delta,
            num_angles,
        )
    else:
        thetas = np.array([cone_center])
    return thetas


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
        method: str = "direction",
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
        method: str, optional
            Method to compute the optimal route. Options are:
            - "direction": Keeps the routes which direction points to the goal
            - "closest": Keeps the closest route to the goal
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

        # Store the other parameters
        self.time_iter = time_iter
        self.time_step = time_step
        self.angle_amplitude = angle_amplitude
        self.num_angles = num_angles
        self.vel = vel
        if method in ["closest", "direction"]:
            self.method = method
        else:
            print("Non recognized method, using 'direction'.")
            self.method = "direction"

    def _optimize_by_closest(
        self, x_start: float, y_start: float, x_end: float, y_end: float
    ) -> List[RouteJax]:
        """
        System of ODE is from Zermelo's Navigation Problem
        https://en.wikipedia.org/wiki/Zermelo%27s_navigation_problem#General_solution)
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
        cone_center = compute_cone_center(x_start, y_start, x_end, y_end)

        # Position now
        x = x_start
        y = y_start
        # Time now
        t = 0

        while dist_to_dest((x, y), (x_end, y_end)) > self.dist_min:
            # Compute time at the end of this step
            t_end = t + self.time_iter

            # Get arrays of initial coordinates for these segments
            arr_x = np.repeat(x, self.num_angles)
            arr_y = np.repeat(y, self.num_angles)
            arr_theta = compute_thetas_in_cone(
                cone_center, self.angle_amplitude, self.num_angles
            )

            list_routes = self.solver(
                self.vectorfield,
                arr_x,
                arr_y,
                arr_theta,
                time_start=t,
                time_end=t_end,
                time_step=self.time_step,
                vel=self.vel,
            )

            # The routes outputted start at the closest point
            # We append those segments to the best route, if we have it
            if "route_best" in locals():
                for idx, route_new in enumerate(list_routes):
                    route: RouteJax = deepcopy(route_best)
                    route.append_points(
                        route_new.x[1:], route_new.y[1:], t=route_new.t[1:]
                    )
                    list_routes[idx] = route

            # Update the closest points and best route
            x_old, y_old = x, y
            idx_best = min_dist_to_dest(list_routes, (x_end, y_end))
            route_best = deepcopy(list_routes[idx_best])
            x, y = route_best.x[-1], route_best.y[-1]
            t = t_end

            # Recompute the cone center
            cone_center = compute_cone_center(x, y, x_end, y_end)

            # Move best route to first position
            list_routes.insert(0, list_routes.pop(idx_best))
            yield list_routes

            if x == x_old and y == y_old:
                break

    def _optimize_by_direction(
        self, x_start: float, y_start: float, x_end: float, y_end: float
    ) -> List[RouteJax]:
        # Compute angle between first and last point
        cone_center = compute_cone_center(x_start, y_start, x_end, y_end)

        # Position now
        x = x_start
        y = y_start
        # Time now
        t = 0

        # Initialize the routes
        # Each one starts with a different angle
        arr_theta = compute_thetas_in_cone(
            cone_center, self.angle_amplitude, self.num_angles
        )
        list_routes: List[RouteJax] = [
            RouteJax(x_start, y_start, t, theta) for theta in arr_theta
        ]

        while dist_to_dest((x, y), (x_end, y_end)) > self.dist_min:
            # Compute time at the end of this step
            t_end = t + self.time_iter

            # Get arrays of initial coordinates for these segments
            arr_x = np.array([route.x[-1] for route in list_routes])
            arr_y = np.array([route.y[-1] for route in list_routes])
            arr_theta = np.array([route.theta[-1] for route in list_routes])

            # Compute the new route segments
            list_segments: List[RouteJax] = self.solver(
                self.vectorfield,
                arr_x,
                arr_y,
                arr_theta,
                time_start=t,
                time_end=t_end,
                time_step=self.time_step,
                vel=self.vel,
            )

            # Develop each route of our previous iteration,
            # following its current heading
            for idx, route in enumerate(list_routes):
                route_new = list_segments[idx]
                # Compute angle between route and goal
                theta_goal = compute_cone_center(
                    route_new.x[-1], route_new.y[-1], x_end, y_end
                )
                # Keep routes which heading is inside search cone
                delta_theta = abs(route_new.theta[-1] - theta_goal)
                if delta_theta <= (self.angle_amplitude / 4):
                    route.append_points(
                        route_new.x[1:],
                        route_new.y[1:],
                        t=route_new.t[1:],
                        theta=route_new.theta[1:],
                    )
                else:
                    list_routes[idx] = None

            # Drop Nones in list
            list_routes = [route for route in list_routes if route is not None]

            if len(list_routes) == 0:
                print("No route has gotten to destination!")
                break

            # The best route will be the one closest to our destination
            x_old, y_old = x, y
            idx_best = min_dist_to_dest(list_routes, (x_end, y_end))
            route_best = list_routes[idx_best]
            x, y = route_best.x[-1], route_best.y[-1]
            t = t_end

            # Move best route to first position
            list_routes.insert(0, list_routes.pop(idx_best))
            yield list_routes

            if x == x_old and y == y_old:
                break

            # If routes were dropped, recompute new ones
            num_missing = self.num_angles - len(list_routes)
            if num_missing > 0:
                # Recompute the cone center using best route
                cone_center = compute_cone_center(x, y, x_end, y_end)
                # Generate new arr_theta
                arr_theta = compute_thetas_in_cone(
                    cone_center, self.angle_amplitude, num_missing
                )
                route_new = deepcopy(route_best)
                for theta in arr_theta:
                    route_new.theta = route_new.theta.at[-1].set(theta)
                    list_routes.append(deepcopy(route_new))

    def optimize_route(
        self, x_start: float, y_start: float, x_end: float, y_end: float
    ) -> List[RouteJax]:
        if self.method == "closest":
            return self._optimize_by_closest(x_start, y_start, x_end, y_end)
        elif self.method == "direction":
            return self._optimize_by_direction(x_start, y_start, x_end, y_end)
