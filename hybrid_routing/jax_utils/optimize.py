from copy import deepcopy
from typing import List, Optional, Tuple

import numpy as np

import hybrid_routing.utils.euclidean as euclidean
import hybrid_routing.utils.spherical as spherical
from hybrid_routing.jax_utils.route import RouteJax
from hybrid_routing.jax_utils.zivp import (
    solve_discretized_zermelo,
    solve_ode_zermelo,
    solve_rk_zermelo,
)
from hybrid_routing.vectorfields.base import Vectorfield


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


class Optimizer:
    def __init__(
        self,
        vectorfield: Vectorfield,
        time_iter: float = 2,
        time_step: float = 0.1,
        angle_amplitude: float = np.pi,
        angle_heading: Optional[float] = None,
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
        angle_heading : float, optional
            Maximum deviation allower when optimizing direction, by default 1/4 angle amplitude
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

        # Define distance metric
        if vectorfield.spherical:
            self.dist_p0_to_p1 = spherical.dist_p0_to_p1
            self.angle_p0_to_p1 = spherical.angle_p0_to_p1
        else:
            self.dist_p0_to_p1 = euclidean.dist_p0_to_p1
            self.angle_p0_to_p1 = euclidean.angle_p0_to_p1

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
        self.angle_heading = (
            angle_amplitude / 4 if angle_heading is None else angle_heading
        )
        self.num_angles = num_angles
        self.vel = vel
        if method in ["closest", "direction"]:
            self.method = method
        else:
            print("Non recognized method, using 'direction'.")
            self.method = "direction"
        self.exploration = None

    def min_dist_p0_to_p1(self, list_routes: List[RouteJax], pt_goal: Tuple) -> int:
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
            dist = self.dist_p0_to_p1((route.x[-1], route.y[-1]), pt_goal)
            if dist < min_dist:
                min_dist = dist
                idx_best_point = idx
        return idx_best_point

    def solve_ivp(
        self, x: np.array, y: np.array, theta: np.array, t: float = 0
    ) -> List[RouteJax]:
        """Solve an initial value problem, given arrays of same length for
        x, y and theta (heading, w.r.t. x-axis)

        Parameters
        ----------
        x : np.array
            Initial coordinate on x-axis
        y : np.array
            Initial coordinate on y-axis
        theta : np.array
            Initial heading w.r.t. x-axis, in radians
        t : float, optional
            Initial time, by default 0

        Returns
        -------
        List[RouteJax]
            Routes generated with this IVP
        """
        return self.solver(
            self.vectorfield,
            x,
            y,
            theta,
            time_start=t,
            time_end=t + self.time_iter,
            time_step=self.time_step,
            vel=self.vel,
        )

    # TODO: Ensure spherical compatibility
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
        cone_center = self.angle_p0_to_p1((x_start, y_start), (x_end, y_end))

        # Position now
        x = x_start
        y = y_start
        # Time now
        t = 0

        while self.dist_p0_to_p1((x, y), (x_end, y_end)) > self.dist_min:
            # Get arrays of initial coordinates for these segments
            arr_x = np.repeat(x, self.num_angles)
            arr_y = np.repeat(y, self.num_angles)
            arr_theta = compute_thetas_in_cone(
                cone_center, self.angle_amplitude, self.num_angles
            )

            list_routes = self.solve_ivp(arr_x, arr_y, arr_theta, t=t)

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
            idx_best = self.min_dist_p0_to_p1(list_routes, (x_end, y_end))
            route_best = deepcopy(list_routes[idx_best])
            x, y = route_best.x[-1], route_best.y[-1]
            t = route_best.t[-1]

            # Recompute the cone center
            cone_center = self.angle_p0_to_p1((x, y), (x_end, y_end))

            # Move best route to first position
            list_routes.insert(0, list_routes.pop(idx_best))
            yield list_routes

            if x == x_old and y == y_old:
                break

    # TODO: Ensure spherical compatibility
    def _optimize_by_direction(
        self, x_start: float, y_start: float, x_end: float, y_end: float
    ) -> List[RouteJax]:
        # Compute angle between first and last point
        cone_center = self.angle_p0_to_p1((x_start, y_start), (x_end, y_end))

        # Position now
        x = x_start
        y = y_start
        t = 0  # Time now
        t_last = -1  # Time of last loop, used to avoid infinite loops

        # Initialize the routes
        # Each one starts with a different angle
        arr_theta = compute_thetas_in_cone(
            cone_center, self.angle_amplitude, self.num_angles
        )
        list_routes: List[RouteJax] = [
            RouteJax(x_start, y_start, t, theta) for theta in arr_theta
        ]

        # Initialize list of routes to stop (outside of angle threshold)
        list_stop: List[int] = []
        # Define whether the next step is exploitation or exploration, and the exploitation index
        # We start in the exploration step, so next step is exploitation
        self.exploration = True  # Exploitation step / Exploration step
        idx_refine = 1  # Where the best segment start + 1
        # The loop continues until the algorithm reaches the end or it gets stuck
        while (self.dist_p0_to_p1((x, y), (x_end, y_end)) > self.dist_min) and (
            t != t_last
        ):
            t_last = t  # Update time of last loop
            # Get arrays of initial coordinates for these segments
            arr_x = np.array([route.x[-1] for route in list_routes])
            arr_y = np.array([route.y[-1] for route in list_routes])
            arr_theta = np.array([route.theta[-1] for route in list_routes])

            # Compute the new route segments
            list_segments = self.solve_ivp(arr_x, arr_y, arr_theta, t=t)

            # Develop each route of our previous iteration,
            # following its current heading
            for idx, route in enumerate(list_routes):
                # If the index is inside the list of stopped routes, skip
                if idx in list_stop:
                    continue
                route_new = list_segments[idx]
                # Compute angle between route and goal
                theta_goal = self.angle_p0_to_p1(
                    (route_new.x[-1], route_new.y[-1]), (x_end, y_end)
                )
                # Keep routes which heading is inside search cone
                delta_theta = abs(route_new.theta[-1] - theta_goal)
                if delta_theta <= (self.angle_heading):
                    route.append_points(
                        route_new.x[1:],
                        route_new.y[1:],
                        t=route_new.t[1:],
                        theta=route_new.theta[1:],
                    )
                else:
                    list_stop.append(idx)

            # If all routes have been stopped, generate new ones
            if len(list_stop) == len(list_routes):
                # Change next step from exploitation <-> exploration
                self.exploration = not self.exploration
                if self.exploration:
                    # Exploration step: New routes are generated starting from
                    # the end of the best segment, using a cone centered
                    # around the direction to the goal
                    # Recompute the cone center using best route
                    cone_center = self.angle_p0_to_p1((x, y), (x_end, y_end))
                    # Generate new arr_theta
                    arr_theta = compute_thetas_in_cone(
                        cone_center, self.angle_amplitude, self.num_angles
                    )
                    route_new = deepcopy(route_best)
                    # Set the new exploitation index
                    idx_refine = len(route_new.x)
                else:
                    # Exploitation step: New routes are generated starting from
                    # the beginning of best segment, using a small cone centered
                    # around the direction of the best segment
                    # Recompute the cone center using best route
                    cone_center = route_best.theta[idx_refine - 1]
                    # Generate new arr_theta
                    arr_theta = compute_thetas_in_cone(
                        cone_center, self.angle_amplitude / 5, self.num_angles
                    )
                    route_new = RouteJax(
                        route_best.x[:idx_refine],
                        route_best.y[:idx_refine],
                        route_best.t[:idx_refine],
                        route_best.theta[:idx_refine],
                    )
                # Reinitialize route lists
                list_routes: List[RouteJax] = []
                list_stop: List[int] = []
                # Fill new list of routes
                for theta in arr_theta:
                    route_new.theta = route_new.theta.at[-1].set(theta)
                    list_routes.append(deepcopy(route_new))
                # Update the time of the last point, will go backwards when changing
                # from exploration to exploitation
                t = route_new.t[-1]
                continue

            # The best route will be the one closest to our destination
            idx_best = self.min_dist_p0_to_p1(list_routes, (x_end, y_end))
            route_best = list_routes[idx_best]
            x, y = route_best.x[-1], route_best.y[-1]
            t = max(route.t[-1] for route in list_routes)

            # Yield list of routes with best route in first position
            list_routes_yield = deepcopy(list_routes)
            list_routes_yield.insert(0, list_routes_yield.pop(idx_best))
            yield list_routes_yield

    def optimize_route(
        self, x_start: float, y_start: float, x_end: float, y_end: float
    ) -> List[RouteJax]:
        if self.method == "closest":
            return self._optimize_by_closest(x_start, y_start, x_end, y_end)
        elif self.method == "direction":
            return self._optimize_by_direction(x_start, y_start, x_end, y_end)
