import os
import shutil
from pathlib import Path
from typing import List, Tuple

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from hybrid_routing.jax_utils.dnj import DNJ
from hybrid_routing.jax_utils.route import RouteJax
from hybrid_routing.vectorfields import FourVortices


class RunnerDNJ:
    def __init__(
        self,
        dnj: DNJ,
        q0: Tuple[float, float],
        q1: Tuple[float, float],
        angle_amplitude: float = np.pi,
        num_points: int = 80,
        num_routes: int = 3,
        num_iter: int = 500,
    ) -> List[RouteJax]:
        x_start, y_start = q0
        x_end, y_end = q1
        list_routes: List[RouteJax] = [None] * num_routes
        # Randomly select number of segments per route
        num_segments = np.random.randint(2, 5, num_routes)
        for idx_route in range(num_routes):
            # We first will choose the bounding points of each segment
            x_pts = [x_start]
            y_pts = [y_start]
            dist = []
            for idx_seg in range(num_segments[idx_route] - 1):
                # The shooting direction is centered on the final destination
                dx = x_end - x_pts[-1]
                dy = y_end - y_pts[-1]
                ang = np.arctan2(dy, dx)
                # Randomly select angle deviation
                ang_dev = np.random.uniform(-0.5, 0.5, 1) * angle_amplitude
                # Randomly select the distance travelled
                d = np.sqrt(dx**2 + dy**2) * np.random.uniform(0.1, 0.9, 1)
                # Get the final point of the segment
                x_pts.append(x_pts[-1] + d * np.cos(ang + ang_dev))
                y_pts.append(y_pts[-1] + d * np.sin(ang + ang_dev))
                dist.append(d)
            # Append final point
            dx = x_end - x_pts[-1]
            dy = y_end - y_pts[-1]
            d = np.sqrt(dx**2 + dy**2)
            x_pts.append(x_end)
            y_pts.append(y_end)
            dist.append(d)
            dist = np.array(dist).flatten()
            # To ensure the points of the route are equi-distant,
            # the number of points per segment will depend on its distance
            # in relation to the total distance travelled
            num_points_seg = (num_points * dist / dist.sum()).astype(int)
            # Start generating the points
            x = np.array([x_start])
            y = np.array([y_start])
            for idx_seg in range(num_segments[idx_route]):
                x_new = np.linspace(
                    x_pts[idx_seg], x_pts[idx_seg + 1], num_points_seg[idx_seg]
                ).flatten()
                x = np.concatenate([x, x_new[1:]])
                y_new = np.linspace(
                    y_pts[idx_seg], y_pts[idx_seg + 1], num_points_seg[idx_seg]
                ).flatten()
                y = np.concatenate([y, y_new[1:]])
            # Add the route to the list
            list_routes[idx_route] = RouteJax(x, y)
        # Store parameters
        self.dnj = dnj
        self.list_routes = list_routes
        self.num_iter = num_iter
        self.total_iter = 0

    def __next__(self):
        for route in self.list_routes:
            self.dnj.optimize_route(route, num_iter=self.num_iter)
        self.total_iter += self.num_iter
        return self.list_routes


def main(time_step: float = 0.1, num_points: int = 80, num_routes: int = 50):
    vectorfield = FourVortices()
    dnj = DNJ(vectorfield, time_step=time_step, optimize_for="time")
    q0 = (0, 0)
    q1 = (6, 2)
    gen_dnj = RunnerDNJ(
        dnj,
        q0,
        q1,
        angle_amplitude=2 * np.pi,
        num_points=num_points,
        num_routes=num_routes,
        num_iter=1,
    )
    # Build folder with images
    path_img = Path("img_dnj")
    if not path_img.exists():
        os.mkdir(path_img)
    images = []
    # Generate one plot every 2 iterations
    for idx in range(100):
        list_routes = next(gen_dnj)
        # Accelerate the number of iterations
        gen_dnj.num_iter = idx + 2
        vectorfield.plot(x_min=-1, x_max=7, y_min=-1, y_max=7)
        for route in list_routes:
            plt.plot(route.x, route.y, color="green")
        plt.scatter([q0[0], q1[0]], [q0[1], q1[1]], c="red")
        plt.title(f"Iterations: {gen_dnj.total_iter:05d}")
        plt.xlim(-1, 7)
        plt.ylim(-1, 7)
        fout = path_img / f"{idx:03d}.png"
        plt.savefig(fout)
        plt.close()
        images.append(imageio.imread(fout))
    # Convert images to gif and delete images
    imageio.mimsave("dnj.gif", images)
    shutil.rmtree(path_img)


if __name__ == "__main__":
    main()
