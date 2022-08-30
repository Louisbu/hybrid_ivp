import os
import shutil
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from hybrid_routing.jax_utils.dnj import DNJ, RunnerDNJ
from hybrid_routing.vectorfields import FourVortices


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
