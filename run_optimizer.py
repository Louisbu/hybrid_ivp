import os
import shutil
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import typer

from hybrid_routing.jax_utils.optimize import Optimizer
from hybrid_routing.vectorfields import *
from hybrid_routing.vectorfields.base import Vectorfield


def main(
    vf: str = "FourVortices",
    discretized: bool = True,
    use_rk: bool = True,
    method: str = "direction",
    time_iter: float = 0.2,
    time_step: float = 0.01,
    angle_amplitude: float = np.pi,
    num_angles: int = 20,
    vel: float = 1,
    dist_min: float = 0.1,
    path_out: str = "output/",
):
    vectorfield: Vectorfield = eval(vf)()
    if discretized:
        vectorfield = vectorfield.discretize(-1, 7, -1, 7)
    q0 = (0, 0)
    q1 = (6, 2)
    optimizer = Optimizer(
        vectorfield,
        time_iter=time_iter,
        time_step=time_step,
        angle_amplitude=angle_amplitude,
        num_angles=num_angles,
        vel=vel,
        dist_min=dist_min,
        use_rk=use_rk,
        method=method,
    )

    title = vf
    title += " Discretized" if discretized else ""
    title += " Runge-Kutta" if use_rk else " ODEINT"
    title += " " + optimizer.method

    # Build folder with images
    path_img = Path("img_optim")
    if not path_img.exists():
        os.mkdir(path_img)
    images = []
    idx = 0
    # Generate one plot every 2 iterations
    for list_routes in optimizer.optimize_route(q0[0], q0[1], q1[0], q1[1]):
        vectorfield.plot(x_min=-1, x_max=7, y_min=-1, y_max=7)
        for route in list_routes:
            plt.plot(route.x, route.y, color="green", alpha=0.6)
        plt.scatter([q0[0], q1[0]], [q0[1], q1[1]], c="red")
        plt.title(title)
        plt.xlim(-1, 7)
        plt.ylim(-1, 7)
        fout = path_img / f"{idx:03d}.png"
        idx += 1
        plt.savefig(fout)
        plt.close()
        images.append(imageio.imread(fout))

    # Plot best for some frames
    for i in range(6):
        route = list_routes[0]
        vectorfield.plot(x_min=-1, x_max=7, y_min=-1, y_max=7)
        plt.plot(route.x, route.y, color="green", alpha=1)
        plt.scatter([q0[0], q1[0]], [q0[1], q1[1]], c="red")
        plt.title(title)
        plt.xlim(-1, 7)
        plt.ylim(-1, 7)
        fout = path_img / f"{idx:03d}.png"
        idx += 1
        plt.savefig(fout)
        plt.close()
        images.append(imageio.imread(fout))

    # Build output folder
    path_out: Path = Path(path_out)
    if not path_out.exists():
        path_out.mkdir()
    # Convert images to gif and delete images
    imageio.mimsave(path_out / "optimizer.gif", images)
    shutil.rmtree(path_img)


if __name__ == "__main__":
    typer.run(main)
