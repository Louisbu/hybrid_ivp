import os
import shutil
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import typer

from hybrid_routing.jax_utils.dnj import DNJRandomGuess
from hybrid_routing.vectorfields import FourVortices


def main(
    time_step: float = 0.1,
    num_points: int = 80,
    num_routes: int = 50,
    path_out: str = "output",
):
    vectorfield = FourVortices()
    q0 = (0, 0)
    q1 = (6, 2)
    dnj_random_guess = DNJRandomGuess(
        vectorfield,
        q0,
        q1,
        time_step=time_step,
        optimize_for="time",
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
        list_routes = next(dnj_random_guess)
        # Accelerate the number of iterations
        dnj_random_guess.num_iter = idx + 2
        vectorfield.plot(x_min=-1, x_max=7, y_min=-1, y_max=7)
        for route in list_routes:
            plt.plot(route.x, route.y, color="green")
        plt.scatter([q0[0], q1[0]], [q0[1], q1[1]], c="red")
        plt.title(f"Iterations: {dnj_random_guess.total_iter:05d}")
        plt.xlim(-1, 7)
        plt.ylim(-1, 7)
        fout = path_img / f"{idx:03d}.png"
        plt.savefig(fout)
        plt.close()
        images.append(imageio.imread(fout))
    # Build output folder
    path_out: Path = Path(path_out)
    if not path_out.exists():
        path_out.mkdir()
    # Convert images to gif and delete images
    imageio.mimsave(path_out / "dnj.gif", images)
    shutil.rmtree(path_img)


if __name__ == "__main__":
    typer.run(main)
