"""
Generate all the figures used in the paper
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import typer

from hybrid_routing.jax_utils.optimize import Optimizer
from hybrid_routing.vectorfields import Circular
from hybrid_routing.vectorfields.base import Vectorfield


def plot_runge_kutta(optimizer: Optimizer, x0: float, y0: float):
    optimizer.vectorfield.plot(x_min=-5, x_max=15, y_min=-5, y_max=15)
    plt.scatter(x0, y0, c="green")

    x = np.repeat(x0, 5)
    y = np.repeat(y0, 5)
    theta = np.linspace(1, 5, 5) * -np.pi / 4

    list_segments = optimizer.solve_ivp(x, y, theta)
    for segment in list_segments:
        plt.plot(segment.x, segment.y)


def main(path_out: str = "output"):
    vectorfield = Circular()

    x0, y0 = 8, 8

    optimizer = Optimizer(
        vectorfield,
        time_iter=2,
        time_step=0.1,
        angle_amplitude=np.pi,
        num_angles=5,
        vel=5,
        dist_min=None,
        use_rk=True,
        method="direction",
    )

    # Build output folder
    path_out: Path = Path(path_out)
    if not path_out.exists():
        path_out.mkdir()

    plot_runge_kutta(optimizer, x0, y0)
    plt.savefig(path_out / "runge-kutta.png")
    plt.close()


if __name__ == "__main__":
    typer.run(main)
