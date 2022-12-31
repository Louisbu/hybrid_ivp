"""
Generate all the figures used in the paper
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import typer

from hybrid_routing.jax_utils.optimize import Optimizer
from hybrid_routing.vectorfields import Circular


def plot_runge_kutta(optimizer: Optimizer, x0: float, y0: float):
    optimizer.vectorfield.plot(x_min=-5, x_max=15, y_min=-5, y_max=15)
    plt.scatter(x0, y0, c="green", s=20)

    x = np.repeat(x0, 5)
    y = np.repeat(y0, 5)
    theta = np.linspace(1, 5, 5) * -np.pi / 4

    list_segments = optimizer.solve_ivp(x, y, theta)
    for segment in list_segments:
        x, y = segment.x, segment.y
        plt.plot(x, y, c="grey", alpha=0.9)
        plt.scatter(x[1:-1], y[1:-1], c="orange", s=12, alpha=0.8)
        plt.scatter(x[-1], y[-1], c="red", s=20)


def main(path_out: str = "output"):
    vectorfield = Circular()

    x0, y0 = 8, 8

    optimizer = Optimizer(
        vectorfield,
        time_iter=5,
        time_step=0.5,
        angle_amplitude=np.pi,
        num_angles=5,
        vel=1.5,
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
