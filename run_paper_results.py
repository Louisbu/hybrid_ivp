"""
Generate all the figures used in the paper. Results section
"""

import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from hybrid_routing.jax_utils.dnj import DNJ
from hybrid_routing.jax_utils.optimize import Optimizer
from hybrid_routing.jax_utils.route import RouteJax
from hybrid_routing.vectorfields import Circular, FourVortices, HillBowl, Swirlys
from hybrid_routing.vectorfields.base import Vectorfield

"""
Create output folder
"""

path_out: Path = Path("output")
if not path_out.exists():
    path_out.mkdir()
# Initialize dict of results
dict_results = {}


def pipeline(
    vectorfield: Vectorfield,
    x0: float,
    y0: float,
    xn: float,
    yn: float,
    xmin: float,
    xmax: float,
    x_text: float,
    y_text: float,
    vel: float = 1,
    textbox_align: str = "top",
) -> Dict:

    # Initialize the optimizer
    optimizer = Optimizer(
        vectorfield,
        time_iter=0.5,
        time_step=0.025,
        angle_amplitude=np.pi,
        angle_heading=np.pi / 2,
        num_angles=20,
        vel=1,
        dist_min=0.1,
        use_rk=True,
        method="direction",
    )

    # Run the optimizer until it converges
    for list_routes in optimizer.optimize_route(x0, y0, xn, yn):
        pass

    # Take the best route
    route: RouteJax = list_routes[0]
    route.append_point_end(x=xn, y=yn, vel=optimizer.vel)

    # Initialize figure with vectorfield
    plt.figure(figsize=(5, 5))
    vectorfield.plot(
        x_min=xmin,
        x_max=xmax,
        y_min=xmin,
        y_max=xmax,
        step=0.25,
        color="grey",
        alpha=0.8,
    )
    plt.gca().set_aspect("equal")
    ticks = np.arange(xmin, xmax, 1)
    plt.xticks(ticks)
    plt.yticks(ticks)

    # Plot source and destination point
    plt.scatter(x0, y0, c="green", s=20, zorder=10)
    plt.scatter(xn, yn, c="green", s=20, zorder=10)
    # Plot route
    plt.plot(route.x, route.y, c="red", linewidth=1, alpha=0.9, zorder=5)
    time_opt = float(route.t[-1])
    # Recompute times
    route.recompute_times(vel, vectorfield)
    time_opt_rec = float(route.t[-1])

    # Apply DNJ
    dnj = DNJ(vectorfield, time_step=0.01, optimize_for="fuel")
    # Apply DNJ in loop
    for n in range(5):
        dnj.optimize_route(route, num_iter=200)
        s = 2 if n == 4 else 1
        c = "black" if n == 4 else "grey"
        alpha = 0.9 if n == 4 else 0.6
        plt.plot(route.x, route.y, c=c, linewidth=s, alpha=alpha, zorder=5)
    route.recompute_times(vel, vectorfield)
    time_dnj = float(route.t[-1])

    # Times
    # Textbox properties
    dict_bbox = dict(boxstyle="round", facecolor="white", alpha=0.95)
    text = (
        r"$\left\langle x_0, y_0 \right\rangle = \left\langle"
        + str(x0)
        + ", "
        + str(y0)
        + r"\right\rangle$"
        + "\n"
        r"$\left\langle x_T, y_T \right\rangle = \left\langle"
        + str(xn)
        + ", "
        + str(yn)
        + r"\right\rangle$"
        + f"\nOptimized (red):\n"
        + f"  t = {time_opt_rec:.3f}\n"
        + "Smoothed (black):\n"
        + f"  t = {time_dnj:.3f}"
    )
    plt.text(
        x_text,
        y_text,
        text,
        fontsize=11,
        verticalalignment=textbox_align,
        bbox=dict_bbox,
    )

    return {
        "Time opt": time_opt,
        "Time opt rec": time_opt_rec,
        "Time DNJ": time_dnj,
    }


"""
Vectorfield - Circular
"""

dict_results["Circular"] = pipeline(
    vectorfield=Circular(),
    x0=3,
    y0=2,
    xn=-7,
    yn=2,
    xmin=-8,
    xmax=8,
    x_text=0,
    y_text=-3.5,
    textbox_align="bottom",
)

# Store plot
plt.xlim(-8, 4)
plt.ylim(-4, 6)
plt.tight_layout()
plt.savefig(path_out / "results-circular.png")
plt.close()

print("Done Circular vectorfield")

"""
Vectorfield - Four Vortices
"""

# We will regenerate the results from Ferraro et al.
dict_results["FourVortices"] = pipeline(
    vectorfield=FourVortices(),
    x0=0,
    y0=0,
    xn=6,
    yn=2,
    xmin=-2,
    xmax=8,
    x_text=0,
    y_text=5.5,
)

# Store plot
plt.xlim(-0.5, 6.5)
plt.ylim(-1.5, 6)
plt.tight_layout()
plt.savefig(path_out / "results-fourvortices.png")
plt.close()

print("Done Four Vortices vectorfield")

"""
Vectorfield - Hillbowl
"""

# We will regenerate the results from Ferraro et al.
dict_results["Hiwllbowl"] = pipeline(
    vectorfield=HillBowl(),
    x0=0,
    y0=0,
    xn=6,
    yn=4,
    xmin=-2,
    xmax=8,
    x_text=0,
    y_text=5.5,
    vel=4,
)

# Store plot
plt.xlim(-0.5, 6.5)
plt.ylim(-1.5, 6)
plt.tight_layout()
plt.savefig(path_out / "results-hillbowl.png")
plt.close()

print("Done Hillbowl vectorfield")

"""
Vectorfield - Swirlys
"""

# We will regenerate the results from Ferraro et al.
dict_results["Swirlys"] = pipeline(
    vectorfield=Swirlys(),
    x0=1,
    y0=1,
    xn=6,
    yn=4,
    xmin=-2,
    xmax=8,
    x_text=0,
    y_text=5.5,
    vel=8,
)

# Store plot
plt.xlim(-0.5, 6.5)
plt.ylim(-1.5, 6)
plt.tight_layout()
plt.savefig(path_out / "results-swirlys.png")
plt.close()

print("Done Swirlys vectorfield")


"""
Store dictionary
"""
with open(path_out / "results.json", "w") as outfile:
    json.dump(dict_results, outfile)
