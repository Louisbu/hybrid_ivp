"""
Generate all the figures used in the paper. Results section
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hybrid_routing.jax_utils.dnj import DNJRandomGuess
from hybrid_routing.vectorfields import FourVortices

"""
Create output folder
"""

path_out: Path = Path("output")
if not path_out.exists():
    path_out.mkdir()

"""
Vectorfield - Four Vortices
"""

vectorfield = FourVortices()

# We will regenerate the results from Ferraro et al.
x0, y0 = 0, 0
xn, yn = 6, 2

dnj = DNJRandomGuess(
    vectorfield=vectorfield,
    q0=(x0, y0),
    q1=(xn, yn),
    time_step=0.025,
    optimize_for="fuel",
    angle_amplitude=np.pi,
    num_points=80,
    num_routes=20,
    num_iter=5000,
)

list_routes = next(dnj)

# Initialize figure with vectorfield
plt.figure(figsize=(5, 5))
vectorfield.plot(
    x_min=-2, x_max=8, y_min=-2, y_max=8, step=0.25, color="grey", alpha=0.8
)
plt.gca().set_aspect("equal")
ticks = np.arange(-2, 7, 1)
plt.xticks(ticks)
plt.yticks(ticks)

for route in list_routes:
    plt.plot(route.x, route.y, c="grey", linewidth=1, alpha=0.9, zorder=5)

# Store plot
plt.xlim(-0.5, 6.5)
plt.ylim(-1.5, 6)
plt.tight_layout()
plt.savefig(path_out / "results-fourvortices-dnj.png")
plt.close()
