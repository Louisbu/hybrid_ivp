"""
Generate all the figures used in the paper. Results section
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hybrid_routing.jax_utils.dnj import DNJ
from hybrid_routing.jax_utils.optimize import Optimizer
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

# Run the optimizer
optimizer = Optimizer(
    vectorfield,
    time_iter=0.1,
    time_step=0.01,
    angle_amplitude=np.pi,
    angle_heading=np.pi / 2,
    num_angles=10,
    vel=1,
    dist_min=0.1,
    use_rk=True,
    method="direction",
)

idx = 0
for list_routes in optimizer.optimize_route(x0, y0, xn, yn):
    idx += 1
    if idx % 20 == 0:
        print(idx)
# Take the best route
route = list_routes[0]

# Initialize figure with vectorfield
plt.figure(figsize=(5, 5))
vectorfield.plot(x_min=-1, x_max=7, y_min=-2, y_max=7, color="grey", alpha=0.8)
plt.gca().set_aspect("equal")
ticks = np.arange(-2, 7, 1)
plt.xticks(ticks)
plt.yticks(ticks)

# Plot source and destination point
plt.scatter(x0, y0, c="green", s=20, zorder=10)
plt.scatter(xn, yn, c="green", s=20, zorder=10)
# Plot route
plt.plot(route.x, route.y, c="red", linewidth=1, alpha=0.9, zorder=5)
print("Time:", route.t[-1])

# Apply DNJ
dnj = DNJ(vectorfield, time_step=0.01, optimize_for="fuel")
dnj.optimize_route(route, num_iter=10000)

# Plot route
plt.plot(route.x, route.y, c="black", linewidth=1, alpha=0.9, zorder=7)
print("Time:", route.t[-1])

# Store plot
plt.xlim(0, 6)
plt.ylim(-1, 6)
plt.tight_layout()
plt.savefig(path_out / "results-fourvortices.png")
plt.close()
