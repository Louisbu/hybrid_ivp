"""
Generate all the figures used in the paper. Results section
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hybrid_routing.jax_utils.dnj import DNJ
from hybrid_routing.jax_utils.optimize import Optimizer
from hybrid_routing.jax_utils.route import RouteJax
from hybrid_routing.vectorfields import FourVortices

"""
Create output folder
"""

path_out: Path = Path("output")
if not path_out.exists():
    path_out.mkdir()

# Initialize dict of results
dict_results = {}

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

for list_routes in optimizer.optimize_route(x0, y0, xn, yn):
    pass

# Take the best route
route: RouteJax = list_routes[0]
route.append_point_end(x=xn, y=yn, vel=optimizer.vel)

# Initialize figure with vectorfield
plt.figure(figsize=(5, 5))
vectorfield.plot(
    x_min=-2, x_max=8, y_min=-2, y_max=8, step=0.25, color="grey", alpha=0.8
)
plt.gca().set_aspect("equal")
ticks = np.arange(-2, 7, 1)
plt.xticks(ticks)
plt.yticks(ticks)

# Plot source and destination point
plt.scatter(x0, y0, c="green", s=20, zorder=10)
plt.scatter(xn, yn, c="green", s=20, zorder=10)
# Plot route
plt.plot(route.x, route.y, c="red", linewidth=1, alpha=0.9, zorder=5)
time_opt = float(route.t[-1])
# Recompute times
route.recompute_times(optimizer.vel, vectorfield)
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
route.recompute_times(optimizer.vel, vectorfield)
time_dnj = float(route.t[-1])

# Times
# Textbox properties
dict_bbox = dict(boxstyle="round", facecolor="white", alpha=0.95)
text = (
    r"""
$\left\langle x_0, y_0 \right\rangle = \left\langle 0, 0 \right\rangle$
$\left\langle x_T, y_T \right\rangle = \left\langle 6, 2 \right\rangle$
"""
    + f"""
Optimized (red):
  t = {time_opt_rec:.3f}
Smoothed (black):
  t = {time_dnj:.3f}
"""
)
plt.text(
    -0.0,
    5.5,
    text,
    fontsize=11,
    verticalalignment="top",
    bbox=dict_bbox,
)

# Store plot
plt.xlim(-0.5, 6.5)
plt.ylim(-1.5, 6)
plt.tight_layout()
plt.savefig(path_out / "results-fourvortices.png")
plt.close()

# Store results
dict_results["FourVortices"] = {
    "Time opt": time_opt,
    "Time opt rec": time_opt_rec,
    "Time DNJ": time_dnj,
}

"""
Store dictionary
"""
with open(path_out / "results.json", "w") as outfile:
    json.dump(dict_results, outfile)
