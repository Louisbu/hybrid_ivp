"""
Generate all the figures used in the paper
"""

from copy import deepcopy
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from hybrid_routing.jax_utils.optimize import (
    Optimizer,
    compute_cone_center,
    compute_thetas_in_cone,
)
from hybrid_routing.jax_utils.route import RouteJax
from hybrid_routing.vectorfields import Circular

"""
Create output folder
"""

path_out: Path = Path("output")
if not path_out.exists():
    path_out.mkdir()

"""
Vectorfield and initial conditions
"""

vectorfield = Circular()

x0, y0 = 8, 8

optimizer = Optimizer(
    vectorfield,
    time_iter=4,
    time_step=0.4,
    angle_amplitude=np.pi,
    num_angles=5,
    vel=1.5,
    dist_min=0.1,
    use_rk=True,
    method="direction",
)


"""
Run Runge-Kutta method and plot its result
"""

# Initialize figure with vectorfield
# We encapsulate this code into a function because we are reusing it later
def plot_vectorfield():
    plt.figure(figsize=(5, 5))
    optimizer.vectorfield.plot(
        x_min=-8, x_max=18, y_min=-8, y_max=18, color="grey", alpha=0.8
    )
    plt.gca().set_aspect("equal")
    ticks = np.arange(-5, 20, 5)
    plt.xticks(ticks)
    plt.yticks(ticks)


plot_vectorfield()

# Plot source point
plt.scatter(x0, y0, c="green", s=20, zorder=10)

# Initial conditions of each segment (only angle varies)
x = np.repeat(x0, 5)
y = np.repeat(y0, 5)
theta = np.linspace(1, 5, 5) * -np.pi / 4

# Run RK method and plot each segment
list_segments = optimizer.solve_ivp(x, y, theta)
for segment in list_segments:
    x, y = segment.x, segment.y
    plt.plot(x, y, c="black", alpha=0.9, zorder=5)
    plt.scatter(x[1:-1], y[1:-1], c="orange", s=10, zorder=10)
    plt.scatter(x[-1], y[-1], c="red", s=20, zorder=10)

# Add equations
bbox = {"boxstyle": "round", "facecolor": "white", "alpha": 1}
eq_rk = r"""
$W(x,y) = \left\langle \frac{y+1}{20}, -\frac{x+3}{20}\right\rangle$
$\left\langle x_0, y_0 \right\rangle = \left\langle 8, 8 \right\rangle$
$V_0 = 1.5$
$\theta_0 = \frac{-\pi}{4}, \frac{-\pi}{2}, \frac{-3\pi}{4}, -\pi, \frac{-5\pi}{4}$
"""
plt.text(-4.5, -4.5, eq_rk, fontsize=10, verticalalignment="bottom", bbox=bbox)

# Store plot
plt.xlim(-5, 15)
plt.ylim(-5, 12)
plt.tight_layout()
plt.savefig(path_out / "runge-kutta.png")
plt.close()

print("Runge-Kutta - Finished")

"""
Exploration step
"""

x0, y0 = 12, -4
xn, yn = 4, 14
optimizer.vel = 1.5
optimizer.time_iter = 0.1
optimizer.time_step = 0.01
optimizer.angle_amplitude = np.pi / 2
optimizer.angle_heading = np.pi / 3
run = optimizer.optimize_route(x0, y0, xn, yn)
list_routes_plot = next(run)

for list_routes in run:
    if optimizer.exploration:
        list_routes_plot = deepcopy(list_routes)
    else:
        break

plot_vectorfield()

# Plot each route segment
# We encapsulate this code into a function because we are reusing it later
def plot_routes(list_routes: List[RouteJax]):
    # Plot source point
    plt.scatter(x0, y0, c="green", s=20, zorder=10)
    plt.scatter(xn, yn, c="green", s=20, zorder=10)
    # Plot routes
    for idx, route in enumerate(list_routes):
        x, y = route.x, route.y
        # Highlight the best route of the bunch
        s = 3 if idx == 0 else 1.5
        plt.plot(x, y, c="black", linewidth=s, alpha=0.9, zorder=5)


plot_routes(list_routes_plot)

# Compute angles
cone_center = compute_cone_center(x0, y0, xn, yn)
arr_theta = compute_thetas_in_cone(
    cone_center, optimizer.angle_amplitude, optimizer.num_angles
)

# Plot original angles
for theta in arr_theta:
    x = x0 + np.cos(theta) * np.array([0, 10])
    y = y0 + np.sin(theta) * np.array([0, 10])
    plt.plot(x, y, linestyle="--", color="orange", alpha=1, zorder=3)

# Add equations
eq_explo = r"""
$W(x,y) = \left\langle \frac{y+1}{20}, -\frac{x+3}{20}\right\rangle$
$\left\langle x_0, y_0 \right\rangle = \left\langle 12, -4 \right\rangle$
$\left\langle x_N, y_N \right\rangle = \left\langle 4, 12 \right\rangle$
$V_0 = 1.5$
$\theta_0 = \frac{3 \pi}{8}, \frac{\pi}{2}, \frac{5\pi}{8}, \mathbf{\frac{3\pi}{4}}, \frac{7\pi}{8}$
"""
plt.text(-6.5, 10, eq_explo, fontsize=10, verticalalignment="top", bbox=bbox)

# Store plot
plt.xlim(-7, 16)
plt.ylim(-7, 16)
plt.tight_layout()
plt.savefig(path_out / "hybrid-exploration.png")
plt.close()

print("Exploration step - Finished")

"""
Exploitation step
"""

for list_routes in run:
    if not optimizer.exploration:
        list_routes_plot = deepcopy(list_routes)
    else:
        break

plot_vectorfield()
plot_routes(list_routes_plot)

# Compute angles
arr_theta = compute_thetas_in_cone(
    3 * np.pi / 4, optimizer.angle_amplitude / 5, optimizer.num_angles
)

# Plot original angles
for theta in arr_theta:
    x = x0 + np.cos(theta) * np.array([0, 10])
    y = y0 + np.sin(theta) * np.array([0, 10])
    plt.plot(x, y, linestyle="--", color="orange", alpha=1, zorder=3)

# Add equations
eq_explo = r"""
$W(x,y) = \left\langle \frac{y+1}{20}, -\frac{x+3}{20}\right\rangle$
$\left\langle x_0, y_0 \right\rangle = \left\langle 12, -4 \right\rangle$
$\left\langle x_N, y_N \right\rangle = \left\langle 4, 12 \right\rangle$
$V_0 = 1.5$
$\theta_0 = \frac{13 \pi}{20}, \frac{7\pi}{10}, \frac{3\pi}{4}, \mathbf{\frac{4\pi}{5}}, \frac{17\pi}{20}$
"""
plt.text(-6.5, -6.5, eq_explo, fontsize=10, verticalalignment="bottom", bbox=bbox)

# Store plot
plt.xlim(-7, 16)
plt.ylim(-7, 16)
plt.tight_layout()
plt.savefig(path_out / "hybrid-exploitation.png")
plt.close()

print("Exploitation step - Finished")

"""
Exploration step #2
"""

for list_routes in run:
    if optimizer.exploration:
        list_routes_plot = deepcopy(list_routes)
    else:
        break

plot_vectorfield()
plot_routes(list_routes_plot)

# Store plot
plt.tight_layout()
plt.savefig(path_out / "hybrid-exploration2.png")
plt.close()
