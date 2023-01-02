"""
Generate all the figures used in the paper. Methods section
"""

from copy import deepcopy
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes

from hybrid_routing.jax_utils.dnj import DNJ
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
$V_{vessel} = 1.5$
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
$\left\langle x_T, y_T \right\rangle = \left\langle 4, 14 \right\rangle$
$V_{vessel} = 1.5$
$\theta_0 = \frac{3 \pi}{8}, \frac{\pi}{2}, \frac{5\pi}{8}, \mathbf{\frac{3\pi}{4}}, \frac{7\pi}{8}$
"""
plt.text(-6.5, 15.5, eq_explo, fontsize=10, verticalalignment="top", bbox=bbox)

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
$\left\langle x_T, y_T \right\rangle = \left\langle 4, 14 \right\rangle$
$V_{vessel} = 1.5$
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
Finish optimization
"""

for list_routes in run:
    list_routes_plot = deepcopy(list_routes)
# Append goal
route: RouteJax = list_routes_plot[0]
route.append_point_end(x=xn, y=yn, vel=optimizer.vel)

plot_vectorfield()
plot_routes([route])

# Add equations
eq_opt = r"""
$W(x,y) = \left\langle \frac{y+1}{20}, -\frac{x+3}{20}\right\rangle$
$\left\langle x_0, y_0 \right\rangle = \left\langle 12, -4 \right\rangle$
$\left\langle x_T, y_T \right\rangle = \left\langle 4, 14 \right\rangle$
$V_{vessel} = 1.5$
"""
plt.text(-6.5, -6.5, eq_opt, fontsize=10, verticalalignment="bottom", bbox=bbox)

# Store plot
plt.xlim(-7, 16)
plt.ylim(-7, 16)
plt.tight_layout()
plt.savefig(path_out / "hybrid-optimized.png")
plt.close()

print("Optimization - Finished")

"""
Discrete Newton-Jacobi
"""

dnj = DNJ(vectorfield, time_step=0.01, optimize_for="fuel")

plot_vectorfield()
plt.scatter(route.x[0], route.y[0], c="green", s=20, zorder=10)

# Prepare zoom
fig = plt.gcf()
ax = plt.gca()
axins = zoomed_inset_axes(ax, zoom=2.5, loc="center right")
optimizer.vectorfield.plot(
    x_min=2, x_max=7, y_min=11, y_max=16, color="grey", alpha=0.8, scale=2
)

# Goal point and original route (plot both in normal and zoom)
for axis in [ax, axins]:
    axis.scatter(route.x[-1], route.y[-1], c="green", s=20, zorder=10)
    axis.plot(route.x, route.y, c="red", linewidth=2, alpha=0.9, zorder=5)

# Apply DNJ in loop
for n in range(5):
    dnj.optimize_route(route, num_iter=2000)
    s = 2 if n == 4 else 1
    c = "black" if n == 4 else "grey"
    alpha = 0.9 if n == 4 else 0.6
    # Plot both in normal and zoom
    for axis in [ax, axins]:
        axis.plot(route.x, route.y, c=c, linewidth=s, alpha=alpha, zorder=5)

# Add equations
ax.text(-6.5, -6.5, eq_opt, fontsize=10, verticalalignment="bottom", bbox=bbox)

# Limit zoom axis
axins.set_xlim(3, 6)
axins.set_ylim(12, 15)
mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")
# Hide ticks in zoomed axis
plt.tick_params(
    axis="both",  # changes apply to the x-axis
    which="both",  # both major and minor ticks are affected
    bottom=False,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
    left=False,
    right=False,
    labelbottom=False,  # labels along the bottom edge are off
    labelleft=False,
)

# Limit normal axis
ax.set_xlim(-7, 16)
ax.set_ylim(-7, 16)

# Store plot
plt.draw()
fig.tight_layout()
plt.savefig(path_out / "hybrid-dnj.png")
plt.close()

print("DNJ - Finished")
