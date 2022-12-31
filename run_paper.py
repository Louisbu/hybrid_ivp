"""
Generate all the figures used in the paper
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hybrid_routing.jax_utils.optimize import Optimizer
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
    dist_min=None,
    use_rk=True,
    method="direction",
)


"""
Run Runge-Kutta method and plot its result
"""

# Initialize figure with vectorfield
# We encapsulate this code into a function because we are reusing it later
def plot_vectorfield():
    plt.figure(figsize=(6, 6))
    optimizer.vectorfield.plot(x_min=-5, x_max=15, y_min=-5, y_max=15)
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
    plt.plot(x, y, c="grey", alpha=0.9, zorder=5)
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
plt.text(-5, -5, eq_rk, fontsize=10, verticalalignment="bottom", bbox=bbox)

# Store plot
plt.tight_layout()
plt.savefig(path_out / "runge-kutta.png")
plt.close()
plt.close()
