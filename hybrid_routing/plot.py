from typing import Callable
import matplotlib.pyplot as plt
import numpy as np


def plot_vector_field(
    vector_field: Callable,
    x_min: float = 0,
    x_max: float = 125,
    y_min: float = 0,
    y_max: float = 125,
    step: float = 10,
):
    x, y = np.meshgrid(np.linspace(x_min, x_max, step), np.linspace(y_min, y_max, step))
    u, v = vector_field(x, y)
    plt.quiver(x, y, u, v)
