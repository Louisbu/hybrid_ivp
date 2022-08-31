import itertools
from typing import Callable

import jax.numpy as jnp
import numpy as np
from hybrid_routing.vectorfields.base import Vectorfield
from scipy.interpolate._rgi import RegularGridInterpolator


def build_interp_fun(interp: RegularGridInterpolator) -> Callable:
    grid_interp = jnp.array(interp.grid)
    values_interp = jnp.array(interp.values)

    def interp_fun(x: jnp.array, y: jnp.array) -> jnp.array:
        xi = jnp.stack((x, y))

        # find relevant edges between which xi are situated
        indices = []
        # compute distance to lower edge in unity units
        norm_distances = []
        # check for out of bounds xi
        out_of_bounds = jnp.zeros((xi.shape[1]), dtype=bool)
        # iterate through dimensions
        for x, grid in zip(xi, grid_interp):
            i = jnp.searchsorted(grid, x) - 1
            i = i.at[i < 0].set(0).at[i > grid.size - 2].set(grid.size - 2)
            indices.append(i)

            # compute norm_distances, incl length-1 grids,
            # where `grid[i+1] == grid[i]`
            denom = grid[i + 1] - grid[i]
            with np.errstate(divide="ignore", invalid="ignore"):
                norm_dist = jnp.where(denom != 0, (x - grid[i]) / denom, 0)
            norm_distances.append(norm_dist)

            out_of_bounds += x < grid[0]
            out_of_bounds += x > grid[-1]

        xi_shape = xi.shape
        xi = xi.reshape(-1, xi_shape[-1])

        # slice for broadcasting over trailing dimensions in self.values
        vslice = (slice(None),) + (None,) * (values_interp.ndim - len(indices))

        # find relevant values
        # each i and i+1 represents a edge
        edges = itertools.product(*[[i, i + 1] for i in indices])
        values = 0.0
        for edge_indices in edges:
            weight = 1.0
            for ei, i, yi in zip(edge_indices, indices, norm_distances):
                weight *= jnp.where(ei == i, 1 - yi, yi)
            values += jnp.asarray(values_interp[edge_indices]) * weight[vslice]
        return jnp.array(values)

    return interp_fun


class Discretized(Vectorfield):
    def __init__(self, vectorfield: Vectorfield):
        self.arr_x = vectorfield.arr_x
        self.arr_y = vectorfield.arr_y
        self.u = vectorfield.u
        self.v = vectorfield.v
        interp_u = RegularGridInterpolator(
            (self.arr_x, self.arr_y), self.u, method="linear"
        )
        interp_v = RegularGridInterpolator(
            (self.arr_x, self.arr_y), self.v, method="linear"
        )
        self.interp_u = build_interp_fun(interp_u)
        self.interp_v = build_interp_fun(interp_v)

    def get_current(self, x: jnp.array, y: jnp.array) -> jnp.array:
        return self.interp_u(x, y), self.interp_v(x, y)
