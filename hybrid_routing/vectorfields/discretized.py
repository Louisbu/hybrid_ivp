import itertools
from typing import Callable

import jax.numpy as jnp
import numpy as np
from hybrid_routing.vectorfields.base import Vectorfield
from scipy.interpolate._rgi import RegularGridInterpolator


def build_interp_fun(
    interp_u: RegularGridInterpolator, interp_v: RegularGridInterpolator
) -> Callable:
    grid_interp = jnp.array(interp_u.grid)
    values_u = jnp.array(interp_u.values)
    values_v = jnp.array(interp_v.values)

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
        vslice = (slice(None),) + (None,) * (values_u.ndim - len(indices))

        # find relevant values
        # each i and i+1 represents a edge
        edges = itertools.product(*[[i, i + 1] for i in indices])
        u = 0.0
        v = 0.0
        for edge_indices in edges:
            weight = 1.0
            for ei, i, yi in zip(edge_indices, indices, norm_distances):
                weight *= jnp.where(ei == i, 1 - yi, yi)
            u += jnp.asarray(values_u[edge_indices]) * weight[vslice]
            v += jnp.asarray(values_v[edge_indices]) * weight[vslice]
        return jnp.array([u, v])

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
        self.interp = build_interp_fun(interp_u, interp_v)

    def get_current(self, x: jnp.array, y: jnp.array) -> jnp.array:
        return self.interp(x, y)
