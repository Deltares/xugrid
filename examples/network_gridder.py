"""
Network gridder example
=======================

In this example, we demonstrate how to interpolate and grid
data from a 1D network grid to a 2D structured grid.
"""

# %%
# We'll start by setting up the structured grid first and converting it to a
# Ugrid2d grid.
import numpy as np
import xarray as xr

import xugrid as xu


def make_structured_grid(nrow, ncol, dx, dy):
    if dy >= 0:
        raise ValueError("dy must be negative.")

    shape = nrow, ncol

    xmin = 0.0
    xmax = dx * ncol
    ymin = 0.0
    ymax = abs(dy) * nrow
    dims = ("y", "x")

    y = np.arange(ymax, ymin, dy) + 0.5 * dy
    x = np.arange(xmin, xmax, dx) + 0.5 * dx
    coords = {"y": y, "x": x}

    return xr.DataArray(np.ones(shape, dtype=np.int32), coords=coords, dims=dims)


structured_grid = make_structured_grid(10, 10, 1.5, -1.5)
uda = xu.UgridDataArray.from_structured2d(structured_grid)
ugrid2d = uda.ugrid.grid

uda

# %%
#
# Next, we create a 1D network. This network consists of 5 nodes and 4 edges.
# At node 2 the network forks to two branches. The data is located no the nodes.

node_xy = np.array(
    [
        [0.0, 0.0],
        [5.0, 5.0],
        [10.0, 5.0],
        [15.0, 0.0],
        [15.0, 10.0],
    ]
)
edge_nodes = np.array(
    [
        [0, 1],
        [1, 2],
        [2, 3],
        [2, 4],
    ]
)
network = xu.Ugrid1d(*node_xy.T, -1, edge_nodes)
data = xr.DataArray(
    np.array([1, 1.5, 2, 4, -4], dtype=float), dims=(network.node_dimension,)
)
uda_1d = xu.UgridDataArray(data, grid=network)

uda_1d

# %%
#
# Let's plot the 1D network on top of the 2D grid. The 1D network is shown in
# gray, the 2D grid in blue. The network's nodes are colored by data values.

uda_1d.ugrid.plot()
uda_1d.ugrid.grid.plot(color="gray", alpha=0.5)
ugrid2d.plot()

# %% Intersect edges
#
# Let's first

edges_coords = ugrid2d.node_coordinates[ugrid2d.edge_node_connectivity]
_, _, intersections_xy = network.intersect_edges(edges_coords)

_intersections_xy = np.unique(intersections_xy, axis=0)
refined_network, refined_node_index = network.refine_by_vertices(
    _intersections_xy, return_index=True
)

# %%
refined_data = xr.DataArray(
    np.empty_like(refined_network.node_x), dims=(refined_network.node_dimension,)
)
uda_1d_refined = xu.UgridDataArray(refined_data, grid=refined_network)

# %%
# Set data
node_dim = uda_1d.ugrid.grid.node_dimension
uda_1d_refined.data[uda_1d[node_dim].data] = uda_1d.data
uda_1d_refined.data[refined_node_index] = np.nan

# Interpolate nodes
uda_1d_interpolated = uda_1d_refined.ugrid.laplace_interpolate()

# %%
# Set data to edge centroids
edge_data = xr.DataArray(
    data=uda_1d_interpolated.data[refined_network.edge_node_connectivity].mean(axis=1),
    dims=(refined_network.edge_dimension,),
)
uda_1d_edge = xu.UgridDataArray(edge_data, grid=refined_network)

# %%
from xugrid.regrid.gridder import NetworkGridder

gridder = NetworkGridder(
    source=uda_1d_edge.ugrid.grid,
    target=ugrid2d,
    method="mean",
)

# %%
network_gridded = gridder.regrid(uda_1d_edge)

# %%
network_gridded.ugrid.plot()

# %%

import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ugrid2d.plot(ax=ax)
plt.scatter(*intersections_xy.T)


# %%
