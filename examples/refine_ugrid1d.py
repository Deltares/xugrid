# %%

import numpy as np
import xugrid as xu
import matplotlib.pyplot as plt

# %%

node_xy = np.array([
    [0.0, 0.0],
    [5.0, 5.0],
    [10.0, 5.0],
    [15.0, 0.0],
    [15.0, 10.0],
])
edge_nodes = np.array([
    [0, 1],
    [1, 2],
    [2, 3],
    [2, 4],
])
grid = xu.Ugrid1d(*node_xy.T, -1, edge_nodes)

fig, ax = plt.subplots()
grid.plot(ax=ax)
ax.scatter(*grid.node_coordinates.T)

# %%

vertices = np.array([
    [7.5, 5.0],
    [12.5, 2.5],
    [12.5, 7.5],
    [1.0, 1.0],
    [4.0, 4.0],
])
edge_index = np.array([1, 2, 3, 0, 0])

# %%
new = grid.refine_by_vertices(vertices, edge_index)

fig, ax = plt.subplots()
new.plot(ax=ax)
ax.scatter(*new.node_coordinates.T)

# %%

vertices = np.array([
    [7.5, 5.0],
    [12.5, 7.5],
])
edge_index = np.array([1, 3])
new = grid.refine_by_vertices(vertices, edge_index)

fig, ax = plt.subplots()
new.plot(ax=ax)
ax.scatter(*new.node_coordinates.T)