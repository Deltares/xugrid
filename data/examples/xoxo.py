"""
Two Part Synthetic Unstructured Grid
====================================

This is a small unstructured grid consisting of two unconnected parts. This is
downloaded via `xugrid.data.xoxo()`. The toplogy data is downloaded to a local
directory if it's not there already.
"""
import xugrid
import matplotlib.pyplot as plt

vertices, triangles = xugrid.data.xoxo()
grid = xugrid.Ugrid2d(
    node_x=vertices[:, 0],
    node_y=vertices[:, 1],
    fill_value=-1,
    face_node_connectivity=triangles,
)

fig, ax = plt.subplots()
xugrid.plot.line(grid, ax=ax, color="#bd0d1f")
ax.set_xlim([0.0, 100.0])
ax.set_ylim([0.0, 85.0])
ax.set_aspect(1)
