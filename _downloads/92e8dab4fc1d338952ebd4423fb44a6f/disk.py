"""
Disk
====

This is a small synthetic unstructured XugridDataset with topology in the shape
of a disk. It contains data on the nodes, faces, and edges.
"""
import matplotlib.pyplot as plt

import xugrid

uds = xugrid.data.disk()

fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(15, 5))
axes = axes.ravel()
for ax in axes:
    ax.set_aspect(1)

uds["node_z"].ugrid.plot(ax=axes[0], add_colorbar=False, cmap="terrain")
uds["face_z"].ugrid.plot(ax=axes[1], add_colorbar=False, cmap="terrain")
uds["edge_z"].ugrid.plot(ax=axes[2], add_colorbar=False, cmap="terrain")
