"""
Plot unstructured mesh data
===========================
Xarray provides a convenient way of plotting your data, provided it is
structured. ``xugrid`` contains a few additional plotting functions to easily
make spatial plots of unstructured grids.
"""
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import xugrid as xu
from xugrid import connectivity

ds = xr.open_dataset("Dommel-tops-bottoms.nc")
uds = xu.UgridDataset(ds)
uda = uds["top_layer_1"]

xmin, xmax = 130_000.0, 140_000.0
ymin, ymax = 380_000.0, 390_000.0

selection = uda.ugrid.sel(x=slice(xmin, xmax), y=slice(ymin, ymax))
tri = selection.ugrid.grid.matplotlib_triangulation()
fig, ax = plt.subplots()
ax.triplot(tri)

fig, ax = plt.subplots()
ax.tripcolor(tri, selection.values)
