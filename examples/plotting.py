"""
Plot unstructured mesh data
===========================
Xarray provides a convenient way of plotting your data, provided it is
structured. ``xugrid`` contains a few additional plotting functions to easily
make spatial plots of unstructured grids.
"""
import matplotlib.pyplot as plt
import xarray as xr

import xugrid as xu

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

# uda.ugrid.plot()
#
# uda.ugrid.plot.edge()
# uda.ugrid.plot.face()
# uda.ugrid.plot.node()
#
# uda.ugrid.plot.edge.line()
#
# uda.ugrid.plot.face.imshow()
# uda.ugrid.plot.face.contour()
# uda.ugrid.plot.face.contourf()
# uda.ugrid.plot.face.scatter()
# uda.ugrid.plot.face.surface()
#
# uda.ugrid.plot.node.tripcolor()
# uda.ugrid.plot.node.contour()
# uda.ugrid.plot.node.contourf()
# uda.ugrid.plot.node.scatter()
# uda.ugrid.plot.node.surface()
