"""
Plot unstructured mesh data
===========================
Xarray provides a convenient way of plotting your data, provided it is
structured. ``xugrid`` contains a few additional plotting functions to easily
make spatial plots of unstructured grids.
"""
import matplotlib.pyplot as plt

import xugrid

ds = xugrid.data.disk()
uda = ds["face_z"]

uda.ugrid.plot()
uda.ugrid.plot.face()
uda.ugrid.plot.face.pcolormesh()
uda.ugrid.plot.face.contour()
uda.ugrid.plot.face.contourf()
uda.ugrid.plot.face.imshow()


fig, ax = plt.subplots()
uda.ugrid.plot(ax=ax)

uda.ugrid.plot.edge(color="black", linewidth=1.0)

# ax.triplot(triangulation)
# selection.ugrid.plot.edge(ax=ax)

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
