"""
OverlapRegridder
================

The overlap regridder works in two stages. First, it searches the source grid
for all faces of the target grid, computes the intersections, and stores all
overlaps between source and target faces. This occurs when the regridder is
initialized. Second, the regridder applies the weights: it reduces the
collection of overlapping faces to a single value for the target face.

There are many reductions possible. The best choice generally differs based on
the physical meaning of the variable, or the application. Xugrid provides a
number of reductions, but it's also possible to use a custom reduction
function. This is demonstrated here.

We start with the same example as in the quick overview.
"""
# %%

import matplotlib.pyplot as plt
import numpy as np

import xugrid as xu

# %%
# We'll use a part of a triangular grid with the surface elevation (including
# some bathymetry) of the Netherlands, and a coarser target grid.


def create_grid(bounds, nx, ny):
    """
    Create a simple grid of triangles covering a rectangle.
    """
    import numpy as np
    from matplotlib.tri import Triangulation

    xmin, ymin, xmax, ymax = bounds
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    x = np.arange(xmin, xmax + dx, dx)
    y = np.arange(ymin, ymax + dy, dy)
    y, x = [a.ravel() for a in np.meshgrid(y, x, indexing="ij")]
    faces = Triangulation(x, y).triangles
    return xu.Ugrid2d(x, y, -1, faces)


uda = xu.data.elevation_nl().ugrid.sel(
    x=slice(125_000, 225_000), y=slice(440_000, 500_000)
)
grid = create_grid(uda.ugrid.total_bounds, nx=7, ny=6)

# %%

fig, ax = plt.subplots()
uda.ugrid.plot(vmin=-20, vmax=90, cmap="terrain", ax=ax)
grid.plot(ax=ax, color="red")

# %%
# Method comparison
# -----------------
#
# Let's compare the different reduction functions that are available in
# xugrid. We'll create a regridder once for every method, and plot the results
# side by side.
#
# .. note::
#   Sum and results in much higher values. The white in the figures are high
#   values, not no data. In contrast, a geometric mean generally only makes
#   sense for physical quantities with a "true zero": surface elevation is not
#   such quantity, as a datum is an arbitrary level. The xugrid geometric mean
#   returns NaN if reducing over negative values.

functions = [
    "mean",
    "harmonic_mean",
    "geometric_mean",
    "sum",
    "minimum",
    "maximum",
    "mode",
    "median",
    "max_overlap",
]

fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(10, 25), sharey=True, sharex=True)
axes = axes.ravel()

for f, ax in zip(functions, axes):
    regridder = xu.OverlapRegridder(source=uda, target=grid, method=f)
    result = regridder.regrid(uda)
    result.ugrid.plot(vmin=-20, vmax=90, cmap="terrain", ax=ax)
    ax.set_title(f)

# %%
# Relative overlap
# ----------------
#
# For some reductions, the relative degree of overlap with the original source
# cell is required rather than the absolute overlap, e.g. for first-order
# conservative methods, such as conductance:

regridder = xu.RelativeOverlapRegridder(source=uda, target=grid, method="conductance")
result = regridder.regrid(uda)
result.ugrid.plot()

# %%
# Custom reductions
# -----------------
#
# It's also possible to define your own reduction methods. Such a method is
# inserted during the ``.regrid`` call and compiled by `Numba`_ for performance.
#
# A valid reduction method must be compileable by Numba, and takes exactly three
# arguments: ``values``, ``indices``, ``weights``.
#
# * ``values``: is the ravelled array containing the (float) source values.
# * ``indices``: contains the flat, or "linear", (integer) indices into the
#   source values for a single target face.
# * ``weights``: contains the (float) overlap between the target face and the
#   source faces. The size of ``weights`` is equal to the size of ``indices``.
#
# Xugrid regridder reduction functions are implemented in such a way. For a example, the area
# could be implemented as follows:


def mean(values, indices, weights):
    subset = values[indices]
    return np.nansum(subset * weights) / np.nansum(weights)


# %%
# .. note::
#    * Custom reductions methods must be able to deal with NaN values as these
#      are commonly encountered in datasets as a "no data value".
#    * If Python features are used that are unsupported by Numba, you will get
#      somewhat obscure errors. In such a case, test your function with
#      synthetic values for ``values, indices, weights``.
#    * The built-in mean is more efficient, avoiding temporary memory
#      allocations.
#
# To use our custom method, we provide at initialization of the OverlapRegridder:

regridder = xu.OverlapRegridder(uda, grid, method=mean)
result = regridder.regrid(uda)
result.ugrid.plot(vmin=-20, vmax=90, cmap="terrain")

# %%
# Not every reduction uses the weights argument. For example, computing an
# arbitrary quantile value requires just the values. Again, make sure the
# function can deal with NaN values! -- hence ``nanpercentile`` rather than
# ``percentile`` here.


def p17(values, indices, weights):
    subset = values[indices]
    return np.nanpercentile(subset, 17)


regridder = xu.OverlapRegridder(uda, grid, method=p17)
result = regridder.regrid(uda)
result.ugrid.plot(vmin=-20, vmax=90, cmap="terrain")

# %%
# .. _Numba: https://numba.pydata.org/
