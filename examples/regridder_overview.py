"""
Regridding overview
===================

`Regridding`_ is the process of converting gridded data from one grid to
another grid. Xugrid provides tools for 2D and 3D regridding of structured
gridded data, represented as xarray objects, as well as (`layered`_)
unstructured gridded data, represented as xugrid objects.

A number of regridding methods are provided, based on area or volume overlap,
as well as interpolation routines. It currently only supports Cartesian
coordinates. See e.g. `xESMF`_ instead for regridding with a spherical Earth
representation (note: EMSF is `not available`_ via conda-forge on Windows).

Here are a number of quick examples of how to get started with regridding.

We'll start by importing a few essential packages.
"""
# %%

import matplotlib.pyplot as plt
import xarray as xr

import xugrid as xu

# %%
# We will take a look at a sample dataset: a triangular grid with the surface
# elevation of the Netherlands.

uda = xu.data.elevation_nl()
uda.ugrid.plot(vmin=-20, vmax=90, cmap="terrain")

# %%
# Xugrid provides several "regridder" classes which can convert gridded data
# from one grid to another grid. Let's generate a very simple coarse mesh that
# covers the entire Netherlands.


def create_grid(bounds, nx, ny):
    """Create a simple grid of triangles covering a rectangle."""
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


grid = create_grid(uda.ugrid.total_bounds, 7, 7)

# %%
# CentroidLocatorRegridder
# ------------------------
#
# An easy way of regridding is by simply looking in which cell of the original
# the centroids of the new grid fall.

fig, ax = plt.subplots()
uda.ugrid.plot(vmin=-20, vmax=90, cmap="terrain", ax=ax)
grid.plot(ax=ax, color="red")
ax.scatter(*grid.centroids.T, color="red")

# %%
# Xugrid provides the CentroidLocatorRegridder for this:

regridder = xu.CentroidLocatorRegridder(source=uda, target=grid)
result = regridder.regrid(uda)
result.ugrid.plot(vmin=-20, vmax=90, cmap="terrain", edgecolor="red")

# %%
# OverlapRegridder
# ----------------
#
# Such a regridding is not appropriate when the new grid cells are
# so large. Let's try the OverlapOverregridder instead.

regridder = xu.OverlapRegridder(source=uda, target=grid)
mean = regridder.regrid(uda)
mean.ugrid.plot(vmin=-20, vmax=90, cmap="terrain", edgecolor="red")

# %%
# By default, the OverlapRegridder computes an area weighted mean.
# Let's try again, now with the minimum:

regridder = xu.OverlapRegridder(source=uda, target=grid, method="minimum")
minimum = regridder.regrid(uda)
minimum.ugrid.plot(vmin=-20, vmax=90, cmap="terrain", edgecolor="red")

# %%
# Or the maximum:

regridder = xu.OverlapRegridder(source=uda, target=grid, method="maximum")
maximum = regridder.regrid(uda)
maximum.ugrid.plot(vmin=-20, vmax=90, cmap="terrain", edgecolor="red")

# %%
# All regridders also work for multi-dimensional data.
#
# Let's pretend our elevation dataset contains multiple layers, for example to
# denote multiple geological strata. We'll generate five layers, each with a
# thickness of 10.0 meters.

thickness = xr.DataArray(
    data=[10.0, 10.0, 10.0, 10.0, 10.0],
    coords={"layer": [1, 2, 3, 4, 5]},
    dims=["layer"],
)

# %%
# We need to make that the face dimension remains last, so we transpose the
# result.

bottom = (uda - thickness.cumsum("layer")).transpose()
bottom

# %%
# We can feed the result to the regridder, which will automatically regrid over
# all additional dimensions.

mean_bottom = xu.OverlapRegridder(source=bottom, target=grid).regrid(bottom)
mean_bottom

# %%
# Let's take a slice to briefly inspect our original layer bottom elevation,
# and the aggregated mean.

section_y = 475_000.0
section = bottom.ugrid.sel(y=section_y)
section_mean = mean_bottom.ugrid.sel(y=section_y)

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 5), sharex=True, sharey=True)
section.plot.line(x="mesh2d_s", hue="layer", ax=ax0)
section_mean.plot.line(x="mesh2d_s", hue="layer", ax=ax1)

# %%
# BarycentricInterpolator
# -----------------------
#
# All examples above show reductions: from a fine grid to a coarse grid.
# However, xugrid also provides interpolation to generate smooth fine
# representations of a coarse grid.
#
# To illustrate, we will zoom in to a part of the Netherlands.

part = uda.ugrid.sel(x=slice(125_000, 225_000), y=slice(440_000, 500_000))
part.ugrid.plot(vmin=-20, vmax=90, cmap="terrain")

# %%
# We can clearly identify the individual triangles that form the grid. To get a
# smooth presentation, we can use the BarycentricInterpolator.
#
# We will generate a fine grid.

grid = create_grid(part.ugrid.total_bounds, nx=100, ny=100)

# %%
# We use the centroids of the fine grid to interpolate between the centroids of
# the triangles.

regridder = xu.BarycentricInterpolator(part, grid)
interpolated = regridder.regrid(part)
interpolated.ugrid.plot(vmin=-20, vmax=90, cmap="terrain")

# %%
# Arbitrary grids
# ---------------
#
# The above examples all feature triangular source and target grids. However,
# the regridders work for any collection of (convex) faces.

grid = create_grid(part.ugrid.total_bounds, nx=20, ny=15)
voronoi_grid = grid.tesselate_centroidal_voronoi()

regridder = xu.CentroidLocatorRegridder(part, voronoi_grid)
result = regridder.regrid(part)

fig, ax = plt.subplots()
result.ugrid.plot(vmin=-20, vmax=90, cmap="terrain")
voronoi_grid.plot(ax=ax, color="red")

# %%
# Re-use
# ------
#
# The most expensive step of the regridding process is finding and computing
# overlaps. A regridder can be used repeatedly, provided the source topology
# is kept the same.

part_other = part - 50.0
result = regridder.regrid(part_other)
result.ugrid.plot(vmin=-20, vmax=90, cmap="terrain")

# %%
# .. _Xarray: https://docs.xarray.dev/en/stable/index.html
# .. _Xugrid: https://deltares.github.io/xugrid/
# .. _Regridding: https://climatedataguide.ucar.edu/climate-tools/regridding-overview
# .. _layered: https://ugrid-conventions.github.io/ugrid-conventions/#3d-layered-mesh-topology
# .. _xESMF: https://xesmf.readthedocs.io/en/latest/index.html
# .. _not available: https://github.com/conda-forge/esmf-feedstock/issues/64

# %%
