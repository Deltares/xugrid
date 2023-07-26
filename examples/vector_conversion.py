"""
Vector geometry conversion
==========================

A great deal of geospatial data is available not in gridded form, but in
vectorized form: as points, lines, and polygons. In the Python data ecosystem,
these geometries and their associated data are generally represented by a
geopandas GeoDataFrame.

Xugrid provides a number of utilities to use such data in combination with
unstructured grids.

#
"""
# %%

import geopandas as gpd
import matplotlib.pyplot as plt

import xugrid as xu

# %%
# We'll once again use the surface elevation data example.

uda = xu.data.elevation_nl()
uda.ugrid.plot(vmin=-20, vmax=90, cmap="terrain")

# %%
# Conversion to GeoDataFrame
# --------------------------
#
# A UgridDataArray or UgridDataset can be directly converted to a GeoDataFrame,
# provided it only contains a spatial dimension (and not a dimension such as
# time). When calling
# ``.to_geodataframe``, a shapely Polygon is created for every face (cell).

gdf = uda.ugrid.to_geodataframe()
gdf

# %%
# Conversion from GeoDataFrame
# ----------------------------
#
# We can also make the opposite conversion: we can create a UgridDataSet from a
# GeoDataFrame.
#
# .. note::
#   Not every GeoDataFrame can be converted to a ``xugrid`` representation!
#   While an unstructured grid topology is generally always a valid collection
#   of polygon geometries, not every collection of polygon geometries is valid
#   grid: polygons should be convex and non-overlapping to create a valid
#   unstructured grid.
#
#   Hence, the ``.from_geodataframe()`` is primarily meant to create ``xugrid``
#   objects from data that were originally created as triangulation or
#   unstructured grid, but that were converted to vector geometry form.

back = xu.UgridDataset.from_geodataframe(gdf)
back

# %%
# "Rasterizing", or "burning" vector geometries
# ---------------------------------------------
#
# Rasterizing is a common operation when working with raster and vector data.
# While we cannot name the operation "rasterizing" when we're dealing with
# unstructured grids, there is a clearly equivalent operation where we mark
# cells that are covered or touched by a polygon.
#
# In this example, we mark the cells that are covered by a certain province.


# %%

provinces = gpd.read_file(r"c:\src\pandamesh\data\provinces-nl.geojson").to_crs(28992)
provinces["value"] = range(len(provinces))
burned = xu.burn_vector_geometry(provinces, uda, column="value")
burned.ugrid.plot()

# %%
# Polygonizing
# ------------
#
# We can also do the opposite operation: turn collections of same-valued grid
# cells into vector polygons. Let's classify the elevation data into below and
# above the boundary of 5 m above mean sea level:

classified = uda > 5
polygonized = xu.polygonize(classified)
polygonized.plot(facecolor="none")

# %%
