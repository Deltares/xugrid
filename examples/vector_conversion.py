"""
Vector geometry conversion
==========================

A great deal of geospatial data is available not in gridded form, but in
vectorized form: as points, lines, and polygons. In the Python data ecosystem,
these geometries and their associated data are generally represented by a
geopandas GeoDataFrame.

Xugrid provides a number of utilities to use such data in combination with
unstructured grids. These are demonstrated below.
"""
# %%

import geopandas as gpd

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
# We see that a GeoDataFrame with 5248 rows is created: one row for each face.
#
# Conversion from GeoDataFrame
# ----------------------------
#
# We can also make the opposite conversion: we can create a UgridDataSet from a
# GeoDataFrame.
#
back = xu.UgridDataset.from_geodataframe(gdf)
back

# %%
# .. note::
#   Not every GeoDataFrame can be converted to a ``xugrid`` representation!
#   While an unstructured grid topology is generally always a valid collection
#   of polygon geometries, not every collection of polygon geometries is a
#   valid grid: polygons should be convex and non-overlapping to create a valid
#   unstructured grid.
#
#   Secondly, each polygon fully owns its vertices (nodes), while the face of a
#   UGRID topology shares its nodes with its neighbors. All the vertices of the
#   polygons must therefore be exactly snapped together to form a connected
#   mesh.
#
#   Hence, the ``.from_geodataframe()`` is primarily meant to create ``xugrid``
#   objects from data that were originally created as triangulation or
#   unstructured grid, but that were converted to vector geometry form.
#
# "Rasterizing", or "burning" vector geometries
# ---------------------------------------------
#
# Rasterizing is a common operation when working with raster and vector data.
# While we cannot name the operation "rasterizing" when we're dealing with
# unstructured grids, there is a clearly equivalent operation where we mark
# cells that are covered or touched by a polygon.
#
# In this example, we mark the faces that are covered by a certain province.

provinces = xu.data.provinces_nl.to_crs(28992)
provinces["value"] = range(len(provinces))
burned = xu.burn_vector_geometry(provinces, uda, column="value")
burned.ugrid.plot()

# %%
# This is a convenient way to create masks and such:

utrecht = provinces[provinces["name"] == "Utrecht"]
burned = xu.burn_vector_geometry(utrecht, uda)
burned.ugrid.plot()

# %%
# This function can also be used to burn points or lines into the faces of an
# unstructured grid.
#
# The exterior boundaries of the province polygons will provide
# a collection of linestrings that we can burn into the grid:

lines = gpd.GeoDataFrame(geometry=provinces.exterior)
burned = xu.burn_vector_geometry(lines, uda)
burned.ugrid.plot()

# %%
# Polygonizing
# ------------
#
# We can also do the opposite operation: turn collections of same-valued grid
# faces into vector polygons. Let's classify the elevation data into below and
# above the boundary of 5 m above mean sea level:

classified = uda > 5
polygonized = xu.polygonize(classified)
polygonized.plot(facecolor="none")

# %%
# We see that the results consists of two large polygons, in which the
# triangles of the triangular grid have been merged to form a single polygon,
# and many smaller polygons, some of which correspond one-to-one to the
# triangles of the grid.
#
# .. note::
#   The produced polygon edges will follow exactly the face boundaries. When
#   the data consists of many unique values (e.g. unbinned elevation data), the
#   result will essentially be one polygon per face. In such cases, it is much
#   more efficient to use ``xugrid.UgridDataArray.to_geodataframe``, which
#   directly converts every face to a polygon.
