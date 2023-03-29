Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog`_, and this project adheres to
`Semantic Versioning`_.

[Unreleased]
------------

[0.3.0] 2023-03-14
------------------

Fixed
~~~~~

Changed
~~~~~~~

- ``pygeos`` has been replaced by ``shapely >= 2.0``.
- :func:`xugrid.snap_to_grid` will now return a UgridDataset and a geopandas
  GeoDataFrame. The UgridDataset contains the snapped data on the edges of the
  the UGRID topology.
- :class:`xugrid.RelativeOverlapRegridder` has been created to separate the
  relative overlap logic from :class:`xugrid.OverlapRegridder`.
- :class:`xugrid.BarycentricInterpolator`,
  :class:`xugrid.CentroidLocatorRegridder`, :class:`xugrid.OverlapRegridder`,
  and :class:`RelativeOverlapRegridder` can now be instantiated from weights
  (``.from_weights``) or from a dataset (``.from_dataset``) containing
  pre-computed weights.
- Regridder classes initiated with method *geometric_mean* now return NaNs for
  negative data.

Added
~~~~~

- :func:`xugrid.Ugrid2d.tesselate_circumcenter_voronoi` has been added to
  provide orthogonal voronoi cells for triangular grids.

[0.2.1] 2023-02-06
------------------

Fixed
~~~~~
- :func:`xugrid.open_dataarray` will now return :class:`xugrid.UgridDataArray`
  instead of only an xarray DataArray without topology.
- Setting wrapped properties of the xarray object (such as ``name``) now works.
- Creating new (subset) topologies via e.g. selection will no longer error when
  datasets contains multiple coordinates systems (such as both longitude and
  latitude next to projected x and y coordinates). 

Changed
~~~~~~~

Added
~~~~~

- Several regridding methods have been added for face associated data:
  :class:`xugrid.BarycentricInterpolator` have been added to interpolate
  smoothly, :class:`xugrid.CentroidLocatorRegridder` has been added to simply
  sample based on face centroid, and :class:`xugrid.OverlapRegridder` supports
  may aggregation methods (e.g. area weighted mean).
- Added :attr:`xugrid.Ugrid1d.edge_node_coordinates`.
- Added :attr:`xugrid.Ugrid2d.edge_node_coordinates` and
  :attr:`xugrid.Ugrid2d.face_node_coordinates`.

[0.2.0] 2023-01-19
------------------

Fixed
~~~~~

- :meth:`xugrid.Ugrid1d.topology_subset`,
  :meth:`xugrid.Ugrid2d.topology_subset`, and therefore also
  :meth:`xugrid.UgridDataArrayAccessor.sel` and
  :meth:`xugrid.UgridDatasetAccessor.sel` now propagate UGRID attributes.
  Before this fix, dimension of the UGRID topology would go out of sync with
  the DataArray, as a subset would return a new UGRID topology with default
  UGRID names. 
- :meth:`xugrid.Ugrid2d.topology_subset`, :meth:`xugrid.UgridDataArrayAccessor.sel`
  :meth:`xugrid.UgridDatasetAccessor.sel` will now return a correct UGRID 2D
  topology when fill values are present in the face node connectivity.
- :meth:`xugrid.plot.contour` and :meth:`xugrid.plot.contourf` will no longer
  plot erratic contours when "island" faces are present (no connections to
  other faces) or when "slivers" are present (where cells have a only a left or
  right neighbor).
- :meth:`xugrid.plot.pcolormesh` will draw all edges around faces now when
  edgecolor is defined, rather than skipping some edges.
- Do not mutate edge_node_connectivity in UGRID2D when the
  face_node_connectivity property is accessed.

Changed
~~~~~~~

- Forwarding to the internal xarray object is now setup at class definition of
  :class:`UgridDataArray` and :class:`UgridDataset` rather than at runtime.
  This means tab completion and docstrings for the xarray methods should work.
- The UGRID dimensions in :class:`UgridDataArray` and :class:`UgridDataset` are
  labelled at initialization. This allows us to track necessary changes to the
  UGRID topology for general xarray operations. Forwarded methods (such as
  :meth:`UgridDataArray.isel`) will now create a subset topology if possible, or
  error if an invalid topology is created by the selection.
- This also means that selection on one facet of the grid (e.g. the face
  dimension) will also result in a valid selection of the data on another facet
  (such as the edge dimension).
- :meth:`xugrid.Ugrid1d.sel` and :meth:`xugrid.Ugrid2d.sel` now take an ``obj``
  argument and return a DataArray or Dataset.
- Consequently, `xugrid.UgridDataArrayAccessor.isel` and
  `xugrid.UgridDatasetAccessor.isel` have been removed.
- :attr:`xugrid.Ugrid1d.dimensions` and
  :attr:`xugrid.Ugrid2d.dimensions` will now return a dictionary with the
  keys the dimension names and as the values the sizes of the dimensions.
- :attr:`xugrid.Ugrid2d.voronoi_topology` will now include exterior vertices to
  also generate a valid 2D topology when when "island" faces are present (no
  connections to other faces) or when "slivers" are present (where cells have a
  only a left or right neighbor).

Added
~~~~~

- :class:`xugrid.Ugrid1d` and :class:`xugrid.Ugrid2d` can now be initialized
  with an ``attrs`` argument to setup non-default UGRID attributes such as
  alternative node, edge, or face dimensions.
- :meth:`xugrid.Ugrid1d.topology_subset`,
  :meth:`xugrid.Ugrid2d.topology_subset`, :meth:`xugrid.Ugrid1d.isel`, and
  :meth:`xugrid.Ugrid2d.isel` now take a ``return_index`` argument and will
  to return UGRID dimension indexes if set to True.
- :meth:`xugrid.UgridDataArrayAccessor.clip_box` and
  :meth:`xugrid.UgridDatasetAccessor.clip_box` have been added to more easily
  select data in a bounding box.
- For convenience, ``.grid``, ``.grids``, ``.obj`` properties are now available
  on all these classes: :class:`UgridDataArray`, :class:`UgridDataset`,
  :class:`UgridDataArrayAccessor`, and :class:`UgridDatasetAccessor`.
- Added :func:`xugrid.merge_partitions` to merge topology and data that have
  been partitioned along UGRID dimensions.

[0.1.10] 2022-12-13
-------------------

Fixed
~~~~~

- Move matplotlib import into a function body so matplotlib remains an optional
  dependency. 

[0.1.9] 2022-12-13
------------------

Changed
~~~~~~~
- Warn instead of error when the UGRID attributes indicate a set of coordinate
  that are not present in the dataset.
- Use `pyproject.toml` for setuptools instead of `setup.cfg`.

Added
~~~~~

- :attr:`xugrid.Ugrid1d.edge_bounds` has been added to get the bounds
  for every edge contained in the grid.
- :attr:`xugrid.Ugrid2d.edge_bounds` has been added to get the bounds
  for every edge contained in the grid.
- :attr:`xugrid.Ugrid2d.face_bounds` has been added to get the bounds
  for face edge contained in the grid.
- :meth:`xugrid.Ugrid1d.from_meshkernel` and
  :meth:`xugrid.Ugrid2d.from_meshkernel` have been added to initialize Ugrid
  topology from a meshkernel object.
- :meth:`xugrid.Ugrid1d.plot` and :meth:`xugrid.Ugrid2d.plot` have been added
  to plot the edges of the grid.

Fixed
~~~~~

- :meth:`xugrid.UgridDataArray.from_structured` will no longer result in
  a flipped grid when the structured coordintes are not ascending.

[0.1.7] 2022-09-06
------------------

Fixed
~~~~~
- The setitem method of :class:`xugrid.UgridDataset` has been updated to check
  the dimensions of grids rather than the dimensions of objects to decide
  whether a new grids should be appended.
- :meth:`xugrid.UgridDataArrayAccessor.assign_edge_coords` and
  :meth:`xugrid.UgridDatasetAccessor.assign_edge_coords` have been added to add
  the UGRID edge coordinates to the xarray object.
- :meth:`xugrid.UgridDataArrayAccessor.assign_face_coords` and
  :meth:`xugrid.UgridDatasetAccessor.assign_face_coords` have been added to add
  the UGRID face coordinates to the xarray object.
- Fixed mixups in ``xugrid.UgridRolesAccessor`` for inferring UGRID dimensions,
  which would result incorrectly in a ``UgridDimensionError`` complaining about
  conflicting dimension names.

[0.1.5] 2022-08-22
------------------

Fixed
~~~~~

- ``list`` and ``dict`` type annotations have been replaced with ``List`` and ``Dict``
  from the typing module to support older versions of Python (<3.9).

Changed
~~~~~~~

- The ``inplace`` argument has been removed from :meth:`xugrid.Ugrid1d.to_crs`
  and :meth:`xugrid.Ugrid2d.to_crs`; A copy is returned when the CRS is already
  as requested.

Added
~~~~~

- :meth:`xugrid.UgridDataArrayAccessor.set_crs` has been added to set the CRS.
- :meth:`xugrid.UgridDataArrayAccessor.to_crs` has been added to reproject the
  grid of the DataArray.
- :meth:`xugrid.UgridDatasetAccessor.set_crs` has been added to set the CRS of
- :meth:`xugrid.UgridDatasetAccessor.to_crs` has been added to reproject a grid
  or all grids of a dataset.
- :attr:`xugrid.UgridDataArrayAccessor.bounds` has been added to get the bounds
  of the grid coordinates.
- :attr:`xugrid.UgridDataArrayAccessor.total_bounds` has been added to get the
  bounds of the grid coordinates.
- :attr:`xugrid.UgridDatasetAccessor.bounds` has been added to get the bounds
  for every grid contained in the dataset.
- :attr:`xugrid.UgridDatasetAccessor.total_bounds` has been added to get the
  total bounds of all grids contained in the dataset.

[0.1.4] 2022-08-16
------------------

Fixed
~~~~~

- A ``start_index`` of 1 in connectivity arrays is handled and will no longer
  result in indexing errors.
- ``levels`` argument is now respected in line and pcolormesh plotting methods.

Changed
~~~~~~~

- UGRID variables are now extracted via :class:`xugrid.UgridRolesAccessor` to
  allow for multiple UGRID topologies in a single dataset.
- Extraction of the UGRID dimensions now proceeds via the dummy variable
  attributes, the connetivity arrays, and finally the coordinates.
- Multiple coordinates can be supported. The UgridRolesAccessor attempts
  to infer valid node coordinates based on their standard names
  (one of``projection_x_coordinate, projection_y_coordinate, longitude,
  latitude``); a warning is raised when these are not found.
- :class:`xugrid.UgridDataset` now supports multiple Ugrid topologies.
  Consequently, its ``.grid`` attribute has been replaced by ``.grids``.
- The xarray object is no longer automatically wrapped when accessing the
  ``.obj`` attribute of a UgridDataArray or UgridDataset.
- Separate UgridAccessors have been created for UgridDataArray and UgridDataset
  as many methods are specific to one but not the other.
- The Ugrid classes have been subtly changed to support multiple topologies
  in a dataset. The ``.dataset`` attribute has been renamed to ``._dataset``,
  as access to the dataset should occur via the ``.to_dataset()`` method
  instead, which can check for consistency with the xarray object.

Added
~~~~~

- :class:`xugrid.UgridRolesAccessor` has been added to extract UGRID variables
  from xarray Datasets.
- :func:`xugrid.merge` and :func:`xugrid.concat` have been added, since the
  xarray functions raise a TypeError on non-xarray objects. 
- :meth:`xugrid.UgridDataArrayAccessor.assign_node_coords` and
  :meth:`xugrid.UgridDatasetAccessor.assign_node_coords` have been added to add
  the UGRID node coordinates to the xarray object.
- :meth:`xugrid.UgridDataArrayAccessor.set_node_coords` and
  :meth:`xugrid.UgridDatasetAccessor.set_node_coords` have been added to set
  other coordinates (e.g. latitude-longitude instead of projected coordinates)
  as the active coordinates of the Ugrid topology.

[0.1.3] 2021-12-23
------------------

.. _Keep a Changelog: https://keepachangelog.com/en/1.0.0/
.. _Semantic Versioning: https://semver.org/spec/v2.0.0.html