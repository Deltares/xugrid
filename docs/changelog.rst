Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog`_, and this project adheres to
`Semantic Versioning`_.

[Unreleased]
------------

Changed
-------

- :meth:`xugrid.Ugrid1d.sel`, :meth:`xugrid.Ugrid1d.sel_points`,
  :meth:`xugrid.Ugrid1d.intersect_lines`, and
  :meth:`xugrid.Ugrid1d.intersect_linestring` now actually conduct spatial
  indexing instead of just returning the object.

Added
~~~~~

- Added :meth:`xugrid.Ugrid1d.format_connectivity_as_dense`,
  :meth:`xugrid.Ugrid1d.format_connectivity_as_sparse`,
  :meth:`xugrid.Ugrid2d.format_connectivity_as_dense`, and
  :meth:`xugrid.Ugrid2d.format_connectivity_as_sparse` utilities to convert
  between sparse and dense rectangular form of the connectivity arrays.
- Added :attr:`xugrid.Ugrid1d.edge_edge_connectivity` and
  :attr:`xugrid.Ugrid2d.edge_edge_connectivity`.
- Added :meth:`xugrid.Ugrid1d.refine_by_vertices` to refine a network with
  inserted vertices.
- Added :class:`xugrid.NetworkGridder` to grid networks (Ugrid1d) to a 2D grid.
  Currently only support gridding to a Ugrid2d grid.

Fixed
-----

- :meth:`xugrid.Ugrid2d.earcut_triangulate_polygons` and
  :func:`xugrid.earcut_triangulate_polygons` will now return grid objects
  with a signed integer type instead of unsigned integer, as the unsigned
  integer type does not support the default negative fill value of -1 and
  would result in an error when writing to a NetCDF file.

[0.12.4] 2025-03-05
-------------------

Changed
-------

- :func:`xugrid.open_dataarray`, :func:`xugrid.load_dataarray`,
  :func:`xugrid.open_dataset`, :func:`xugrid.load_dataset` now error when no
  UGRID conventions data is present in the file or object.

Added
~~~~~

- Added :attr:`xugrid.Ugrid1d.is_cyclic` property to check if grid topology
  contains cycles.

[0.12.3] 2025-02-17
-------------------

Changed
~~~~~~~

- :meth:`xugrid.UgridDataset.from_structured` and
  :meth:`xugrid.UgridDataArray.from_structured` are deprecated and will be
  removed in the future; calling them will raise a FutureWarning. They have
  been replaced by :meth:`xugrid.UgridDataset.from_structured2d` and
  :meth:`xugrid.UgridDataArray.from_structured2d` respectively.

Added
~~~~~

- :meth:`xugrid.Ugrid2d.from_structured_bounds` now accepts 3D bounds to allow
  conversion of grids with non-monotonic x and y coordinates, such as strongly
  curvilinear grids.
- :meth:`xugrid.Ugrid2d.from_structured_bounds` now takes an optional
  ``return_index`` argument to return the indices of invalid grid faces,
  identified by one or more NaNs in its bounds.
- This method is used in :meth:`xugrid.UgridDataArray.from_structured2d` and
  :meth:`xugrid.UgridDataset.from_structured2d` when the optional arguments
  ``x_bounds`` and ``y_bounds`` are provided.
- Added :attr:`xugrid.Ugrid1d.directed_edge_edge_connectivity` and
  :attr:`xugrid.Ugrid2d.directed_edge_edge_connectivity`.
- Added :func:`xugrid.load_dataset` and :func:`xugrid.load_dataarray`.

[0.12.2] 2025-01-31
-------------------

Changed
~~~~~~~

- :meth:`xugrid.UgridDataArrayAccessor.from_structured` previously required the
  literal dimensions ``("y", "x")``. This requirement has been relaxed, it will
  now infer the dimensions from the provided coordinates.
- :meth:`xugrid.Ugrid2d.from_structured` previously only supported 1D
  coordinates; it now detects whether coordinates are 1D or 2D automatically.
  Accordingly, :meth:`xugrid.Ugrid2d.from_structured_multicoord` should no
  longer be used, and calling it will give a FutureWarning.
- The first argument of the ``.regrid`` methods of
  :class:`xugrid.CentroidLocatorRegridder`, :class:`xugrid.OverlapRegridder`,
  :class:`xugrid.RelativeOverlapRegridder`, and
  :class:`xugrid.BarycentricInterpolator` has been renamed. The method now
  takes a ``data`` argument instead of ``object``.

Added
~~~~~

- :meth:`xugrid.UgridDataset.from_structured` has been added to create
  UgriDatasets from xarray Datasets.

Fixed
~~~~~

- The ``.regrid`` methods of :class:`xugrid.CentroidLocatorRegridder`,
  :class:`xugrid.OverlapRegridder`, :class:`xugrid.RelativeOverlapRegridder`,
  and :class:`xugrid.BarycentricInterpolator` now raise a TypeError if an
  inappropriate type is provided.
- Fixed file handling in :meth:`xugrid.UgridDataArray.close` and
  :meth:`xugrid.UgridDataset.close`. Previously, files opened with
  :func:`xugrid.open_dataarray` or :func:`xugrid.open_dataset` could not be
  properly closed, and new UgridDataset or UgridDataArray objects were not
  correctly associated with their source files. Now, calling the close methods
  will properly close the associated files.

[0.12.1] 2024-09-09
-------------------

Fixed
~~~~~

- Release 0.12.0 changed the return type of the face node connectivity of
  :attr:`xugrid.Ugrid2d.voronoi_topology` from a `scipy.sparse.coo_matrix` to
  an ordinary `np.array` of integers (and similarly for internal voronoi
  tesselations); this dense array had fill (hard-coded) values of -1,
  potentially differing from the grid's fill value. This lead to a number of
  errors for methods relying on voronoi tesselations (such as contour plots)
  if the fill value of the grid was not -1. Internally, a ``FILL_VALUE = -1``
  is now used everywhere in connectivity arrays, and fill values are no longer
  passed for internal methods; a value of -1 is always assumed. When converting
  the grid (back) to a dataset with :meth:`xugrid.Ugrid1d.to_dataset` or
  :meth:`xugrid.Ugrid2d.to_dataset`, the fill value is set back to its original
  value; the fill value is also set when calling
  :meth:`xugrid.UgridDataArrayAccessor.to_netcdf` or
  :meth:`xugrid.UgridDatasetAccessor.to_netcdf`.
 
Added
~~~~~

- :class:`xugrid.Ugrid1d` and :class:`xugrid.Ugrid2d` now take an optional
  ``start_index`` which controls the start index for the UGRID connectivity
  arrays.
- :attr:`xugrid.Ugrid1d.fill_value`, :attr:`xugrid.Ugrid1d.start_index`,
  :attr:`xugrid.Ugrid2d.fill_value`, and :attr:`xugrid.Ugrid2d.start_index`,
  have been added to get and set the fill value and start index for the UGRID
  connectivity arrays. (Internally, every array is 0-based, and has a fill
  value of -1.)
  
Changed
~~~~~~~

- :class:`xugrid.Ugrid1d` and :class:`xugrid.Ugrid2d` will generally preserve
  the fill value and start index of grids when roundtripping from and to xarray
  Dataset. An exception is when the start index or fill value varies per
  connectivity: ``xugrid`` will enforce a single start index and a single fill
  value per grid. In case of inconsistent values across connectivity arrays,
  the values associated with the core connectivity are used: for Ugrid2d, this
  is the face node connectivity.

[0.12.0] 2024-09-03
-------------------

Fixed
~~~~~

- The :class:`xugrid.BarycentricInterpolator` now interpolates according to
  linear weights within the full bounds of the source grid, rather than only
  within the centroids of the source grid. Previously, it would give no results
  beyond the centroids for structured to structured regridding, and it would
  give nearest results (equal to :class:`xugrid.CentroidLocatorRegridder`) otherwise.

Added
~~~~~

- :meth:`xugrid.UgridDataArrayAccessor.interpolate_na` has been added to fill missing
  data. Currently, the only supported method is ``"nearest"``.
- :attr:`xugrid.Ugrid1.dims` and :attr:`xugrid.Ugrid2.dims` have been added to
  return a set of the UGRID dimensions.
- :meth:`xugrid.UgridDataArrayAccessor.laplace_interpolate` now uses broadcasts
  over non-UGRID dimensions and support lazy evaluation.

Changed
~~~~~~~

- Selection operations such as :meth:`UgridDataArrayAccessor.sel_points` will
  now also return points that are located on the edges of 2D topologies.
- :attr:`xugrid.Ugrid1d.dimensions` and :attr:`xugrid.Ugrid2d.dimensions` now
  give a FutureWarning; use ``.dims`` or ``.sizes`` instead.
- Improved performance of :func:`xugrid.open_dataset` and
  :func:`xugrid.merge_partitions` when handling datasets with a large number
  of variables (>100).

[0.11.2] 2024-08-16
-------------------

Fixed
~~~~~

- The regridders will no longer flip around data along an axis when regridding
  from data from structured to unstructured form when the coordinates along the
  dimensions is decreasing. (Decreasing y-axis is a common occurence in
  geospatial rasters.)
- The regridders will no longer error on ``.regrid()`` if a structured target
  grid is non-equidistant, and contains an array delta (``d``) coordinate
  rather than a single delta to denote cell sizes along a dimension (i.e.
  ``dy`` along ``y`` midpoints, and ``dx`` along ``x``.)

Added
~~~~~

- :func:`xugrid.snap_nodes` to snap neighboring vertices together that are
  located within a maximum snapping distance from each other. If vertices are
  located within a maximum distance, some of them are snapped to their
  neighbors ("targets"), thereby guaranteeing a minimum distance between nodes
  in the result. The determination of whether a point becomes a target itself
  or gets snapped to another point is primarily based on the order in which
  points are processed and their spatial relationships.

[0.11.1] 2024-08-13
-------------------

Fixed
~~~~~

- The reduction methods for the overlap regridders now behave consistently when
  all values are NaN or when all weights (overlaps) are zero, and all methods
  give the same answer irrespective of the order in which the values are
  encountered.
- :meth:`xugrid.merge_partitions` will now raise a ValueError if zero
  partitions are provided.
- :meth:`xugrid.merge_partitions` will no longer error when chunks are
  inconsistent across variables in a dataset, but now returns a merged dataset
  while keeping the chunking per variable. (Note that if chunks are inconstent
  for a variable **across partitions** that they are still and always unified
  for the variable.)

Added
~~~~~

- Percentiles (5, 10, 25, 50, 75, 90, 95) have been added to the
  :class:`xugrid.OverlapRegridder` as standard available reduction methods
  (available as ``"p5", "p10"``, etc.). Custom percentile values (e.g. 2.5, 42) can be
  setup using :meth:`xugrid.OverlapRegridder.create_percentile_method`.

Changed
~~~~~~~

- Custom reduction functions provide to the overlap regridders no longer require
  an ``indices`` argument.
- :meth:`xugrid.Ugrid2d.sel_points`,
  :meth:`xugrid.UgridDataArrayAccessor.sel_points` and
  :meth:`xugrid.UgridDatasetAccessor.sel_points` now take an ``out_of_bounds``
  and ``fill_value`` argument to determine what to with points that do not fall
  inside of any grid feature. Previously, the method silently dropped these
  points. The method now takes a ``fill_value`` argument to assign to
  out-of-bounds points. It gives a warning return uses ``fill_value=np.nan`` by
  default. To enable the old behavior, set ``out_of_bounds="drop"``.

[0.11.0] 2024-08-05
-------------------

Fixed
~~~~~

- :func:`xugrid.merge_partitions` now automatically merges chunks (if defined
  in the partition datasets). This removes the commonly seen
  ``PerformanceWarning: Slicing with an out-of-order index is generating ...
  times more chunks`` warning in subsequent operations, and also greatly
  improves the performance of subsequent operations (roughly scaling linearly
  with the number of partitions). The previous behavior can be maintained by
  setting ``merge_ugrid_chunks=False``. This keyword will likely be deprecated
  in the future as merging the UGRID dimension chunks should be superior for
  (almost all?) subsquent operations.
- :func:`xugrid.snap_to_grid` now returns proper line indexes when multiple
  linestrings are snapped. Snapping previously could result in correct
  linestring locations, but wrong line indexes.

Added
~~~~~

- Included ``edge_node_connectivity`` in :meth:`xugrid.Ugrid2d.from_meshkernel`,
  so the ordering of edges is consistent with ``meshkernel``.
- Added :meth:`xugrid.Ugrid1d.create_data_array`,
  :meth:`xugrid.Ugrid2d.create_data_array`, and
  :meth:`xugrid.UgridDataArray.from_data` to more easily instantiate a
  UgridDataArray from a grid topology and an array of values.
- Added :func:`xugrid.create_snap_to_grid_dataframe` to provide
  more versatile snapping, e.g. with custom reductions to assign_edge_coords
  aggregated properties to grid edges.

Changed
~~~~~~~

- :meth:`xugrid.UgridDataArrayAccessor.laplace_interpolate` now uses ``rtol``
  and ``atol`` keywords instead of ``tol``, to match changes in
  ``scipy.linalg.sparse.cg``.

[0.10.0] 2024-05-01
-------------------

Fixed
~~~~~

- Fixed indexing bug in the ``"mode"`` method in
  :class:`xugrid.CentroidLocatorRegridder`, :class:`xugrid.OverlapRegridder`,
  :class:`xugrid.RelativeOverlapRegridder`, which gave the method the tendency
  to repeat the first value in the source grid across the target grid.

Added
~~~~~

- :func:`xugrid.earcut_triangulate_polygons` and
  :meth:`xugrid.Ugrid2d.earcut_triangulate_polygons` have been added to break
  down polygon geodataframes into a triangular mesh for further processing.
- :meth:`xugrid.OverlapRegridder.weights_as_dataframe` has been added to
  extract regridding weights (overlaps) from the regridders. This method is
  also available for :class:`BarycentricInterpolator`,
  :class:`CentroidLocatorRegridder`, and :class:`RelativeOverlapRegridder`.

[0.9.0] 2024-02-15
------------------

Fixed
~~~~~

- :meth:`xugrid.Ugrid2d.equals` and :meth:`xugrid.Ugrid1d.equals` test if
  dataset is equal instead of testing type.
- Fixed bug in :func:`xugrid.concat` and :func:`xugrid.merge` where multiple
  grids were returned if grids did not point to the same object id (i.e.
  copies).
- Fixed bug in :meth:`xugrid.Ugrid1d.merge_partitions`, which caused
  ``ValueError: indexes must be provided for attrs``.
- Fixed ``from_structured`` methods: the generated faces are now always in
  counterclockwise direction, also for increasing y-coordinates or decreasing
  x-coordinates.

Added
~~~~~

- :meth:`xugrid.Ugrid2d.from_structured_multicoord` has been added
  to generate UGRID topologies from rotated or approximated curvilinear grids.
- :meth:`xugrid.Ugrid2d.from_structured_intervals1d` has been added to generate
  UGRID topologies from "intervals": the N + 1 vertex coordinates for N faces.
- :meth:`xugrid.Ugrid2d.from_structured_intervals2d` has been added to generate
  UGRID topologies from "intervals": the (M + 1, N + 1) vertex coordinates for N faces.
- :meth:`xugrid.UgridDataArrayAccessor.from_structured` now takes ``x`` and ``y``
  arguments to specify which coordinates to use as the UGRID x and y coordinates.
- :attr:`xugrid.UgridDataset.sizes` as an alternative to :attr:`xugrid.UgridDataset.dimensions`
- :attr:`xugrid.Ugrid2d.max_face_node_dimension` which returns the dimension
  name designating nodes per face.
- :attr:`xugrid.AbstractUgrid.max_connectivity_sizes` which returns all
  maximum connectivity dimensions and their corresponding size.
- :attr:`xugrid.AbstractUgrid.max_connectivity_dimensions` which returns all
  maximum connectivity dimensions.

Changed
~~~~~~~

- :meth:`xugrid.Ugrid2d.from_structured` now takes ``x`` and ``y`` arguments instead
  of ``x_bounds`` and ``y_bounds`` arguments.
- :func:`xugrid.merge_partitions` now also merges datasets with grids that are
  only contained in some of the partition datasets.

[0.8.1] 2024-01-19
------------------

Fixed
~~~~~

- :meth:`xugrid.UgridDataArrayAccessor.reindex_like` will now take the tolerance
  argument into account before sorting. In the past, near ties could be resolved
  differently between otherwise similar grid topologies due to roundoff.

Added
~~~~~

- :meth:`xugrid.UgridDataArrayAccessor.laplace_interpolate` now also supports
  interpolation of node associated data, and Ugrid1d topologies.
- :meth:`xugrid.Ugrid1d.from_shapely` and :meth:`xugrid.Ugrid2d.from_shapely` have
  been added to directly instantiate UGRID topologies from arrays of shapely geometries.

Changed
~~~~~~~

- :meth:`xugrid.UgridDataArrayAccessor.laplace_interpolate` no longer uses scipy's
  ILU decomposition as a preconditioner. A simpler and more effective preconditioner
  is automatically used instead. The arguments have changed accordingly.
  ``direct_solve`` is now by default ``False``.
- :meth:`xugrid.Ugrid1d.from_geodataframe` and :meth:`xugrid.Ugrid2d.from_geodataframe`
  now check whether the geodataframe argument is a geopandas GeoDataFrame, and whether
  the geometry types are appropriate (LineStrings for Ugrid1d, Polygons for Ugrid2d).

[0.8.0] 2023-12-11
------------------

Changed
~~~~~~~

- Initialize Meshkernel with a spherical projection if the coordinate reference
  system (crs) is geographic.
- Minimum Python version increased to 3.9.

[0.7.1] 2023-11-17
------------------

Fixed
~~~~~
- Support for Meshkernel 3 (#171). Initialize Meshkernel
  with defaults, setting it to cartesian projection.

[0.7.0] 2023-10-19
------------------

Added
~~~~~

- :meth:`xugrid.Ugrid2d.to_nonperiodic`,
  :meth:`xugrid.UgridDataArrayAccessor.to_nonperiodic` and
  :meth:`xugrid.UgridDatasetAccessor.to_nonperiodic` have been added to convert
  a "periodid grid" (where the leftmost nodes are the same as the rightmost
  nodes, e.g. a mesh for the globe) to an "ordinary" grid.
- Conversely, :meth:`xugrid.Ugrid2d.to_periodic`,
  :meth:`xugrid.UgridDataArrayAccessor.to_periodic` and
  :meth:`xugrid.UgridDatasetAccessor.to_periodic` have been added to convert an
  ordinary grid to a periodic grid.
- :attr:`xugrid.Ugrid2d.perimeter` has been added the compute the length of the
  face perimeters.
- :meth:`xugrid.Ugrid1d.reindex_like`,
  :meth:`xugrid.Ugrid2d.reindex_like`,
  :meth:`xugrid.UgridDataArrayAccessor.reindex_like` and
  :meth:`xugrid.UgridDatasetAccessor.reindex_like` have been added to deal with
  equivalent but differently ordered topologies and data.

Changed
~~~~~~~

- UGRID 2D topologies are no longer automatically forced in counterclockwise
  orientation during initialization.

Fixed
~~~~~

- Using an index which only reorders but does not change the size in
  :meth:`xugrid.Ugrid1d.topology_subset` or
  :meth:`xugrid.Ugrid2d.topology_subset` would erroneously result in the
  original grid being returned, rather than a new grid with the faces or edges
  shuffled. This breaks the link the between topology and data when using
  ``.isel`` on a UgridDataset or UgridDataArray. This has been fixed: both data
  and the topology are now shuffled accordingly.

[0.6.5] 2023-09-30
------------------

Added
~~~~~

- :meth:`xugrid.Ugrid2d.intersect_line`,
  :meth:`xugrid.Ugrid2d.intersect_linestring`
  :meth:`xugrid.UgridDataArrayAccessor.intersect_line`,
  :meth:`xugrid.UgridDataArrayAccessor.intersect_linestring`,
  :meth:`xugrid.UgridDatasetAccessor.intersect_line`, and
  :meth:`xugrid.UgridDatasetAccessor.intersect_linestring` have been added to
  intersect line and linestrings and extract the associated face data.

Changed
~~~~~~~

- Selection operations along a line, or at point locations, will now prefix the
  name of the grid in the x and y coordinates. This avoids name collisions when
  multiple topologies are present in a dataset.
- Xugrid now contains a partial copy of the xarray plot utils module, and its
  tests. The latest xarray release broke xugrid (on import), since (private)
  parts of xarray were used which no longer existed.

Fixed
~~~~~

- :meth:`xugrid.UgridDatasetAccessor.sel` would return only a single grid
  topology even when the selection subject contains more than one grid. It now
  correctly returns subsets of all topologies.

[0.6.4] 2023-08-22
------------------

Fixed
~~~~~

- Bug in :func:`xugrid.snap_to_grid`, which caused an ``IndexError``.
  See `#122 <https://github.com/Deltares/xugrid/issues/122>`_.


[0.6.3] 2023-08-12
------------------

Added
~~~~~

- Added :func:`xugrid.burn_vector_geometries` to burn vector geometries in the
  form of geopandas GeoDataFrames into a Ugrid2d topology.
- Added :func:`xugrid.polygonize` to create vector polygons for all connected
  regions of a Ugrid2d topology sharing a common value. The result is a
  geopandas GeoDataFrame.
- :meth:`xugrid.Ugrid2d.validate_edge_node_connectivity` has been added to
  validate edge_node_connectivity by comparing with the face_node_connectivity.
  The result can be used to define a valid subselection.
- :meth:`xugrid.Ugrid2d.from_structured_bounds` can be used to generate
  a Ugrid2d topology from x and y bounds arrays.
- :attr:`xugrid.UgridDatasetAccessor.name`,
  :attr:`xugrid.UgridDatasetAccessor.names`,
  :attr:`xugrid.UgridDatasetAccessor.topology`; and
  :attr:`xugrid.UgridDataArrayAccessor.name`,
  :attr:`xugrid.UgridDataArrayAccessor.names`,
  :attr:`xugrid.UgridDataArrayAccessor.topology` have been added to provide
  easier access to the names of the UGRID topologies.
- :meth:`xugrid.UgridDatasetAccessor.rename` and
  :meth:`xugrid.UgridDataArrayAccessor.rename` to rename both topology and the
  associated dimensions.
- :meth:`xugrid.Ugrid2d.bounding_polygon` has been added to get a polygon
  describing the bounds of the grid.

Fixed
~~~~~

- :class:`xugrid.CentroidLocatorRegridder`, :class:`xugrid.OverlapRegridder`,
  and :class:`xugrid.BarycentricInterpolator` will now also regrid structured
  to unstructured grid topologies.
- :meth:`xugrid.Ugrid1d.to_dataset` and :meth:`xugrid.Ugrid2d.to_dataset` no
  longer write unused connectivity variables into the attributes of the UGRID
  dummy variable.
- Conversion from and to GeoDataFrame will now conserve the CRS (coordinate
  reference system).
- :meth:`xugrid.UgridDatasetAccessor.to_geodataframe` will no longer error when
  converting a UgridDataset that does not contain any variables.
- :meth:`xugrid.OverlapRegridder.regrid` will no longer give incorrect results
  on repeated calls with the "mode" method.

Changed
~~~~~~~

- Initializing a Ugrid2d topology with an invalid edge_node_connectivity will
  no longer raise an error.
- :attr:`xugrid.Ugrid1d.node_node_connectivity`,
  :attr:`xugrid.Ugrid1d.directed_node_node_connectivity`,
  :attr:`xugrid.Ugrid2d.node_node_connectivity`,
  :attr:`xugrid.Ugrid2d.directed_node_node_connectivity`, and
  :attr:`xugrid.Ugrid2d.face_face_connectivity` now contain the associated edge
  index in the ``.data`` attribute of the resulting CSR matrix.

[0.6.2] 2023-07-26
------------------

Fixed
~~~~~

- Computing indexer to avoid dask array of unknown shape upon plotting.
  See `#117 <https://github.com/Deltares/xugrid/issues/117>`_.
- Bug where chunked dask arrays could not be regridded.
  See `#119 <https://github.com/Deltares/xugrid/issues/99>`_.
- Bug where error was thrown in the RelativeOverlapRegridder upon
  flipping the y coordinate.


[0.6.1] 2023-07-07
------------------

Fixed
~~~~~

- Fillvalue was not properly replaced in cast.
  See `#113 <https://github.com/Deltares/xugrid/issues/113>`_.


[0.6.0] 2023-07-05
------------------

Added
~~~~~

- :meth:`xugrid.Ugrid2d.label_partitions`, :meth:`xugrid.Ugrid2d.partition`,
  :meth:`xugrid.Ugrid2d.merge_partitions` have been added to partition and merge
  a grid.
- :meth:`xugrid.UgridDataArrayAccessor.partition`,
  :meth:`xugrid.UgridDataArrayAccessor.partition_by_label`,
  :meth:`xugrid.UgridDatasetAccessor.partition`, and
  :meth:`xugrid.UgridDatasetAccessor.partition_by_label` have been added to
  part a grid and its associated data.
- :meth:`xugrid.Ugrid1d.rename` and :meth:`xugrid.Ugrid2d.rename` have been
  added to rename a grid, including the attributes that are created when the
  grid is converted into an xarray dataset.
- :meth:`xugrid.Ugrid1d.node_node_connectivity` and
  :meth:`xugrid.Ugrid2.node_node_connectivity` properties have been added.
- :meth:`xugrid.Ugrid1d.topological_sort_by_dfs` has been added.
- :meth:`xugrid.Ugrid1d.contract_vertices` has been added.

Fixed
~~~~~

- Regridding is possible again with regridders initiated ``from_weights``.
  See `#90 <https://github.com/Deltares/xugrid/issues/90>`_.
  This was a broken feature in the 0.5.0 release.
- Computed weights for structured grids regridders now decrease with distance
  instead of increase.
- Fixed edge case for regridding structured grids, where midpoints of the
  source and target grid are equal.
- Fixed numba typing error for regridders.

Changed
~~~~~~~

- Regridding structured grids now throws error if computed weights < 0.0 or >
  1.0, before these weights were clipped to 0.0 and 1.0 respectively.


[0.5.0] 2023-05-25
------------------

Added
~~~~~

- :class:`xugrid.BarycentricInterpolator`,
  :class:`xugrid.CentroidLocatorRegridder`, :class:`xugrid.OverlapRegridder`,
  and :class:`RelativeOverlapRegridder`, now accept structured grids, in the
  form of a ``xr.DataArray`` with a ``"x"`` and a ``"y"`` coordinate.

[0.4.0] 2023-05-05
------------------

Fixed
~~~~~

- :meth:`xugrid.Ugrid2d.tesselate_centroidal_voronoi` and
  :meth:`xugrid.Ugrid2d.tesselate_circumcenter_voronoi` will only include
  relevant centroids, rather than all the original centroids when
  ``add_exterior=False``. Previously, a scrambled voronoi grid could result
  from the tesselation when the original grid contained cells with only one
  neighbor.
- ``import xugrid`` now does not throw ``ImportError`` anymore when the
  optional package ``geopandas`` was missing in the environment.

Changed
~~~~~~~

- :meth:`xugrid.Ugrid2d.sel_points` and
  :meth:`xugrid.UgridDataArrayAccessor.sel_points` now return a result with an
  "index" coordinate, containing the (integer) index of the points.
- :class:`xugrid.Ugrid2d` will now error during initialization if the
  node_edge_connectivity is invalid (i.e. contains nodes that are not used in
  any face).
- :meth:`xugrid.UgridDataArrayAccessor.plot.pcolormesh` now defaults to
  ``edgecolors="face"`` to avoid white lines (which can be become relatively
  dominant in when plotting large grids).

Added
~~~~~

- :meth:`xugrid.Ugrid2d.tesselate_circumcenter_voronoi` has been added to
  provide orthogonal voronoi cells for triangular grids.
- :meth:`xugrid.Ugrid1d.to_dataset`, :meth:`xugrid.Ugrid2d.to_dataset`,
  :meth:`xugrid.UgridDataArrayAccessor.to_dataset`, and
  :meth:`xugrid.UgridDatasetAccessor.to_dataset` now take an
  ``optional_attributes`` keyword argument to generate the optional UGRID
  attributes.
- :class:`xugrid.Ugrid1d` and :class:`xugrid.Ugrid2d` now have an ``attrs``
  property.
- :meth:`xugrid.UgridDatasetAccessor.rasterize` and
  :meth:`xugrid.UgridDatasetAccessor.rasterize_like` have been added to
  rasterize all face variables in a UgridDataset.

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
