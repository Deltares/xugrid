Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog`_, and this project adheres to
`Semantic Versioning`_.

[Unreleased]
------------

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