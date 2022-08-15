.. currentmodule:: xugrid

.. _api:

API Reference
=============

This page provides an auto-generated summary of xugrid's API.

Top-level functions
-------------------

.. autosummary::
   :toctree: api/

    open_dataarray
    open_dataset
    open_mfdataset
    open_zarr
    full_like
    ones_like
    zeros_like
    concat
    merge

UgridDataArray
--------------

.. autosummary::
   :toctree: api/

    UgridDataArray
    UgridDataArray.ugrid
    UgridDataArray.from_structured

UgridDataset
------------

.. autosummary::
   :toctree: api/

    UgridDataset
    UgridDataset.ugrid
    UgridDataset.from_geodataframe

UGRID Accessor
--------------

These methods and attributes are available under the ``.ugrid`` attribute of a
UgridDataArray or UgridDataset.

.. autosummary::
   :toctree: api/

    UgridDatasetAccessor
    UgridDatasetAccessor.assign_node_coords
    UgridDatasetAccessor.set_node_coords
    UgridDatasetAccessor.crs
    UgridDatasetAccessor.isel
    UgridDatasetAccessor.sel
    UgridDatasetAccessor.sel_points
    UgridDatasetAccessor.to_geodataframe
    UgridDatasetAccessor.to_dataset
    UgridDatasetAccessor.to_netcdf
    UgridDatasetAccessor.to_zarr

    UgridDataArrayAccessor
    UgridDataArrayAccessor.assign_node_coords
    UgridDataArrayAccessor.set_node_coords
    UgridDataArrayAccessor.crs
    UgridDataArrayAccessor.isel
    UgridDataArrayAccessor.sel
    UgridDataArrayAccessor.sel_points
    UgridDataArrayAccessor.rasterize
    UgridDataArrayAccessor.rasterize_like
    UgridDataArrayAccessor.to_geodataframe
    UgridDataArrayAccessor.binary_dilation
    UgridDataArrayAccessor.binary_erosion
    UgridDataArrayAccessor.connected_components
    UgridDataArrayAccessor.reverse_cuthill_mckee
    UgridDataArrayAccessor.laplace_interpolate
    UgridDataArrayAccessor.to_dataset
    UgridDataArrayAccessor.to_netcdf
    UgridDataArrayAccessor.to_zarr

Plotting
--------

These methods are also available under the ``.ugrid.plot`` attribute of a
UgridDataArray.

.. autosummary::
   :toctree: api/

   plot.contour
   plot.contourf
   plot.imshow
   plot.line
   plot.pcolormesh
   plot.scatter
   plot.surface
   plot.tripcolor

UGRID1D Topology
----------------

.. autosummary::
   :toctree: api/

    Ugrid1d

    Ugrid1d.topology_dimension
    Ugrid1d.dimensions

    Ugrid1d.n_node
    Ugrid1d.node_dimension
    Ugrid1d.node_coordinates
    Ugrid1d.set_node_coords
    Ugrid2d.assign_node_coords

    Ugrid1d.n_edge
    Ugrid1d.edge_dimension
    Ugrid1d.edge_coordinates
    Ugrid1d.edge_x
    Ugrid1d.edge_y

    Ugrid1d.bounds

    Ugrid1d.node_edge_connectivity

    Ugrid1d.copy

    Ugrid1d.isel
    Ugrid1d.sel
    Ugrid1d.topology_subset

    Ugrid1d.mesh
    Ugrid1d.meshkernel

    Ugrid1d.set_crs
    Ugrid1d.to_crs

    Ugrid1d.from_dataset
    Ugrid1d.to_dataset
    Ugrid1d.from_geodataframe
    Ugrid1d.to_pygeos

UGRID2D Topology
----------------

.. autosummary::
   :toctree: api/

    Ugrid2d

    Ugrid2d.topology_dimension
    Ugrid2d.dimensions

    Ugrid2d.n_node
    Ugrid2d.node_dimension
    Ugrid2d.node_coordinates
    Ugrid2d.set_node_coords
    Ugrid2d.assign_node_coords

    Ugrid2d.n_edge
    Ugrid2d.edge_dimension
    Ugrid2d.edge_coordinates
    Ugrid2d.edge_x
    Ugrid2d.edge_y

    Ugrid2d.n_face
    Ugrid2d.face_dimension
    Ugrid2d.face_coordinates
    Ugrid2d.centroids
    Ugrid2d.face_x
    Ugrid2d.face_y

    Ugrid2d.bounds
    Ugrid2d.centroids

    Ugrid2d.node_edge_connectivity
    Ugrid2d.node_face_connectivity
    Ugrid2d.edge_node_connectivity
    Ugrid2d.face_edge_connectivity
    Ugrid2d.face_face_connectivity

    Ugrid2d.exterior_edges
    Ugrid2d.exterior_faces

    Ugrid2d.copy

    Ugrid2d.triangulate
    Ugrid2d.triangulation
    Ugrid2d.voronoi_topology
    Ugrid2d.centroid_triangulation
    Ugrid2d.tesselate_centroidal_voronoi
    Ugrid2d.reverse_cuthill_mckee

    Ugrid2d.isel
    Ugrid2d.sel
    Ugrid2d.sel_points
    Ugrid2d.celltree
    Ugrid2d.locate_points
    Ugrid2d.intersect_edges
    Ugrid2d.locate_bounding_box
    Ugrid2d.rasterize
    Ugrid2d.rasterize_like
    Ugrid2d.topology_subset

    Ugrid2d.mesh
    Ugrid2d.meshkernel

    Ugrid2d.set_crs
    Ugrid2d.to_crs

    Ugrid2d.from_dataset
    Ugrid2d.to_dataset
    Ugrid2d.from_geodataframe
    Ugrid2d.from_structured
    Ugrid2d.to_pygeos

UGRID Roles Accessor
--------------------

.. autosummary::
   :toctree: api/

    UgridRolesAccessor
    UgridRolesAccessor.topology
    UgridRolesAccessor.connectivity
    UgridRolesAccessor.coordinates
    UgridRolesAccessor.dimensions
