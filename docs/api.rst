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

UGRID Accessor
--------------

.. autosummary::
    :toctree: api/
    
    UgridAccessor
    UgridAccessor.crs
    UgridAccessor.isel
    UgridAccessor.sel
    UgridAccessor.sel_points
    UgridAccessor.rasterize
    UgridAccessor.rasterize_like
    UgridAccessor.to_geodataframe
    UgridAccessor.binary_dilation
    UgridAccessor.binary_erosion
    UgridAccessor.connected_components
    UgridAccessor.reverse_cuthill_mckee
    UgridAccessor.laplace_interpolate
    UgridAccessor.to_dataset
    UgridAccessor.to_netcdf
    UgridAccessor.to_zarr

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
    UgridDataArray.ugrid
    UgridDataset.from_geodataframe

UGRID1D Topology
----------------

.. autosummary::
    :toctree: api/
    
    Ugrid1d

    Ugrid1d.topology_dimension

    Ugrid1d.n_node
    Ugrid1d.node_dimension
    Ugrid1d.node_coordinates

    Ugrid1d.n_edge
    Ugrid1d.edge_dimension
    Ugrid1d.edge_coordinates
    Ugrid1d.edge_x
    Ugrid1d.edge_y

    Ugrid1d.bounds
    
    Ugrid1d.node_edge_connectivity

    Ugrid1d.copy

    Ugrid1d.remove_topology
    Ugrid1d.topology_coords
    Ugrid1d.topology_dataset

    Ugrid1d.sel
    Ugrid1d.topology_subset

    Ugrid1d.mesh
    Ugrid1d.meshkernel
    
    Ugrid1d.set_crs
    Ugrid1d.to_crs

    Ugrid1d.from_dataset
    Ugrid1d.from_geodataframe
    Ugrid1d.to_pygeos

UGRID2D Topology
----------------

.. autosummary::
    :toctree: api/

    Ugrid2d

    Ugrid2d.topology_dimension

    Ugrid2d.n_node
    Ugrid2d.node_dimension
    Ugrid2d.node_coordinates

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

    Ugrid2d.remove_topology
    Ugrid2d.topology_coords
    Ugrid2d.topology_dataset

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
    Ugrid2d.from_geodataframe
    Ugrid2d.from_structured
    Ugrid2d.to_pygeos
