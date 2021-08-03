from typing import Union

import matplotlib.tri as tri
import numpy as np
import xarray as xr
import pandas as pd

IntArray = np.ndarray
FloatArray = np.ndarray


def create_trimesh_steady_state(nx, ny):
   '''
   creates a xarray dataset containing an irregular grid geometry (triangles) with a single dataarray containing time-invariant data
   associated to faces
   '''
   x = np.linspace(0.0, 100.0, nx)
   y = np.linspace(0.0, 100.0, ny)
   return _assign_data(triangular_dataset(x, y))
    
def create_trimesh_transient(nx, ny, ntime):
   '''
   creates a xarray dataset containing an irregular grid geometry (triangles) with a single dataarray containing transient data
   associated to faces
   '''
   x = np.linspace(0.0, 100.0, nx)
   y = np.linspace(0.0, 100.0, ny)
   return _assign_time_data(triangular_dataset(x, y), ntime)
    
def create_quadmesh_steady_state(nx, ny):
   '''
   creates a xarray dataset containing an irregular grid geometry (quadrilaterals ) with a single dataarray containing  time-invariant data
   associated to faces
   '''
   x = np.linspace(0.0, 100.0, nx)
   y = np.linspace(0.0, 100.0, ny)
   return _assign_data(quadrilateral_dataset(x, y))
    
def create_quadmesh_transient(nx, ny, ntime): 
   '''
   creates a xarray dataset containing an irregular grid geometry (quadrilaterals ) with a single dataarray containing  transient data
   associated to faces
   '''  
   x = np.linspace(0.0, 100.0, nx)
   y = np.linspace(0.0, 100.0, ny)
   return _assign_time_data(quadrilateral_dataset(x, y), ntime)
    
def create_hexmesh_steady_state(m,n):
   '''
   creates a xarray dataset containing an irregular grid geometry (hexagons ) with a single dataarray containing  time-invariant data
   associated to faces
   '''  
   return _assign_data(hexagonal_dataset(m, n))

def  create_hexmesh_transient(m,n, ntime):
   '''
   creates a xarray dataset containing an irregular grid geometry (hexagons ) with a single dataarray containing  transient data
   associated to faces
   '''   
   return _assign_time_data(hexagonal_dataset(m, n), ntime)


def ugrid2d_dataset(
    node_x: FloatArray,
    node_y: FloatArray,
    face_node_connectivity: IntArray,
) -> xr.Dataset:
    """
    Create a dataset containing UGRID 2D mesh topology.

    Parameters
    ----------
    node_x: FloatArray
    node_y: FloatArray
    node_x: FloatArray
    face_y: FloatArray
    face_node_connectivity: IntArray

    Returns
    -------
    ds: xr.Dataset
    """
    ds = xr.Dataset()
    ds["mesh2d"] = xr.DataArray(
        data=0,
        attrs={
            "cf_role": "mesh_topology",
            "long_name": "Topology data of 2D mesh",
            "topology_dimension": 2,
            "node_coordinates": "node_x node_y",
            "face_node_connectivity": "face_nodes",
            "edge_node_connectivity": "edge_nodes",
        },
    )
    ds = ds.assign_coords(
        node_x=xr.DataArray(
            data=node_x,
            dims=["node"],
        )
    )
    ds = ds.assign_coords(
        node_y=xr.DataArray(
            data=node_y,
            dims=["node"],
        )
    )
    ds["face_nodes"] = xr.DataArray(
        data=face_node_connectivity,

        dims=["face", "nmax_face"],
        attrs={
            "cf_role": "face_node_connectivity",
            "long_name": "Vertex nodes of mesh faces (counterclockwise)",
            "start_index": 0,
            "_FillValue": -1,
        },
    )
    ds.attrs = {"Conventions": "CF-1.8 UGRID-1.0"}
    return ds


def ugrid2d_topology(data: Union[xr.DataArray, xr.Dataset]) -> xr.Dataset:
    """
    Derive the 2D-UGRID quadrilateral mesh topology from a structured DataArray
    or Dataset, with (2D-dimensions) "y" and "x".

    Parameters
    ----------
    data: Union[xr.DataArray, xr.Dataset]
        Structured data from which the "x" and "y" coordinate will be used to
        define the UGRID-2D topology.

    Returns
    -------
    ugrid_topology: xr.Dataset
        Dataset with the required arrays describing 2D unstructured topology:
        node_x, node_y, face_x, face_y, face_nodes (connectivity).
    """
    from imod.prepare import common

    # Transform midpoints into vertices
    # These are always returned monotonically increasing
    x = data["x"].values
    xcoord = common._coord(data, "x")
    if not data.indexes["x"].is_monotonic_increasing:
        xcoord = xcoord[::-1]
    y = data["y"].values
    ycoord = common._coord(data, "y")
    if not data.indexes["y"].is_monotonic_increasing:
        ycoord = ycoord[::-1]
    # Compute all vertices, these are the ugrid nodes
    node_y, node_x = (a.ravel() for a in np.meshgrid(ycoord, xcoord, indexing="ij"))
    face_y, face_x = (a.ravel() for a in np.meshgrid(y, x, indexing="ij"))
    linear_index = np.arange(node_x.size, dtype=np.int).reshape(
        ycoord.size, xcoord.size
    )
    # Allocate face_node_connectivity
    nfaces = (ycoord.size - 1) * (xcoord.size - 1)
    face_nodes = np.empty((nfaces, 4))
    # Set connectivity in counterclockwise manner
    face_nodes[:, 0] = linear_index[:-1, 1:].ravel()  # upper right
    face_nodes[:, 1] = linear_index[:-1, :-1].ravel()  # upper left
    face_nodes[:, 2] = linear_index[1:, :-1].ravel()  # lower left
    face_nodes[:, 3] = linear_index[1:, 1:].ravel()  # lower right
    # Tie it together
    ds = ugrid2d_dataset(node_x, node_y, face_nodes)
    return ds


def quadrilateral_dataset(x: FloatArray, y: FloatArray) -> xr.Dataset:
    """
    Create a UGRID layered dataset of quadrilaterals.

    Parameters
    ----------
    x: FloatArray
    y: FloatArray

    Returns
    -------
    ds: xr.Dataset
    """
    nrow = y.size
    ncol = x.size

    da = xr.DataArray(
        data=np.random.rand( nrow, ncol),
        coords={ "y": y, "x": x},
        dims=("y", "x"),
    )
    return ugrid2d_topology(da)


def triangular_dataset(x: FloatArray, y: FloatArray) -> xr.Dataset:
    """
    Create a UGRID layered dataset of triangles.

    Parameters
    ----------
    x: FloatArray
    y: FloatArray

    Returns
    -------
    ds: xr.Dataset
    """
    node_y, node_x = [a.ravel() for a in np.meshgrid(y, x, indexing="ij")]
    face_node_connectivity = tri.Triangulation(node_x, node_y).triangles
    return ugrid2d_dataset(node_x, node_y, face_node_connectivity)


def hexagonal_dataset(m: int, n: int) -> xr.Dataset:
    """
    Create a UGRID layered dataset of hexagons.

    Parameters
    ----------
    m: int
        Half the number of rows of hexagons
    n: int
        Half the number of columns of hexagons

    Returns
    -------
    ds: xr.Dataset
    """
    # Create a single hexagon
    x = np.array([0.0, 0.5, 1.0, 1.0, 0.5, 0.0])
    y = np.array([0.0, -0.5, 0.0, 1.0, 1.5, 1.0])
    # Tile coordinates and add offsets to create a mesh of hexagons
    xs = np.tile(x, m * n).reshape(m, n, 6) + np.arange(n)[np.newaxis, :, np.newaxis]
    ys = np.tile(y, m * n).reshape(m, n, 6) + (
        3.0 * np.arange(m)[:, np.newaxis, np.newaxis]
    )
    # Create the "inbetween" rows
    ys_inbetween = ys + 1.5
    xs_inbetween = xs + 0.5
    xs = np.vstack((xs, xs_inbetween))
    ys = np.vstack((ys, ys_inbetween))
    # Get rid of all the duplicated nodes
    nodes = np.vstack((xs.ravel(), ys.ravel())).T
    face_node_connectivity = np.arange(nodes.shape[0]).reshape(m * n * 2, 6)
    nodes, inverse_indices = np.unique(nodes, return_inverse=True, axis=0)
    face_node_connectivity = inverse_indices[face_node_connectivity]
    # Get the ugrid arrays
    node_x = nodes[:, 0]
    node_y = nodes[:, 1]

    return ugrid2d_dataset(node_x, node_y,  face_node_connectivity)


def _assign_data(ds: xr.Dataset) -> xr.Dataset:
    ds["data_time_invariant"] = (("face"), np.arange(0, ds["face"].size)*3.0)
    return ds



def _assign_time_data(ds: xr.Dataset, ntime: int) -> xr.Dataset:
    ds = _assign_data(ds)
    times = pd.date_range(start="1/1/2018", periods=ntime, freq="D")
    da_ls = []
    multiplier = 1.0
    for time in times:
        da =ds["data_time_invariant"] *multiplier
        da = da.assign_coords(time=time)
        da_ls.append(da)
        multiplier = multiplier + 1.0
    ds["data_transient"] = xr.concat(da_ls, dim="time").cumsum(dim="time")

    #%% Due to a bug in MDAL, we have to encode the times as floats
    # instead of integers
    # when this is fixed: https://github.com/lutraconsulting/MDAL/issues/348
    ds["time"].encoding["dtype"] = np.float64
    ds = ds.drop_vars("data_time_invariant")

    return ds

