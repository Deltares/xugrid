"""
Conversion from and to other data structures:

* GIS vector data (e.g. geopackage, shapefile)
* Structured data (e.g. rasters)

"""
from typing import Tuple

import geopandas as gpd
import numpy as np
import pygeos
import xarray as xr

from .connectivity import ragged_index
from .typing import IntDType
from .ugrid import Ugrid1d, Ugrid2d

FloatArray = np.ndarray
IntArray = np.ndarray
PointArray = np.ndarray
LineArray = np.ndarray
PolygonArray = np.ndarray


def contiguous_xy(xy: FloatArray) -> Tuple[FloatArray, FloatArray]:
    x, y = [np.ascontiguousarray(a) for a in xy.T]
    return x, y


def nodes_to_points(x: FloatArray, y: FloatArray) -> PointArray:
    return pygeos.points(x, y)


def edges_to_linestrings(
    x: FloatArray, y: FloatArray, edge_node_connectivity: IntArray
) -> LineArray:
    c = edge_node_connectivity.ravel()
    xy = np.column_stack((x[c], y[c]))
    i = np.repeat(np.arange(len(edge_node_connectivity)), 2)
    return pygeos.linestrings(xy, indices=i)


def faces_to_polygons(
    x: FloatArray, y: FloatArray, face_node_connectivity: IntArray, fill_value: int
) -> PolygonArray:
    is_data = face_node_connectivity != fill_value
    m_per_row = is_data.sum(axis=1)
    i = np.repeat(np.arange(len(face_node_connectivity)), m_per_row)
    c = face_node_connectivity.ravel()[is_data.ravel()]
    xy = np.column_stack((x[c], y[c]))
    rings = pygeos.linearrings(xy, indices=i)
    return pygeos.polygons(rings)


def points_to_nodes(points: PointArray) -> Tuple[FloatArray, FloatArray]:
    return contiguous_xy(pygeos.get_coordinates(points))


def linestrings_to_edges(edges: LineArray) -> Tuple[FloatArray, FloatArray, IntArray]:
    xy = pygeos.get_coordinates(edges)
    unique, inverse = np.unique(xy, axis=0, return_inverse=True)
    return *contiguous_xy(unique), inverse.reshape((-1, 2))


def _remove_last_vertex(xy: FloatArray, indices: IntArray):
    """
    GEOS polygons are always closed: the first and last vertex are the same.
    UGRID faces are by definition closed. So we'll have to remove the last,
    repeated, vertex of every polygon here.
    """
    no_repeats = np.diff(indices, append=-1) == 0
    xy = xy[no_repeats]
    indices = indices[no_repeats]
    return xy, indices


def polygons_to_faces(
    polygons: PolygonArray,
) -> Tuple[FloatArray, FloatArray, IntArray, int]:
    xy, indices = _remove_last_vertex(
        *pygeos.get_coordinates(polygons, return_index=True)
    )
    unique, inverse = np.unique(xy, axis=0, return_inverse=True)
    n = len(polygons)
    m_per_row = np.bincount(indices)
    m = m_per_row.max()
    fill_value = -1
    # Allocate 2D array and create a flat view of the dense connectivity
    conn = np.empty((n, m), dtype=IntDType)
    flat_conn = conn.ravel()
    if (n * m) == indices.size:
        # Shortcut if fill_value is not present, when all of same geom. type
        # e.g. all triangles or all quadrangles
        valid = slice(None)  # a[:] equals a[slice(None)]
    else:
        valid = ragged_index(n, m, m_per_row).ravel()
        flat_conn[~valid] = fill_value
    flat_conn[valid] = inverse
    return *contiguous_xy(unique), conn, fill_value


def geodataframe_to_ugrid1d(geodataframe: gpd.GeoDataFrame):
    ds = xr.Dataset.from_dataframe(geodataframe.drop("geometry", axis=1)).rename_dims(
        {"index": "edge"}
    )
    coords, edge_node_connectivity = linestrings_to_edges(geodataframe.geometry.values)
    grid = Ugrid1d(Ugrid1d.topology_dataset(coords, edge_node_connectivity))
    return ds, grid


def geodataframe_to_ugrid2d(geodataframe: gpd.GeoDataFrame):
    ds = xr.Dataset.from_dataframe(geodataframe.drop("geometry", axis=1)).rename_dims(
        {"index": "face"}
    )
    coords, face_node_connectivity = polygons_to_faces(geodataframe.geometry.values)
    grid = Ugrid2d(Ugrid2d.topology_dataset(coords, face_node_connectivity))
    return ds, grid


def geodataframe_to_ugrid(geodataframe: gpd.GeoDataFrame):
    """
    Convert a geodataframe into the appropriate Ugrid topology and dataset.

    Parameters
    ----------
    geodataframe: gpd.GeoDataFrame

    Returns
    -------
    grid: UgridTopology
    dataset: xr.Dataset
        Contains the data of the columns.
    """
    gdf = geodataframe
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError(f"Cannot convert a {type(gdf)}, expected a GeoDataFrame")

    geom_types = gdf.geom_type.unique()
    if len(geom_types) == 0:
        raise ValueError("geodataframe contains no geometry")
    elif len(geom_types) > 1:
        message = ", ".join(geom_types)
        raise ValueError(f"Multiple geometry types detected: {message}")

    geom_type = geom_types[0]
    if geom_type == "Linestring":
        return geodataframe_to_ugrid1d(gdf)
    elif geom_type == "Polygon":
        return geodataframe_to_ugrid2d(gdf)
    else:
        raise ValueError(
            f"Invalid geometry type: {geom_type}. Expected Linestring or Polygon."
        )
