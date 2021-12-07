"""
Conversion from and to other data structures:

* GIS vector data (e.g. geopackage, shapefile)
* Structured data (e.g. rasters)

"""
from typing import Any, Tuple

import numpy as np
import pygeos

from .connectivity import ragged_index
from .typing import IntDType

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


def _to_pygeos(geometry: Any):
    first = geometry[0]
    if not isinstance(first, pygeos.Geometry):
        # might be shapely
        try:
            geometry = pygeos.from_shapely(geometry)
        except:
            raise TypeError(
                "geometry should be pygeos or shapely type. "
                f"Received instead {type(first)}"
            )
    return geometry


def points_to_nodes(points: PointArray) -> Tuple[FloatArray, FloatArray]:
    points = _to_pygeos(points)
    return contiguous_xy(pygeos.get_coordinates(points))


def linestrings_to_edges(edges: LineArray) -> Tuple[FloatArray, FloatArray, IntArray]:
    edges = _to_pygeos(edges)
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
    polygons = _to_pygeos(polygons)
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
