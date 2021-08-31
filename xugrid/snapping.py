"""
Snapes nodes at an arbitrary distance together.
"""
from typing import Tuple, Union

from numba_celltree import CellTree2d
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

import geopandas as gpd
import imod
import numba as nb
import numpy as np
import pandas as pd
import pygeos
import xarray as xr

from .typing import (
    FloatArray,
    IntArray,
    LineArray,
    Point,
    Vector,
)
from . import connectivity
from .connectivity import AdjacencyMatrix


def snap(
    x: FloatArray, y: FloatArray, max_distance: float
) -> Tuple[FloatArray, FloatArray, IntArray]:
    """
    Snap neigbhoring vertices together that are located within a maximum
    distance from each other.

    If vertices are located within a maximum distance, they are merged into a
    single vertex. The coordinates of the merged coordinates are given by the
    centroid of the merging vertices.

    Note that merging is a communicative process: vertex A might lie close to
    vertex B, vertex B might lie close to vertex C, and so on. These points are
    grouped and merged into a single new vertex.

    This function also return an inverse index array. In case of a connectivity
    array, ``inverse`` can be used to index into, yielding the updated
    numbers. E.g.:

    ``updated_face_nodes = inverse[face_nodes]``

    Parameters
    ----------
    x: 1D nd array of floats of size N
    y: 1D nd array of floats of size N
    max_distance: float

    Returns
    -------
    inverse: 1D nd array of ints of size N
        Inverse index array: the new vertex number for every old vertex. Is
        None when no vertices within max_distance of each other.
    x_merged: 1D nd array of floats of size M
        Returns a copy of ``x`` when no vertices within max_distance of each
        other.
    y_merged: 1D nd array of floats of size M
        Returns a copy of ``y`` when no vertices within max_distance of each
        other.
    """
    # First, find all the points that lie within max_distance of each other
    coords = np.column_stack((x, y))
    n = len(coords)
    tree = cKDTree(coords)
    coo_distances = tree.sparse_distance_matrix(
        tree, max_distance=max_distance, output_type="coo_matrix"
    )
    # Get rid of diagonal
    i = coo_distances.row
    j = coo_distances.col
    off_diagonal = i != j
    i = i[off_diagonal]
    j = j[off_diagonal]
    # If no pairs are found, there is nothing to snap
    if len(i) > 0:
        # Next, group the points together.
        # Point A might lie close to point B, point B might lie close to
        # Point C, and so on. These points are grouped, and their centroid
        # is computed.
        coo_content = (np.ones(i.size), (i, j))
        coo_matrix = sparse.coo_matrix(coo_content, shape=(n, n))
        # Directed is true: this matrix is symmetrical
        _, inverse = connected_components(coo_matrix, directed=True)
        new = (
            pd.DataFrame({"label": inverse, "x": x, "y": y})
            .groupby("label")
            .agg(
                {
                    "x": "mean",
                    "y": "mean",
                }
            )
        )
        return inverse, new["x"].values, new["y"].values
    else:
        return None, x.copy(), y.copy()


def snap_to(
    x: FloatArray,
    y: FloatArray,
    to_x: FloatArray,
    to_y: FloatArray,
    max_distance: float,
    tiebreaker=None,
) -> Tuple[FloatArray, FloatArray, IntArray]:
    """
    Snap vertices (x, y) to another set of vertices (to_x, to_y) if within a
    specified maximum distance.

    Parameters
    ----------
    x: 1D nd array of floats of size N
    y: 1D nd array of floats of size N
    to_x: 1D nd array of floats of size M
    to_y: 1D nd array of floats of size M
    max_distance: float

    Returns
    -------
    x_snapped: 1D nd array of floats of size N
    y_snapped: 1D nd array of floats of size N
    """
    if tiebreaker not in (None, "nearest"):
        raise ValueError(
            f"Invalid tiebreaker: {tiebreaker}, should be one of "
            '{None, "nearest"} instead.'
        )
    # Build KDTrees
    coords = np.column_stack((x, y))
    to_coords = np.column_stack((to_x, to_y))
    tree = cKDTree(coords)
    to_tree = cKDTree(to_coords)

    # Build distance matrix, identify points to snap (update) and possible ties
    # row (i) is index in x, y; column (j) is index in to_x, to_y
    # Convert to csr for quicker row indexing, and row reductions.
    distances = tree.sparse_distance_matrix(
        to_tree, max_distance=max_distance, output_type="coo_matrix"
    ).tocsr()
    n_per_row = distances.getnnz(axis=1)
    update = n_per_row == 1
    tie = n_per_row > 1

    # Update the points that snap to a single (to_x, to_y) vertex
    xnew = x.copy()
    ynew = y.copy()
    j_update = distances[update].indices
    xnew[update] = to_x[j_update]
    ynew[update] = to_y[j_update]

    # Resolve ties with multiple (to_x, to_y) candidates
    if tie.any():
        if tiebreaker == "nearest":
            ties = distances[tie].tocoo()
            j_nearest = (
                pd.DataFrame({"i": ties.row, "distance": ties.data}, index=ties.col)
                .groupby("i")["distance"]
                .idxmin()
                .values
            )
            xnew[tie] = to_x[j_nearest]
            ynew[tie] = to_y[j_nearest]
        elif tiebreaker is None:
            raise ValueError(
                "Ties detected: multiple options to snap to, given max distance: "
                "set a smaller tolerance or specify a tiebreaker."
            )
        else:
            raise ValueError("Invalid tiebreaker")

    return xnew, ynew


@nb.njit(inline="always")
def to_vector(a: Point, b: Point) -> Vector:
    return Vector(b.x - a.x, b.y - a.y)


@nb.njit(inline="always")
def as_point(a: FloatArray) -> Point:
    return Point(a[0], a[1])


@nb.njit(inline="always")
def cross_product(u: Vector, v: Vector) -> float:
    return u.x * v.y - u.y * v.x


@nb.njit(inline="always")
def dot_product(u: Vector, v: Vector) -> float:
    return u.x * v.x + u.y * v.y


def structured_centroids(grid: xr.DataArray) -> FloatArray:
    active = grid.values != 0
    y = grid["y"].values
    x = grid["x"].values
    yy, xx = np.meshgrid(y, x, indexing="ij")
    active = grid.values != 0
    centroids = np.column_stack((xx[active], yy[active]))
    return centroids


def lines_as_edges(line_coords, line_index) -> FloatArray:
    edges = np.empty((len(line_coords) - 1, 2, 2))
    edges[:, 0, :] = line_coords[:-1]
    edges[:, 1, :] = line_coords[1:]
    return edges[np.diff(line_index) == 0]


@nb.njit(inline="always")
def lines_intersect(a: Point, V: Vector, N: Vector, p: Point, q: Point) -> bool:
    # V: a -> b
    # N: norm of U: p -> q
    # r, N form clipping plane
    W = Vector(p.x - a.x, p.y - a.y)
    nw = dot_product(N, W)
    nv = dot_product(N, V)
    if nv != 0:
        tv = nw / nv
        if tv <= 0.0 or tv >= 1.0:
            return False

        U = Vector(q.x - p.x, q.y - p.y)
        if U.x != 0:
            tu = (tv * V.x - W.x) / U.x
        elif U.y != 0:
            tu = (tv * V.x - W.x) / U.x
        else:  # no dx, no dy: just a point
            return False
        if tu <= 0.0 or tu >= 1.0:
            return False

        return True
    else:
        # parallel lines
        return False


@nb.njit
def separated_faces(
    segment_indices: IntArray,
    face_indices: IntArray,
    intersection_edges: FloatArray,
    face_face_connectivity: AdjacencyMatrix,
    centroids: FloatArray,
) -> Tuple[IntArray, IntArray]:
    # a line can only snap to two edges of a quad
    max_n_new_edges = len(face_indices) * 2
    face_to_face = np.empty((max_n_new_edges, 2), dtype=np.intp)
    segment_index = np.empty(max_n_new_edges, dtype=np.intp)

    count = 0
    for i in range(len(segment_indices)):
        segment = segment_indices[i]
        face = face_indices[i]
        a = as_point(centroids[face])
        p = as_point(intersection_edges[i, 0])
        q = as_point(intersection_edges[i, 1])
        U = to_vector(p, q)
        N = Vector(-U.y, U.x)

        for neighbor in connectivity.neighbors(face_face_connectivity, face):
            b = as_point(centroids[neighbor])
            V = to_vector(a, b)
            if lines_intersect(a, V, N, p, q):
                face_to_face[count, 0] = face
                face_to_face[count, 1] = neighbor
                segment_index[count] = segment
                count += 1

    return face_to_face[:count], segment_index[:count]


def create_geometry(
    vertices: FloatArray, faces: IntArray, face_to_face: IntArray
) -> LineArray:
    face_to_face.sort(axis=1)
    edge_node_connectivity, face_edge_connectivity = connectivity.edge_connectivity(
        faces, -1
    )
    edge_face_connectivity = connectivity.invert_dense(
        face_edge_connectivity, fill_value=-1
    )

    df = pd.DataFrame(
        {
            "edge": np.arange(len(edge_face_connectivity)),
            "face0": edge_face_connectivity[:, 0],
            "face1": edge_face_connectivity[:, 1],
        }
    ).set_index(["face0", "face1"])
    idx = pd.MultiIndex.from_arrays([face_to_face[:, 0], face_to_face[:, 1]])
    edge_index = df.loc[idx, "edge"].values

    new_edges = edge_node_connectivity[edge_index]
    new_coords = vertices[new_edges]
    return pygeos.creation.linestrings(new_coords)


def coerce_geometry(lines: gpd.GeoDataFrame) -> LineArray:
    geometry = lines.geometry.values
    first = geometry[0]
    if not isinstance(first, pygeos.Geometry):
        # might be shapely
        try:
            geometry = pygeos.from_shapely(geometry)
        except TypeError:
            raise TypeError(
                "lines geometry should only contain either shapely or pygeos "
                "geometries."
            )
    geom_type = pygeos.get_type_id(geometry)
    if not (geom_type == 1).all():
        raise ValueError("Geometry should contain only LineStrings")
    return geometry


def snap_to_grid(
    lines: gpd.GeoDataFrame, grid: xr.DataArray, return_geometry: bool = False
) -> Tuple[IntArray, Union[pd.DataFrame, gpd.GeoDataFrame]]:
    """
    Snap a collection of lines to a grid.
    
    A line is included and snapped to a grid edge when the line separates when
    it separates the cell in which it is located from the centroid of a
    neighboring cell.
    
    When a line is located in a cell, but does not separate the cell centroid
    from any neighbor, it is not included: the line is not snapped to e.g.
    exterior boundaries.
    
    Parameters
    ----------
    lines: gpd.GeoDataFrame
        Line data. Geometry colum should contain exclusively LineStrings.
    grid: xr.DataArray of integers
        Grid of cells to snap lines to. Cells with a value of 0 are not
        included.
    return_geometry: bool, optional. Default: False.
        Whether to return geometry. In this case a GeoDataFrame is returned
        with the snapped line segments, rather than a DataFrame without
        geometry.
    
    Returns
    -------
    cell_to_cell: ndarray of integers with shape ``(N, 2)``
        Cells whose centroids are separated from each other by a line.
    segment_data: pd.DataFrame or gpd.DataFrame
        Data for every segment. GeoDataFrame if ``return_geometry`` is
        ``True``.
    """
    active = grid.values != 0
    nrow, ncol = active.shape
    topology = imod.util.ugrid2d_topology(grid)
    faces = topology["face_nodes"].values[active.ravel()].astype(int)
    vertices = np.column_stack((topology["node_x"].values, topology["node_y"].values))
    face_to_cell = np.arange(nrow * ncol)[active.ravel()]
    face_face_connectivity = connectivity.structured_connectivity(active)
    centroids = structured_centroids(grid)

    line_geometry = coerce_geometry(lines)
    line_coords, line_index = pygeos.get_coordinates(
        pygeos.from_shapely(line_geometry), return_index=True
    )
    line_edges = lines_as_edges(line_coords, line_index)

    celltree = CellTree2d(vertices, faces, -1)
    segment_indices, face_indices, intersection_edges = celltree.intersect_edges(
        line_edges
    )
    face_to_face, segment_index = separated_faces(
        segment_indices,
        face_indices,
        intersection_edges,
        face_face_connectivity,
        centroids,
    )
    line_index = line_index[segment_index]
    cell_to_cell = face_to_cell[face_to_face]

    if return_geometry:
        geometry = create_geometry(vertices, faces, face_to_face)
        gdf = gpd.GeoDataFrame(
            lines.drop(columns="geometry").iloc[line_index], geometry=geometry
        )
        return cell_to_cell, gdf
    else:
        df = lines.drop(columns="geometry").iloc[line_index]
        return cell_to_cell, df
