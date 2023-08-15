"""
Snapes nodes at an arbitrary distance together.
"""
from typing import Tuple, TypeVar, Union

import numba as nb
import numpy as np
import pandas as pd
import xarray as xr
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

import xugrid as xu
from xugrid.constants import (
    FloatArray,
    IntArray,
    IntDType,
    LineArray,
    MissingOptionalModule,
    Point,
    Vector,
)
from xugrid.ugrid import connectivity
from xugrid.ugrid.connectivity import AdjacencyMatrix
from xugrid.ugrid.ugrid2d import Ugrid2d

try:
    import geopandas as gpd

    GeoDataFrameType = gpd.GeoDataFrame
except ImportError:
    gpd = MissingOptionalModule("geopandas")
    # https://stackoverflow.com/questions/61384752/how-to-type-hint-with-an-optional-import
    GeoDataFrameType = TypeVar("GeoDataFrameType")  # avoid ImportError in typehints

try:
    import shapely
except ImportError:
    shapely = MissingOptionalModule("shapely")


def snap_nodes(
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


def snap_to_nodes(
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


def lines_as_edges(line_coords, line_index) -> FloatArray:
    edges = np.empty((len(line_coords) - 1, 2, 2))
    edges[:, 0, :] = line_coords[:-1]
    edges[:, 1, :] = line_coords[1:]
    return edges[np.diff(line_index) == 0]


@nb.njit(inline="always")
def left_of(a: Point, p: Point, U: Vector) -> bool:
    # Whether point a is left of vector U
    # U: p -> q direction vector
    # TODO: maybe add epsilon for floating point
    return U.x * (a.y - p.y) > U.y * (a.x - p.x)


def coerce_geometry(lines: GeoDataFrameType) -> LineArray:
    geometry = lines.geometry.values
    geom_type = shapely.get_type_id(geometry)
    if not ((geom_type == 1) | (geom_type == 2)).all():
        raise ValueError("Geometry should contain only LineStrings and/or LinearRings")
    return geometry


@nb.njit(cache=True)
def snap_to_edges(
    face_indices: IntArray,
    intersection_edges: FloatArray,
    face_edge_connectivity: AdjacencyMatrix,
    centroids: FloatArray,
    edge_centroids: FloatArray,
    edges: IntArray,
    segment_index: IntArray,
) -> Tuple[IntArray, IntArray]:
    """
    This algorithm works as follows:

    * It takes the intersected edges; any edge (p to q) to test falls fully
      within a single face.
    * For a face, we take the centroid (a).
    * We loop through every edge of the face.
    * If the edge separates the centroid (a) from the centroid of the edge (b)
      we store that edge as a separating edge.

    We test for separation by:

    * Finding whether a and b are located at opposite sides of the half-plane
      created by the edge p -> q (U).
    * Finding whether p and q are located at opposide sides of the half-plane
      created by a -> b (V).
    * The separation test will return False if the lines are collinear. This is
      desirable here, if U runs collinear with V, U doesn't separate a from b.

    Do a minimum amount of work: reuse a_left, only compute V if needed.
    """
    count = 0
    for i in range(len(face_indices)):
        face = face_indices[i]
        a = as_point(centroids[face])
        p = as_point(intersection_edges[i, 0])
        q = as_point(intersection_edges[i, 1])
        U = to_vector(p, q)
        if U.x == 0 and U.y == 0:
            continue

        a_left = left_of(a, p, U)
        for edge in connectivity.neighbors(face_edge_connectivity, face):
            b = as_point(edge_centroids[edge])
            b_left = left_of(b, p, U)
            if a_left != b_left:
                V = to_vector(a, b)
                if left_of(p, a, V) != left_of(q, a, V):
                    segment_index[count] = i
                    edges[count] = edge
                    count += 1

    return edges[:count], segment_index[:count]


def _find_largest_edges(
    edges: FloatArray,
    edge_index: IntArray,
    line_index: IntArray,
):
    max_edge_index = (
        pd.DataFrame(
            data={
                "edge_index": edge_index,
                "length": ((edges[:, 1] - edges[:, 0]) ** 2).sum(axis=1),
            }
        )
        .groupby("edge_index")
        .idxmax()["length"]
        .values
    )

    edge_index = edge_index[max_edge_index]
    line_index = line_index[max_edge_index]
    return edge_index, line_index


def _create_output_dataset(
    lines: GeoDataFrameType,
    topology: "xu.Ugrid2d",
    edges: IntArray,
    line_index: IntArray,
) -> xu.UgridDataset:
    uds = xu.UgridDataset(grids=[topology])
    data = np.full(topology.n_edge, np.nan)
    data[edges] = line_index
    uds["line_index"] = xr.DataArray(
        data=data,
        dims=[topology.edge_dimension],
    )
    for column in lines.columns:
        if column == "geometry":
            continue
        data = np.full(topology.n_edge, np.nan)
        data[edges] = lines[column].iloc[line_index]
        uds[column] = xr.DataArray(
            data=data,
            dims=[topology.edge_dimension],
        )
    return uds


def _create_output_gdf(
    lines,
    vertices,
    edge_node_connectivity,
    edges,
    shapely_index,
):
    edge_vertices = vertices[edge_node_connectivity[edges]]
    geometry = shapely.linestrings(edge_vertices)
    return gpd.GeoDataFrame(
        lines.drop(columns="geometry").iloc[shapely_index], geometry=geometry
    )


def snap_to_grid(
    lines: GeoDataFrameType,
    grid: Union[xr.DataArray, xu.UgridDataArray],
    max_snap_distance: float,
) -> Tuple[IntArray, Union[pd.DataFrame, GeoDataFrameType]]:
    """
    Snap a collection of lines to a grid.

    A line is included and snapped to a grid edge when the line separates
    the centroid of the cell with the centroid of the edge.

    When a line in a cell is snapped to an edge that is **not** shared with
    another cell, this is denoted with a value of -1 in the second column of
    ``cell_to_cell``.

    Parameters
    ----------
    lines: gpd.GeoDataFrame
        Line data. Geometry colum should contain exclusively LineStrings.
    grid: xr.DataArray or xu.UgridDataArray of integers
        Grid of cells to snap lines to. Cells with a value of 0 are not
        included.
    max_snap_distance: float

    Returns
    -------
    cell_to_cell: ndarray of integers with shape ``(N, 2)``
        Cells whose centroids are separated from each other by a line.
    segment_data: pd.DataFrame or gpd.DataFrame
        Data for every segment. GeoDataFrame if ``return_geometry`` is
        ``True``.
    """
    if isinstance(grid, Ugrid2d):
        topology = grid
    elif isinstance(grid, xr.DataArray):
        # Convert structured to unstructured representation
        topology = Ugrid2d.from_structured(grid)
    elif isinstance(grid, xu.UgridDataArray):
        topology = grid.ugrid.grid
    else:
        raise TypeError(
            "Expected xarray.DataArray or xugrid.UgridDataArray, received: "
            f" {type(grid).__name__}"
        )

    vertices = topology.node_coordinates
    edge_centroids = topology.edge_coordinates
    edge_node_connectivity = topology.edge_node_connectivity
    face_edge_connectivity = topology.face_edge_connectivity
    A = connectivity.to_sparse(face_edge_connectivity, fill_value=-1)
    n, m = A.shape
    face_edge_connectivity = AdjacencyMatrix(A.indices, A.indptr, A.nnz, n, m)

    # Create geometric data
    line_geometry = coerce_geometry(lines)
    line_coords, shapely_index = shapely.get_coordinates(
        line_geometry, return_index=True
    )
    # Snap line_coords to grid
    x, y = snap_to_nodes(
        *line_coords.T, *vertices.T, max_snap_distance, tiebreaker="nearest"
    )
    line_edges = lines_as_edges(np.column_stack([x, y]), shapely_index)

    # Search for intersections. Every edge is potentially divided into smaller
    # segments: The segment_indices contain (repeated) values of the
    #
    # * line_index: for each segment, the index of the shapely geometry.
    # * face_indices: for each segment, the index of the topology face.
    # * segment_edges: for each segment, start and end xy-coordinates.
    #
    line_index, face_indices, segment_edges = topology.celltree.intersect_edges(
        line_edges
    )

    # Create edges from the intersected lines a: line can only snap to N - 1
    # edges for N edges of a face. Pre-allocate the arrays here. For some
    # reason, recent versions of numba refuse np.empty or np.zeros calls in
    # this module (!).
    # TODO: investigate...
    #
    # * edge_index: which edge of the topology (may contain duplicates)
    # * segment_index: which segment

    max_n_new_edges = len(face_indices) * topology.n_max_node_per_face - 1
    edge_index = np.empty(max_n_new_edges, dtype=IntDType)
    segment_index = np.empty(max_n_new_edges, dtype=IntDType)
    edge_index, segment_index = snap_to_edges(
        face_indices,
        segment_edges,
        face_edge_connectivity,
        topology.centroids,
        edge_centroids,
        edge_index,  # out
        segment_index,  # out
    )
    line_index = line_index[segment_index]
    segment_edges = segment_edges[segment_index]

    # When multiple line parts are snapped to the same edge, use the ones with
    # the greatest length inside the cell.
    edges, line_index = _find_largest_edges(segment_edges, edge_index, line_index)
    shapely_index = shapely_index[line_index]

    uds = _create_output_dataset(lines, topology, edges, shapely_index)
    gdf = _create_output_gdf(
        lines, vertices, edge_node_connectivity, edges, shapely_index
    )
    return uds, gdf
