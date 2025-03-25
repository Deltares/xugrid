"""Logic for snapping points and lines to face nodes and face edges."""

from typing import Tuple, TypeVar, Union

import numba as nb
import numpy as np
import pandas as pd
import xarray as xr
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
from xugrid.core.sparse import MatrixCSR, columns_and_values, row_slice
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


@nb.njit(cache=True)
def _snap_to_nearest(A: MatrixCSR, snap_candidates: IntArray, max_distance) -> IntArray:
    """
    Find a closest target for each node.

    The kD tree distance matrix will have stored for each node the other nodes
    that are within snapping distance. These are the rows in the sparse matrix
    that have more than one entry: the snap_candidates.

    The first condition for a point to become a TARGET is if it hasn't been
    connected to another point yet, i.e. it is UNVISITED. Once a point becomes
    an TARGET, it looks for nearby points within the max_distance. These nearby
    points are connected if: they are UNVISITED (i.e. they don't have a target
    yet), or the current target is closer than the previous.
    """
    UNVISITED = -1
    TARGET = -2
    nearest = np.full(A.n, max_distance + 1.0)
    visited = np.full(A.n, UNVISITED)

    for i in snap_candidates:
        if visited[i] != UNVISITED:
            continue
        visited[i] = TARGET

        # Now iterate through every node j that is within max_distance of node i.
        for j, dist in columns_and_values(A, row_slice(A, i)):
            if i == j or visited[j] == TARGET:
                # Continue if we're looking at the distance to ourselves
                # (i==j), or other node is a target.
                continue
            if visited[j] == UNVISITED or dist < nearest[j]:
                # If unvisited node, or already visited but we're closer, set
                # to i.
                visited[j] = i
                nearest[j] = dist

    return visited


def snap_nodes(
    x: FloatArray, y: FloatArray, max_snap_distance: float
) -> Tuple[FloatArray, FloatArray, IntArray]:
    """
    Snap neigbhoring vertices together that are located within a maximum
    snapping distance from each other.

    If vertices are located within a maximum distance, some of them are snapped
    to their neighbors ("targets"), thereby guaranteeing a minimum distance
    between nodes in the result. The determination of whether a point becomes a
    target itself or gets snapped to another point is primarily based on the
    order in which points are processed and their spatial relationships.

    This function also return an inverse index array. In case of a connectivity
    array, ``inverse`` can be used to index into, yielding the updated
    numbers. E.g.:

    ``updated_face_nodes = inverse[face_nodes]``

    Parameters
    ----------
    x: 1D nd array of floats of size N
    y: 1D nd array of floats of size N
    max_snap_distance: float

    Returns
    -------
    inverse: 1D nd array of ints of size N
        Inverse index array: the new vertex number for every old vertex. Is
        None when no vertices within max_distance of each other.
    x_snapped: 1D nd array of floats of size M
        Returns a copy of ``x`` when no vertices within max_distance of each
        other.
    y_snapped: 1D nd array of floats of size M
        Returns a copy of ``y`` when no vertices within max_distance of each
        other.
    """
    # First, find all the points that lie within max_distance of each other
    coords = np.column_stack((x, y))
    tree = cKDTree(coords)
    distances = tree.sparse_distance_matrix(
        tree, max_distance=max_snap_distance, output_type="coo_matrix"
    ).tocsr()
    should_snap = distances.getnnz(axis=1) > 1

    if should_snap.any():
        index = np.arange(x.size)
        visited = _snap_to_nearest(
            A=MatrixCSR.from_csr_matrix(distances),
            snap_candidates=index[should_snap],
            max_distance=max_snap_distance,
        )
        targets = visited < 0  # i.e. still UNVISITED or TARGET valued.
        visited[targets] = index[targets]
        deduplicated, inverse = np.unique(visited, return_inverse=True)
        return inverse, x[deduplicated], y[deduplicated]
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
                .to_numpy()
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
    keep = np.diff(line_index) == 0
    return edges[keep], line_index[1:][keep]


@nb.njit(inline="always")
def left_of(a: Point, p: Point, U: Vector) -> bool:
    # Whether point a is left of vector U
    # U: p -> q direction vector
    # TODO: maybe add epsilon for floating point
    return U.x * (a.y - p.y) > U.y * (a.x - p.x)


def coerce_geometry(lines: GeoDataFrameType) -> LineArray:
    geometry = lines.geometry.to_numpy()
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
    Snap the intersected edges to the edges of the surrounding face.

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


def create_snap_to_grid_dataframe(
    lines: GeoDataFrameType,
    grid: Union[xr.DataArray, xu.UgridDataArray],
    max_snap_distance: float,
) -> pd.DataFrame:
    """
    Create a dataframe required to snap line geometries to a Ugrid2d topology.

    A line is included and snapped to a grid edge when the line separates
    the centroid of the cell with the centroid of the edge.

    Parameters
    ----------
    lines: gpd.GeoDataFrame
        Line data. Geometry colum should contain exclusively LineStrings.
    grid: xugrid.Ugrid2d
        Grid of cells to snap lines to.
    max_snap_distance: float

    Returns
    -------
    result: pd.DataFrame
        DataFrame with columns:

        * line_index: the index of the geodataframe geometry.
        * edge_index: the index of the
        * x0: start x-coordinate of edge segment.
        * y0: start y-coordinate of edge segment.
        * x1: end x-coordinate of edge segment.
        * y1: end y-coordinate of edge segment.
        * length: length of the edge.

    Examples
    --------
    First create data frame:

    >>> snapping_df = create_snap_to_grid_dataframe(lines, grid2d, max_snap_distance=0.5)

    Use the ``line_index`` column to assign values from ``lines`` to this new dataframe:

    >>> snapping_df["my_variable"] = lines["my_variable"].iloc[snapping_df["line_index"]].to_numpy()

    Run some reduction on the variable, to create an aggregated value per grid edge:

    >>> aggregated = snapping_df.groupby("edge_index").sum()

    Assign the aggregated values to a Ugrid2d topology:

    >>> new = xu.full_like(edge_data, np.nan)
    >>> new.data[aggregated.index] = aggregated["my_variable"]

    """
    if not isinstance(grid, Ugrid2d):
        raise TypeError(f"Expected Ugrid2d, received: {type(grid).__name__}")

    topology = grid
    vertices = topology.node_coordinates
    edge_centroids = topology.edge_coordinates
    face_edge_connectivity = topology.face_edge_connectivity
    A = connectivity.to_sparse(face_edge_connectivity)
    n, m = A.shape
    face_edge_connectivity = AdjacencyMatrix(A.indices, A.indptr, A.nnz, n, m)

    # Create geometric data
    line_geometry = coerce_geometry(lines)
    line_coords, shapely_vertex_index = shapely.get_coordinates(
        line_geometry, return_index=True
    )
    # Snap line_coords to grid
    x, y = snap_to_nodes(
        *line_coords.T, *vertices.T, max_snap_distance, tiebreaker="nearest"
    )
    line_edges, shapely_line_index = lines_as_edges(
        np.column_stack([x, y]), shapely_vertex_index
    )

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

    return pd.DataFrame(
        data={
            "line_index": shapely_line_index[line_index],
            "edge_index": edge_index,
            "x0": segment_edges[:, 0, 0],
            "y0": segment_edges[:, 0, 1],
            "x1": segment_edges[:, 1, 0],
            "y1": segment_edges[:, 1, 1],
            "length": ((segment_edges[:, 1] - segment_edges[:, 0]) ** 2).sum(axis=1),
        }
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

    Parameters
    ----------
    lines: gpd.GeoDataFrame
        Line data. Geometry colum should contain exclusively LineStrings.
    grid: xr.DataArray or xu.UgridDataArray of integers
        Grid of cells to snap lines to.
    max_snap_distance: float

    Returns
    -------
    uds: UgridDataset
        Snapped line geometries as edges in a Ugrid2d topology. Contains a
        ``line_index`` variable identifying the original geodataframe line.
    gdf: gpd.GeoDataFrame
        Snapped line geometries.
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

    result = create_snap_to_grid_dataframe(lines, topology, max_snap_distance)

    # When multiple line parts are snapped to the same edge, use the ones with
    # the greatest length inside the cell.
    max_edge_index = result.groupby("edge_index").idxmax()["length"].to_numpy()
    line_index = result["line_index"].to_numpy()[max_edge_index]
    edges = result["edge_index"].to_numpy()[max_edge_index]

    uds = _create_output_dataset(lines, topology, edges, line_index)
    gdf = _create_output_gdf(
        lines,
        topology.node_coordinates,
        topology.edge_node_connectivity,
        edges,
        line_index,
    )
    return uds, gdf
