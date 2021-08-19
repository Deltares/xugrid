"""
Snapes nodes at an arbitrary distance together.
"""
from typing import Tuple

import numpy as np
import pandas as pd
from numpy.lib.shape_base import column_stack
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .typing import FloatArray, IntArray


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
