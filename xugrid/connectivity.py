from typing import NamedTuple, Tuple

import numba as nb
import numpy as np
from scipy import sparse

from .typing import BoolArray, FloatArray, IntArray, IntDType


class AdjacencyMatrix(NamedTuple):
    indices: IntArray
    indptr: IntArray
    nnz: int


@nb.njit(inline="always")
def neighbors(A: AdjacencyMatrix, cell: int) -> IntArray:
    start = A.indptr[cell]
    end = A.indptr[cell + 1]
    return A.indices[start:end]


# Conversion between dense and sparse
# -----------------------------------
def _to_ij(conn: IntArray, fill_value: int, invert: bool) -> Tuple[IntArray, IntArray]:
    n, m = conn.shape
    j = conn.ravel()
    valid = j != fill_value
    i = np.repeat(np.arange(n), m)[valid]
    j = j[valid]
    if invert:
        return j, i
    else:
        return i, j


def _to_sparse(conn: IntArray, fill_value: int, invert: bool) -> sparse.csr_matrix:
    i, j = _to_ij(conn, fill_value, invert)
    coo_content = (j, (i, j))
    coo_matrix = sparse.coo_matrix(coo_content)
    return coo_matrix.tocsr()


def ragged_index(n: int, m: int, m_per_row: IntArray) -> BoolArray:
    """
    Given an array of n rows by m columns, starting from left mark the values
    True such that the number of True values equals m_per_row.

    For example:

    n = 3
    m = 4
    m_per_row = np.array([1, 2, 3])

    Then the result of _ragged_index(n, m, m_per_row) is:

    np.array([
        [True, False, False, False],
        [True, True, False, False],
        [True, True, True, False],
    ])

    This can be used as boolean index to set a variable number of values per
    row.
    """
    column_number = np.tile(np.arange(m), n).reshape((n, m))
    return (column_number.T < m_per_row).T


def to_sparse(conn: IntArray, fill_value: int) -> sparse.csr_matrix:
    return _to_sparse(conn, fill_value, invert=False)


def to_dense(conn: sparse.coo_matrix, fill_value: int) -> IntArray:
    n, _ = conn.shape
    m_per_row = conn.getnnz(axis=1)
    m = m_per_row.max()
    # Allocate 2D array and create a flat view of the dense connectivity
    dense_conn = np.empty((n, m), dtype=IntDType)
    flat_conn = dense_conn.ravel()
    if (n * m) == conn.nnz:
        # Shortcut if fill_value is not present, when all of same geom. type
        # e.g. all triangles or all quadrangles
        valid = slice(None)  # a[:] equals a[slice(None)]
    else:
        valid = ragged_index(n, m, m_per_row).ravel()
        flat_conn[~valid] = fill_value

    if isinstance(conn, sparse.csr_matrix):
        flat_conn[valid] = conn.indices
    elif isinstance(conn, sparse.coo_matrix):
        flat_conn[valid] = conn.col
    else:
        raise TypeError("Can only invert coo or csr matrix")
    return dense_conn


# Inverting connectivities
# ------------------------
def invert_dense_to_sparse(conn: IntArray, fill_value: int) -> sparse.csr_matrix:
    return _to_sparse(conn, fill_value, invert=True)


def invert_dense(conn: IntArray, fill_value: int) -> IntArray:
    sparse_inverted = _to_sparse(conn, fill_value, invert=True)
    return to_dense(sparse_inverted, fill_value)


def invert_sparse(conn: sparse.csr_matrix) -> sparse.csr_matrix:
    coo = conn.tocoo()
    j = coo.row
    i = coo.col
    coo_content = (j, (i, j))
    inverted = sparse.coo_matrix(coo_content)
    return inverted.tocsr()


def invert_sparse_to_dense(conn: sparse.csr_matrix, fill_value: int) -> IntArray:
    inverted = invert_sparse(conn)
    return to_dense(inverted, fill_value)


# Renumbering
# -----------
def renumber(a: IntArray) -> IntArray:
    # Taken from https://github.com/scipy/scipy/blob/v1.7.1/scipy/stats/stats.py#L8631-L8737
    # (scipy is BSD-3-Clause License)
    arr = np.ravel(np.asarray(a))
    sorter = np.argsort(arr, kind="quicksort")
    inv = np.empty(sorter.size, dtype=IntDType)
    inv[sorter] = np.arange(sorter.size, dtype=IntDType)
    arr = arr[sorter]
    obs = np.r_[True, arr[1:] != arr[:-1]]
    dense = obs.cumsum()[inv] - 1
    return dense.reshape(a.shape)


# Derived connectivities
# ----------------------
def close_polygons(face_node_connectivity: IntArray, fill_value: int) -> IntArray:
    # Wrap around and create closed polygon: put the first node at the end of the row
    # In case of fill values, replace all fill values
    n, m = face_node_connectivity.shape
    closed = np.full((n, m + 1), fill_value, dtype=IntDType)
    closed[:, :-1] = face_node_connectivity
    first_node = face_node_connectivity[:, 0]
    # Identify fill value, and replace by first node also
    isfill = closed == fill_value
    closed.ravel()[isfill.ravel()] = np.repeat(first_node, isfill.sum(axis=1))
    return closed, isfill


def edge_connectivity(
    face_node_connectivity: IntArray, fill_value: int
) -> Tuple[IntArray, IntArray]:
    n, m = face_node_connectivity.shape
    # Close the polygons: [0 1 2 3] -> [0 1 2 3 0]
    closed, isfill = close_polygons(face_node_connectivity, fill_value)
    # Allocate array for edge_node_connectivity: includes duplicate edges
    edge_node_connectivity = np.empty((n * m, 2), dtype=IntDType)
    edge_node_connectivity[:, 0] = closed[:, :-1].ravel()
    edge_node_connectivity[:, 1] = closed[:, 1:].ravel()
    # Cleanup: delete invalid edges (same node to same node)
    edge_node_connectivity = edge_node_connectivity[
        edge_node_connectivity[:, 0] != edge_node_connectivity[:, 1]
    ]
    # Now find the unique rows == unique edges
    edge_node_connectivity.sort(axis=1)
    edge_node_connectivity, inverse_indices = np.unique(
        ar=edge_node_connectivity, return_inverse=True, axis=0
    )
    # Create face_edge_connectivity
    face_edge_connectivity = np.full((n, m), fill_value, dtype=np.int64)
    isnode = ~isfill[:, :-1]
    face_edge_connectivity.ravel()[isnode.ravel()] = inverse_indices
    return edge_node_connectivity, face_edge_connectivity


def face_face_connectivity(
    edge_face_connectivity: IntArray,
    fill_value: int,
) -> sparse.csr_matrix:
    """
    An edge can be shared by two faces at most. If this is the case, they are
    neighbors.
    """
    i = edge_face_connectivity[:, 0]
    j = edge_face_connectivity[:, 1]
    is_connection = j != fill_value
    i = i[is_connection]
    j = j[is_connection]
    coo_content = (j, (i, j))
    coo_matrix = sparse.coo_matrix(coo_content)
    return coo_matrix.tocsr()


def structured_connectivity(active: IntArray) -> AdjacencyMatrix:
    nrow, ncol = active.shape
    nodes = np.arange(nrow * ncol).reshape(nrow, ncol)
    nodes[~active] = -1
    left = nodes[:, :-1].ravel()
    right = nodes[:, 1:].ravel()
    front = nodes[:-1].ravel()
    back = nodes[1:].ravel()
    # valid x connection
    valid = (left != -1) & (right != -1)
    left = left[valid]
    right = right[valid]
    # valid y connection
    valid = (front != -1) & (back != -1)
    front = front[valid]
    back = back[valid]
    i = renumber(np.concatenate([left, right, front, back]))
    j = renumber(np.concatenate([right, left, back, front]))
    coo_content = (j, (i, j))
    A = sparse.coo_matrix(coo_content).tocsr()
    return AdjacencyMatrix(A.indices, A.indptr, A.nnz)


def centroids(
    face_node_connectivity: IntArray,
    fill_value: int,
    node_x: FloatArray,
    node_y: FloatArray,
) -> FloatArray:
    n_face, n_max_node = face_node_connectivity.shape
    nodes = np.column_stack([node_x, node_y])
    # Check if it's fully triangular
    if n_max_node == 3:
        # Since it's triangular, computing the average of the vertices suffices
        coordinates = nodes[face_node_connectivity]
        coordinates[face_node_connectivity == fill_value] = np.nan
        return np.nanmean(coordinates, axis=1)
    else:
        # This is mathematically equivalent to triangulating, computing triangle centroids
        # and computing the area weighted average of those centroids
        centroid_coordinates = np.empty((n_face, 2), dtype=np.float64)
        coordinates = nodes[close_polygons(face_node_connectivity, fill_value)]
        a = coordinates[:, :-1]
        b = coordinates[:, 1:]
        c = a + b
        determinant = np.cross_product(a, b)
        area_weight = 1.0 / (3.0 * determinant.sum(axis=1))
        centroid_coordinates[:, 0] = area_weight * (c[..., 0] * determinant).sum(axis=1)
        centroid_coordinates[:, 1] = area_weight * (c[..., 1] * determinant).sum(axis=1)
        return centroid_coordinates


@nb.njit
def _triangulate_dense(faces: IntArray, fill_value: int, triangles: IntArray) -> None:
    n_face, n_max = faces.shape
    count = 0
    for i in range(n_face):
        first = faces[i, j]
        second = faces[i, j + 1]
        for j in range(2, n_max):
            third = faces[i, j]
            if third == fill_value:
                break
            triangles[count, 0] = first
            triangles[count, 1] = second
            triangles[count, 2] = third
            second = third
            count += 1
    return


@nb.njit
def _triangulate_coo(
    i: IntArray, j: IntArray, n_face: int, triangles: IntArray
) -> None:
    n = i.size
    face_i = 0
    count = 0
    while face_i < (n - 2):
        face = i[face_i]
        first = j[face_i]
        second = j[face_i + 1]
        face_i += 2
        while face_i < n and i[face_i] == face:
            third = j[face_i]
            triangles[count, 0] = first
            triangles[count, 1] = second
            triangles[count, 2] = third
            second = third
            face_i += 1
            count += 1
    return


def triangulate(face_node_connectivity, fill_value: int = None) -> IntArray:
    """
    Convert polygons into its constituent triangles.

    Triangulation runs from the first node of every face:

    * first, second, third,
    * first, third, fourth,
    * and so forth ...

    If the grid is already triangular, a copy is returned.

    Returns
    -------
    triangles: ndarray of integers with shape ``(n_triangle, 3)``
    triangle_face_connectivity: ndarray of integers with shape ``(n_triangle,)``
    """
    if isinstance(face_node_connectivity, IntArray):
        if face_node_connectivity.shape[1] == 3:
            triangles = face_node_connectivity.copy()
            return triangles, np.arange(len(triangles))
        if fill_value is None:
            raise ValueError("fill_value is required for dense connectivity")
        valid = face_node_connectivity != fill_value
        n_triangle_per_row = valid.sum(axis=1) - 2
        n_triangle = n_triangle_per_row.sum()
        triangles = np.empty((n_triangle, 3), IntDType)
        _triangulate_dense(face_node_connectivity, fill_value, triangles)
    elif isinstance(face_node_connectivity, sparse.coo_matrix):
        ncol_per_row = face_node_connectivity.getnnz(axis=1)
        n_triangle_per_row = ncol_per_row - 2
        if ncol_per_row.max() == 3:
            triangles = face_node_connectivity.row.copy().reshape((-1, 3))
            return triangles, np.arange(len(triangles))
        n_triangle = n_triangle_per_row.sum()
        triangles = np.empty((n_triangle, 3), IntDType)
        coo = face_node_connectivity
        _triangulate_coo(coo.row, coo.col, n_triangle, triangles)
    else:
        raise TypeError("connectivity must be ndarray or sparse matrix")

    n_face = face_node_connectivity.shape[0]
    triangle_face_connectivity = np.repeat(
        np.arange(n_face), repeats=n_triangle_per_row
    )
    return triangles, triangle_face_connectivity
