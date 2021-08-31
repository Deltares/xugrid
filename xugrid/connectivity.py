from typing import NamedTuple, Tuple

import numpy as np
import numba as nb
from scipy import sparse

from .typing import BoolArray, IntArray, IntDType


class AdjacencyMatrix(NamedTuple):
    indices: IntArray
    indptr: IntArray
    nnz: int


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


def to_dense(conn: sparse.csr_matrix, fill_value: int) -> IntArray:
    n, _ = conn.shape
    m_per_row = np.diff(conn.indptr)
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
    flat_conn[valid] = conn.indices
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


# Renumbering
# -----------
def renumber(a: IntArray) -> IntArray:
    # Taken from https://github.com/scipy/scipy/blob/v1.7.1/scipy/stats/stats.py#L8631-L8737
    # (scipy is BSD-3-Clause License)
    arr = np.ravel(np.asarray(a))
    sorter = np.argsort(arr, kind="quicksort")
    inv = np.empty(sorter.size, dtype=INT_DTYPE)
    inv[sorter] = np.arange(sorter.size, dtype=INT_DTYPE)
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
    closed = np.full((n, m + 1), fill_value, dtype=INT_DTYPE)
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
    edge_node_connectivity = np.empty((n * m, 2), dtype=INT_DTYPE)
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
    i = edge_face_connectivity[:, 0]
    j = edge_face_connectivity[:, 1]
    is_connection = j != fill_value
    i = i[is_connection]
    j = j[is_connection]
    coo_content = (j, (i, j))
    coo_matrix = sparse.coo_matrix(coo_content)
    return coo_matrix.tocsr()
