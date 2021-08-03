from typing import Tuple

import numpy as np
from scipy import sparse

INT_DTYPE = np.int32
FLOAT_DTYPE = np.float64
IntArray = np.ndarray
BoolArray = np.ndarray


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


def _ragged_index(n: int, m: int, m_per_row: IntArray) -> BoolArray:
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
    dense_conn = np.empty((n, m), dtype=INT_DTYPE)
    flat_conn = dense_conn.ravel()
    if (n * m) == conn.nnz:
        # Shortcut if fill_value is not present, when all of same geom. type
        # e.g. all triangles or all quadrangles
        valid = slice(None)  # a[:] equals a[slice(None)]
    else:
        valid = _ragged_index(n, m, m_per_row).ravel()
        flat_conn[~valid] = fill_value
    flat_conn.ravel()[valid] = conn.indices
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
    return np.argsort(a.ravel()).reshape(a.shape)
