"""
Custom Sparse Matrix utilities.

Numba cannot deal with scipy.sparse objects directly. The data
structures are mostly a collection of numpy arrays, which can
be neatly represented by (typed) namedtuples, which numba accepts.
"""
from typing import NamedTuple

import numba
import numpy as np
from scipy import sparse

from xugrid.constants import FloatArray, IntArray, IntDType


class MatrixCOO(NamedTuple):
    """
    A sparse matrix in COOrdinate format, also known as the "ijv" or "triplet"
    format. More or less matches the scipy.sparse.coo_matrix.

    NamedTuple for easy ingestion by numba.

    Parameters
    ----------
    data: np.ndarray of floats
        The regridding weights.
    row: np.ndarray of integers
        The linear index into the source grid.
    col: np.ndarray of integers
        The linear index into the target grid.
    n: int
        The number of rows.
    m: int
        The number of columns.
    nnz: int
        The number of non-zero values.
    """

    data: FloatArray
    row: IntArray
    col: IntArray
    n: int
    m: int
    nnz: int

    @staticmethod
    def from_triplet(row, col, data, n=None, m=None) -> "MatrixCOO":
        if n is None:
            n = row.max() + 1
        if m is None:
            m = col.max() + 1
        nnz = data.size
        return MatrixCOO(data, row, col, n, m, nnz)

    def to_csr(self) -> "MatrixCSR":
        """
        Convert COO matrix to CSR matrix.

        Assumes the COO matrix indices are already sorted by row number!
        """
        i = np.cumsum(np.bincount(self.row, minlength=self.n))
        indptr = np.empty(i.size + 1, dtype=IntDType)
        indptr[0] = 0
        indptr[1:] = i
        return MatrixCSR(
            self.data,
            self.col,
            indptr,
            self.n,
            self.m,
            self.nnz,
        )


class MatrixCSR(NamedTuple):
    """
    Compressed Sparse Row matrix. The row indices are compressed; all values
    must therefore be sorted by row number. More or less matches the
    scipy.sparse.csr_matrix.

    NamedTuple for easy ingestion by numba.

    Parameters
    ----------
    data: np.ndarray of floats
        The values of the matrix.
    indices: np.ndarray of integers
        The column numbers of the CSR format.
    indptr: inp.ndarray of integers
        The row index CSR pointer array.
        Values for row i (target index i) are stored in:
        indices[indptr[i]: indptr[i + 1]]
    n: int
        The number of rows.
    m: int
        The number of columns.
    nnz: int
        The number of non-zero values.
    """

    data: FloatArray
    indices: IntArray
    indptr: IntArray
    n: int
    m: int
    nnz: int

    @staticmethod
    def from_csr_matrix(A: sparse.csr_matrix) -> "MatrixCSR":
        n, m = A.shape
        return MatrixCSR(A.data, A.indices, A.indptr, n, m, A.nnz)

    @staticmethod
    def from_triplet(
        row,
        col,
        data,
        n=None,
        m=None,
    ) -> "MatrixCSR":
        return MatrixCOO.from_triplet(row, col, data, n, m).to_csr()

    def to_coo(self) -> MatrixCOO:
        """
        Convert CSR matrix to COO matrix.

        Expand the indtpr to full row numbers.
        """
        n_repeat = np.diff(self.indptr)
        row = np.repeat(np.arange(self.n), n_repeat)
        return MatrixCOO(self.data, row, self.indices, self.n, self.m, self.nnz)


@numba.njit(inline="always")
def nzrange(A: MatrixCSR, row: int) -> range:
    """Return the non-zero indices of a single row."""
    start = A.indptr[row]
    end = A.indptr[row + 1]
    return range(start, end)


@numba.njit(inline="always")
def row_slice(A, row: int) -> slice:
    """Return the indices or data slice of a single row."""
    start = A.indptr[row]
    end = A.indptr[row + 1]
    return slice(start, end)


@numba.njit(inline="always")
def columns_and_values(A, slice):
    return zip(A.indices[slice], A.data[slice])
