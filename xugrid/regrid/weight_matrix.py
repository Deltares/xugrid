"""
The weight matrix contains the target indices (rows), the source indices
(columns), and the weights. These are stored in compressed sparse row format
(CSR).

We store the linear target index in the column: then all the source indices may
be found together for a single target index. This is useful during querying,
and during regridding: in both case, process on target cell at a time.
"""
from typing import NamedTuple, Tuple

import numba
import numpy as np

from xugrid.constants import FloatArray, IntArray, IntDType


class WeightMatrixCOO(NamedTuple):
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
    nnz: int
        The number of non-zero values.
    """

    data: FloatArray
    row: IntArray
    col: IntArray
    nnz: int


class WeightMatrixCSR(NamedTuple):
    """
    Compressed Sparse Row matrix. The row indices are compressed; all values
    must therefore be sorted by row number. More or less matches the
    scipy.sparse.csr_matrix.

    NamedTuple for easy ingestion by numba.

    Parameters
    ----------
    data: np.ndarray of floats
        The regridding weights.
    indices: np.ndarray of integers
        The column numbers of the CSR format. The linear index into the source
        grid.
    indptr: inp.ndarray of integers
        The row index. Values for row i (target index i) are stored in:
        indices[indptr[i]: indptr[i + 1]]
    n: int
        The number of rows.
    nnz: int
        The number of non-zero values.
    """

    data: FloatArray
    indices: IntArray
    indptr: IntArray
    n: int
    nnz: int


@numba.njit(inline="always")
def nzrange(A: WeightMatrixCSR, column: int) -> Tuple[IntArray, FloatArray]:
    """Return the indices and values of a single row."""
    start = A.indptr[column]
    end = A.indptr[column + 1]
    return A.indices[start:end], A.data[start:end]


def weight_matrix_coo(
    source_index: IntArray,
    target_index: IntArray,
    weight_values: FloatArray,
) -> WeightMatrixCOO:
    return WeightMatrixCOO(
        weight_values,
        target_index,
        source_index,
        weight_values.size,
    )


def weight_matrix_csr(
    source_index: IntArray,
    target_index: IntArray,
    weight_values: FloatArray,
) -> WeightMatrixCSR:
    coo = weight_matrix_coo(source_index, target_index, weight_values)
    return coo_to_csr(coo)


def coo_to_csr(matrix: WeightMatrixCOO) -> WeightMatrixCSR:
    """
    Convert COO matrix to CSR matrix.

    Assumes the COO matrix indices are already sorted by row number!
    """
    i = np.cumsum(np.bincount(matrix.row))
    indptr = np.empty(i.size + 1, dtype=IntDType)
    indptr[0] = 0
    indptr[1:] = i
    return WeightMatrixCSR(
        matrix.data,
        matrix.col,
        indptr,
        indptr.size - 1,
        matrix.data.size,
    )


def csr_to_coo(matrix: WeightMatrixCSR) -> WeightMatrixCOO:
    """
    Convert CSR matrix to COO matrix.

    Expand the indtpr to full row numbers.
    """
    n_repeat = np.diff(matrix.indptr)
    row = np.repeat(np.arange(matrix.n), n_repeat)
    return WeightMatrixCOO(matrix.data, row, matrix.indices, matrix.nnz)
