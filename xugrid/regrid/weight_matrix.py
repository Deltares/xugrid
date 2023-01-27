"""
The weight matrix contains the target indices (rows), the source indices
(columns), and the weights. These are stored in compressed sparse row format
(CSR).
"""
from typing import NamedTuple, Tuple

import numba
import numpy as np

from xugrid.constants import FloatArray, IntArray, IntDType


class WeightMatrixCSR(NamedTuple):
    """
    NamedTuple for easy ingestion by numba.

    More or less matches the scipy.sparse.csr_matrix.
    """

    indptr: IntArray
    indices: IntArray
    weights: FloatArray
    n: int
    nnz: int


def create_weight_matrix(
    target_index: IntArray,
    source_index: IntArray,
    weights: FloatArray,
) -> WeightMatrixCSR:
    i = np.cumsum(np.bincount(target_index))
    indptr = np.empty(i.size + 1, dtype=IntDType)
    indptr[0] = 0
    indptr[1:] = i
    return WeightMatrixCSR(
        indptr,
        source_index,
        weights,
        indptr.size - 1,
        source_index.size,
    )


@numba.njit(inline="always")
def nzrange(A: WeightMatrixCSR, row: int) -> Tuple[IntArray, FloatArray]:
    """
    Return the indices and values of a single row
    """
    start = A.indptr[row]
    end = A.indptr[row + 1]
    return A.indices[start:end], A.weights[start:end]
