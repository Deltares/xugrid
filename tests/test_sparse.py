import os

import numpy as np
import pytest

from xugrid.core import sparse


def numba_enabled() -> bool:
    return os.environ.get("NUMBA_DISABLE_JIT") != "1"


@pytest.fixture(scope="function")
def coo_matrix():
    source_index = np.arange(10)
    target_index = np.repeat(np.arange(5), 2)
    weights = np.full(10, 0.5)
    return sparse.MatrixCOO.from_triplet(target_index, source_index, weights)


@pytest.fixture(scope="function")
def csr_matrix():
    source_index = np.arange(10)
    target_index = np.repeat(np.arange(5), 2)
    weights = np.full(10, 0.5)
    return sparse.MatrixCSR.from_triplet(target_index, source_index, weights)


def test_weight_matrix_coo(coo_matrix):
    assert isinstance(coo_matrix, sparse.MatrixCOO)
    assert np.allclose(coo_matrix.data, np.full(10, 0.5))
    assert np.array_equal(coo_matrix.row, [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    assert np.array_equal(coo_matrix.col, np.arange(10))
    assert coo_matrix.nnz == 10


def test_weight_matrix_csr(csr_matrix):
    assert isinstance(csr_matrix, sparse.MatrixCSR)
    assert np.allclose(csr_matrix.data, np.full(10, 0.5))
    assert np.array_equal(csr_matrix.indices, np.arange(10))
    assert np.array_equal(csr_matrix.indptr, [0, 2, 4, 6, 8, 10])
    assert csr_matrix.n == 5
    assert csr_matrix.nnz == 10


@pytest.mark.skipif(
    numba_enabled(),
    reason="Numba cannot convert native range_state_int64 to Python object.",
)
def test_nzrange(csr_matrix):
    # These functions work fine if called inside of other numba functions when
    # numba is enabled.
    i = sparse.nzrange(csr_matrix, 0)
    assert np.array_equal(i, range(0, 2))
    i = sparse.nzrange(csr_matrix, 1)
    assert np.array_equal(i, range(2, 4))


@pytest.mark.skipif(
    numba_enabled(),
    reason="Function returns a slice object; python and no-python slices don't mix.",
)
def test_row_slice(csr_matrix):
    # These functions work fine if called inside of other numba functions when
    # numba is enabled.
    assert sparse.row_slice(csr_matrix, 0) == slice(0, 2, None)


@pytest.mark.skipif(
    numba_enabled(),
    reason="Function returns a zip object; python and no-python zips don't mix.",
)
def test_columns_and_values(csr_matrix):
    # These functions work fine if called inside of other numba functions when
    # numba is enabled.
    zipped = sparse.columns_and_values(csr_matrix, sparse.row_slice(csr_matrix, 0))
    result = list(zipped)
    assert result == [(0, 0.5), (1, 0.5)]


def test_coo_to_csr(coo_matrix):
    csr_matrix = coo_matrix.to_csr()
    assert isinstance(csr_matrix, sparse.MatrixCSR)
    assert np.allclose(csr_matrix.data, np.full(10, 0.5))
    assert np.array_equal(csr_matrix.indices, np.arange(10))
    assert np.array_equal(csr_matrix.indptr, [0, 2, 4, 6, 8, 10])
    assert csr_matrix.n == 5
    assert csr_matrix.nnz == 10


def test_csr_to_coo(csr_matrix):
    coo_matrix = csr_matrix.to_coo()
    assert isinstance(coo_matrix, sparse.MatrixCOO)
    assert np.allclose(coo_matrix.data, np.full(10, 0.5))
    assert np.array_equal(coo_matrix.row, [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    assert np.array_equal(coo_matrix.col, np.arange(10))
    assert coo_matrix.nnz == 10


def test_shape():
    source_index = np.arange(10)
    target_index = np.repeat(np.arange(5), 2)
    weights = np.full(10, 0.5)
    matrix = sparse.MatrixCSR.from_triplet(target_index, source_index, weights, n=20)
    assert matrix.n == 20
    assert matrix.m == 10
    matrix = sparse.MatrixCSR.from_triplet(target_index, source_index, weights, m=20)
    assert matrix.n == 5
    assert matrix.m == 20
