import numpy as np
import pytest

from xugrid.core import sparse


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


def test_nzrange(csr_matrix):
    i = sparse.nzrange(csr_matrix, 0)
    assert np.array_equal(i, range(0, 2))
    i = sparse.nzrange(csr_matrix, 1)
    assert np.array_equal(i, range(2, 4))


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
