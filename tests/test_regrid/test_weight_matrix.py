import numpy as np
import pytest

from xugrid.regrid import weight_matrix


@pytest.fixture(scope="function")
def matrix():
    source_index = np.arange(10)
    target_index = np.repeat(np.arange(5), 2)
    weights = np.full(10, 0.5)
    return weight_matrix.create_weight_matrix(target_index, source_index, weights)


def test_create_weight_matrix(matrix):
    assert isinstance(matrix, weight_matrix.WeightMatrixCSR)
    assert np.array_equal(matrix.indptr, [0, 2, 4, 6, 8, 10])
    assert np.array_equal(matrix.indices, np.arange(10))
    assert np.allclose(matrix.weights, np.full(10, 0.5))
    assert matrix.n == 5
    assert matrix.nnz == 10


def test_nzrange(matrix):
    i, w = weight_matrix.nzrange(matrix, 0)
    assert np.array_equal(i, [0, 1])
    assert np.allclose(w, [0.5, 0.5])
    i, w = weight_matrix.nzrange(matrix, 1)
    assert np.array_equal(i, [2, 3])
    assert np.allclose(w, [0.5, 0.5])
