import numpy as np
import pytest
from scipy import sparse
from xugrid import connectivity


@pytest.fixture(scope="function")
def triangle_mesh():
    fill_value = -1
    # Two triangles
    faces = np.array(
        [
            [0, 1, 2],
            [1, 3, 2],
        ]
    )
    return faces, fill_value


@pytest.fixture(scope="function")
def mixed_mesh():
    fill_value = -1
    # Triangle, quadrangle
    faces = np.array(
        [
            [0, 1, 2, fill_value],
            [1, 3, 4, 2],
        ]
    )
    return faces, fill_value


def test_to_ij(triangle_mesh, mixed_mesh):
    faces, fill_value = triangle_mesh
    ac_i, ac_j = connectivity._to_ij(faces, fill_value, invert=False)
    ex_i = [0, 0, 0, 1, 1, 1]
    ex_j = [0, 1, 2, 1, 3, 2]
    assert np.array_equal(ac_i, ex_i)
    assert np.array_equal(ac_j, ex_j)

    # Inverted
    ac_i, ac_j = connectivity._to_ij(faces, fill_value, invert=True)
    assert np.array_equal(ac_i, ex_j)
    assert np.array_equal(ac_j, ex_i)

    faces, fill_value = mixed_mesh
    ac_i, ac_j = connectivity._to_ij(faces, fill_value, invert=False)
    ex_i = [0, 0, 0, 1, 1, 1, 1]
    ex_j = [0, 1, 2, 1, 3, 4, 2]
    assert np.array_equal(ac_i, ex_i)
    assert np.array_equal(ac_j, ex_j)

    # Inverted
    ac_i, ac_j = connectivity._to_ij(faces, fill_value, invert=True)
    assert np.array_equal(ac_i, ex_j)
    assert np.array_equal(ac_j, ex_i)


def test_ragged_index():
    n = 3
    m = 4
    m_per_row = np.array([1, 2, 3])
    actual = connectivity.ragged_index(n, m, m_per_row)
    expected = np.array(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
        ]
    )
    assert np.array_equal(actual, expected)


def test_sparse_dense_conversion_roundtrip(triangle_mesh, mixed_mesh):
    faces, fill_value = triangle_mesh
    sparse = connectivity.to_sparse(faces, fill_value)
    back = connectivity.to_dense(sparse, fill_value)
    # Note: roundtrip does not preserve CW/CCW orientation, since orientation
    # does not apply to node_face_connectivity, but the sorted rows should
    # contain the same elements.
    assert np.array_equal(faces.sort(axis=1), back.sort(axis=1))

    faces, fill_value = mixed_mesh
    sparse = connectivity.to_sparse(faces, fill_value)
    back = connectivity.to_dense(sparse, fill_value)
    assert np.array_equal(faces.sort(axis=1), back.sort(axis=1))


def test_invert_dense(triangle_mesh, mixed_mesh):
    faces, fill_value = triangle_mesh
    actual = connectivity.invert_dense(faces, fill_value)
    expected = np.array(
        [
            [0, -1],  # 0
            [0, 1],  # 1
            [0, 1],  # 2
            [1, -1],  # 3
        ]
    )
    assert np.array_equal(actual, expected)

    faces, fill_value = mixed_mesh
    actual = connectivity.invert_dense(faces, fill_value)
    expected = np.array(
        [
            [0, -1],  # 0
            [0, 1],  # 1
            [0, 1],  # 2
            [1, -1],  # 3
            [1, -1],  # 4
        ]
    )
    assert np.array_equal(actual, expected)


def test_invert_sparse(triangle_mesh, mixed_mesh):
    faces, fill_value = triangle_mesh
    sparse = connectivity.to_sparse(faces, fill_value)
    inverted = connectivity.invert_sparse(sparse)
    actual = connectivity.to_dense(inverted, fill_value)
    expected = np.array(
        [
            [0, -1],  # 0
            [0, 1],  # 1
            [0, 1],  # 2
            [1, -1],  # 3
        ]
    )
    assert np.array_equal(actual, expected)

    faces, fill_value = mixed_mesh
    sparse = connectivity.to_sparse(faces, fill_value)
    inverted = connectivity.invert_sparse(sparse)
    actual = connectivity.to_dense(inverted, fill_value)
    expected = np.array(
        [
            [0, -1],  # 0
            [0, 1],  # 1
            [0, 1],  # 2
            [1, -1],  # 3
            [1, -1],  # 4
        ]
    )
    assert np.array_equal(actual, expected)


def test_renumber():
    a = np.array(
        [
            [0, 1, 2],
            [10, 11, 12],
            [30, 31, 32],
        ]
    )
    actual = connectivity.renumber(a)
    expected = np.array(
        [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
        ]
    )
    assert np.array_equal(actual, expected)

    a = np.array(
        [
            [0, 1, 2],
            [10, 11, 2],
            [30, 31, 2],
        ]
    )
    actual = connectivity.renumber(a)
    expected = np.array(
        [
            [0, 1, 2],
            [3, 4, 2],
            [5, 6, 2],
        ]
    )
    assert np.array_equal(actual, expected)
