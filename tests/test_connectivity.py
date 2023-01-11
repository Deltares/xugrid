import numpy as np
import pytest
from scipy import sparse

from xugrid.ugrid import connectivity


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


def test_neighbors():
    i = [0, 0, 0, 1, 1, 1]
    j = [0, 1, 2, 1, 3, 2]
    coo_content = (j, (i, j))
    A = sparse.coo_matrix(coo_content).tocsr()
    A = connectivity.AdjacencyMatrix(A.indices, A.indptr, A.nnz)
    assert np.array_equal(connectivity.neighbors(A, 0), [0, 1, 2])
    assert np.array_equal(connectivity.neighbors(A, 1), [1, 2, 3])


def test_to_ij(triangle_mesh, mixed_mesh):
    faces, fill_value = triangle_mesh
    actual_i, actual_j = connectivity._to_ij(faces, fill_value, invert=False)
    expected_i = [0, 0, 0, 1, 1, 1]
    expected_j = [0, 1, 2, 1, 3, 2]
    assert np.array_equal(actual_i, expected_i)
    assert np.array_equal(actual_j, expected_j)

    # Inverted
    actual_i, actual_j = connectivity._to_ij(faces, fill_value, invert=True)
    assert np.array_equal(actual_i, expected_j)
    assert np.array_equal(actual_j, expected_i)

    faces, fill_value = mixed_mesh
    actual_i, actual_j = connectivity._to_ij(faces, fill_value, invert=False)
    expected_i = [0, 0, 0, 1, 1, 1, 1]
    expected_j = [0, 1, 2, 1, 3, 4, 2]
    assert np.array_equal(actual_i, expected_i)
    assert np.array_equal(actual_j, expected_j)

    # Inverted
    actual_i, actual_j = connectivity._to_ij(faces, fill_value, invert=True)
    assert np.array_equal(actual_i, expected_j)
    assert np.array_equal(actual_j, expected_i)


def test_to_sparse(mixed_mesh):
    faces, fill_value = mixed_mesh
    csr = connectivity._to_sparse(faces, fill_value, invert=False, sort_indices=True)
    expected_j = np.array([0, 1, 2, 1, 2, 3, 4])
    assert np.array_equal(csr.indices, expected_j)
    assert csr.has_sorted_indices

    csr = connectivity._to_sparse(faces, fill_value, invert=False, sort_indices=False)
    expected_j = np.array([0, 1, 2, 1, 3, 4, 2])
    assert np.array_equal(csr.indices, expected_j)
    assert not csr.has_sorted_indices


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


def test_close_polygons(mixed_mesh):
    faces, fill_value = mixed_mesh
    closed, isfill = connectivity.close_polygons(faces, fill_value)
    expected = np.array(
        [
            [0, 1, 2, 0, 0],
            [1, 3, 4, 2, 1],
        ]
    )
    expected_isfill = np.full((2, 5), False)
    expected_isfill[0, -2:] = True
    expected_isfill[1, -1] = True
    assert np.array_equal(closed, expected)
    assert np.array_equal(isfill, expected_isfill)


def test_reverse_orientation(mixed_mesh):
    faces, fill_value = mixed_mesh
    reverse = connectivity.reverse_orientation(faces, fill_value)
    expected = np.array(
        [
            [2, 1, 0, fill_value],
            [2, 4, 3, 1],
        ]
    )
    assert np.array_equal(reverse, expected)


def test_counterclockwise():
    nodes = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 2.0],
        ]
    )
    fill_value = -1

    # Already ccw, nothing should be changed.
    faces = np.array([[0, 2, 3, -1]])
    actual = connectivity.counterclockwise(faces, fill_value, nodes)
    assert np.array_equal(actual, faces)

    # Clockwise with a fill value, reverse.
    faces_cw = np.array([[3, 2, 0, -1]])
    actual = connectivity.counterclockwise(faces_cw, fill_value, nodes)
    assert np.array_equal(actual, faces)

    # Including a hanging node, ccw, nothing changed.
    hanging_ccw = np.array([[0, 1, 2, 3, -1]])
    actual = connectivity.counterclockwise(hanging_ccw, fill_value, nodes)
    assert np.array_equal(actual, hanging_ccw)

    # Including a hanging node, reverse.
    hanging_cw = np.array([[3, 2, 1, 0, -1]])
    actual = connectivity.counterclockwise(hanging_cw, fill_value, nodes)
    assert np.array_equal(actual, hanging_ccw)


def test_edge_connectivity(mixed_mesh):
    faces, fill_value = mixed_mesh
    edge_nodes, face_edges = connectivity.edge_connectivity(faces, fill_value)
    expected_edge_nodes = np.array(
        [
            [0, 1],
            [0, 2],
            [1, 2],
            [1, 3],
            [2, 4],
            [3, 4],
        ]
    )
    expected_face_edges = np.array(
        [
            [0, 2, 1, -1],
            [3, 5, 4, 2],
        ]
    )
    assert np.array_equal(edge_nodes, expected_edge_nodes)
    assert np.array_equal(face_edges, expected_face_edges)


def test_face_face_connectivity():
    edge_faces = np.array(
        [
            [0, -1],
            [0, -1],
            [0, 1],
            [1, -1],
            [1, -1],
            [1, -1],
        ]
    )
    face_face = connectivity.face_face_connectivity(edge_faces, fill_value=-1)
    assert isinstance(face_face, sparse.csr_matrix)
    assert np.array_equal(face_face.indices, [1, 0])
    assert np.array_equal(face_face.indptr, [0, 1, 2])


def test_centroids(mixed_mesh):
    faces, fill_value = mixed_mesh
    nodes = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [2.0, 1.0],
        ]
    )
    actual = connectivity.centroids(faces, fill_value, nodes[:, 0], nodes[:, 1])
    expected = np.array(
        [
            [2.0 / 3.0, 1.0 / 3.0],
            [1.5, 0.5],
        ]
    )
    assert np.allclose(actual, expected)


def test_structured_connectivity():
    active = np.array(
        [
            [True, True, False],
            [True, True, True],
            [True, False, True],
        ]
    )
    A = connectivity.structured_connectivity(active)
    assert A.nnz == 14
    assert np.array_equal(connectivity.neighbors(A, 0), [1, 2])
    assert np.array_equal(connectivity.neighbors(A, 1), [0, 3])
    assert np.array_equal(connectivity.neighbors(A, 2), [0, 3, 5])
    assert np.array_equal(connectivity.neighbors(A, 3), [1, 2, 4])
    assert np.array_equal(connectivity.neighbors(A, 4), [3, 6])
    assert np.array_equal(connectivity.neighbors(A, 5), [2])
    assert np.array_equal(connectivity.neighbors(A, 6), [4])


def test_triangulate(mixed_mesh):
    faces, fill_value = mixed_mesh
    actual_triangles, actual_faces = connectivity.triangulate_dense(faces, fill_value)
    expected_triangles = np.array(
        [
            [0, 1, 2],
            [1, 3, 4],
            [1, 4, 2],
        ]
    )
    expected_faces = np.array([0, 1, 1])
    assert np.array_equal(actual_triangles, expected_triangles)
    assert np.array_equal(actual_faces, expected_faces)

    sparse_faces = connectivity.to_sparse(faces, -1, sort_indices=False).tocoo()
    actual_triangles, actual_faces = connectivity.triangulate_coo(sparse_faces)
    assert np.array_equal(actual_triangles, expected_triangles)
    assert np.array_equal(actual_faces, expected_faces)


def test_binary_erosion():
    i = np.array([0, 1, 1, 2, 2, 3, 3])
    j = np.array([1, 0, 2, 1, 3, 2, 4])
    coo_content = (j, (i, j))
    con = sparse.coo_matrix(coo_content).tocsr()
    a = np.full(5, True)

    actual = connectivity.binary_erosion(con, a)
    assert actual.all()

    exterior = np.array([0, 4])
    actual = connectivity.binary_erosion(con, a, exterior=exterior)
    expected = np.array([False, True, True, True, False])
    assert np.array_equal(actual, expected)
    # Check for mutation
    assert a.all()

    actual = connectivity.binary_erosion(con, a, exterior=exterior, iterations=3)
    assert (~actual).all()

    mask = np.array([False, False, False, True, True])
    actual = connectivity.binary_erosion(
        con, a, exterior=exterior, iterations=3, mask=mask
    )
    assert np.array_equal(actual, mask)

    a = np.array([False, True, True, True, False])
    actual = connectivity.binary_erosion(con, a)
    expected = np.array([False, False, True, False, False])
    assert np.array_equal(actual, expected)


def test_binary_dilation():
    i = np.array([0, 1, 1, 2, 2, 3, 3])
    j = np.array([1, 0, 2, 1, 3, 2, 4])
    coo_content = (j, (i, j))
    con = sparse.coo_matrix(coo_content).tocsr()
    a = np.full(5, False)

    # No change
    actual = connectivity.binary_dilation(con, a)
    assert (~actual).all()

    exterior = np.array([0, 4])
    actual = connectivity.binary_dilation(con, a, exterior=exterior)
    assert (~actual).all()

    actual = connectivity.binary_dilation(con, a, exterior=exterior, border_value=True)
    expected = np.array([True, False, False, False, True])
    assert np.array_equal(actual, expected)
    # Check for mutation
    assert (~a).all()

    actual = connectivity.binary_dilation(
        con, a, exterior=exterior, iterations=3, border_value=True
    )
    assert actual.all()

    mask = np.array([False, False, False, True, True])
    actual = connectivity.binary_dilation(
        con, a, exterior=exterior, iterations=3, mask=mask, border_value=True
    )
    assert np.array_equal(actual, ~mask)

    a = np.array([False, False, True, False, False])
    actual = connectivity.binary_dilation(con, a)
    expected = np.array([False, True, True, True, False])
    assert np.array_equal(actual, expected)
