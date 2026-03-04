import numpy as np
import pytest
from scipy import sparse

from xugrid.constants import FILL_VALUE
from xugrid.ugrid import connectivity


@pytest.fixture(scope="function")
def triangle_mesh():
    # Two triangles
    faces = np.array(
        [
            [0, 1, 2],
            [1, 3, 2],
        ]
    )
    return faces


@pytest.fixture(scope="function")
def mixed_mesh():
    # Triangle, quadrangle
    faces = np.array(
        [
            [0, 1, 2, FILL_VALUE],
            [1, 3, 4, 2],
        ]
    )
    return faces


def test_argsort_rows():
    with pytest.raises(ValueError, match="Array is not 2D"):
        connectivity.argsort_rows(np.array([3, 2, 1, 0]))

    array = np.array(
        [
            [1, 0],
            [0, 1],
            [2, 2],
            [2, 1],
            [0, 2],
            [2, 0],
        ]
    )
    _, expected = np.unique(array, axis=0, return_index=True)
    actual = connectivity.argsort_rows(array)
    assert np.array_equal(actual, expected)


def test_index_like():
    xy_a = np.array([[0.0, 0.0], [1.0, 1.0]])
    xy_b = np.array([[0.0, 0.0]])
    with pytest.raises(ValueError, match="coordinates do not match in shape"):
        connectivity.index_like(xy_a, xy_b, tolerance=0.0)

    xy_b = np.array([[0.0, 0.0], [1.1, 1.0]])
    with pytest.raises(ValueError, match="coordinates are not identical after sorting"):
        connectivity.index_like(xy_a, xy_b, tolerance=0.0)
    # Now with higher tolerance
    connectivity.index_like(xy_a, xy_b, tolerance=0.2)

    xy_a = np.array([[3.0, 3.0], [1.0, 1.0], [2.0, 2.0], [0.0, 0.0]])
    xy_b = np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0], [2.0, 2.0]])
    actual = connectivity.index_like(xy_a, xy_b, tolerance=0.0)
    expected = [3, 1, 0, 2]
    assert np.array_equal(actual, expected)

    # With tolerance
    xy_a = np.array([[3.0, 3.0001], [1.0, 1.0], [2.0, 2.0], [-0.0001, 0.0]])
    xy_b = np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0], [2.0, 2.0]])
    actual = connectivity.index_like(xy_a, xy_b, tolerance=0.001)
    expected = [3, 1, 0, 2]
    assert np.array_equal(actual, expected)


def test_neighbors():
    i = [0, 0, 0, 1, 1, 1]
    j = [0, 1, 2, 1, 3, 2]
    coo_content = (j, (i, j))
    A = sparse.coo_matrix(coo_content).tocsr()
    n, m = A.shape
    A = connectivity.AdjacencyMatrix(A.indices, A.indptr, A.nnz, n, m)
    assert np.array_equal(connectivity.neighbors(A, 0), [0, 1, 2])
    assert np.array_equal(connectivity.neighbors(A, 1), [1, 2, 3])


def test_to_ij(triangle_mesh, mixed_mesh):
    faces = triangle_mesh
    actual_i, actual_j = connectivity._to_ij(faces, invert=False)
    expected_i = [0, 0, 0, 1, 1, 1]
    expected_j = [0, 1, 2, 1, 3, 2]
    assert np.array_equal(actual_i, expected_i)
    assert np.array_equal(actual_j, expected_j)

    # Inverted
    actual_i, actual_j = connectivity._to_ij(faces, invert=True)
    assert np.array_equal(actual_i, expected_j)
    assert np.array_equal(actual_j, expected_i)

    faces = mixed_mesh
    actual_i, actual_j = connectivity._to_ij(faces, invert=False)
    expected_i = [0, 0, 0, 1, 1, 1, 1]
    expected_j = [0, 1, 2, 1, 3, 4, 2]
    assert np.array_equal(actual_i, expected_i)
    assert np.array_equal(actual_j, expected_j)

    # Inverted
    actual_i, actual_j = connectivity._to_ij(faces, invert=True)
    assert np.array_equal(actual_i, expected_j)
    assert np.array_equal(actual_j, expected_i)


def test_to_sparse(mixed_mesh):
    faces = mixed_mesh
    csr = connectivity._to_sparse(faces, invert=False, sort_indices=True)
    expected_j = np.array([0, 1, 2, 1, 2, 3, 4])
    assert np.array_equal(csr.indices, expected_j)
    assert csr.has_sorted_indices

    csr = connectivity._to_sparse(faces, invert=False, sort_indices=False)
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
    faces = triangle_mesh
    sparse = connectivity.to_sparse(faces)
    back = connectivity.to_dense(sparse)
    # Note: roundtrip does not preserve CW/CCW orientation, since orientation
    # does not apply to node_face_connectivity, but the sorted rows should
    # contain the same elements.
    assert np.array_equal(faces.sort(axis=1), back.sort(axis=1))

    faces = mixed_mesh
    sparse = connectivity.to_sparse(faces)
    back = connectivity.to_dense(sparse)
    assert np.array_equal(faces.sort(axis=1), back.sort(axis=1))


def test_invert_dense(triangle_mesh, mixed_mesh):
    faces = triangle_mesh
    actual = connectivity.invert_dense(faces)
    expected = np.array(
        [
            [0, -1],  # 0
            [0, 1],  # 1
            [0, 1],  # 2
            [1, -1],  # 3
        ]
    )
    assert np.array_equal(actual, expected)

    faces = mixed_mesh
    actual = connectivity.invert_dense(faces)
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
    faces = triangle_mesh
    sparse = connectivity.to_sparse(faces)
    inverted = connectivity.invert_sparse(sparse)
    actual = connectivity.to_dense(inverted)
    expected = np.array(
        [
            [0, -1],  # 0
            [0, 1],  # 1
            [0, 1],  # 2
            [1, -1],  # 3
        ]
    )
    assert np.array_equal(actual, expected)

    faces = mixed_mesh
    sparse = connectivity.to_sparse(faces)
    inverted = connectivity.invert_sparse(sparse)
    actual = connectivity.to_dense(inverted)
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


def test_to_dense(triangle_mesh):
    faces = triangle_mesh
    sparse = connectivity.to_sparse(faces)
    actual = connectivity.to_dense(sparse)
    assert np.array_equal(actual, np.sort(faces, axis=1))

    with pytest.raises(ValueError, match="n_columns 2 is too small"):
        connectivity.to_dense(sparse, n_columns=2)

    # now pad
    actual = connectivity.to_dense(sparse, n_columns=4)
    expected = np.array(
        [
            [0, 1, 2, FILL_VALUE],
            [1, 2, 3, FILL_VALUE],
        ]
    )
    assert np.array_equal(actual, expected)

    # and twice
    actual = connectivity.to_dense(sparse, n_columns=5)
    expected = np.array(
        [
            [0, 1, 2, FILL_VALUE, FILL_VALUE],
            [1, 2, 3, FILL_VALUE, FILL_VALUE],
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


def test_renumber_with_fill_value():
    a = np.array(
        [
            [0, 1, FILL_VALUE],
            [10, 11, 12],
            [30, FILL_VALUE, 32],
        ]
    )
    actual = connectivity.renumber(a)
    expected = np.array(
        [
            [0, 1, -1],
            [2, 3, 4],
            [5, -1, 6],
        ]
    )
    assert np.array_equal(actual, expected)

    a = np.array(
        [
            [0, -1, 2],
            [10, 11, 2],
            [30, -1, 2],
        ]
    )
    actual = connectivity.renumber(a)
    expected = np.array(
        [
            [0, -1, 1],
            [2, 3, 1],
            [4, -1, 1],
        ]
    )
    assert np.array_equal(actual, expected)


def test_close_polygons(mixed_mesh):
    faces = mixed_mesh
    closed, isfill = connectivity.close_polygons(faces)
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
    faces = mixed_mesh
    reverse = connectivity.reverse_orientation(faces)
    expected = np.array(
        [
            [2, 1, 0, FILL_VALUE],
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

    # Already ccw, nothing should be changed.
    faces = np.array([[0, 2, 3, -1]])
    actual = connectivity.counterclockwise(faces, nodes)
    assert np.array_equal(actual, faces)

    # Clockwise with a fill value, reverse.
    faces_cw = np.array([[3, 2, 0, -1]])
    actual = connectivity.counterclockwise(faces_cw, nodes)
    assert np.array_equal(actual, faces)

    # Including a hanging node, ccw, nothing changed.
    hanging_ccw = np.array([[0, 1, 2, 3, -1]])
    actual = connectivity.counterclockwise(hanging_ccw, nodes)
    assert np.array_equal(actual, hanging_ccw)

    # Including a hanging node, reverse.
    hanging_cw = np.array([[3, 2, 1, 0, -1]])
    actual = connectivity.counterclockwise(hanging_cw, nodes)
    assert np.array_equal(actual, hanging_ccw)


def test_edge_connectivity(mixed_mesh):
    faces = mixed_mesh
    edge_nodes, face_edges = connectivity.edge_connectivity(faces)
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


def test_validate_edge_connectivity(mixed_mesh):
    faces = mixed_mesh
    edges = np.array([[0, 1]])
    with pytest.raises(ValueError, match="face_node_connectivity defines 6 edges"):
        connectivity.validate_edge_node_connectivity(faces, edges)

    edges = np.array(
        [
            [0, 1],  # T
            [0, 1],  # F
            [1, 0],  # F
            [0, 2],  # T
            [1, 2],  # T
            [1, 3],  # T
            [2, 4],  # T
            [3, 4],  # T
            [0, 4],  # F
        ]
    )
    actual = connectivity.validate_edge_node_connectivity(faces, edges)
    expected = np.array([True, False, False, True, True, True, True, True, False])
    assert np.array_equal(actual, expected)


def test_node_node_connectivity():
    edge_nodes = np.array(
        [
            [0, 1],  # 0
            [1, 2],  # 1
            [2, 3],  # 2
            [3, 0],  # 3
            [1, 4],  # 4
            [4, 5],  # 5
            [5, 2],  # 6
        ]
    )
    csr = connectivity.node_node_connectivity(edge_nodes)
    coo = csr.tocoo()
    actual = np.column_stack([coo.row, coo.col])
    expected = np.array(
        [
            [0, 1],
            [0, 3],
            [1, 0],
            [1, 2],
            [1, 4],
            [2, 1],
            [2, 3],
            [2, 5],
            [3, 0],
            [3, 2],
            [4, 1],
            [4, 5],
            [5, 2],
            [5, 4],
        ]
    )
    expected_edges = np.array([0, 3, 0, 1, 4, 1, 2, 6, 3, 2, 4, 5, 6, 5])
    assert np.array_equal(actual, expected)
    assert np.array_equal(expected_edges, coo.data)


def test_directed_node_node_connectivity():
    edge_nodes = np.array(
        [
            [0, 1],  # 0
            [1, 2],  # 1
            [2, 3],  # 2
            [3, 0],  # 3
            [1, 4],  # 4
            [4, 5],  # 5
            [5, 2],  # 6
            [0, 6],  # 7
        ]
    )
    csr = connectivity.directed_node_node_connectivity(edge_nodes)
    coo = csr.tocoo()
    actual = np.column_stack([coo.row, coo.col])
    expected = np.array(
        [
            [0, 1],
            [0, 6],
            [1, 2],
            [1, 4],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 2],
        ]
    )
    expected_edges = np.array([0, 7, 1, 4, 2, 3, 5, 6])
    assert np.array_equal(actual, expected)
    assert np.array_equal(expected_edges, coo.data)


class TestEdgeConnectivity:
    r"""
    Test networks:
    
    Simple:

                    3
                   /
                  / 2
       0     1   /
    0 --- 1 --- 2 
                 \
                  \ 3
                   \
                    4 

    And:

                             5
                          4 --- 6
                         /
                        / 3
       0     1     2   /
    0 --- 1 --- 2 --- 3 
                       \
                        \ 4
                         \
                          5 
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.simple_edge_node_connectivity = np.array(
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [2, 4],
            ]
        )
        self.simple_node_edge_connectivity = connectivity.invert_dense_to_sparse(
            self.simple_edge_node_connectivity
        )
        self.edge_node_connectivity = np.array(
            [
                [0, 1],  # edge 0
                [1, 2],  # edge 1
                [2, 3],  # edge 2
                [3, 4],  # edge 3
                [3, 5],  # edge 4
                [4, 6],  # edge 5
            ]
        )
        self.node_edge_connectivity = connectivity.invert_dense_to_sparse(
            self.edge_node_connectivity
        )

    def test_edge_edge_connectivity_simple(self):
        csr = connectivity.edge_edge_connectivity(
            self.simple_edge_node_connectivity, self.simple_node_edge_connectivity
        )
        assert isinstance(csr, sparse.csr_matrix)
        coo = csr.tocoo()
        actual = np.column_stack([coo.row, coo.col])
        expected = np.array(
            [[0, 1], [1, 0], [1, 2], [1, 3], [2, 1], [2, 3], [3, 1], [3, 2]]
        )
        assert np.array_equal(actual, expected)
        # Test through which node the connection is formed.
        assert np.array_equal(coo.data, [1, 1, 2, 2, 2, 2, 2, 2])

    def test_direct_edge_edge_connectivity_simple(self):
        csr = connectivity.directed_edge_edge_connectivity(
            self.simple_edge_node_connectivity, self.simple_node_edge_connectivity
        )
        coo = csr.tocoo()
        actual = np.column_stack([coo.row, coo.col])
        expected = np.array([[0, 1], [1, 2], [1, 3]])
        assert np.array_equal(actual, expected)
        # Test through which node the connection is formed.
        assert np.array_equal(coo.data, [1, 2, 2])

    def test_directed_edge_edge_connectivity(self):
        csr = connectivity.directed_edge_edge_connectivity(
            self.edge_node_connectivity, self.node_edge_connectivity
        )
        assert isinstance(csr, sparse.csr_matrix)

        coo = csr.tocoo()
        actual = np.column_stack([coo.row, coo.col])
        expected = np.array([[0, 1], [1, 2], [2, 3], [2, 4], [3, 5]])
        assert np.array_equal(actual, expected)
        # Test through which node the connection is formed.
        assert np.array_equal(coo.data, [1, 2, 3, 3, 4])


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
    face_face = connectivity.face_face_connectivity(edge_faces, n_face=2)
    assert isinstance(face_face, sparse.csr_matrix)
    assert np.array_equal(face_face.indices, [1, 0])
    assert np.array_equal(face_face.indptr, [0, 1, 2])
    assert np.array_equal(face_face.data, [2, 2])

    # Test two isolated faces
    # Two triangles:
    edge_faces = np.array(
        [
            [0, -1],
            [0, -1],
            [0, -1],
            [1, -1],
            [1, -1],
            [1, -1],
        ]
    )
    face_face = connectivity.face_face_connectivity(edge_faces, n_face=2)
    assert isinstance(face_face, sparse.csr_matrix)
    assert face_face.shape == (2, 2)


def test_centroids(mixed_mesh):
    faces = mixed_mesh
    nodes = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [2.0, 1.0],
        ]
    )
    actual = connectivity.centroids(faces, nodes[:, 0], nodes[:, 1])
    expected = np.array(
        [
            [2.0 / 3.0, 1.0 / 3.0],
            [1.5, 0.5],
        ]
    )
    assert np.allclose(actual, expected)


def test_circumcenters_error(mixed_mesh):
    faces = mixed_mesh
    nodes = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [2.0, 1.0],
        ]
    )
    with pytest.raises(NotImplementedError):
        connectivity.circumcenters(faces, nodes[:, 0], nodes[:, 1])


def test_circumcenters(triangle_mesh):
    faces = triangle_mesh
    nodes = np.array(
        [
            [0.0, 1.0],
            [0.0, 0.0],
            [2.0, 0.0],
            [1.0, -2.0],
        ]
    )
    actual = connectivity.circumcenters(faces, nodes[:, 0], nodes[:, 1])
    expected = np.array([[1.0, 0.5], [1.0, -0.75]])
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


def test_area(mixed_mesh):
    node_x = np.array([0.0, 1.0, 1.0, 2.0, 2.0])
    node_y = np.array([0.0, 0.0, 1.0, 0.0, 1.0])
    actual = connectivity.area(mixed_mesh, node_x, node_y)
    assert np.allclose(actual, [0.5, 1.0])


def test_perimeter(mixed_mesh):
    node_x = np.array([0.0, 1.0, 1.0, 2.0, 2.0])
    node_y = np.array([0.0, 0.0, 1.0, 0.0, 1.0])
    actual = connectivity.perimeter(mixed_mesh, node_x, node_y)
    assert np.allclose(actual, [2.0 + np.sqrt(2), 4.0])


def test_triangulate(mixed_mesh):
    faces = mixed_mesh
    actual_triangles, actual_faces = connectivity.triangulate_dense(faces)
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

    sparse_faces = connectivity.to_sparse(faces, sort_indices=False).tocoo()
    actual_triangles, actual_faces = connectivity.triangulate_coo(sparse_faces)
    assert np.array_equal(actual_triangles, expected_triangles)
    assert np.array_equal(actual_faces, expected_faces)


def test_binary_erosion():
    i = np.array([0, 1, 1, 2, 2, 3, 3])
    j = np.array([1, 0, 2, 1, 3, 2, 4])
    coo_content = (j, (i, j))
    con = sparse.coo_matrix(coo_content).tocsr()
    a = np.full(5, True)

    with pytest.raises(ValueError, match="Binary operations"):
        connectivity.binary_erosion(con, a[np.newaxis, :])

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

    with pytest.raises(ValueError, match="Binary operations"):
        connectivity.binary_dilation(con, a[np.newaxis, :])

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


@pytest.fixture(scope="function")
def dag() -> sparse.csr_matrix:
    #
    #    0──►2──►3
    #    │   ▲
    #    │   │
    #    └──►1
    #
    i = [0, 1, 0, 2]
    j = [1, 2, 2, 3]
    csr = sparse.coo_matrix((j, (i, j)), shape=(4, 4)).tocsr()
    return csr


@pytest.fixture(scope="function")
def cycle() -> sparse.csr_matrix:
    # same as dag fixture, but bidirectional
    i = [0, 1, 0, 2]
    j = [1, 2, 2, 3]
    ij = np.concatenate((i, j))
    ji = np.concatenate((j, i))
    csr = sparse.coo_matrix((ji, (ij, ji)), shape=(4, 4)).tocsr()
    return csr


def test_topological_sort_by_dfs(dag):
    actual = connectivity.topological_sort_by_dfs(dag)
    assert np.array_equal(actual, [0, 1, 2, 3])


def test_topological_sort_by_dfs__cycle_error(cycle):
    with pytest.raises(ValueError, match="The graph contains at least one cycle"):
        connectivity.topological_sort_by_dfs(cycle)


def test_contract_vertices(dag):
    actual = connectivity.contract_vertices(dag, [0, 1, 3])
    expected = np.array([[0, 1], [1, 3]])
    assert np.array_equal(actual, expected)


def test_contract_vertices__cycle_error(cycle):
    with pytest.raises(ValueError, match="The graph contains at least one cycle"):
        connectivity.contract_vertices(cycle, indices=[1, 3])
