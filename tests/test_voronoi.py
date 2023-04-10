import numpy as np
import pytest

import xugrid as xu
from xugrid.ugrid import connectivity, voronoi


def rowsort(a):
    return a[np.lexsort(a.T)]


def polygon_area(p):
    area = 0.0
    n = len(p)
    for i in range(n):
        v0 = p[i]
        v1 = p[(i + 1) % n]
        area += v0[0] * v1[1]
        area -= v0[1] * v1[0]
    return area


def mesh_area(vertices, faces):
    area_sum = 0.0
    for face in faces:
        polygon = vertices[face[face != -1]]
        area_sum += polygon_area(polygon)
    return 0.5 * abs(area_sum)


def test_dot_product2d():
    U = np.array([[1.0, 2.0], [3.0, 4.0]])
    V = np.array([[5.0, 6.0], [7.0, 8.0]])
    assert np.allclose(voronoi.dot_product2d(U, V), [17.0, 53.0])


def test_compute_centroid():
    x = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 2.0, 2.0, 1.0])
    y = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0])
    i = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    expected_x = np.array([0.5, 1.5])
    expected_y = np.array([0.5, 1.5])

    actual_x, actual_y = voronoi._centroid_pandas(i, x, y)
    assert np.allclose(actual_x, expected_x)
    assert np.allclose(actual_y, expected_y)

    actual_x, actual_y = voronoi._centroid_scipy(i, x, y)
    assert np.allclose(actual_x, expected_x)
    assert np.allclose(actual_y, expected_y)

    actual_x, actual_y = voronoi.compute_centroid(i, x, y)
    assert np.allclose(actual_x, expected_x)
    assert np.allclose(actual_y, expected_y)


class TestVoronoi:
    """
    Note: Arguably the best way to check these tests is by plotting the
    resulting polygons.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        # Make a simple rectangular mesh of 2 rows, 3 columns
        self.vertices = np.array(
            [
                [0.0, 0.0],  # 0
                [1.0, 0.0],  # 1
                [2.0, 0.0],  # 2
                [3.0, 0.0],  # 3
                [0.0, 1.0],  # 4
                [1.0, 1.0],  # 5
                [2.0, 1.0],  # 6
                [3.0, 1.0],  # 7
                [0.0, 2.0],  # 8
                [1.0, 2.0],  # 9
                [2.0, 2.0],  # 10
                [3.0, 2.0],  # 11
            ]
        )
        self.fill_value = -1
        self.face_node_connectivity = np.array(
            [
                [0, 1, 5, 4],
                [1, 2, 6, 5],
                [2, 3, 7, 6],
                [4, 5, 9, 8],
                [5, 6, 10, 9],
                [6, 7, 11, 10],
            ]
        )
        self.node_face_connectivity = connectivity.invert_dense_to_sparse(
            self.face_node_connectivity, self.fill_value
        )
        (
            self.edge_node_connectivity,
            face_edge_connectivity,
        ) = connectivity.edge_connectivity(
            self.face_node_connectivity,
            self.fill_value,
        )
        self.edge_face_connectivity = connectivity.invert_dense(
            face_edge_connectivity, self.fill_value
        )
        self.centroids = np.array(
            [
                [0.5, 0.5],
                [1.5, 0.5],
                [2.5, 0.5],
                [0.5, 1.5],
                [1.5, 1.5],
                [2.5, 1.5],
            ]
        )
        self.expected_vertices = rowsort(self.centroids)

        self.exterior_vertices = rowsort(
            np.array(
                [
                    [0.0, 0.5],  # left
                    [0.0, 1.5],  # left
                    [3.0, 0.5],  # right
                    [3.0, 1.5],  # right
                    [0.5, 0.0],  # bottom
                    [1.5, 0.0],  # bottom
                    [2.5, 0.0],  # bottom
                    [0.5, 2.0],  # top
                    [1.5, 2.0],  # top
                    [2.5, 2.0],  # top
                ]
            )
        )
        self.additional_vertices = rowsort(
            np.array(
                [
                    [0.0, 0.0],  # 0
                    [1.0, 0.0],  # 1
                    [2.0, 0.0],  # 2
                    [3.0, 0.0],  # 3
                    [0.0, 1.0],  # 4
                    [3.0, 1.0],  # 7
                    [0.0, 2.0],  # 8
                    [1.0, 2.0],  # 9
                    [2.0, 2.0],  # 10
                    [3.0, 2.0],  # 11
                ]
            )
        )

    def test_exterior_centroids(self):
        actual_i, actual_j = voronoi.exterior_centroids(self.node_face_connectivity)
        expected_i = np.array([0, 3, 8, 11])
        expected_j = np.array([0, 2, 3, 5])
        assert np.array_equal(actual_i, expected_i)
        assert np.array_equal(actual_j, expected_j)

    def test_interior_centroids(self):
        actual_i, actual_j = voronoi.interior_centroids(
            self.node_face_connectivity,
            self.edge_face_connectivity,
            self.edge_node_connectivity,
            fill_value=self.fill_value,
        )
        expected_i = np.array([1, 1, 2, 2, 4, 4, 7, 7, 9, 9, 10, 10])
        expected_j = np.array([0, 1, 1, 2, 0, 3, 2, 5, 3, 4, 4, 5])
        assert np.array_equal(actual_i, expected_i)
        assert np.array_equal(actual_j, expected_j)

    def test_exterior_vertices(self):
        _, _, actual_vertices, actual_face, n = voronoi.exterior_vertices(
            self.edge_face_connectivity,
            self.edge_node_connectivity,
            self.fill_value,
            self.vertices,
            self.centroids,
            add_vertices=False,
        )
        assert n == 0
        assert np.allclose(rowsort(actual_vertices), self.exterior_vertices)
        assert np.isin(np.arange(6), actual_face).all()

    def test_voronoi_topology(self):
        vertices, faces, face_i = voronoi.voronoi_topology(
            self.node_face_connectivity,
            self.vertices,
            self.centroids,
        )
        actual_faces = connectivity.to_dense(faces, fill_value=-1)
        expected_faces = np.array(
            [
                [0, 1, 4, 3],
                [1, 2, 5, 4],
            ]
        )
        assert actual_faces.shape == (2, 4)
        assert np.allclose(rowsort(vertices), self.expected_vertices)
        assert np.array_equal(face_i, [0, 1, 2, 3, 4, 5])
        assert np.array_equal(actual_faces, expected_faces)
        assert np.allclose(mesh_area(vertices, actual_faces), 2.0)

    def test_voronoi_topology__add_exterior(self):
        with pytest.raises(
            ValueError, match="must be provided if add_exterior is True"
        ):
            voronoi.voronoi_topology(
                self.node_face_connectivity,
                self.vertices,
                self.centroids,
                add_exterior=True,
            )

        vertices, faces, face_i = voronoi.voronoi_topology(
            self.node_face_connectivity,
            self.vertices,
            self.centroids,
            self.edge_face_connectivity,
            self.edge_node_connectivity,
            self.fill_value,
            add_exterior=True,
        )
        actual_faces = connectivity.to_dense(faces, fill_value=-1)
        expected_vertices = rowsort(
            np.concatenate([self.expected_vertices, self.exterior_vertices])
        )
        assert actual_faces.shape == (12, 4)
        assert np.allclose(rowsort(vertices), expected_vertices)
        assert (face_i != -1).all()
        assert np.allclose(mesh_area(vertices, actual_faces), 5.5)

        vertices, faces, face_i = voronoi.voronoi_topology(
            self.node_face_connectivity,
            self.vertices,
            self.centroids,
            self.edge_face_connectivity,
            self.edge_node_connectivity,
            self.fill_value,
            add_exterior=True,
            add_vertices=True,
        )
        actual_faces = connectivity.to_dense(faces, fill_value=-1)
        expected_vertices = rowsort(
            np.concatenate([expected_vertices, self.additional_vertices])
        )
        # This introduces hanging nodes
        assert actual_faces.shape == (12, 5)
        assert np.allclose(rowsort(vertices), expected_vertices)
        assert (face_i == -1).sum() == 10
        assert np.allclose(mesh_area(vertices, actual_faces), 6.0)


def test_projected_vertices_on_edge():
    """
    For certain triangles, the voronoi projection falls exactly on the edge.
         x
        ---
      -------
    x -- o -- x
         |
         |
         v

    Where:

    * x: the triangle vertices
    * o: the circumcenter
    * -->: the orthogonal voronoi ray

    This results in a centroid which is identical to the voronoi ray edge
    intersection. This will then create a zero length edge, which is obviously
    problematic.
    """
    nodes = np.array(
        [
            [0.0, 0.0],  # 0
            [0.0, 2.0],  # 1
            [2.0, 2.0],  # 2
            [0.0, 2.0],  # 3
            [1.0, 1.0],  # 4
        ]
    )
    faces = np.array(
        [
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],
        ]
    )
    grid = xu.Ugrid2d(nodes[:, 0], nodes[:, 1], -1, faces)
    voronoi_grid = grid.tesselate_circumcenter_voronoi()
    assert voronoi_grid.n_face
