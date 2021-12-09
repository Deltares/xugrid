import geopandas as gpd
import numpy as np
import pygeos
import pytest

from xugrid import conversion as cv


@pytest.fixture(scope="function")
def line():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 0.0, 0.0])
    edge_node_connectivity = np.array(
        [
            [0, 1],
            [1, 2],
        ]
    )
    return x, y, edge_node_connectivity


@pytest.fixture(scope="function")
def line_gdf():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 0.0, 0.0])
    gdf = gpd.GeoDataFrame(geometry=[pygeos.creation.linestrings(x, y)])
    return gdf


@pytest.fixture(scope="function")
def triangle_mesh():
    x = np.array([0.0, 1.0, 1.0, 2.0])
    y = np.array([0.0, 0.0, 1.0, 0.0])
    fill_value = -1
    # Two triangles
    faces = np.array(
        [
            [0, 1, 2],
            [1, 3, 2],
        ]
    )
    return x, y, faces, fill_value


@pytest.fixture(scope="function")
def mixed_mesh():
    x = np.array([0.0, 1.0, 1.0, 2.0, 2.0])
    y = np.array([0.0, 0.0, 1.0, 0.0, 1.0])
    fill_value = -1
    # Triangle, quadrangle
    faces = np.array(
        [
            [0, 1, 2, fill_value],
            [1, 3, 4, 2],
        ]
    )
    return x, y, faces, fill_value


def test_nodes_geos_roundtrip(line):
    x, y, _ = line
    actual = cv.nodes_to_points(x, y)
    x_back, y_back = cv.points_to_nodes(actual)
    points_back = cv.nodes_to_points(x_back, y_back)
    assert np.array_equal(x, x_back)
    assert np.array_equal(y, y_back)
    assert np.array_equal(actual, points_back)


def test_linestrings_to_edges(line_gdf):
    x, y, segments = cv.linestrings_to_edges(line_gdf.geometry.values)
    assert np.allclose(x, [0.0, 1.0, 2.0])
    assert np.allclose(y, [0.0, 0.0, 0.0])
    assert np.array_equal(segments, [[0, 1], [1, 2]])


def test_edges_geos_roundtrip(line):
    x, y, c = line
    actual = cv.edges_to_linestrings(x, y, c)
    x_back, y_back, c_back = cv.linestrings_to_edges(actual)
    lines_back = cv.edges_to_linestrings(x_back, y_back, c_back)
    assert np.array_equal(x, x_back)
    assert np.array_equal(y, y_back)
    assert np.array_equal(c, c_back)
    assert np.array_equal(actual, lines_back)


# Cannot use fixtures in parametrize:
# https://github.com/pytest-dev/pytest/issues/349
def _faces_geos_roundtrip(mesh):
    x, y, c, fv = mesh
    actual = cv.faces_to_polygons(x, y, c, fv)
    print(actual)
    x_back, y_back, c_back, fv_back = cv.polygons_to_faces(actual)
    polygons_back = cv.faces_to_polygons(x_back, y_back, c_back, fv_back)
    assert np.array_equal(x, x_back)
    assert np.array_equal(y, y_back)
    assert np.array_equal(c, c_back)
    assert fv == fv_back
    assert np.array_equal(actual, polygons_back)


def test_faces_geos_roundtrip__triangle(triangle_mesh):
    _faces_geos_roundtrip(triangle_mesh)


def test_faces_geos_roundtrip__mixed(mixed_mesh):
    _faces_geos_roundtrip(mixed_mesh)
