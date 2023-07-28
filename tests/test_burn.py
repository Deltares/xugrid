import os

os.environ["NUMBA_DISABLE_JIT"] = "1"
import geopandas as gpd
import numpy as np
import pytest
import shapely
from numba_celltree.constants import Point, Triangle

import xugrid as xu
from xugrid.ugrid import burn


@pytest.fixture(scope="function")
def grid():
    """Three by three squares"""
    x = np.arange(0.0, 4.0)
    y = np.arange(0.0, 4.0)
    node_y, node_x = [a.ravel() for a in np.meshgrid(y, x, indexing="ij")]
    nx = ny = 3
    # Define the first vertex of every face, v.
    v = (np.add.outer(np.arange(nx), nx * np.arange(ny)) + np.arange(ny)).T.ravel()
    faces = np.column_stack((v, v + 1, v + nx + 2, v + nx + 1))
    return xu.Ugrid2d(node_x, node_y, -1, faces)


def test_point_in_triangle():
    a = Point(0.1, 0.1)
    b = Point(0.7, 0.5)
    c = Point(0.4, 0.7)
    # Should work for clockwise and ccw orientation.
    triangle = Triangle(a, b, c)
    rtriangle = Triangle(c, b, a)
    p = Point(0.5, 0.5)
    assert burn.point_in_triangle(p, triangle)
    assert burn.point_in_triangle(p, rtriangle)

    p = Point(0.0, 0.0)
    assert not burn.point_in_triangle(p, triangle)
    assert not burn.point_in_triangle(p, rtriangle)


def test_points_in_triangle():
    vertices = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
        ]
    )
    faces = np.array(
        [
            [0, 1, 2],
            [1, 3, 2],
        ]
    )
    points = np.array(
        [
            [-0.5, 0.25],
            [0.0, 0.0],  # on vertex
            [0.5, 0.5],  # on edge
            [0.5, 0.25],
            [1.5, 0.25],
            [2.5, 0.25],
        ]
    )
    face_indices = np.array([0, 0, 0, 0, 1, 1])
    expected = [False, True, True, True, True, False]
    actual = burn.points_in_triangles(
        points=points,
        face_indices=face_indices,
        faces=faces,
        vertices=vertices,
    )
    assert np.array_equal(expected, actual)


def test_locate_polygon(grid):
    polygon = shapely.Polygon(shell=[(0.5, 0.5), (2.5, 0.5), (0.5, 2.5)])
    exterior = shapely.get_coordinates(polygon.exterior)
    interiors = [shapely.get_coordinates(interior) for interior in polygon.interiors]

    actual = burn._locate_polygon(grid, exterior, interiors, all_touched=False)
    assert np.array_equal(np.sort(actual), [0, 1, 2, 3, 4, 6])
    actual = burn._locate_polygon(grid, exterior, interiors, all_touched=True)
    assert np.array_equal(np.sort(actual), [0, 1, 2, 3, 4, 6])

    polygon = shapely.Polygon(shell=[(0.75, 0.5), (2.5, 0.5), (0.75, 2.5)])
    exterior = shapely.get_coordinates(polygon.exterior)
    interiors = [shapely.get_coordinates(interior) for interior in polygon.interiors]

    actual = burn._locate_polygon(grid, exterior, interiors, all_touched=False)
    assert np.array_equal(np.sort(actual), [1, 2, 4])
    actual = burn._locate_polygon(grid, exterior, interiors, all_touched=True)
    assert np.array_equal(np.sort(actual), [0, 1, 2, 3, 4, 5, 6, 7])


def test_locate_polygon_with_hole(grid):
    # The hole omits the centroid at (1.5, 1.5)
    polygon = shapely.Polygon(
        shell=[(0.7, 0.7), (2.3, 0.7), (1.5, 2.3)],
        holes=[[(1.4, 1.6), (1.5, 1.4), (1.6, 1.6)]],
    )
    exterior = shapely.get_coordinates(polygon.exterior)
    interiors = [shapely.get_coordinates(interior) for interior in polygon.interiors]

    actual = burn._locate_polygon(grid, exterior, interiors, all_touched=False)
    assert np.array_equal(actual, [])
    actual = burn._locate_polygon(grid, exterior, interiors, all_touched=True)
    assert np.array_equal(np.unique(actual), [0, 1, 2, 3, 4, 5, 7])


def test_burn_polygons(grid):
    output = np.full(grid.n_face, np.nan)
    values = [0, 1]
    polygons = gpd.GeoSeries(
        [
            shapely.Polygon(shell=[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]),
            shapely.Polygon(
                shell=[
                    (0.0, 2.0),
                    (2.0, 2.0),
                    (2.0, 0.0),
                    (3.0, 0.0),
                    (3.0, 3.0),
                    (0.0, 3.0),
                ]
            ),
        ]
    )
    burn._burn_polygons(polygons, grid, values, all_touched=False, output=output)
    expected = np.array([0, 0, 1, 0, 0, 1, 1, 1, 1])
    assert np.allclose(output, expected)
