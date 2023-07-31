import geopandas as gpd
import numpy as np
import pytest
import shapely
from numba_celltree.constants import Point, Triangle
from shapely.geometry import Polygon

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


@pytest.fixture(scope="function")
def points_and_values():
    xy = np.array(
        [
            [0.5, 0.5],
            [1.5, 0.5],
            [2.5, 2.5],
        ]
    )
    points = gpd.points_from_xy(*xy.T)
    values = np.array([0.0, 1.0, 3.0])
    return points, values


@pytest.fixture(scope="function")
def lines_and_values():
    xy = np.array(
        [
            [0.5, 0.5],
            [2.5, 0.5],
            [1.2, 1.5],
            [1.8, 1.5],
            [0.2, 2.2],
            [0.8, 2.8],
            [1.2, 2.2],
            [1.8, 2.8],
        ]
    )
    indices = np.array([0, 0, 1, 1, 2, 2, 2, 2])
    values = np.array([0, 1, 2])
    lines = gpd.GeoSeries(shapely.linestrings(xy, indices=indices))
    return lines, values


@pytest.fixture(scope="function")
def polygons_and_values():
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
    return polygons, values


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


def test_burn_polygons(grid, polygons_and_values):
    polygons, values = polygons_and_values
    output = np.full(grid.n_face, np.nan)
    burn._burn_polygons(polygons, grid, values, all_touched=False, output=output)
    expected = np.array([0, 0, 1, 0, 0, 1, 1, 1, 1])
    assert np.allclose(output, expected)


def test_burn_points(grid, points_and_values):
    points, values = points_and_values
    output = np.full(grid.n_face, -1.0)

    burn._burn_points(points, grid, values, output=output)
    expected = np.array([0, 1, -1, -1, -1, -1, -1, -1, 3])
    assert np.allclose(output, expected)


def test_burn_lines(grid, lines_and_values):
    lines, values = lines_and_values
    output = np.full(grid.n_face, -1.0)

    burn._burn_lines(lines, grid, values, output=output)
    expected = np.array([0, 0, 0, -1, 1, -1, 2, 2, -1])
    assert np.allclose(output, expected)


def test_burn_vector_geometry__errors(grid, points_and_values):
    with pytest.raises(TypeError, match="gdf must be GeoDataFrame"):
        xu.burn_vector_geometry(0, grid)

    points, values = points_and_values
    gdf = gpd.GeoDataFrame({"values": values}, geometry=points)
    with pytest.raises(TypeError, match="Like must be Ugrid2d, UgridDataArray"):
        xu.burn_vector_geometry(gdf, gdf)

    # This seems like the easiest way to generate a multi-polygon inside a
    # GeoDataFrame, since it won't initialize with a multi-polygon.
    p1 = Polygon([(0, 0), (1, 0), (1, 1)])
    p2 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    p3 = Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])
    gdf = gpd.GeoDataFrame({"values": [0, 0, 0]}, geometry=[p1, p2, p3]).dissolve(
        by="values"
    )
    with pytest.raises(
        TypeError, match="GeoDataFrame contains unsupported geometry types"
    ):
        xu.burn_vector_geometry(gdf, grid)


def test_burn_vector_geometry(
    grid, points_and_values, lines_and_values, polygons_and_values
):
    polygons, poly_values = polygons_and_values
    gdf = gpd.GeoDataFrame({"values": poly_values}, geometry=polygons)
    actual = xu.burn_vector_geometry(gdf, grid)
    assert isinstance(actual, xu.UgridDataArray)
    assert np.allclose(actual.to_numpy(), 1)
    actual = xu.burn_vector_geometry(gdf, grid, all_touched=True)
    assert np.allclose(actual.to_numpy(), 1)

    expected = np.array([0, 0, 1, 0, 0, 1, 1, 1, 1])
    actual = xu.burn_vector_geometry(gdf, grid, column="values")
    assert np.allclose(actual.to_numpy(), expected)

    points, point_values = points_and_values
    lines, line_values = lines_and_values
    line_values += 10
    point_values += 20
    values = np.concatenate([poly_values, line_values, point_values])
    geometry = np.concatenate(
        [polygons.to_numpy(), lines.to_numpy(), points.to_numpy()]
    )
    gdf = gpd.GeoDataFrame({"values": values}, geometry=geometry)
    actual = xu.burn_vector_geometry(gdf, grid, column="values")
    expected = np.array([20.0, 21.0, 10.0, 0.0, 11.0, 1.0, 12.0, 12.0, 23.0])
    assert np.allclose(actual.to_numpy(), expected)

    # All touched should give the same answer for this specific example.
    actual = xu.burn_vector_geometry(gdf, grid, column="values", all_touched=True)
    assert np.allclose(actual.to_numpy(), expected)
