import geopandas as gpd
import numpy as np
import pytest
import shapely
import shapely.geometry as sg
import xarray as xr

import xugrid as xu
from xugrid.ugrid.snapping import snap_nodes, snap_to_grid, snap_to_nodes


@pytest.fixture(scope="function")
def structured():
    shape = nlay, nrow, ncol = 3, 9, 9
    dx = 10.0
    dy = -10.0
    xmin = 0.0
    xmax = dx * ncol
    ymin = 0.0
    ymax = abs(dy) * nrow
    dims = ("layer", "y", "x")

    layer = np.arange(1, nlay + 1)
    y = np.arange(ymax, ymin, dy) + 0.5 * dy
    x = np.arange(xmin, xmax, dx) + 0.5 * dx
    coords = {"layer": layer, "y": y, "x": x}

    return xr.DataArray(np.ones(shape, dtype=np.int32), coords=coords, dims=dims)


def test_snap__three_points():
    x = y = np.array([0.0, 1.0, 1.5])
    inv_perm, snap_x, snap_y = snap_nodes(x, y, 0.1)
    assert inv_perm is None
    assert np.array_equal(x, snap_x)
    assert np.array_equal(y, snap_y)

    # hypot(0.5, 0.5) = 0.707...
    inv_perm, snap_x, snap_y = snap_nodes(x, y, 0.71)
    expected_inv_perm = np.array([0, 1, 1])
    expected_x = expected_y = np.array([0.0, 1.25])
    assert np.array_equal(inv_perm, expected_inv_perm)
    assert np.array_equal(snap_x, expected_x)
    assert np.array_equal(snap_y, expected_y)

    # hypot(1, 1) = 1.414...
    inv_perm, snap_x, snap_y = snap_nodes(x, y, 1.42)
    expected_inv_perm = np.array([0, 0, 0])
    assert np.array_equal(inv_perm, expected_inv_perm)
    assert np.allclose(snap_x, np.array([2.5 / 3]))
    assert np.allclose(snap_y, np.array([2.5 / 3]))


def test_snap__two_lines():
    x = np.array([0.0, 1.0, 1.02, 2.0])
    y = np.array([1.0, 0.0, 0.0, 1.0])
    edge_node_connectivity = np.array(
        [
            [0, 1],
            [2, 3],
        ]
    )
    inv_perm, snap_x, snap_y = snap_nodes(x, y, 0.1)
    c = inv_perm[edge_node_connectivity]

    expected_inv_perm = np.array([0, 1, 1, 2])
    expected_x = np.array([0.0, 1.01, 2.0])
    expected_y = np.array([1.0, 0.0, 1.0])
    expected_c = np.array(
        [
            [0, 1],
            [1, 2],
        ]
    )
    assert np.array_equal(inv_perm, expected_inv_perm)
    assert np.array_equal(snap_x, expected_x)
    assert np.array_equal(snap_y, expected_y)
    assert np.array_equal(c, expected_c)


def test_snap_to_nodes():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0])
    to_x = x + 0.1
    to_y = y + 0.1

    # None snapped
    snap_x, snap_y = snap_to_nodes(x, y, to_x, to_y, 0.1)
    assert np.array_equal(snap_x, x)
    assert np.array_equal(snap_y, y)

    # All snapped
    snap_x, snap_y = snap_to_nodes(x, y, to_x, to_y, 0.2)
    assert np.array_equal(snap_x, to_x)
    assert np.array_equal(snap_y, to_y)

    # Ties, no tiebreaker
    with pytest.raises(ValueError):
        snap_x, snap_y = snap_to_nodes(x, y, to_x, to_y, 3.0)

    # Take nearest
    snap_x, snap_y = snap_to_nodes(x, y, to_x, to_y, 3.0, tiebreaker="nearest")
    assert np.array_equal(snap_x, to_x)
    assert np.array_equal(snap_y, to_y)

    # More ties
    to_x = np.array([1.01, 2.01, 2.002, 3.01])
    to_y = np.array([1.01, 2.01, 2.002, 3.01])
    snap_x, snap_y = snap_to_nodes(x, y, to_x, to_y, 0.5, tiebreaker="nearest")
    expected_x = np.array([1.01, 2.002, 3.01])
    expected_y = np.array([1.01, 2.002, 3.01])
    assert np.array_equal(snap_x, expected_x)
    assert np.array_equal(snap_y, expected_y)

    # Exact ties
    to_x = np.array([1.01, 2.002, 2.002, 3.01])
    to_y = np.array([1.01, 2.002, 2.002, 3.01])
    snap_x, snap_y = snap_to_nodes(x, y, to_x, to_y, 0.5, tiebreaker="nearest")
    assert np.array_equal(snap_x, expected_x)
    assert np.array_equal(snap_y, expected_y)

    # Multiple ties
    to_x = np.array([1.01, 2.01, 2.002, 3.002, 3.01])
    to_y = np.array([1.01, 2.01, 2.002, 3.002, 3.01])
    snap_x, snap_y = snap_to_nodes(x, y, to_x, to_y, 0.5, tiebreaker="nearest")
    expected_x = np.array([1.01, 2.002, 3.002])
    expected_y = np.array([1.01, 2.002, 3.002])
    assert np.array_equal(snap_x, expected_x)
    assert np.array_equal(snap_y, expected_y)


def test_snap_to_grid():
    idomain = xr.DataArray(
        data=[[1, 1], [1, 1]],
        coords={"y": [0.5, 1.5], "x": [0.5, 1.5]},
        dims=["y", "x"],
    )
    line = sg.LineString([[0.5, 0.0], [1.5, 2.0]])
    line_gdf = gpd.GeoDataFrame({"resistance": [100.0]}, geometry=[line])
    uds, gdf = snap_to_grid(line_gdf, idomain, 2.0)
    assert isinstance(uds, xu.UgridDataset)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert uds["resistance"].dims == (uds.ugrid.grid.edge_dimension,)
    # TODO test for returned values...


def test_snap_to_grid_with_data(structured):
    # This caused a failure in 0.6.3
    line_x = [2.2, 2.2, 2.2]
    line_y = [82.0, 40.0, 0.0]
    geometry = gpd.GeoDataFrame(
        geometry=[shapely.linestrings(line_x, line_y)], data={"a": [1.0]}
    )

    uds, gdf = snap_to_grid(geometry, structured, max_snap_distance=0.5)
    assert isinstance(uds, xu.UgridDataset)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert uds["a"].dims == (uds.ugrid.grid.edge_dimension,)
    assert uds["a"].notnull().sum() == 8
    assert uds["line_index"].notnull().sum() == 8
    assert uds["line_index"].sum() == 0  # all values should be 0


def test_snap_parallel_linestrings_to_grid(structured):
    line_x1 = [10.2, 10.2, 10.2]
    line_x2 = [30.2, 30.2, 30.2]
    line_y = [82.0, 40.0, 0.0]

    line1 = shapely.linestrings(line_x1, line_y)
    line2 = shapely.linestrings(line_x2, line_y)

    geometry = gpd.GeoDataFrame(geometry=[line1, line2], data={"a": [1.0, 1.0]})

    uds, gdf = snap_to_grid(geometry, structured, max_snap_distance=0.5)
    assert isinstance(uds, xu.UgridDataset)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert uds["a"].dims == (uds.ugrid.grid.edge_dimension,)
    expected_unique_values = np.array([ 0.,  1., np.nan])
    expected_line_counts = np.array([  8,   8, 164])
    actual_unique_values, actual_line_counts = np.unique(uds["line_index"], return_counts=True)
    np.testing.assert_array_equal(expected_unique_values, actual_unique_values)
    np.testing.assert_array_equal(expected_line_counts, actual_line_counts)


def test_snap_series_linestrings_to_grid(structured):
    # This caused a failure up to 0.10.0
    line_x = [40.2, 40.2]
    line_y1 = [82.0, 60.0]
    line_y2 = [60.0, 40.0]
    line_y3 = [40.0, 20.0]
    line_y4 = [20.0, 0.0]
    line1 = shapely.linestrings(line_x, line_y1)
    line2 = shapely.linestrings(line_x, line_y2)
    line3 = shapely.linestrings(line_x, line_y3)
    line4 = shapely.linestrings(line_x, line_y4)
    geometry = gpd.GeoDataFrame(
        geometry=[line1, line2, line3, line4], data={"a": [1.0, 1.0, 1.0, 1.0]}
    )
    uds, gdf = snap_to_grid(geometry, structured, max_snap_distance=0.5)
    assert isinstance(uds, xu.UgridDataset)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert uds["a"].dims == (uds.ugrid.grid.edge_dimension,)
    expected_unique_values = np.array([ 0.,  1., 2., 3., np.nan])
    expected_line_counts = np.array([ 2, 2, 2, 2, 172])
    actual_unique_values, actual_line_counts = np.unique(uds["line_index"], return_counts=True)
    np.testing.assert_array_equal(expected_unique_values, actual_unique_values)
    np.testing.assert_array_equal(expected_line_counts, actual_line_counts)


def test_snap_crossing_linestrings_to_grid(structured):
    # This caused a failure up to 0.10.0
    line_x = [40.2, 40.2, 40.2]
    line_y = [82.0, 40.0, 0.0]
    line1 = shapely.linestrings(line_x, line_y)
    line2 = shapely.linestrings(line_y, line_x)
    geometry = gpd.GeoDataFrame(geometry=[line1, line2], data={"a": [1.0, 2.0]})
    uds, gdf = snap_to_grid(geometry, structured, max_snap_distance=0.5)

    assert isinstance(uds, xu.UgridDataset)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert uds["a"].dims == (uds.ugrid.grid.edge_dimension,)
    expected_unique_values = np.array([ 0.,  1., np.nan])
    expected_line_counts = np.array([  8,   8, 164])
    actual_unique_values, actual_line_counts = np.unique(uds["line_index"], return_counts=True)
    np.testing.assert_array_equal(expected_unique_values, actual_unique_values)
    np.testing.assert_array_equal(expected_line_counts, actual_line_counts)


def test_snap_closely_parallel_linestrings_to_grid(structured):
    """
    Closely parallel lines, are snapped to same edge, the first one should be taken.
    We can use this test to monitor if this behaviour changes.
    """
    line_x1 = [19.0, 19.0, 19.0]
    line_x2 = [21.0, 21.0, 21.0]
    line_y = [82.0, 40.0, 0.0]

    line1 = shapely.linestrings(line_x1, line_y)
    line2 = shapely.linestrings(line_x2, line_y)

    geometry = gpd.GeoDataFrame(geometry=[line1, line2], data={"a": [1.0, 1.0]})

    uds, gdf = snap_to_grid(geometry, structured, max_snap_distance=0.5)
    assert isinstance(uds, xu.UgridDataset)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert uds["a"].dims == (uds.ugrid.grid.edge_dimension,)
    expected_unique_values = np.array([0., np.nan])
    expected_line_counts = np.array([  8, 172])
    actual_unique_values, actual_line_counts = np.unique(uds["line_index"], return_counts=True)
    np.testing.assert_array_equal(expected_unique_values, actual_unique_values)
    np.testing.assert_array_equal(expected_line_counts, actual_line_counts)
