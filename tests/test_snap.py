import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely.geometry as sg
import xarray as xr

from xugrid.snapping import snap_nodes, snap_to_grid, snap_to_nodes


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
        coords={"y": [1.5, 0.5], "x": [0.5, 1.5]},
        dims=["y", "x"],
    )
    line = sg.LineString([[0.5, 0.0], [1.5, 2.0]])
    line_gdf = gpd.GeoDataFrame({"resistance": [100.0]}, geometry=[line])
    cell_to_cell, df = snap_to_grid(line_gdf, idomain)
    assert np.array_equal(cell_to_cell, [[0, 1], [2, 3], [2, -1]])
    assert isinstance(df, pd.DataFrame)
    assert np.allclose(df["resistance"], 100.0)

    # Return geometry
    cell_to_cell, df = snap_to_grid(line_gdf, idomain, return_geometry=True)
    assert np.array_equal(cell_to_cell, [[0, 1], [2, 3], [2, -1]])
    assert isinstance(df, gpd.GeoDataFrame)
    assert np.allclose(df["resistance"], 100.0)

    # Test a case where multiple segments occur within a cell; the answer
    # should not change. This is also a degenerate case:
    # 1. Line goes through cell vertices
    # 2. Line starts and stops right below and above cell centroids.
    line = sg.LineString([[0.5, 0.0], [0.75, 0.5], [1.0, 1.0], [1.25, 1.5], [1.5, 2.0]])
    line_gdf = gpd.GeoDataFrame({"resistance": [100.0]}, geometry=[line])
    cell_to_cell, df = snap_to_grid(line_gdf, idomain)

    assert np.array_equal(cell_to_cell, [[2, -1], [2, 3], [0, 1]])
    assert isinstance(df, pd.DataFrame)
    assert np.allclose(df["resistance"], 100.0)

    # Now the same case, but with line rotated to horizontal
    line = sg.LineString([[0.0, 0.5], [0.5, 0.75], [1.0, 1.0], [1.5, 1.25], [2.0, 1.5]])
    line_gdf = gpd.GeoDataFrame({"resistance": [100.0]}, geometry=[line])
    cell_to_cell, df = snap_to_grid(line_gdf, idomain)
    assert np.array_equal(cell_to_cell, [[0, 2], [1, 3]])
