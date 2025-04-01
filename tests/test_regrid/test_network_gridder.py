import numpy as np
import pytest
import xarray as xr

import xugrid as xu


@pytest.fixture(scope="function")
def structured_grid():
    dims = ("y", "x")
    y = np.arange(3.5, -0.5, -1.0)
    x = np.arange(0.5, 4.5, 1.0)
    coords = {"y": y, "x": x}

    return xr.DataArray(np.ones((4, 4), dtype=np.int32), coords=coords, dims=dims)


@pytest.fixture(scope="function")
def unstructured_grid(structured_grid):
    return xu.UgridDataArray.from_structured2d(structured_grid)


@pytest.fixture(scope="function")
def network():
    node_xy = np.array(
        [
            [0.0, 0.0],
            [1.5, 1.5],
            [2.5, 1.5],
            [4.0, 0.0],
            [4.0, 3.0],
        ]
    )
    edge_nodes = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [2, 4],
        ]
    )
    ugrid1d = xu.Ugrid1d(*node_xy.T, -1, edge_nodes)
    data = xr.DataArray(
        np.array([1, 2, 4, -4], dtype=float), dims=(ugrid1d.edge_dimension,)
    )
    return xu.UgridDataArray(data, grid=ugrid1d)


@pytest.fixture(scope="function")
def points_to_sample():
    x_loc = np.array([0.5, 1.5, 2.5, 3.5, 3.5])
    y_loc = np.array([0.5, 1.5, 1.5, 2.5, 0.5])
    diag = 0.5 * np.sqrt(2)
    expected_values = np.array(
        [
            1.0,
            (diag * 1 + 0.5 * 2) / (diag + 0.5),  # 1 diagonal edge, 1 horizontal edge
            (0.5 * 2 + diag * -4 + diag * 4)
            / (2 * diag + 0.5),  # 2 diagonal edge, 1 horizontal edge
            -4.0,
            4.0,
        ]
    )

    return x_loc, y_loc, expected_values


def test_network_gridder_init__unstructured(network, unstructured_grid):
    gridder = xu.NetworkGridder(network, unstructured_grid, method="mean")

    assert isinstance(gridder, xu.NetworkGridder)
    assert gridder._source.ugrid_topology == network.grid
    assert gridder._target.ugrid_topology == unstructured_grid.grid
    assert gridder._weights.n == unstructured_grid.grid.n_face
    assert gridder._weights.m == network.grid.n_edge
    assert gridder._weights.nnz == 8


def test_network_gridder_regrid__unstructured(
    network, unstructured_grid, points_to_sample
):
    gridder = xu.NetworkGridder(network, unstructured_grid, method="mean")
    gridded = gridder.regrid(network)

    assert isinstance(gridded, type(unstructured_grid))
    assert gridded.shape == unstructured_grid.shape
    assert np.count_nonzero(np.isnan(gridded)) == 11

    x_loc, y_loc, expected_values = points_to_sample
    grid_values = gridded.ugrid.sel_points(x=x_loc, y=y_loc)

    np.testing.assert_allclose(grid_values, expected_values)


def test_network_gridder_regrid__unstructured_transient(
    network, unstructured_grid, points_to_sample
):
    # Make transient network
    times = [np.datetime64("2022-01-01"), np.datetime64("2023-01-01")]
    time_multiplier = xr.DataArray([1.0, 2.0], dims="time", coords={"time": times})
    network = (network * time_multiplier).transpose(
        "time", network.ugrid.grid.core_dimension
    )

    gridder = xu.NetworkGridder(network, unstructured_grid, method="mean")
    gridded = gridder.regrid(network)

    assert isinstance(gridded, type(unstructured_grid))
    assert np.count_nonzero(np.isnan(gridded)) == 22

    x_loc, y_loc, expected_values = points_to_sample
    grid_values_t0 = gridded.isel(time=0).ugrid.sel_points(x=x_loc, y=y_loc)
    grid_values_t1 = gridded.isel(time=1).ugrid.sel_points(x=x_loc, y=y_loc)

    np.testing.assert_allclose(grid_values_t0, expected_values)
    np.testing.assert_allclose(grid_values_t1, 2 * expected_values)


def test_network_gridder_init__structured(network, structured_grid):
    with pytest.raises(NotImplementedError):
        xu.NetworkGridder(network, structured_grid, method="mean")
