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


def test_network_gridder__init(network, unstructured_grid):
    gridder = xu.NetworkGridder(network, unstructured_grid, method="mean")

    assert isinstance(gridder, xu.NetworkGridder)
    assert gridder._source.ugrid_topology == network.grid
    assert gridder._target.ugrid_topology == unstructured_grid.grid
    assert gridder._weights.n == unstructured_grid.grid.n_face
    assert gridder._weights.m == network.grid.n_edge
    assert gridder._weights.nnz == 8


def test_network_gridder__regrid(network, unstructured_grid):
    gridder = xu.NetworkGridder(network, unstructured_grid, method="mean")
    gridded = gridder.regrid(network)

    assert isinstance(gridded, type(unstructured_grid))
    assert gridded.shape == unstructured_grid.shape
    assert np.count_nonzero(np.isnan(gridded)) == 11

    x_loc = [0.5, 1.5, 2.5, 3.5, 3.5]
    y_loc = [0.5, 1.5, 1.5, 2.5, 0.5]
    diag = 0.5 * np.sqrt(2)
    grid_values = gridded.ugrid.sel_points(x=x_loc, y=y_loc)

    np.testing.assert_allclose(grid_values[0], 1.0)
    np.testing.assert_allclose(grid_values[1], (diag * 1 + 0.5 * 2) / (diag + 0.5))
    np.testing.assert_allclose(
        grid_values[2], (0.5 * 2 + diag * -4 + diag * 4) / (2 * diag + 0.5)
    )
    np.testing.assert_allclose(grid_values[3], -4.0)
    np.testing.assert_allclose(grid_values[4], 4.0)
