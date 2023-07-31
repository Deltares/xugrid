import geopandas as gpd
import numpy as np
import pytest
import xarray as xr

import xugrid as xu


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


def test_polygonize__errors(grid):
    uda = xu.UgridDataArray(
        xr.DataArray(np.ones(grid.n_edge), dims=[grid.edge_dimension]), grid=grid
    )
    with pytest.raises(ValueError, match="Cannot polygonize non-face dimension"):
        xu.polygonize(uda)

    uda = xu.UgridDataArray(
        xr.DataArray(np.ones((3, grid.n_face)), dims=["layer", grid.face_dimension]),
        grid=grid,
    )
    with pytest.raises(ValueError, match="Cannot polygonize non-face dimension"):
        xu.polygonize(uda)


def test_polygonize(grid):
    a = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0])
    uda = xu.UgridDataArray(xr.DataArray(a, dims=grid.face_dimension), grid)
    actual = xu.polygonize(uda)
    assert isinstance(actual, gpd.GeoDataFrame)
    assert len(actual) == 3

    # With a hole in the 1-valued polygon.
    a = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1])
    uda = xu.UgridDataArray(xr.DataArray(a, dims=grid.face_dimension), grid)
    actual = xu.polygonize(uda)
    assert isinstance(actual, gpd.GeoDataFrame)
    assert len(actual) == 2
