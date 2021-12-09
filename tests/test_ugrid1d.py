import geopandas as gpd
import meshkernel as mk
import numpy as np
import pygeos
import pytest
import xarray as xr

import xugrid

NAME = xugrid.ugrid.ugrid_io.UGRID1D_DEFAULT_NAME


def grid1d(dataset=None, name=None, crs=None):
    grid = xugrid.Ugrid1d(
        node_x=np.array([0.0, 1.0, 2.0]),
        node_y=np.array([0.0, 1.0, 2.0]),
        fill_value=-1,
        edge_node_connectivity=np.array([[0, 1], [1, 2]]),
        dataset=dataset,
        name=name,
        crs=crs,
    )
    return grid


def test_ugrid1d_init():
    grid = grid1d()
    assert grid.name == NAME
    assert isinstance(grid.dataset, xr.Dataset)


def test_ugrid1d_from_dataset():
    grid = grid1d()
    grid2 = xugrid.Ugrid1d.from_dataset(grid.dataset)
    assert grid.dataset == grid2.dataset


def test_remove_topology():
    grid = grid1d()
    ds = grid.dataset.copy()
    ds["a"] = xr.DataArray(0)
    actual = grid.remove_topology(ds)
    print(actual)
    assert set(actual.data_vars) == set(["a", NAME])


def test_topology_coords():
    grid = grid1d()
    ds = xr.Dataset()
    ds["a"] = xr.DataArray([1, 2, 3], dims=["network1d_nNodes"])
    ds["b"] = xr.DataArray([1, 2], dims=["network1d_nEdges"])
    coords = grid.topology_coords(ds)
    assert isinstance(coords, dict)
    assert "network1d_edge_x" in coords
    assert "network1d_edge_y" in coords
    assert "network1d_node_x" in coords
    assert "network1d_node_y" in coords


def test_topology_dataset():
    grid = grid1d()
    ds = grid.topology_dataset()
    assert isinstance(ds, xr.Dataset)
    name = NAME
    assert f"{name}" in ds
    assert f"{name}_nNodes" in ds.dims
    assert f"{name}_nEdges" in ds.dims
    assert f"{name}_node_x" in ds.coords
    assert f"{name}_node_y" in ds.coords
    assert f"{name}_edge_nodes" in ds


def test_topology_dataset():
    grid = grid1d()
    for attr in [
        "_mesh",
        "_meshkernel",
        "_celltree",
        "_xmin",
        "_xmax",
        "_ymin",
        "_ymax",
        "_edge_x",
        "_edge_y",
    ]:
        setattr(grid, attr, 1)
        grid._clear_geometry_properties()
        assert getattr(grid, attr) is None


def test_topology_dimension():
    grid = grid1d()
    assert grid.topology_dimension == 1


def test_get_dimension():
    grid = grid1d()
    assert grid._get_dimension("node") == "network1d_nNodes"
    assert grid._get_dimension("edge") == "network1d_nEdges"


def test_mesh():
    grid = grid1d()
    assert isinstance(grid.mesh, mk.Mesh1d)


def test_meshkernel():
    grid = grid1d()
    assert isinstance(grid.meshkernel, mk.MeshKernel)


def test_from_geodataframe():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 0.0, 0.0])
    gdf = gpd.GeoDataFrame(geometry=[pygeos.creation.linestrings(x, y)])
    grid = xugrid.Ugrid1d.from_geodataframe(gdf)
    assert isinstance(grid, xugrid.Ugrid1d)


def test_topology_subset():
    grid = grid1d()
    edge_indices = np.array([1])
    actual = grid.topology_subset(edge_indices)
    assert np.array_equal(actual.edge_node_connectivity, [[0, 1]])
    assert np.array_equal(actual.node_x, [1.0, 2.0])
    assert np.array_equal(actual.node_y, [1.0, 2.0])
