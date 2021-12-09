import geopandas as gpd
import meshkernel as mk
import numpy as np
import pygeos
import pyproj
import pytest
import xarray as xr
from scipy import sparse

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


def test_ugrid1d_properties():
    # These are defined in the base class
    grid = grid1d()
    assert grid.node_dimension == "network1d_nNodes"
    assert grid.edge_dimension == "network1d_nEdges"
    expected_coords = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    assert np.allclose(grid.node_coordinates, expected_coords)
    assert np.allclose(grid.edge_x, [0.5, 1.5])
    assert np.allclose(grid.edge_y, [0.5, 1.5])
    assert np.allclose(grid.edge_coordinates, np.column_stack([[0.5, 1.5], [0.5, 1.5]]))
    assert grid.bounds == (0.0, 0.0, 2.0, 2.0)
    node_edges = grid.node_edge_connectivity
    assert isinstance(node_edges, sparse.csr_matrix)


def test_set_crs():
    grid = grid1d()

    with pytest.raises(ValueError, match="Must pass either"):
        grid.set_crs()

    grid.set_crs("epsg:28992")
    assert grid.crs == pyproj.CRS.from_epsg(28992)

    # This is allowed
    grid.set_crs("epsg:28992")
    assert grid.crs == pyproj.CRS.from_epsg(28992)

    # This is not allowed ...
    with pytest.raises(ValueError, match="The Ugrid already has a CRS"):
        grid.set_crs("epsg:4326")

    # Unless explicitly set with allow_override
    grid.set_crs("epsg:4326", allow_override=True)
    assert grid.crs == pyproj.CRS.from_epsg(4326)

    # Test espg alternative arg
    grid.crs = None
    grid.set_crs(epsg=28992)
    assert grid.crs == pyproj.CRS.from_epsg(28992)


def test_to_crs():
    grid = grid1d()

    with pytest.raises(ValueError, match="Cannot transform naive geometries"):
        grid.to_crs("epsg:28992")

    grid.set_crs("epsg:4326")

    # Skip reprojection
    same = grid.to_crs("epsg:4326")
    assert np.allclose(same.node_coordinates, grid.node_coordinates)

    reprojected = grid.to_crs("epsg:28992")
    assert reprojected.crs == pyproj.CRS.from_epsg(28992)
    assert (~(grid.node_coordinates == reprojected.node_coordinates)).all()

    # Test inplace
    grid.to_crs("epsg:28992", inplace=True)
    assert np.allclose(reprojected.node_coordinates, grid.node_coordinates)

    # Test epsg alternative arg
    grid.to_crs(epsg=4326, inplace=True)
    assert grid.crs == pyproj.CRS.from_epsg(4326)


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


def test_to_pygeos():
    grid = grid1d()

    points = grid.to_pygeos("network1d_nNodes")
    assert isinstance(points[0], pygeos.Geometry)

    lines = grid.to_pygeos("network1d_nEdges")
    assert isinstance(lines[0], pygeos.Geometry)


def test_topology_subset():
    grid = grid1d()
    edge_indices = np.array([1])
    actual = grid.topology_subset(edge_indices)
    assert np.array_equal(actual.edge_node_connectivity, [[0, 1]])
    assert np.array_equal(actual.node_x, [1.0, 2.0])
    assert np.array_equal(actual.node_y, [1.0, 2.0])
