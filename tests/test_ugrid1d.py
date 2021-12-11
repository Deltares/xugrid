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
    xy = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ]
    )
    grid = xugrid.Ugrid1d(
        node_x=xy[:, 0],
        node_y=xy[:, 1],
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
    assert grid.node_x.flags["C_CONTIGUOUS"]
    assert grid.node_y.flags["C_CONTIGUOUS"]


def test_ugrid1d_properties():
    # These are defined in the base class
    grid = grid1d()
    assert grid.node_dimension == f"{NAME}_nNodes"
    assert grid.edge_dimension == f"{NAME}_nEdges"
    assert grid.n_node == 3
    assert grid.n_edge == 2
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
    assert set(actual.data_vars) == set(["a"])


def test_topology_coords():
    grid = grid1d()
    ds = xr.Dataset()
    ds["a"] = xr.DataArray([1, 2, 3], dims=[f"{NAME}_nNodes"])
    ds["b"] = xr.DataArray([1, 2], dims=[f"{NAME}_nEdges"])
    coords = grid.topology_coords(ds)
    assert isinstance(coords, dict)
    assert f"{NAME}_edge_x" in coords
    assert f"{NAME}_edge_y" in coords
    assert f"{NAME}_node_x" in coords
    assert f"{NAME}_node_y" in coords


def test_topology_dataset():
    grid = grid1d()
    ds = grid.topology_dataset()
    assert isinstance(ds, xr.Dataset)
    assert f"{NAME}" in ds
    assert f"{NAME}_nNodes" in ds.dims
    assert f"{NAME}_nEdges" in ds.dims
    assert f"{NAME}_node_x" in ds.coords
    assert f"{NAME}_node_y" in ds.coords
    assert f"{NAME}_edge_nodes" in ds


def test_clear_geometry_properties():
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
    assert grid._get_dimension("node") == f"{NAME}_nNodes"
    assert grid._get_dimension("edge") == f"{NAME}_nEdges"


def test_dimensions():
    grid = grid1d()
    assert grid.node_dimension == f"{NAME}_nNodes"
    assert grid.edge_dimension == f"{NAME}_nEdges"


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

    points = grid.to_pygeos(f"{NAME}_nNodes")
    assert isinstance(points[0], pygeos.Geometry)

    lines = grid.to_pygeos(f"{NAME}_nEdges")
    assert isinstance(lines[0], pygeos.Geometry)


def test_sel():
    grid = grid1d()
    with pytest.raises(ValueError, match="Ugrid1d only supports slice indexing"):
        grid.sel(x=1.0, y=1.0)
    with pytest.raises(ValueError, match="Ugrid1d does not support steps"):
        grid.sel(x=slice(0, 2, 1), y=slice(0, 2, 1))
    with pytest.raises(ValueError, match="slice start should be smaller"):
        grid.sel(x=slice(2, 0), y=slice(0, 2))
    dim, as_ugrid, index, coords = grid.sel(x=slice(0, 1), y=slice(0, 1))
    assert dim == f"{NAME}_nEdges"
    assert as_ugrid
    assert np.allclose(index, [0])
    assert coords == {}


def test_topology_subset():
    grid = grid1d()
    edge_indices = np.array([1])
    actual = grid.topology_subset(edge_indices)
    assert np.array_equal(actual.edge_node_connectivity, [[0, 1]])
    assert np.array_equal(actual.node_x, [1.0, 2.0])
    assert np.array_equal(actual.node_y, [1.0, 2.0])
