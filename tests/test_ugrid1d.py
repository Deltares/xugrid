from typing import NamedTuple

import geopandas as gpd
import numpy as np
import pygeos
import pyproj
import pytest
import xarray as xr
from matplotlib.collections import LineCollection
from scipy import sparse

import xugrid

from . import requires_meshkernel

try:
    import meshkernel as mk
except ImportError:
    pass

NAME = "network1d"


def grid1d(crs=None):
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
        crs=crs,
    )
    return grid


def test_ugrid1d_init():
    grid = grid1d()
    assert grid.name == NAME
    assert grid._dataset is None
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


def test_ugrid1d_egde_bounds():
    grid = grid1d()
    expected = np.array(
        [
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 2.0, 2.0],
        ]
    )
    actual = grid.edge_bounds
    assert actual.shape == (2, 4)
    assert np.allclose(actual, expected)


def test_set_crs():
    grid = grid1d()

    with pytest.raises(ValueError, match="Must pass either"):
        grid.set_crs()

    grid.set_crs("epsg:28992")
    assert grid.crs == pyproj.CRS.from_epsg(28992)

    # This is allowed: the same crs
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


def test_to_dataset():
    grid = grid1d()
    ds = grid.to_dataset()
    assert isinstance(ds, xr.Dataset)
    assert f"{NAME}" in ds
    assert f"{NAME}_nNodes" in ds.dims
    assert f"{NAME}_nEdges" in ds.dims
    assert f"{NAME}_node_x" in ds.coords
    assert f"{NAME}_node_y" in ds.coords
    assert f"{NAME}_edge_nodes" in ds


def test_ugrid1d_dataset_roundtrip():
    grid = grid1d()
    ds = grid.to_dataset()
    grid2 = xugrid.Ugrid1d.from_dataset(grid.to_dataset())
    assert isinstance(grid2._dataset, xr.Dataset)
    assert grid2._dataset == ds


def test_ugrid1d_from_meshkernel():
    class Mesh1d(NamedTuple):
        node_x: np.ndarray
        node_y: np.ndarray
        edge_nodes: np.ndarray

    mesh1d = Mesh1d(
        node_x=np.array(
            [
                0.0,
                0.8975979,
                1.7951958,
                2.6927937,
                3.5903916,
                4.48798951,
                5.38558741,
                6.28318531,
            ]
        ),
        node_y=np.array(
            [
                0.00000000e00,
                7.81831482e-01,
                9.74927912e-01,
                4.33883739e-01,
                -4.33883739e-01,
                -9.74927912e-01,
                -7.81831482e-01,
                -2.44929360e-16,
            ]
        ),
        edge_nodes=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    )

    grid = xugrid.Ugrid1d.from_meshkernel(mesh1d)
    assert grid.n_edge == 7
    assert np.allclose(mesh1d.node_x, grid.node_x)
    assert np.allclose(mesh1d.node_y, grid.node_y)
    assert np.allclose(grid.edge_node_connectivity, mesh1d.edge_nodes.reshape((7, 2)))


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


def test_dimensions():
    grid = grid1d()
    assert grid.node_dimension == f"{NAME}_nNodes"
    assert grid.edge_dimension == f"{NAME}_nEdges"
    assert grid.dimensions == (f"{NAME}_nNodes", f"{NAME}_nEdges")


@requires_meshkernel
def test_mesh():
    grid = grid1d()
    assert isinstance(grid.mesh, mk.Mesh1d)


@requires_meshkernel
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


def test_ugrid1d_plot():
    grid = grid1d()
    primitive = grid.plot()
    assert isinstance(primitive, LineCollection)
