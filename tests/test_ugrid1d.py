from typing import NamedTuple

import geopandas as gpd
import numpy as np
import pyproj
import pytest
import shapely
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


def grid1d(dataset=None, indexes=None, crs=None, attrs=None):
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
        indexes=indexes,
        crs=crs,
        attrs=attrs,
    )
    return grid


def test_ugrid1d_init():
    grid = grid1d()
    assert grid.name == NAME
    assert grid._dataset is None
    assert grid.node_x.flags["C_CONTIGUOUS"]
    assert grid.node_y.flags["C_CONTIGUOUS"]


def test_safe_attrs():
    # .attrs should return a copy
    grid = grid1d()
    assert grid.attrs == grid.attrs
    assert grid._attrs is not grid.attrs


def test_ugrid1d_alternative_init():
    custom_attrs = {
        "node_dimension": "nNetNode",
        "name": "mesh1d",
        "node_coordinates": "mesh1d_node_x mesh1d_node_y",
    }
    indexes = {"node_x": "mesh1d_node_x", "node_y": "mesh1d_node_y"}
    grid = grid1d(indexes=indexes, attrs=custom_attrs)
    assert grid.node_dimension == "nNetNode"
    assert grid.name == NAME
    # name in attrs should be overwritten by given name.
    assert grid._attrs["name"] == NAME

    with pytest.raises(ValueError, match="Provide either dataset or attrs, not both"):
        grid1d(dataset=xr.Dataset, attrs=custom_attrs)

    with pytest.raises(ValueError, match="indexes must be provided for dataset"):
        grid1d(dataset=xr.Dataset, indexes=None)

    with pytest.raises(ValueError, match="indexes must be provided for attrs"):
        grid1d(attrs=custom_attrs)


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
    assert isinstance(grid.node_edge_connectivity, sparse.csr_matrix)
    assert isinstance(grid.node_node_connectivity, sparse.csr_matrix)

    expected_coords = [
        [[0.0, 0.0], [1.0, 1.0]],
        [[1.0, 1.0], [2.0, 2.0]],
    ]
    actual_coords = grid.edge_node_coordinates
    assert actual_coords.shape == (2, 2, 2)
    assert np.allclose(actual_coords, expected_coords)
    assert isinstance(grid.attrs, dict)

    coords = grid.coords
    assert isinstance(coords, dict)
    assert np.array_equal(coords[grid.node_dimension], grid.node_coordinates)
    assert np.array_equal(coords[grid.edge_dimension], grid.edge_coordinates)


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
    def check_attrs(ds):
        attrs = ds[NAME].attrs.copy()
        attrs.pop("cf_role")
        attrs.pop("long_name")
        attrs.pop("topology_dimension")
        ds_contents = tuple(ds.dims) + tuple(ds.coords) + tuple(ds.data_vars)
        for values in attrs.values():
            # e.g node_coordinates are joined by a whitespace.
            for value in values.split(" "):
                assert value in ds_contents

    grid = grid1d()
    ds = grid.to_dataset()
    assert isinstance(ds, xr.Dataset)
    assert f"{NAME}" in ds
    assert f"{NAME}_nNodes" in ds.dims
    assert f"{NAME}_nEdges" in ds.dims
    assert f"{NAME}_node_x" in ds.coords
    assert f"{NAME}_node_y" in ds.coords
    assert f"{NAME}_edge_nodes" in ds
    check_attrs(ds)

    ds = grid.to_dataset(optional_attributes=True)
    assert f"{NAME}_edge_x" in ds.coords
    assert f"{NAME}_edge_y" in ds.coords
    check_attrs(ds)


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
        edge_nodes=np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 0, 0]),
    )

    grid = xugrid.Ugrid1d.from_meshkernel(mesh1d)
    assert grid.n_edge == 8
    assert np.allclose(mesh1d.node_x, grid.node_x)
    assert np.allclose(mesh1d.node_y, grid.node_y)
    assert np.allclose(grid.edge_node_connectivity, mesh1d.edge_nodes.reshape((8, 2)))


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
    assert grid.dimensions == {
        f"{NAME}_nNodes": 3,
        f"{NAME}_nEdges": 2,
    }


@requires_meshkernel
def test_mesh():
    grid = grid1d()
    assert isinstance(grid.mesh, mk.Mesh1d)


@requires_meshkernel
def test_meshkernel():
    grid = grid1d()
    assert isinstance(grid.meshkernel, mk.MeshKernel)


def test_from_shapely():
    with pytest.raises(TypeError):
        xy = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 0.0],
            ]
        )
        xugrid.Ugrid1d.from_shapely(geometry=[shapely.polygons(xy)])

    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 0.0, 0.0])
    grid = xugrid.Ugrid1d.from_shapely(geometry=shapely.linestrings(x, y))
    assert isinstance(grid, xugrid.Ugrid1d)


def test_from_geodataframe():
    with pytest.raises(TypeError, match="Expected GeoDataFrame"):
        xugrid.Ugrid1d.from_geodataframe(1)

    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 0.0, 0.0])
    gdf = gpd.GeoDataFrame(geometry=[shapely.linestrings(x, y)])
    grid = xugrid.Ugrid1d.from_geodataframe(gdf)
    assert isinstance(grid, xugrid.Ugrid1d)


def test_to_shapely():
    grid = grid1d()

    points = grid.to_shapely(f"{NAME}_nNodes")
    assert isinstance(points[0], shapely.Geometry)

    lines = grid.to_shapely(f"{NAME}_nEdges")
    assert isinstance(lines[0], shapely.Geometry)


def test_sel():
    grid = grid1d()
    obj = xr.DataArray(
        data=[0, 1],
        dims=[grid.edge_dimension],
    )
    with pytest.raises(ValueError, match="Ugrid1d only supports slice indexing"):
        grid.sel(obj=obj, x=1.0, y=1.0)
    with pytest.raises(ValueError, match="Ugrid1d does not support steps"):
        grid.sel(obj=obj, x=slice(0, 2, 1), y=slice(0, 2, 1))
    with pytest.raises(ValueError, match="slice start should be smaller"):
        grid.sel(obj, x=slice(2, 0), y=slice(0, 2))

    actual = grid.sel(obj=obj, x=slice(0, 1), y=slice(0, 1))
    assert isinstance(actual, tuple)
    new_obj, new_grid = actual

    assert isinstance(new_obj, xr.DataArray)
    assert isinstance(new_grid, xugrid.Ugrid1d)
    assert new_obj.dims[0] == f"{NAME}_nEdges"
    assert new_grid.edge_dimension == f"{NAME}_nEdges"
    assert np.array_equal(new_obj.values, [0])


def test_sel_points():
    grid = grid1d()
    obj = xr.DataArray(
        data=[0, 1],
        dims=[grid.edge_dimension],
    )
    # For now, this function does nothing so it'll work for multi-topology
    # UgridDatasets.
    actual = grid.sel_points(
        obj=obj, x=None, y=None, out_of_bounds=None, fill_value=None
    )
    assert actual.identical(obj)


def test_topology_subset():
    grid = grid1d()
    edge_indices = np.array([1])
    actual = grid.topology_subset(edge_indices)
    assert np.array_equal(actual.edge_node_connectivity, [[0, 1]])
    assert np.array_equal(actual.node_x, [1.0, 2.0])
    assert np.array_equal(actual.node_y, [1.0, 2.0])


def test_reindex_like():
    grid = grid1d()
    index = np.array([1, 0])
    reordered = grid.topology_subset(index)
    obj = xr.DataArray(index, dims=(reordered.edge_dimension))
    reindexed = reordered.reindex_like(grid, obj=obj)
    assert np.array_equal(reindexed, [0, 1])


def test_ugrid1d_plot():
    grid = grid1d()
    primitive = grid.plot()
    assert isinstance(primitive, LineCollection)


def test_ugrid1d_rename():
    grid = grid1d()
    original_indexes = grid._indexes.copy()
    original_attrs = grid._attrs.copy()

    renamed = grid.rename("__renamed")

    # Check that original is unchanged
    assert grid._attrs == original_attrs
    assert grid._indexes == original_indexes
    assert renamed._attrs == {
        "cf_role": "mesh_topology",
        "long_name": "Topology data of 1D network",
        "topology_dimension": 1,
        "node_dimension": "__renamed_nNodes",
        "edge_dimension": "__renamed_nEdges",
        "edge_node_connectivity": "__renamed_edge_nodes",
        "node_coordinates": "__renamed_node_x __renamed_node_y",
        "edge_coordinates": "__renamed_edge_x __renamed_edge_y",
    }
    assert renamed._indexes == {
        "node_x": "__renamed_node_x",
        "node_y": "__renamed_node_y",
    }
    assert renamed.name == "__renamed"


def test_ugrid1d_rename_with_dataset():
    grid = grid1d()
    grid2 = xugrid.Ugrid1d.from_dataset(grid.to_dataset())
    original_dataset = grid2._dataset.copy()

    renamed2 = grid2.rename("__renamed")
    dataset = renamed2._dataset
    assert grid2._dataset.equals(original_dataset)
    assert sorted(dataset.data_vars) == ["__renamed", "__renamed_edge_nodes"]
    assert sorted(dataset.dims) == ["__renamed_nEdges", "__renamed_nNodes", "two"]
    assert sorted(dataset.coords) == ["__renamed_node_x", "__renamed_node_y"]


def test_topology_sort_by_dfs():
    grid = grid1d()
    vertices = grid.topological_sort_by_dfs()
    assert isinstance(vertices, np.ndarray)
    assert np.array_equal(vertices, [0, 1, 2])


def test_contract_vertices():
    grid = grid1d()
    new = grid.contract_vertices(indices=[0, 2])
    assert isinstance(new, xugrid.Ugrid1d)
    assert new.n_node == 2
    # The nodes have been renumbered
    assert np.array_equal(new.edge_node_connectivity, [[0, 1]])


def test_get_connectivity_matrix():
    xy = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [3.0, 0.0],
        ]
    )
    grid = xugrid.Ugrid1d(
        node_x=xy[:, 0],
        node_y=xy[:, 1],
        fill_value=-1,
        edge_node_connectivity=np.array([[0, 1], [1, 2]]),
    )
    with pytest.raises(ValueError, match="Expected network1d_nNodes; got: abc"):
        grid.get_connectivity_matrix(dim="abc", xy_weights=True)

    connectivity = grid.get_connectivity_matrix(grid.node_dimension, True)
    assert isinstance(connectivity, sparse.csr_matrix)
    assert np.allclose(connectivity.data, [1.5, 1.5, 0.75, 0.75])
    assert np.array_equal(connectivity.indices, [1, 0, 2, 1])


def test_get_coordinates():
    xy = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [3.0, 0.0],
        ]
    )
    grid = xugrid.Ugrid1d(
        node_x=xy[:, 0],
        node_y=xy[:, 1],
        fill_value=-1,
        edge_node_connectivity=np.array([[0, 1], [1, 2]]),
    )
    with pytest.raises(
        ValueError, match="Expected network1d_nNodes or network1d_nEdges; got: abc"
    ):
        grid.get_coordinates(dim="abc")

    assert isinstance(grid.get_coordinates(grid.node_dimension), np.ndarray)
    assert isinstance(grid.get_coordinates(grid.edge_dimension), np.ndarray)


def test_equals():
    grid = grid1d()
    grid_copy = grid1d()
    assert grid.equals(grid)
    assert grid.equals(grid_copy)
    xr_grid = grid.to_dataset()
    assert not grid.equals(xr_grid)
    grid_copy._attrs["attr"] = "something_else"
    # Dataset.identical is called so returns False
    assert not grid.equals(grid_copy)


def test_ugrid1d_create_data_array():
    grid = grid1d()

    uda = grid.create_data_array(np.zeros(grid.n_node), facet="node")
    assert isinstance(uda, xugrid.UgridDataArray)

    uda = grid.create_data_array(np.zeros(grid.n_edge), facet="edge")
    assert isinstance(uda, xugrid.UgridDataArray)

    # Error on facet
    with pytest.raises(ValueError, match="Invalid facet"):
        grid.create_data_array([1, 2, 3], facet="face")

    # Error on on dimensions
    with pytest.raises(ValueError, match="Can only create DataArrays from 1D arrays"):
        grid.create_data_array([[1, 2, 3]], facet="node")

    # Error on size
    with pytest.raises(ValueError, match="Conflicting sizes"):
        grid.create_data_array([1, 2, 3, 4], facet="node")
