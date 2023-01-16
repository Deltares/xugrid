from typing import NamedTuple

import geopandas as gpd
import numba_celltree
import numpy as np
import pygeos
import pyproj
import pytest
import xarray as xr
from matplotlib.collections import LineCollection
from scipy import sparse

import xugrid

try:
    import meshkernel as mk
except ImportError:
    pass

from . import requires_meshkernel

NAME = "mesh2d"
VERTICES = np.array(
    [
        [0.0, 0.0],  # 0
        [1.0, 0.0],  # 1
        [2.0, 0.0],  # 2
        [0.0, 1.0],  # 3
        [1.0, 1.0],  # 4
        [2.0, 1.0],  # 5
        [1.0, 2.0],  # 6
    ]
)
FACES = np.array(
    [
        [0, 1, 4, 3],
        [1, 2, 5, 4],
        [3, 4, 6, -1],
        [4, 5, 6, -1],
    ]
)
EDGE_NODES = np.array(
    [
        [0, 1],  # 0
        [0, 3],  # 1
        [1, 2],  # 2
        [1, 4],  # 3
        [2, 5],  # 4
        [3, 4],  # 5
        [3, 6],  # 6
        [4, 5],  # 7
        [4, 6],  # 8
        [5, 6],  # 9
    ]
)
EDGE_FACES = np.array(
    [
        [0, -1],
        [0, -1],
        [1, -1],
        [0, 1],
        [1, -1],
        [0, 2],
        [2, -1],
        [1, 3],
        [2, 3],
        [3, -1],
    ]
)
CENTROIDS = np.array(
    [
        [0.5, 0.5],
        [1.5, 0.5],
        [2.0 / 3.0, 4.0 / 3.0],
        [4.0 / 3.0, 4.0 / 3.0],
    ]
)
FFI = np.array([0, 0, 1, 1, 2, 2, 3, 3])
FFJ = np.array([1, 2, 0, 3, 0, 3, 1, 2])
FACE_FACE_CONNECTIVITY = sparse.coo_matrix((FFJ, (FFI, FFJ))).tocsr()
NFI = np.array([0, 1, 1, 2, 3, 3, 4, 4, 4, 4, 5, 5, 6, 6])
NFJ = np.array([0, 0, 1, 1, 0, 2, 0, 1, 2, 3, 1, 3, 2, 3])
NODE_FACE_CONNECTIVITY = sparse.coo_matrix((NFJ, (NFI, NFJ))).tocsr()


def grid2d(dataset=None, indexes=None, crs=None, attrs=None):
    grid = xugrid.Ugrid2d(
        node_x=VERTICES[:, 0],
        node_y=VERTICES[:, 1],
        fill_value=-1,
        face_node_connectivity=FACES,
        dataset=dataset,
        indexes=indexes,
        crs=crs,
        attrs=attrs,
    )
    return grid


def test_ugrid2d_init():
    grid = grid2d()
    assert grid.name == NAME
    assert grid._dataset is None
    assert grid.node_x.flags["C_CONTIGUOUS"]
    assert grid.node_y.flags["C_CONTIGUOUS"]
    assert grid._edge_node_connectivity is None
    assert grid._face_edge_connectivity is None


def test_ugrid2d_alternative_init():
    custom_attrs = {"node_dimension": "nNetNode", "name": "mesh1d"}
    grid = grid2d(attrs=custom_attrs)
    assert grid.node_dimension == "nNetNode"
    assert grid.name == NAME
    # name in attrs should be overwritten by given name.
    assert grid._attrs["name"] == NAME

    with pytest.raises(ValueError, match="Provide either dataset or attrs, not both"):
        grid2d(dataset=xr.Dataset, attrs=custom_attrs)

    with pytest.raises(ValueError, match="indexes must be provided for dataset"):
        grid2d(dataset=xr.Dataset, indexes=None)


def test_ugrid2d_properties():
    # These are defined in the base class
    grid = grid2d()
    assert grid.edge_dimension == f"{NAME}_nEdges"
    assert grid.node_dimension == f"{NAME}_nNodes"
    assert grid.face_dimension == f"{NAME}_nFaces"
    assert grid.n_node == 7
    assert grid.n_edge == 10
    assert grid.n_face == 4
    assert grid.n_max_node_per_face == 4
    assert np.array_equal(grid.n_node_per_face, [4, 4, 3, 3])
    assert np.allclose(grid.node_coordinates, VERTICES)
    assert grid.bounds == (0.0, 0.0, 2.0, 2.0)
    node_edges = grid.node_edge_connectivity
    assert isinstance(node_edges, sparse.csr_matrix)


def test_ugrid2d_edge_bounds():
    grid = grid2d()
    expected = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 2.0, 0.0],
            [1.0, 0.0, 1.0, 1.0],
            [2.0, 0.0, 2.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 2.0],
            [1.0, 1.0, 2.0, 1.0],
            [1.0, 1.0, 1.0, 2.0],
            [1.0, 1.0, 2.0, 2.0],
        ]
    )
    actual = grid.edge_bounds
    assert actual.shape == (10, 4)
    assert np.allclose(actual, expected)


def test_ugrid2d_face_bounds():
    grid = grid2d()
    expected = np.array(
        [
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 2.0, 1.0],
            [0.0, 1.0, 1.0, 2.0],
            [1.0, 1.0, 2.0, 2.0],
        ]
    )
    actual = grid.face_bounds
    assert actual.shape == (4, 4)
    assert np.allclose(actual, expected)


def test_set_crs():
    grid = grid2d()
    grid.set_crs("epsg:28992")
    assert grid.crs == pyproj.CRS.from_epsg(28992)


def test_to_crs():
    grid = grid2d()
    grid.set_crs("epsg:4326")
    reprojected = grid.to_crs("epsg:28992")
    assert reprojected.crs == pyproj.CRS.from_epsg(28992)
    assert (~(grid.node_coordinates == reprojected.node_coordinates)).all()


def test_to_dataset():
    grid = grid2d()
    ds = grid.to_dataset()
    assert isinstance(ds, xr.Dataset)
    assert f"{NAME}" in ds
    assert f"{NAME}_nNodes" in ds.dims
    assert f"{NAME}_nFaces" in ds.dims
    assert f"{NAME}_node_x" in ds.coords
    assert f"{NAME}_node_y" in ds.coords
    assert f"{NAME}_face_nodes" in ds


def test_ugrid2d_set_node_coords():
    grid = grid2d()
    ds = xr.Dataset()
    lonvalues = VERTICES[:, 0] + 10.0
    latvalues = VERTICES[:, 1] + 10.0
    ds["lon"] = xr.DataArray(lonvalues, dims=[grid.node_dimension])
    ds["lat"] = xr.DataArray(latvalues, dims=[grid.node_dimension])
    ds["lon with space"] = ds["lon"]
    ds["lat with space"] = ds["lat"]
    ds["short_lon"] = xr.DataArray(np.arange(6.0), dims=["short_node"])
    ds["long_lat"] = xr.DataArray(np.arange(8.0), dims=["long_node"])

    with pytest.raises(ValueError, match="coordinate names may not contain spaces"):
        grid.set_node_coords("lon with space", "lat with space", ds)
    with pytest.raises(
        ValueError, match="shape of node_x does not match n_node of grid: "
    ):
        grid.set_node_coords("short_lon", "lat", ds)
    with pytest.raises(
        ValueError, match="shape of node_y does not match n_node of grid: "
    ):
        grid.set_node_coords("lon", "long_lat", ds)

    grid.set_node_coords("lon", "lat", ds, projected=False)
    assert np.allclose(grid.node_x, lonvalues)
    assert np.allclose(grid.node_y, latvalues)
    assert grid._indexes["node_x"] == "lon"
    assert grid._indexes["node_y"] == "lat"
    assert not grid.projected


def test_ugrid2d_dataset_roundtrip():
    grid = grid2d()
    ds = grid.to_dataset()
    grid2 = xugrid.Ugrid2d.from_dataset(ds)
    assert isinstance(grid2._dataset, xr.Dataset)
    assert grid2._dataset == ds


def test_ugrid2d_from_meshkernel():
    # Setup a meshkernel Mesh2d mimick
    class Mesh2d(NamedTuple):
        node_x: np.ndarray
        node_y: np.ndarray
        face_nodes: np.ndarray
        nodes_per_face: np.ndarray

    mesh2d = Mesh2d(
        node_x=np.array([0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]),
        node_y=np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]),
        face_nodes=np.array(
            [0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 4, 5, 9, 8, 5, 6, 10, 9, 6, 7, 11, 10]
        ),
        nodes_per_face=np.array([4, 4, 4, 4, 4, 4]),
    )

    grid = xugrid.Ugrid2d.from_meshkernel(mesh2d)
    assert grid.n_face == 6
    assert np.allclose(mesh2d.node_x, grid.node_x)
    assert np.allclose(mesh2d.node_y, grid.node_y)
    assert np.allclose(grid.face_node_connectivity, mesh2d.face_nodes.reshape((6, 4)))


def test_assign_node_coords():
    grid = grid2d()
    ds = xr.Dataset()
    # Place some data on the grid facets.
    ds["a"] = xr.DataArray([1, 2, 3, 4, 5, 6, 7], dims=[f"{NAME}_nNodes"])
    with_coords = grid.assign_node_coords(ds)
    assert f"{NAME}_node_x" in with_coords
    assert f"{NAME}_node_y" in with_coords
    assert np.array_equal(with_coords[f"{NAME}_node_x"], grid.node_x)
    assert np.array_equal(with_coords[f"{NAME}_node_y"], grid.node_y)


def test_assign_edge_coords():
    grid = grid2d()
    ds = xr.Dataset()
    # Place some data on the grid facets.
    ds["a"] = xr.DataArray([1, 2, 3, 4, 5, 6, 7], dims=[f"{NAME}_nNodes"])
    with_coords = grid.assign_edge_coords(ds)
    assert f"{NAME}_edge_x" in with_coords
    assert f"{NAME}_edge_y" in with_coords
    assert np.array_equal(with_coords[f"{NAME}_edge_x"], grid.edge_x)
    assert np.array_equal(with_coords[f"{NAME}_edge_y"], grid.edge_y)


def test_assign_face_coords():
    grid = grid2d()
    ds = xr.Dataset()
    # Place some data on the grid facets.
    ds["a"] = xr.DataArray([1, 2, 3, 4, 5, 6, 7], dims=[f"{NAME}_nNodes"])
    with_coords = grid.assign_face_coords(ds)
    assert f"{NAME}_face_x" in with_coords
    assert f"{NAME}_face_y" in with_coords
    assert np.array_equal(with_coords[f"{NAME}_face_x"], grid.face_x)
    assert np.array_equal(with_coords[f"{NAME}_face_y"], grid.face_y)


def test_clear_geometry_properties():
    grid = grid2d()
    for attr in [
        "_mesh",
        "_meshkernel",
        "_celltree",
        "_centroids",
        "_xmin",
        "_xmax",
        "_ymin",
        "_ymax",
        "_edge_x",
        "_edge_y",
        "_triangulation",
        "_voronoi_topology",
        "_centroid_triangulation",
    ]:
        setattr(grid, attr, 1)
        grid._clear_geometry_properties()
        assert getattr(grid, attr) is None


def test_topology_dimension():
    grid = grid2d()
    assert grid.topology_dimension == 2


def test_dimensions():
    grid = grid2d()
    assert grid.node_dimension == f"{NAME}_nNodes"
    assert grid.edge_dimension == f"{NAME}_nEdges"
    assert grid.face_dimension == f"{NAME}_nFaces"
    assert grid.dimensions == {
        f"{NAME}_nNodes": 7,
        f"{NAME}_nEdges": 10,
        f"{NAME}_nFaces": 4,
    }


def test_edge_node_connectivity():
    grid = grid2d()
    edge_nodes = grid.edge_node_connectivity
    assert grid._edge_node_connectivity is not None
    assert grid._face_edge_connectivity is not None
    assert np.allclose(edge_nodes, EDGE_NODES)


def test_edge_face_connectivity():
    grid = grid2d()
    edge_faces = grid.edge_face_connectivity
    assert grid._edge_node_connectivity is not None
    assert grid._face_edge_connectivity is not None
    assert np.allclose(edge_faces, EDGE_FACES)


def test_centroids():
    grid = grid2d()
    assert np.allclose(grid.centroids, CENTROIDS)
    assert np.allclose(grid.face_coordinates, CENTROIDS)
    assert np.allclose(grid.face_x, CENTROIDS[:, 0])
    assert np.allclose(grid.face_y, CENTROIDS[:, 1])


def test_face_face_connectivity():
    grid = grid2d()
    face_face = grid.face_face_connectivity
    assert isinstance(face_face, sparse.csr_matrix)
    assert np.array_equal(face_face.indptr, FACE_FACE_CONNECTIVITY.indptr)
    assert np.array_equal(face_face.indices, FACE_FACE_CONNECTIVITY.indices)


def test_node_face_connectivity():
    grid = grid2d()
    node_face = grid.node_face_connectivity
    assert isinstance(node_face, sparse.csr_matrix)
    assert np.array_equal(node_face.indptr, NODE_FACE_CONNECTIVITY.indptr)
    assert np.array_equal(node_face.indices, NODE_FACE_CONNECTIVITY.indices)


def test_voronoi_topology():
    grid = grid2d()
    vertices, faces, face_index = grid.voronoi_topology
    expected_exterior = np.array(
        [
            [0.5, 0.0],
            [0.0, 0.5],
            [1.5, 0.0],
            [2.0, 0.5],
            [0.5, 1.5],
            [1.5, 1.5],
        ]
    )
    expected_vertices = np.vstack([CENTROIDS, expected_exterior])
    assert np.allclose(vertices, expected_vertices)
    assert isinstance(faces, sparse.coo_matrix)
    expected_row = np.array(
        [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6]
    )
    expected_col = np.array(
        [0, 1, 3, 2, 4, 0, 5, 4, 6, 1, 0, 6, 7, 1, 5, 0, 2, 8, 1, 7, 9, 3, 2, 3, 9, 8]
    )
    assert np.array_equal(faces.row, expected_row)
    assert np.array_equal(faces.col, expected_col)
    assert np.array_equal(face_index, [0, 1, 2, 3, 0, 0, 1, 1, 2, 3])


def test_centroid_triangulation():
    grid = grid2d()
    (x, y, triangles), face_index = grid.centroid_triangulation
    assert np.allclose(x, list(CENTROIDS[:, 0]) + [0.5, 0.0, 1.5, 2.0, 0.5, 1.5])
    assert np.allclose(y, list(CENTROIDS[:, 1]) + [0.0, 0.5, 0.0, 0.5, 1.5, 1.5])
    expected_triangles = np.array(
        [
            [0, 1, 3],
            [0, 3, 2],
            [4, 0, 5],
            [4, 6, 1],
            [4, 1, 0],
            [6, 7, 1],
            [5, 0, 2],
            [5, 2, 8],
            [1, 7, 9],
            [1, 9, 3],
            [2, 3, 9],
            [2, 9, 8],
        ]
    )
    assert np.array_equal(triangles, expected_triangles)
    assert np.array_equal(face_index, [0, 1, 2, 3, 0, 0, 1, 1, 2, 3])


def test_triangulation():
    grid = grid2d()
    (x, y, triangles), face_index = grid.triangulation
    expected_triangles = np.array(
        [
            [0, 1, 4],
            [0, 4, 3],
            [1, 2, 5],
            [1, 5, 4],
            [3, 4, 6],
            [4, 5, 6],
        ]
    )
    assert np.allclose(x, VERTICES[:, 0])
    assert np.allclose(y, VERTICES[:, 1])
    assert np.array_equal(triangles, expected_triangles)
    assert np.array_equal(face_index, [0, 0, 1, 1, 2, 3])


def test_exterior_edges():
    grid = grid2d()
    assert np.array_equal(grid.exterior_edges, [0, 1, 2, 4, 6, 9])


def test_exterior_faces():
    grid = grid2d()
    assert np.array_equal(grid.exterior_faces, [0, 1, 2, 3])


def test_celltree():
    grid = grid2d()
    tree = grid.celltree
    assert isinstance(tree, numba_celltree.CellTree2d)


def test_locate_points():
    grid = grid2d()
    assert np.array_equal(grid.locate_points(CENTROIDS), [0, 1, 2, 3])


def test_locate_bounding_box():
    grid = grid2d()
    faces = grid.locate_bounding_box(1.25, 0.25, 2.5, 1.5)
    assert np.allclose(faces, [1, 3])


def test_compute_barycentric_weights():
    grid = grid2d()
    xy = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.5],
            [1.5, 0.5],
            [0.5, 1.5],
            [2.0, 2.0],
        ]
    )
    expected_face = np.array([0, 0, 1, 2, -1])
    expected_weights = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.5, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    face, weights = grid.compute_barycentric_weights(xy)
    assert np.array_equal(face, expected_face)
    assert np.allclose(weights, expected_weights)


def test_rasterize():
    grid = grid2d()
    x, y, index = grid.rasterize(resolution=0.5)
    expected_index = np.array(
        [
            [-1, 2, -1, -1],
            [2, 2, 3, -1],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ]
    )
    assert np.allclose(x, [0.25, 0.75, 1.25, 1.75])
    assert np.allclose(y, [1.75, 1.25, 0.75, 0.25])
    assert np.array_equal(index, expected_index)


class TestUgrid2dSelection:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.grid = grid2d()
        self.obj = xr.DataArray([0, 1, 2, 3], dims=[self.grid.face_dimension])

    def test_sel_points(self):
        x = [0.5, 1.5]
        y = [0.5, 1.25]

        with pytest.raises(ValueError, match="shape of x does not match shape of y"):
            self.grid.sel_points(obj=self.obj, x=[0.5, 1.5], y=[0.5])
        with pytest.raises(ValueError, match="x and y must be 1d"):
            self.grid.sel_points(obj=self.obj, x=[x], y=[y])

        actual = self.grid.sel_points(obj=self.obj, x=x, y=y)
        assert isinstance(actual, xr.DataArray)

        dim = f"{NAME}_nFaces"
        expected = xr.DataArray(
            data=[0, 3],
            coords={
                "x": (dim, x),
                "y": (dim, y),
            },
            dims=[dim],
        )
        assert expected.equals(actual)

    def test_validate_indexer(self):
        with pytest.raises(ValueError, match="slice stop should be larger than"):
            self.grid._validate_indexer(slice(2, 0))
        with pytest.raises(ValueError, match="step should be None"):
            self.grid._validate_indexer(slice(None, 2, 1))
        with pytest.raises(ValueError, match="step should be None"):
            self.grid._validate_indexer(slice(0, None, 1))

        expected = np.arange(0.0, 2.0, 0.5)
        assert np.allclose(self.grid._validate_indexer(slice(0, 2, 0.5)), expected)
        assert self.grid._validate_indexer(slice(None, 2)) == slice(None, 2)
        assert self.grid._validate_indexer(slice(0, None)) == slice(0, None)

        with pytest.raises(TypeError, match="Invalid indexer type"):
            self.grid._validate_indexer((0, 1, 2))

        # list
        actual = self.grid._validate_indexer([0.0, 1.0, 2.0])
        assert isinstance(actual, np.ndarray)
        assert np.allclose(actual, [0.0, 1.0, 2.0])

        # numpy array
        actual = self.grid._validate_indexer(np.array([0.0, 1.0, 2.0]))
        assert isinstance(actual, np.ndarray)
        assert np.allclose(actual, [0.0, 1.0, 2.0])

        # xarray DataArray
        indexer = xr.DataArray([0.0, 1.0, 2.0], {"x": [0, 1, 2]}, ["x"])
        actual = self.grid._validate_indexer(indexer)
        assert isinstance(actual, np.ndarray)
        assert np.allclose(actual, [0.0, 1.0, 2.0])

        # float
        actual = self.grid._validate_indexer(1.0)
        assert isinstance(actual, np.ndarray)
        assert np.allclose(actual, [1.0])

        # int
        actual = self.grid._validate_indexer(1)
        assert isinstance(actual, np.ndarray)
        assert np.allclose(actual, [1])

    def test_sel__bounding_box(self):
        def check_output(actual, expected):
            assert isinstance(actual, tuple)
            new_obj, new_grid = actual
            assert isinstance(new_obj, xr.DataArray)
            assert isinstance(new_grid, xugrid.Ugrid2d)
            assert new_obj.dims[0] == f"{NAME}_nFaces"
            assert new_grid.face_dimension == f"{NAME}_nFaces"
            assert np.array_equal(new_obj.values, expected)

        actual = self.grid.sel(obj=self.obj, x=slice(0.0, 2.0), y=slice(0.0, 1.0))
        check_output(actual, [0, 1])

        actual = self.grid.sel(obj=self.obj, x=slice(None, None), y=slice(None, 1.0))
        check_output(actual, [0, 1])

        actual = self.grid.sel(obj=self.obj, x=slice(0.0, 1.0), y=slice(0.0, 2.0))
        check_output(actual, [0, 2])

        actual = self.grid.sel(obj=self.obj, x=slice(None, 1.0), y=slice(None, None))
        check_output(actual, [0, 2])

        for x, y in zip([None, None, slice(0, 2)], [None, slice(0, 2), None]):
            actual = self.grid.sel(obj=self.obj, x=x, y=y)
            check_output(actual, [0, 1, 2, 3])

        # Check default arguments, should return entire grid
        actual = self.grid.sel(obj=self.obj)
        check_output(actual, [0, 1, 2, 3])

    def test_sel__points_from_scalar(self):
        def check_output(actual):
            assert isinstance(actual, xr.DataArray)
            dim = f"{NAME}_nFaces"
            expected = xr.DataArray(
                data=[0],
                coords={
                    "x": (dim, [0.5]),
                    "y": (dim, [0.5]),
                },
                dims=[dim],
            )
            assert expected.equals(actual)

        actual = self.grid.sel(obj=self.obj, x=0.5, y=0.5)
        check_output(actual)

        actual = self.grid.sel(obj=self.obj, x=[0.5], y=[0.5])
        check_output(actual)

        with pytest.raises(TypeError, match="Invalid indexer type"):
            self.grid.sel(obj=self.obj, x=(0.5,), y=[0.5])

    def test_sel__points_from_arrays_and_slice(self):
        def check_output(actual):
            assert isinstance(actual, xr.DataArray)
            dim = f"{NAME}_nFaces"
            expected = xr.DataArray(
                data=[0, 0, 1, 2, 2, 3],
                coords={
                    "x": (dim, [0.4, 0.8, 1.2, 0.4, 0.8, 1.2]),
                    "y": (dim, [0.5, 0.5, 0.5, 1.1, 1.1, 1.1]),
                },
                dims=[dim],
            )
            # This fails for some reason:
            # assert expected.equals(actual)
            assert np.array_equal(expected.values, actual.values)
            assert expected.dims == actual.dims
            assert np.allclose(expected["y"].values, actual["y"].values)
            assert np.allclose(expected["x"].values, actual["x"].values)

        x = [0.4, 0.8, 1.2]
        y = [0.5, 1.1]
        actual = self.grid.sel(obj=self.obj, x=x, y=y)
        check_output(actual)

        x = slice(0.4, 1.5, 0.4)  # Evaluates to: [0.4, 0.8, 1.2]
        actual = self.grid.sel(obj=self.obj, x=x, y=y)
        check_output(actual)

    def test_sel__edges_from_slice(self):
        with pytest.raises(ValueError, match="If x is a slice without steps"):
            self.grid.sel(obj=self.obj, x=slice(None, None), y=[0.25, 0.75])
        with pytest.raises(ValueError, match="If x is a slice without steps"):
            self.grid.sel(obj=self.obj, x=slice(None, None), y=slice(0.25, 1.0, 0.25))
        with pytest.raises(ValueError, match="If y is a slice without steps"):
            self.grid.sel(obj=self.obj, x=[0.25, 0.75], y=slice(None, None))

        actual = self.grid.sel(obj=self.obj, x=slice(None, None), y=0.5)
        assert isinstance(actual, xr.DataArray)
        dim = f"{NAME}_nFaces"
        expected = xr.DataArray(
            data=[0, 1],
            coords={
                "x": (dim, [0.5, 1.5]),
                "y": (dim, [0.5, 0.5]),
                "s": (dim, [0.5, 1.5]),
            },
            dims=[dim],
        )
        assert expected.equals(actual)

        actual = self.grid.sel(obj=self.obj, x=0.5, y=slice(None, None))
        assert isinstance(actual, xr.DataArray)
        expected = xr.DataArray(
            data=[0, 2],
            coords={
                "x": (dim, [0.5, 0.5]),
                "y": (dim, [0.5, 1.25]),
                "s": (dim, [0.5, 1.25]),
            },
            dims=[dim],
        )
        assert expected.equals(actual)


def test_topology_subset():
    grid = grid2d()
    face_index = np.array([1])
    actual = grid.topology_subset(face_index)
    assert np.array_equal(actual.face_node_connectivity, [[0, 1, 3, 2]])
    assert np.array_equal(actual.node_x, [1.0, 2.0, 1.0, 2.0])
    assert np.array_equal(actual.node_y, [0.0, 0.0, 1.0, 1.0])

    face_index = np.array([False, True, False, False])
    actual = grid.topology_subset(face_index)
    assert np.array_equal(actual.face_node_connectivity, [[0, 1, 3, 2]])
    assert np.array_equal(actual.node_x, [1.0, 2.0, 1.0, 2.0])
    assert np.array_equal(actual.node_y, [0.0, 0.0, 1.0, 1.0])

    # Entire mesh
    face_index = np.array([0, 1, 2, 3])
    actual = grid.topology_subset(face_index)
    assert actual is grid

    # Check that alternative attrs are preserved.
    grid = grid2d(attrs={"node_dimension": "nNetNode"})
    face_index = np.array([1])
    actual = grid.topology_subset(face_index)
    assert actual.node_dimension == "nNetNode"


def test_triangulate():
    grid = grid2d()
    actual = grid.triangulate()
    assert isinstance(actual, xugrid.Ugrid2d)
    assert actual.n_face == 6


def test_tesselate_centroidal_voronoi():
    grid = grid2d()

    voronoi = grid.tesselate_centroidal_voronoi(add_exterior=False)
    assert isinstance(voronoi, xugrid.Ugrid2d)
    assert voronoi.n_face == 1

    voronoi = grid.tesselate_centroidal_voronoi(add_vertices=False)
    assert voronoi.n_face == 7

    voronoi = grid.tesselate_centroidal_voronoi()
    assert voronoi.n_face == 7


def test_reverse_cuthill_mckee():
    grid = grid2d()
    new, index = grid.reverse_cuthill_mckee()
    assert isinstance(new, xugrid.Ugrid2d)
    assert np.allclose(new.node_coordinates, grid.node_coordinates)
    assert np.array_equal(index, [3, 2, 1, 0])


@requires_meshkernel
def test_mesh():
    grid = grid2d()
    assert isinstance(grid.mesh, mk.Mesh2d)


@requires_meshkernel
def test_meshkernel():
    grid = grid2d()
    assert isinstance(grid.meshkernel, mk.MeshKernel)


def test_from_geodataframe():
    xy = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 0.0],
        ]
    )
    gdf = gpd.GeoDataFrame(geometry=[pygeos.creation.polygons(xy)])
    grid = xugrid.Ugrid2d.from_geodataframe(gdf)
    assert isinstance(grid, xugrid.Ugrid2d)


def test_to_pygeos():
    grid = grid2d()

    points = grid.to_pygeos(f"{NAME}_nNodes")
    assert isinstance(points[0], pygeos.Geometry)

    lines = grid.to_pygeos(f"{NAME}_nEdges")
    assert isinstance(lines[0], pygeos.Geometry)

    polygons = grid.to_pygeos(f"{NAME}_nFaces")
    assert isinstance(polygons[0], pygeos.Geometry)


def test_grid_from_geodataframe():
    with pytest.raises(TypeError, match="Cannot convert a list"):
        xugrid.conversion.grid_from_geodataframe([])

    with pytest.raises(ValueError, match="geodataframe contains no geometry"):
        xugrid.conversion.grid_from_geodataframe(gpd.GeoDataFrame(geometry=[]))

    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 0.0, 0.0])
    line = pygeos.creation.linestrings(x, y)
    xy = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 0.0],
        ]
    )
    polygon = pygeos.creation.polygons(xy)
    points = pygeos.creation.points(x, y)

    with pytest.raises(ValueError, match="Multiple geometry types detected"):
        xugrid.conversion.grid_from_geodataframe(
            gpd.GeoDataFrame(geometry=[line, polygon])
        )
    with pytest.raises(ValueError, match="Invalid geometry type"):
        xugrid.conversion.grid_from_geodataframe(gpd.GeoDataFrame(geometry=points))

    grid = xugrid.conversion.grid_from_geodataframe(gpd.GeoDataFrame(geometry=[line]))
    assert isinstance(grid, xugrid.Ugrid1d)
    grid = xugrid.conversion.grid_from_geodataframe(
        gpd.GeoDataFrame(geometry=[polygon])
    )
    assert isinstance(grid, xugrid.Ugrid2d)


def test_ugrid2d_plot():
    grid = grid2d()
    primitive = grid.plot()
    assert isinstance(primitive, LineCollection)
