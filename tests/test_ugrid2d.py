from typing import NamedTuple

import geopandas as gpd
import numba_celltree
import numpy as np
import pyproj
import pytest
import shapely
import xarray as xr
from matplotlib.collections import LineCollection
from scipy import sparse, spatial

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


def test_safe_attrs():
    # .attrs should return a copy
    grid = grid2d()
    assert grid.attrs == grid.attrs
    assert grid._attrs is not grid.attrs


def test_ugrid2d_alternative_init():
    custom_attrs = {
        "node_dimension": "nNetNode",
        "name": "mesh1d",
        "node_coordinates": "mesh1d_node_x mesh1d_node_y",
    }
    indexes = {"node_x": "mesh1d_node_x", "node_y": "mesh1d_node_y"}
    grid = grid2d(attrs=custom_attrs, indexes=indexes)
    assert grid.node_dimension == "nNetNode"
    assert grid.name == NAME
    # name in attrs should be overwritten by given name.
    assert grid._attrs["name"] == NAME

    with pytest.raises(ValueError, match="Provide either dataset or attrs, not both"):
        grid2d(dataset=xr.Dataset, attrs=custom_attrs)

    with pytest.raises(ValueError, match="indexes must be provided for dataset"):
        grid2d(dataset=xr.Dataset, indexes=None)

    with pytest.raises(ValueError, match="indexes must be provided for attrs"):
        grid = grid2d(attrs=custom_attrs)


def test_ugrid2d_properties():
    grid = grid2d()
    assert grid.edge_dimension == f"{NAME}_nEdges"
    assert grid.node_dimension == f"{NAME}_nNodes"
    assert grid.face_dimension == f"{NAME}_nFaces"
    assert grid.n_node == 7
    assert grid.n_edge == 10
    assert grid.n_face == 4
    assert grid.n_max_node_per_face == 4
    assert grid.facets == {
        "node": grid.node_dimension,
        "edge": grid.edge_dimension,
        "face": grid.face_dimension,
    }
    assert np.array_equal(grid.n_node_per_face, [4, 4, 3, 3])
    assert np.allclose(grid.node_coordinates, VERTICES)
    assert grid.bounds == (0.0, 0.0, 2.0, 2.0)
    assert isinstance(grid.node_node_connectivity, sparse.csr_matrix)
    assert isinstance(grid.node_edge_connectivity, sparse.csr_matrix)
    assert isinstance(grid.directed_node_node_connectivity, sparse.csr_matrix)
    assert isinstance(grid.directed_edge_edge_connectivity, sparse.csr_matrix)
    edge_node_coords = grid.edge_node_coordinates
    face_node_coords = grid.face_node_coordinates
    assert edge_node_coords.shape == (10, 2, 2)
    assert face_node_coords.shape == (4, 4, 2)
    assert grid.edge_length.shape == (grid.n_edge,)
    assert grid.area.shape == (grid.n_face,)
    assert grid.perimeter.shape == (grid.n_face,)
    are_nan = np.isnan(face_node_coords)
    assert are_nan[2:, -1:, :].all()
    assert not are_nan[:, :-1, :].any()
    assert isinstance(grid.attrs, dict)
    coords = grid.coords
    assert isinstance(coords, dict)
    assert np.array_equal(coords[grid.node_dimension], grid.node_coordinates)
    assert np.array_equal(coords[grid.edge_dimension], grid.edge_coordinates)
    assert np.array_equal(coords[grid.face_dimension], grid.face_coordinates)

    with pytest.raises(ValueError, match="start_index must be 0 or 1, received: 2"):
        grid.start_index = 2
    grid.start_index = 1
    assert grid._start_index == 1

    assert isinstance(grid.node_kdtree, spatial.KDTree)
    assert isinstance(grid.edge_kdtree, spatial.KDTree)
    assert isinstance(grid.face_kdtree, spatial.KDTree)


def test_validate_edge_node_connectivity():
    # Full test at test_connectivity
    grid = grid2d()
    valid = grid.validate_edge_node_connectivity()
    assert isinstance(valid, np.ndarray)
    assert valid.size == grid.n_edge
    assert valid.all()


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


def test_ugrid2d_update_coordinate_attrs():
    grid = grid2d()
    obj = xr.DataArray(np.ones(grid.n_face), dims=(grid.face_dimension,))
    obj = grid.assign_face_coords(obj)
    grid._indexes["face_x"] = "mesh2d_face_x"
    grid._indexes["face_y"] = "mesh2d_face_y"
    grid.set_crs(epsg=4326)
    grid._update_coordinate_attrs(obj)
    assert obj["mesh2d_face_x"].attrs["standard_name"] == "longitude"
    assert obj["mesh2d_face_y"].attrs["standard_name"] == "latitude"


def test_ugrid2d_assign_derived_coordinates():
    grid = grid2d()
    obj = xr.DataArray(np.ones(grid.n_face), dims=(grid.face_dimension,))
    obj = grid._assign_derived_coords(obj)
    assert "mesh2d_face_x" in obj.coords
    assert "mesh2d_face_y" in obj.coords


def test_to_crs():
    grid = grid2d()
    grid.set_crs("epsg:4326")
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

    grid = grid2d()
    ds = grid.to_dataset()
    assert isinstance(ds, xr.Dataset)
    assert f"{NAME}" in ds
    assert f"{NAME}_nNodes" in ds.dims
    assert f"{NAME}_nFaces" in ds.dims
    assert f"{NAME}_node_x" in ds.coords
    assert f"{NAME}_node_y" in ds.coords
    assert f"{NAME}_face_nodes" in ds
    check_attrs(ds)

    ds = grid.to_dataset(optional_attributes=True)
    assert f"{NAME}_edge_nodes" in ds
    assert f"{NAME}_face_nodes" in ds
    assert f"{NAME}_face_edges" in ds
    assert f"{NAME}_face_faces" in ds
    assert f"{NAME}_edge_faces" in ds
    assert f"{NAME}_boundary_nodes" in ds
    assert f"{NAME}_face_x" in ds
    assert f"{NAME}_face_y" in ds
    assert f"{NAME}_edge_x" in ds
    assert f"{NAME}_edge_y" in ds
    check_attrs(ds)


def test_find_ugrid_dim():
    grid = grid2d()
    da = xr.DataArray(data=np.ones((grid.n_face,)), dims=[grid.face_dimension])
    assert grid.find_ugrid_dim(da) == grid.face_dimension

    weird = xr.DataArray(
        data=np.ones((grid.n_face, grid.n_node)),
        dims=[grid.face_dimension, grid.node_dimension],
    )
    with pytest.raises(
        ValueError,
        match="UgridDataArray should contain exactly one of the UGRID dimension",
    ):
        grid.find_ugrid_dim(weird)


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


def test_ugrid2d_dataset_no_mutation():
    grid = grid2d()
    ds = grid.to_dataset()
    # Test a non-default fill value
    face_nodes = ds["mesh2d_face_nodes"]
    face_nodes = face_nodes.where(face_nodes != -1, other=-999)
    face_nodes.attrs["_FillValue"] = -999
    ds["mesh2d_face_nodes"] = face_nodes
    reference = ds.copy(deep=True)
    xugrid.Ugrid2d.from_dataset(ds)
    assert ds.identical(reference)


@pytest.mark.parametrize("edge_start_index", [0, 1])
@pytest.mark.parametrize("face_start_index", [0, 1])
def test_ugrid2d_from_dataset__different_start_index(
    face_start_index, edge_start_index
):
    grid = grid2d()
    ds = grid.to_dataset(optional_attributes=True)  # include edge_nodes
    faces = ds["mesh2d_face_nodes"].to_numpy()
    faces[faces != -1] += face_start_index
    ds["mesh2d_face_nodes"].attrs["start_index"] = face_start_index
    ds["mesh2d_edge_nodes"] += edge_start_index
    ds["mesh2d_edge_nodes"].attrs["start_index"] = edge_start_index
    new = xugrid.Ugrid2d.from_dataset(ds)
    assert new.start_index == face_start_index
    assert np.array_equal(new.face_node_connectivity, grid.face_node_connectivity)
    assert np.array_equal(new.edge_node_connectivity, grid.edge_node_connectivity)


def test_ugrid2d_from_meshkernel():
    # Setup a meshkernel Mesh2d mimick
    class Mesh2d(NamedTuple):
        node_x: np.ndarray
        node_y: np.ndarray
        face_nodes: np.ndarray
        nodes_per_face: np.ndarray
        edge_nodes: np.ndarray

    mesh2d = Mesh2d(
        node_x=np.array([0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]),
        node_y=np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]),
        face_nodes=np.array(
            [0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 4, 5, 9, 8, 5, 6, 10, 9, 6, 7, 11, 10]
        ),
        nodes_per_face=np.array([4, 4, 4, 4, 4, 4]),
        edge_nodes=np.array(
            [
                4,
                8,
                5,
                6,
                5,
                9,
                6,
                7,
                6,
                10,
                7,
                11,
                8,
                9,
                9,
                10,
                10,
                11,
                0,
                1,
                0,
                4,
                1,
                2,
                1,
                5,
                2,
                3,
                2,
                6,
                3,
                7,
                4,
                5,
            ]
        ),
    )

    grid = xugrid.Ugrid2d.from_meshkernel(mesh2d)
    grid.plot()
    assert grid.n_face == 6
    assert np.allclose(mesh2d.node_x, grid.node_x)
    assert np.allclose(mesh2d.node_y, grid.node_y)
    assert np.allclose(grid.face_node_connectivity, mesh2d.face_nodes.reshape((6, 4)))
    assert np.allclose(grid.edge_node_connectivity, mesh2d.edge_nodes.reshape((-1, 2)))


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
        "_node_kdtree",
        "_edge_kdtree",
        "_face_kdtree",
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
    assert grid.dims == {
        f"{NAME}_nNodes",
        f"{NAME}_nEdges",
        f"{NAME}_nFaces",
    }
    assert grid.sizes == {
        f"{NAME}_nNodes": 7,
        f"{NAME}_nEdges": 10,
        f"{NAME}_nFaces": 4,
    }
    with pytest.warns(FutureWarning):
        assert grid.dimensions == grid.sizes


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


def test_connectivity_matrix():
    grid = grid2d()
    with pytest.raises(
        ValueError, match="Expected mesh2d_nNodes or mesh2d_nFaces; got: mesh2d_nEdges"
    ):
        grid.get_connectivity_matrix(dim=grid.edge_dimension, xy_weights=False)

    connectivity = grid.get_connectivity_matrix(grid.face_dimension, xy_weights=True)
    assert isinstance(connectivity, sparse.csr_matrix)
    assert np.array_equal(connectivity.indices, [1, 2, 0, 3, 0, 3, 1, 2])

    connectivity = grid.get_connectivity_matrix(grid.node_dimension, xy_weights=True)
    assert isinstance(connectivity, sparse.csr_matrix)
    assert np.array_equal(
        connectivity.indices,
        [1, 3, 0, 2, 4, 1, 5, 0, 4, 6, 1, 3, 5, 6, 2, 4, 6, 3, 4, 5],
    )


def test_get_coordinates():
    grid = grid2d()
    with pytest.raises(
        ValueError,
        match="Expected mesh2d_nNodes, mesh2d_nEdges, or mesh2d_nFaces; got: abc",
    ):
        grid.get_coordinates(dim="abc")

    assert isinstance(grid.get_coordinates(grid.node_dimension), np.ndarray)
    assert isinstance(grid.get_coordinates(grid.edge_dimension), np.ndarray)
    assert isinstance(grid.get_coordinates(grid.face_dimension), np.ndarray)


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
    assert isinstance(faces, np.ndarray)
    expected_faces = np.array(
        [
            [0, 1, 3, 2],
            [4, 0, 5, -1],
            [4, 6, 1, 0],
            [6, 7, 1, -1],
            [5, 0, 2, 8],
            [1, 7, 9, 3],
            [2, 3, 9, 8],
        ]
    )
    assert np.array_equal(faces, expected_faces)
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
    # Test tolerance
    centroids_offset = [[-0.01, 1.0], [-0.01, 0.5]]
    assert np.array_equal(grid.locate_points(centroids_offset, 0.011), [0, 0])


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
    # Test with tolerance. First point goes out of bounds, tolerance shouldn't
    # matter.
    xy[:, 0] -= 0.01
    face, weights = grid.compute_barycentric_weights(xy, tolerance=0.01)
    expected_face = np.array([-1, 0, 1, 2, -1])
    expected_weights = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.5, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    assert np.array_equal(face, expected_face)
    assert np.allclose(weights, expected_weights, atol=0.05)


def test_rasterize():
    grid = grid2d()
    x, y, index = grid.rasterize(resolution=0.5)
    expected_index = np.array(
        [
            [-1, 2, 3, -1],
            [2, 2, 3, 3],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ]
    )
    assert np.allclose(x, [0.25, 0.75, 1.25, 1.75])
    assert np.allclose(y, [1.75, 1.25, 0.75, 0.25])
    assert np.array_equal(index, expected_index)

    # Test with alternative bounds
    bounds = (-1.0, -1.0, 2.0, 2.0)
    x, y, index = grid.rasterize(resolution=0.5, bounds=bounds)
    expected_index = np.array(
        [
            [-1, -1, -1, 2, 3, -1],
            [-1, -1, 2, 2, 3, 3],
            [-1, -1, 0, 0, 1, 1],
            [-1, -1, 0, 0, 1, 1],
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
        ]
    )
    assert np.allclose(x, [-0.75, -0.25, 0.25, 0.75, 1.25, 1.75])
    assert np.allclose(y, [1.75, 1.25, 0.75, 0.25, -0.25, -0.75])
    assert np.array_equal(index, expected_index)


class TestUgrid2dSelection:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.grid = grid2d()
        self.obj = xr.DataArray([0, 1, 2, 3], dims=[self.grid.face_dimension])

    def test_sel_points(self):
        x = [0.5, 1.5]
        y = [0.5, 1.25]

        with pytest.raises(ValueError, match="out_of_bounds must be one of"):
            self.grid.sel_points(obj=self.obj, x=x, y=y, out_of_bounds="nothing")
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
                f"{NAME}_index": (dim, [0, 1]),
                f"{NAME}_x": (dim, x),
                f"{NAME}_y": (dim, y),
            },
            dims=[dim],
        )
        assert expected.equals(actual)

    def test_sel_points_out_of_bounds(self):
        x = [-10.0, 0.5, -20.0, 1.5, -30.0]
        y = [-10.0, 0.5, -20.0, 1.25, -30.0]

        with pytest.raises(
            ValueError, match="Not all points are located inside of the grid"
        ):
            self.grid.sel_points(obj=self.obj, x=x, y=y, out_of_bounds="raise")

        actual = self.grid.sel_points(obj=self.obj, x=x, y=y, out_of_bounds="drop")
        assert np.array_equal(actual[f"{NAME}_index"], [1, 3])

        with pytest.warns(
            UserWarning, match="Not all points are located inside of the grid"
        ):
            actual = self.grid.sel_points(obj=self.obj, x=x, y=y, out_of_bounds="warn")
            assert np.allclose(actual, [np.nan, 0, np.nan, 3, np.nan], equal_nan=True)

        actual = self.grid.sel_points(obj=self.obj, x=x, y=y, out_of_bounds="ignore")
        assert np.allclose(actual, [np.nan, 0, np.nan, 3, np.nan], equal_nan=True)

        actual = self.grid.sel_points(
            obj=self.obj, x=x, y=y, out_of_bounds="ignore", fill_value=-1
        )
        assert np.allclose(actual, [-1, 0, -1, 3, -1])
        # Case with tolerance, tolerance shouldn't affect results since points
        # are out of bounds
        actual = self.grid.sel_points(
            obj=self.obj, x=x, y=y, out_of_bounds="drop", tolerance=11.0
        )
        assert np.array_equal(actual[f"{NAME}_index"], [1, 3])

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
                    f"{NAME}_index": (dim, [0]),
                    f"{NAME}_x": (dim, [0.5]),
                    f"{NAME}_y": (dim, [0.5]),
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
                    f"{NAME}_x": (dim, [0.4, 0.8, 1.2, 0.4, 0.8, 1.2]),
                    f"{NAME}_y": (dim, [0.5, 0.5, 0.5, 1.1, 1.1, 1.1]),
                },
                dims=[dim],
            )
            # This fails for some reason:
            # assert expected.equals(actual)
            assert np.array_equal(expected.values, actual.values)
            assert expected.dims == actual.dims
            x = f"{NAME}_x"
            y = f"{NAME}_y"
            assert np.allclose(expected[y].values, actual[y].values)
            assert np.allclose(expected[x].values, actual[x].values)

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
                f"{NAME}_x": (dim, [0.5, 1.5]),
                f"{NAME}_y": (dim, [0.5, 0.5]),
                f"{NAME}_s": (dim, [0.5, 1.5]),
            },
            dims=[dim],
        )
        assert expected.equals(actual)

        actual = self.grid.sel(obj=self.obj, x=0.5, y=slice(None, None))
        assert isinstance(actual, xr.DataArray)
        expected = xr.DataArray(
            data=[0, 2],
            coords={
                f"{NAME}_x": (dim, [0.5, 0.5]),
                f"{NAME}_y": (dim, [0.5, 1.25]),
                f"{NAME}_s": (dim, [0.5, 1.25]),
            },
            dims=[dim],
        )
        assert expected.equals(actual)

    def test_intersect_line_error(self):
        with pytest.raises(ValueError, match="Start and end coordinate pairs"):
            self.grid.intersect_line(
                obj=None, start=(0.0, 0.0, 0.0), end=(1.0, 1.0, 1.0)
            )

    def test_intersect_line(self):
        grid = self.grid
        obj = xr.DataArray([0, 1, 2, 3], dims=[grid.face_dimension])

        p0 = (0.0, 0.0)
        p1 = (2.0, 2.0)
        actual = grid.intersect_line(obj, start=p0, end=p1)
        sqrt2 = np.sqrt(2.0)
        assert isinstance(actual, xr.DataArray)
        assert actual.dims == (grid.face_dimension,)
        assert np.array_equal(actual.to_numpy(), [0, 3])
        assert np.allclose(actual[f"{NAME}_x"], [0.5, 1.25])
        assert np.allclose(actual[f"{NAME}_y"], [0.5, 1.25])
        assert np.allclose(actual[f"{NAME}_s"], [0.5 * sqrt2, 1.25 * sqrt2])

        actual = grid.intersect_line(obj, start=p1, end=p0)
        assert np.array_equal(actual.to_numpy(), [3, 0])

    def test_intersect_linestring(self):
        grid = self.grid
        obj = xr.DataArray([0, 1, 2, 3], dims=[grid.face_dimension])
        linestring = shapely.geometry.LineString(
            [
                [0.5, 0.5],
                [1.5, 0.5],
                [1.5, 1.5],
            ]
        )
        actual = grid.intersect_linestring(obj, linestring)
        assert isinstance(actual, xr.DataArray)
        assert actual.dims == (grid.face_dimension,)
        assert np.array_equal(actual.to_numpy(), [0, 1, 1, 3])
        assert np.allclose(actual[f"{NAME}_x"], [0.75, 1.25, 1.5, 1.5])
        assert np.allclose(actual[f"{NAME}_y"], [0.5, 0.5, 0.75, 1.25])
        assert np.allclose(actual[f"{NAME}_s"], [0.25, 0.75, 1.25, 1.75])


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

    # Reordering
    face_index = np.array([3, 2, 1, 0])
    actual = grid.topology_subset(face_index)
    assert np.array_equal(
        actual.face_node_connectivity,
        grid.face_node_connectivity[::-1],
    )

    # Check that alternative attrs are preserved.
    grid = grid2d(
        attrs={"node_dimension": "nNetNode"},
        indexes={"node_x": "mesh2d_node_x", "node_y": "mesh2d_node_y"},
    )
    face_index = np.array([1])
    actual = grid.topology_subset(face_index)
    assert actual.node_dimension == "nNetNode"


def test_reindex_like():
    grid = grid2d()
    # Change face and edge_index.
    index = np.arange(grid.n_face)
    rev_index = index[::-1]
    edge_index = np.arange(grid.n_edge)
    rev_edge_index = edge_index[::-1]
    reordered = grid.topology_subset(rev_index)
    reordered._edge_node_connectivity = reordered._edge_node_connectivity[
        rev_edge_index
    ]

    face_da = xr.DataArray(rev_index, dims=(reordered.face_dimension,))
    edge_da = xr.DataArray(rev_edge_index, dims=(reordered.edge_dimension,))
    ds = xr.Dataset({"edge": edge_da, "face": face_da})
    reindexed = reordered.reindex_like(grid, obj=ds)
    assert np.array_equal(reindexed["face"], index)
    assert np.array_equal(reindexed["edge"], edge_index)


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

    faces = FACES.copy()
    faces[faces == -1] = -999
    grid = xugrid.Ugrid2d(
        node_x=VERTICES[:, 0],
        node_y=VERTICES[:, 1],
        fill_value=-999,
        face_node_connectivity=faces,
    )
    voronoi = grid.tesselate_centroidal_voronoi(add_exterior=True)
    vfaces = voronoi.face_node_connectivity
    fill_nodes = vfaces[vfaces < 0]
    assert (fill_nodes == -1).all()


def test_tesselate_circumcenter_voronoi():
    grid = grid2d()

    # Can only deal with triangular grids
    with pytest.raises(NotImplementedError):
        grid.tesselate_circumcenter_voronoi()

    # Now test with triangular grid
    vertices = np.array(
        [
            [0.0, 0.0],  # 0
            [2.0, 0.0],  # 1
            [1.0, 1.0],  # 2
            [2.0, 2.0],  # 3
            [0.0, 2.0],  # 4
        ]
    )
    faces = np.array(
        [
            [0, 1, 2],
            [1, 3, 2],
            [3, 4, 2],
            [4, 0, 2],
        ]
    )
    grid = xugrid.Ugrid2d(
        node_x=vertices[:, 0],
        node_y=vertices[:, 1],
        fill_value=-1,
        face_node_connectivity=faces,
    )
    voronoi = grid.tesselate_circumcenter_voronoi()
    assert voronoi.n_face == 5


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


@pytest.mark.parametrize("xflip", [False, True])
@pytest.mark.parametrize("yflip", [False, True])
def test_from_structured_intervals1d(xflip: bool, yflip: bool):
    x = y = np.array([0.0, 1.0, 2.0])
    if xflip:
        x = np.flip(x)
    if yflip:
        y = np.flip(y)

    grid = xugrid.Ugrid2d.from_structured_intervals1d(x_intervals=x, y_intervals=y)
    assert isinstance(grid, xugrid.Ugrid2d)
    assert grid.n_face == 4

    # Make sure the orientation is still ccw if x is decreasing, y is increasing.
    dxy = np.diff(grid.face_node_coordinates, axis=1)
    clockwise = (
        xugrid.ugrid.connectivity.cross2d(dxy[:, :-1], dxy[:, 1:]).sum(axis=1) < 0
    )
    assert not clockwise.any()


def test_from_structured_intervals2d():
    with pytest.raises(ValueError, match="Dimensions of intervals must be 2D."):
        xugrid.Ugrid2d.from_structured_intervals2d(
            x_intervals=[0.0, 1.0, 2.0],
            y_intervals=[2.0, 1.0, 0.0],
        )
    with pytest.raises(ValueError, match="Interval shapes must match."):
        xugrid.Ugrid2d.from_structured_intervals2d(
            x_intervals=[[0.0, 1.0, 2.0]],
            y_intervals=[[2.0, 1.0, 0.0, 4.0]],
        )

    grid = xugrid.Ugrid2d.from_structured_intervals2d(
        x_intervals=[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
        y_intervals=[
            [2.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
        ],
    )
    assert isinstance(grid, xugrid.Ugrid2d)
    assert grid.n_face == 4


def test_from_structured_bounds():
    x_vertices = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    y_vertices = np.array([2.5, 7.5, 12.5, 17.5])
    # Ascending
    x_bounds = np.column_stack((x_vertices[:-1], x_vertices[1:]))
    y_bounds = np.column_stack((y_vertices[:-1], y_vertices[1:]))
    grid = xugrid.Ugrid2d.from_structured_bounds(x_bounds, y_bounds)
    assert isinstance(grid, xugrid.Ugrid2d)
    assert grid.n_face == 12

    x_bounds = np.array(
        [[[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0], [4.0, 4.0, 5.0, 5.0]]]
    )
    y_bounds = np.array(
        [[[0.0, 1.0, 1.0, 0.0], [2.0, 3.0, 3.0, 2.0], [4.0, 5.0, 5.0, 4.0]]]
    )
    grid = xugrid.Ugrid2d.from_structured_bounds(x_bounds, y_bounds)
    assert grid.n_face == 3
    assert np.allclose(grid.area, 1.0)
    assert grid.bounds == (0.0, 0.0, 5.0, 5.0)

    with pytest.raises(ValueError, match="Bounds shapes do not match"):
        xugrid.Ugrid2d.from_structured_bounds(x_bounds, y_bounds.T)


def test_from_structured():
    da = xr.DataArray(
        data=np.ones((2, 2)),
        coords={"y": [12.0, 11.0], "x": [1.0, 2.0]},
        dims=("y", "x"),
    )
    grid = xugrid.Ugrid2d.from_structured(da)
    assert isinstance(grid, xugrid.Ugrid2d)
    assert grid.n_face == 4

    da = xr.DataArray(
        data=np.ones((2, 2)),
        coords={"lat": [12.0, 11.0], "lon": [1.0, 2.0]},
        dims=("lat", "lon"),
    )
    grid = xugrid.Ugrid2d.from_structured(da, x="lon", y="lat")
    assert isinstance(grid, xugrid.Ugrid2d)
    assert grid.n_face == 4


def test_from_structured_multicoord():
    da = xr.DataArray(
        data=np.ones((2, 2)),
        coords={
            "yc": (("y", "x"), [[12.0, 11.0], [13.0, 12.0]]),
            "xc": (("y", "x"), [[1.0, 2.0], [2.0, 3.0]]),
        },
        dims=("y", "x"),
    )
    grid = xugrid.Ugrid2d._from_structured_multicoord(da, x="xc", y="yc")
    assert isinstance(grid, xugrid.Ugrid2d)
    assert grid.n_face == 4


def test_from_shapely():
    with pytest.raises(TypeError):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 0.0, 0.0])
        xugrid.Ugrid2d.from_shapely(geometry=shapely.linestrings(x, y))

    xy = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 0.0],
        ]
    )
    grid = xugrid.Ugrid2d.from_shapely(geometry=[shapely.polygons(xy)])
    assert isinstance(grid, xugrid.Ugrid2d)


def test_from_geodataframe():
    with pytest.raises(TypeError, match="Expected GeoDataFrame"):
        xugrid.Ugrid2d.from_geodataframe(1)

    xy = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 0.0],
        ]
    )
    gdf = gpd.GeoDataFrame(geometry=[shapely.polygons(xy)])
    grid = xugrid.Ugrid2d.from_geodataframe(gdf)
    assert isinstance(grid, xugrid.Ugrid2d)


def test_bounding_polygon():
    grid = grid2d()
    polygon = grid.bounding_polygon()
    assert isinstance(polygon, shapely.Polygon)
    assert np.allclose(grid.bounds, polygon.bounds)


def test_to_shapely():
    grid = grid2d()

    points = grid.to_shapely(f"{NAME}_nNodes")
    assert isinstance(points[0], shapely.Geometry)

    lines = grid.to_shapely(f"{NAME}_nEdges")
    assert isinstance(lines[0], shapely.Geometry)

    polygons = grid.to_shapely(f"{NAME}_nFaces")
    assert isinstance(polygons[0], shapely.Geometry)


def test_grid_from_geodataframe():
    with pytest.raises(TypeError, match="Cannot convert a list"):
        xugrid.conversion.grid_from_geodataframe([])

    with pytest.raises(ValueError, match="geodataframe contains no geometry"):
        xugrid.conversion.grid_from_geodataframe(gpd.GeoDataFrame(geometry=[]))

    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 0.0, 0.0])
    line = shapely.linestrings(x, y)
    xy = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 0.0],
        ]
    )
    polygon = shapely.polygons(xy)
    points = shapely.points(x, y)

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


def test_ugrid2d_rename():
    grid = grid2d()
    original_indexes = grid._indexes.copy()
    original_attrs = grid._attrs.copy()

    renamed = grid.rename("__renamed")

    # Check that original is unchanged
    assert grid._attrs == original_attrs
    assert grid._indexes == original_indexes
    assert renamed._attrs == {
        "cf_role": "mesh_topology",
        "long_name": "Topology data of 2D mesh",
        "topology_dimension": 2,
        "node_dimension": "__renamed_nNodes",
        "edge_dimension": "__renamed_nEdges",
        "face_dimension": "__renamed_nFaces",
        "max_face_nodes_dimension": "__renamed_nMax_face_nodes",
        "boundary_edge_dimension": "__renamed_nBoundary_edges",
        "edge_node_connectivity": "__renamed_edge_nodes",
        "face_node_connectivity": "__renamed_face_nodes",
        "face_edge_connectivity": "__renamed_face_edges",
        "edge_face_connectivity": "__renamed_edge_faces",
        "boundary_node_connectivity": "__renamed_boundary_nodes",
        "face_face_connectivity": "__renamed_face_faces",
        "node_coordinates": "__renamed_node_x __renamed_node_y",
        "edge_coordinates": "__renamed_edge_x __renamed_edge_y",
        "face_coordinates": "__renamed_face_x __renamed_face_y",
    }
    assert renamed._indexes == {
        "node_x": "__renamed_node_x",
        "node_y": "__renamed_node_y",
    }
    assert renamed.name == "__renamed"


def test_ugrid2d_rename_with_dataset():
    grid = grid2d()
    grid2 = xugrid.Ugrid2d.from_dataset(grid.to_dataset())
    original_dataset = grid2._dataset.copy()

    renamed2 = grid2.rename("__renamed")
    dataset = renamed2._dataset
    assert grid2._dataset.equals(original_dataset)
    assert sorted(dataset.data_vars) == [
        "__renamed",
        "__renamed_edge_nodes",
        "__renamed_face_nodes",
    ]
    assert sorted(dataset.dims) == [
        "__renamed_nEdges",
        "__renamed_nFaces",
        "__renamed_nMax_face_nodes",
        "__renamed_nNodes",
        "two",
    ]
    assert sorted(dataset.coords) == ["__renamed_node_x", "__renamed_node_y"]


class TestPeriodicGridConversion:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.vertices = np.array(
            [
                [0.0, 0.0],  # 0
                [1.0, 0.0],  # 1
                [2.0, 0.0],  # 2
                [3.0, 0.0],  # 3
                [0.0, 1.0],  # 4
                [1.0, 1.0],  # 5
                [2.0, 1.0],  # 6
                [3.0, 1.0],  # 7
                [0.0, 2.0],  # 8
                [1.0, 2.0],  # 9
                [2.0, 2.0],  # 10
                [3.0, 2.0],  # 11
            ]
        )
        self.faces = np.array(
            [
                [0, 1, 5, 4],
                [1, 2, 6, 5],
                [2, 3, 7, 6],
                [4, 5, 9, 8],
                [5, 6, 10, 9],
                [6, 7, 11, 10],
            ]
        )
        grid = xugrid.Ugrid2d(*self.vertices.T, -1, self.faces)
        ds = xr.Dataset()
        ds["a"] = xr.DataArray(np.arange(grid.n_node), dims=(grid.node_dimension,))
        ds["b"] = xr.DataArray(np.arange(grid.n_edge), dims=(grid.edge_dimension,))
        ds["c"] = xr.DataArray(np.arange(grid.n_face), dims=(grid.face_dimension,))
        self.ds = ds
        self.grid = grid

    def test_to_periodic(self):
        grid = self.grid.copy()

        # Trigger edge node connectivity
        _ = grid.edge_node_connectivity
        # Convert
        new, new_ds = grid.to_periodic(obj=self.ds)

        # Absent vertices: 3, 7, 11
        expected_vertices = self.vertices[[0, 1, 2, 4, 5, 6, 8, 9, 10]]
        expected_faces = np.array(
            [
                [0, 1, 4, 3],
                [1, 2, 5, 4],
                [2, 0, 3, 5],
                [3, 4, 7, 6],
                [4, 5, 8, 7],
                [5, 3, 6, 8],
            ]
        )
        expected_edges = np.array(
            [
                [0, 1],
                [0, 3],
                [1, 2],
                [1, 4],
                [0, 2],
                [2, 5],
                [3, 4],
                [3, 6],
                [4, 5],
                [4, 7],
                [3, 5],
                [5, 8],
                [6, 7],
                [7, 8],
                [6, 8],
            ]
        )
        assert np.array_equal(
            new.face_node_connectivity,
            expected_faces,
        )
        assert np.allclose(
            new.node_coordinates,
            expected_vertices,
        )
        assert np.array_equal(
            new.edge_node_connectivity,
            expected_edges,
        )
        # Remove nodes (3 & 7 & 11) and edges (6 & 13)
        expected_a = np.arange(grid.n_node).tolist()
        expected_a.remove(3)
        expected_a.remove(7)
        expected_a.remove(11)
        expected_b = np.arange(grid.n_edge).tolist()
        expected_b.remove(6)
        expected_b.remove(13)
        assert np.array_equal(new_ds["a"], expected_a)
        assert np.array_equal(new_ds["b"], expected_b)
        assert np.array_equal(new_ds["c"], [0, 1, 2, 3, 4, 5])

        # Test whether it also works without an object provided.
        new = grid.to_periodic()
        assert np.array_equal(
            new.face_node_connectivity,
            expected_faces,
        )
        assert np.allclose(new.node_coordinates, expected_vertices)
        assert np.array_equal(new.edge_node_connectivity, expected_edges)

    def test_to_nonperiodic(self):
        grid = self.grid.copy()
        _ = grid.edge_node_connectivity  # trigger generation of edge nodes
        periodic_grid, new_ds = grid.to_periodic(obj=self.ds)
        back = periodic_grid.to_nonperiodic(xmax=3.0)

        expected_vertices = self.vertices[[0, 1, 2, 4, 5, 6, 8, 9, 10, 3, 7, 11]]
        expected_faces = np.array(
            [
                [0, 1, 4, 3],
                [1, 2, 5, 4],
                [2, 9, 10, 5],
                [3, 4, 7, 6],
                [4, 5, 8, 7],
                [5, 10, 11, 8],
            ]
        )
        back, back_ds = periodic_grid.to_nonperiodic(xmax=3.0, obj=new_ds)
        assert np.allclose(back.node_coordinates, expected_vertices)
        assert np.array_equal(back.face_node_connectivity, expected_faces)
        assert back.edge_node_connectivity.shape == (17, 2)
        assert np.array_equal(back_ds["a"], [0, 1, 2, 4, 5, 6, 8, 9, 10, 0, 4, 8])
        assert np.array_equal(
            back_ds["b"], [0, 1, 2, 3, 3, 4, 5, 7, 8, 9, 10, 10, 11, 12, 14, 15, 16]
        )
        assert np.array_equal(back_ds["c"], [0, 1, 2, 3, 4, 5])

        back = periodic_grid.to_nonperiodic(xmax=3.0)
        assert np.allclose(back.node_coordinates, expected_vertices)
        assert np.array_equal(back.face_node_connectivity, expected_faces)
        assert back.edge_node_connectivity.shape == (17, 2)


def test_equals():
    grid = grid2d()
    grid_copy = grid2d()
    assert grid.equals(grid)
    assert grid.equals(grid_copy)
    xr_grid = grid.to_dataset()
    assert not grid.equals(xr_grid)
    grid_copy._attrs["attr"] = "something_else"
    assert not grid.equals(grid_copy)


def test_earcut_triangulate_polygons():
    with pytest.raises(TypeError):
        xugrid.Ugrid2d.earcut_triangulate_polygons("abc")

    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 0.0, 0.0])
    gdf = gpd.GeoDataFrame(geometry=[shapely.linestrings(x, y)])
    with pytest.raises(TypeError):
        xugrid.Ugrid2d.earcut_triangulate_polygons(gdf)

    xy = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )
    hole = np.array(
        [
            [
                [0.25, 0.25],
                [0.75, 0.25],
                [0.75, 0.75],
                [0.25, 0.25],
            ]
        ]
    )
    polygon = shapely.polygons(xy, holes=hole)
    gdf = gpd.GeoDataFrame(geometry=[polygon])

    grid = xugrid.Ugrid2d.earcut_triangulate_polygons(polygons=gdf)
    assert isinstance(grid, xugrid.Ugrid2d)
    assert np.issubdtype(grid.face_node_connectivity.dtype, np.signedinteger)
    assert np.allclose(polygon.area, grid.area.sum())

    grid, index = xugrid.Ugrid2d.earcut_triangulate_polygons(
        polygons=gdf, return_index=True
    )
    assert isinstance(grid, xugrid.Ugrid2d)
    assert isinstance(index, np.ndarray)
    assert (index == 0).all()


def test_ugrid2d_create_data_array():
    grid = grid2d()

    uda = grid.create_data_array(np.zeros(grid.n_node), facet="node")
    assert isinstance(uda, xugrid.UgridDataArray)

    uda = grid.create_data_array(np.zeros(grid.n_edge), facet="edge")
    assert isinstance(uda, xugrid.UgridDataArray)

    uda = grid.create_data_array(np.zeros(grid.n_face), facet="face")
    assert isinstance(uda, xugrid.UgridDataArray)

    # Error on facet
    with pytest.raises(ValueError, match="Invalid facet"):
        grid.create_data_array([1, 2, 3, 4], facet="volume")

    # Error on on dimensions
    with pytest.raises(ValueError, match="Can only create DataArrays from 1D arrays"):
        grid.create_data_array([[1, 2, 3, 4]], facet="face")

    # Error on size
    with pytest.raises(ValueError, match="Conflicting sizes"):
        grid.create_data_array([1, 2, 3, 4, 5], facet="face")


def test_ugrid2d_format_connectivity():
    grid = grid2d()
    assert isinstance(grid.face_node_connectivity, np.ndarray)
    assert isinstance(
        grid.format_connectivity_as_sparse(grid.face_node_connectivity),
        sparse.csr_matrix,
    )
    assert isinstance(grid.node_node_connectivity, sparse.csr_matrix)
    assert isinstance(
        grid.format_connectivity_as_dense(grid.node_node_connectivity), np.ndarray
    )
    assert isinstance(
        grid.format_connectivity_as_sparse(grid.node_node_connectivity.tocoo()),
        sparse.csr_matrix,
    )


def test_nearest_interpolate():
    node_x = np.array([-0.5, -0.5, 0.5, 0.5, 1.5, 1.5, 2.5, 2.5, 3.5, 3.5, 4.5, 4.5])
    node_y = np.array(
        [-1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    )
    face_node_connectivity = np.array(
        [
            [0, 1, 2, 3],  # First quad
            [3, 2, 4, 5],  # Second quad
            [5, 4, 6, 7],  # Third quad
            [7, 6, 8, 9],  # Fourth quad
            [9, 8, 10, 11],  # Fifth quad
        ]
    )
    grid = xugrid.Ugrid2d(node_x, node_y, -1, face_node_connectivity)

    # Centroids are:
    # x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    # y = np.zeros_like(x)
    data = np.array([0.0, np.nan, np.nan, np.nan, 4.0])
    facedim = grid.face_dimension
    actual = grid._nearest_interpolate(data, facedim, np.inf)
    assert np.allclose(actual, np.array([0.0, 0.0, 0.0, 4.0, 4.0]))

    actual = grid._nearest_interpolate(data, facedim, 1.1)
    assert np.allclose(actual, np.array([0.0, 0.0, np.nan, 4.0, 4.0]), equal_nan=True)

    with pytest.raises(ValueError, match="All values are NA."):
        grid._nearest_interpolate(np.full_like(data, np.nan), facedim, np.inf)

    # Test for node_dimension
    data = np.full(grid.n_node, np.nan)
    data[0] = 1.0
    actual = grid._nearest_interpolate(data, grid.node_dimension, np.inf)
    assert np.allclose(actual, 1.0)

    # Test for edge_dimension
    data = np.full(grid.n_edge, np.nan)
    data[0] = 1.0
    actual = grid._nearest_interpolate(data, grid.edge_dimension, np.inf)
    assert np.allclose(actual, 1.0)


def test_locate_nearest():
    grid = grid2d()

    indices = grid.locate_nearest_node(grid.node_coordinates)
    assert np.array_equal(indices, [0, 1, 2, 3, 4, 5, 6])

    indices = grid.locate_nearest_edge(grid.edge_coordinates)
    assert np.array_equal(indices, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    indices = grid.locate_nearest_face(grid.face_coordinates)
    assert np.array_equal(indices, [0, 1, 2, 3])

    assert np.array_equal(grid.locate_nearest_node([[-10.0, 0.0]], 1.0), [-1])
    assert np.array_equal(grid.locate_nearest_edge([[-10.0, 0.0]], 1.0), [-1])
    assert np.array_equal(grid.locate_nearest_face([[-10.0, 0.0]], 1.0), [-1])
