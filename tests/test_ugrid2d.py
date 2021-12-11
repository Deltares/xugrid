import geopandas as gpd
import meshkernel as mk
import numba_celltree
import numpy as np
import pygeos
import pyproj
import pytest
import xarray as xr
from scipy import sparse

import xugrid

NAME = xugrid.ugrid.ugrid_io.UGRID2D_DEFAULT_NAME
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
        [0, 1],
        [0, 3],
        [1, 2],
        [1, 4],
        [2, 5],
        [3, 4],
        [3, 6],
        [4, 5],
        [4, 6],
        [5, 6],
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


def grid2d(dataset=None, name=None, crs=None):
    grid = xugrid.Ugrid2d(
        node_x=VERTICES[:, 0],
        node_y=VERTICES[:, 1],
        fill_value=-1,
        face_node_connectivity=FACES,
        dataset=dataset,
        name=name,
        crs=crs,
    )
    return grid


def test_ugrid2d_init():
    grid = grid2d()
    assert grid.name == NAME
    assert isinstance(grid.dataset, xr.Dataset)
    assert grid.node_x.flags["C_CONTIGUOUS"]
    assert grid.node_y.flags["C_CONTIGUOUS"]
    assert grid._edge_node_connectivity is None
    assert grid._face_edge_connectivity is None


def test_ugrid1d_properties():
    # These are defined in the base class
    grid = grid2d()
    assert grid.edge_dimension == f"{NAME}_nEdges"
    assert grid.node_dimension == f"{NAME}_nNodes"
    assert grid.face_dimension == f"{NAME}_nFaces"
    assert grid.n_node == 7
    assert grid.n_edge == 10
    assert grid.n_face == 4
    assert np.allclose(grid.node_coordinates, VERTICES)
    assert grid.bounds == (0.0, 0.0, 2.0, 2.0)
    node_edges = grid.node_edge_connectivity
    assert isinstance(node_edges, sparse.csr_matrix)


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


def test_ugrid2d_from_dataset():
    grid = grid2d()
    grid2 = xugrid.Ugrid2d.from_dataset(grid.dataset)
    assert grid.dataset == grid2.dataset


def test_remove_topology():
    grid = grid2d()
    ds = grid.dataset.copy()
    ds["a"] = xr.DataArray(0)
    actual = grid.remove_topology(ds)
    print(actual)
    assert set(actual.data_vars) == set(["a"])


def test_topology_coords():
    grid = grid2d()
    ds = xr.Dataset()
    ds["a"] = xr.DataArray([1, 2, 3], dims=[f"{NAME}_nNodes"])
    ds["b"] = xr.DataArray([1, 2], dims=[f"{NAME}_nEdges"])
    ds["c"] = xr.DataArray([1, 2], dims=[f"{NAME}_nFaces"])
    coords = grid.topology_coords(ds)
    assert isinstance(coords, dict)
    assert f"{NAME}_edge_x" in coords
    assert f"{NAME}_edge_y" in coords
    assert f"{NAME}_node_x" in coords
    assert f"{NAME}_node_y" in coords
    assert f"{NAME}_face_x" in coords
    assert f"{NAME}_face_y" in coords


def test_topology_dataset():
    grid = grid2d()
    ds = grid.topology_dataset()
    assert isinstance(ds, xr.Dataset)
    assert f"{NAME}" in ds
    assert f"{NAME}_nNodes" in ds.dims
    assert f"{NAME}_nFaces" in ds.dims
    assert f"{NAME}_node_x" in ds.coords
    assert f"{NAME}_node_y" in ds.coords
    assert f"{NAME}_face_nodes" in ds


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


def test_get_dimension():
    grid = grid2d()
    assert grid._get_dimension("node") == f"{NAME}_nNodes"
    assert grid._get_dimension("edge") == f"{NAME}_nEdges"
    assert grid._get_dimension("face") == f"{NAME}_nFaces"


def test_dimensions():
    grid = grid2d()
    assert grid.node_dimension == f"{NAME}_nNodes"
    assert grid.edge_dimension == f"{NAME}_nEdges"
    assert grid.face_dimension == f"{NAME}_nFaces"


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
    assert np.allclose(vertices, CENTROIDS)
    assert isinstance(faces, sparse.coo_matrix)
    assert np.array_equal(faces.row, [0, 0, 0, 0])
    assert np.array_equal(faces.col, [0, 1, 3, 2])
    assert np.array_equal(face_index, [0, 1, 2, 3])


def test_centroid_triangulation():
    grid = grid2d()
    (x, y, triangles), face_index = grid.centroid_triangulation
    assert np.allclose(x, CENTROIDS[:, 0])
    assert np.allclose(y, CENTROIDS[:, 1])
    assert np.array_equal(triangles, [[0, 1, 3], [0, 3, 2]])
    assert np.array_equal(face_index, [0, 1, 2, 3])


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


def test_sel_points():
    grid = grid2d()
    x = [0.5, 1.5]
    y = [0.5, 1.25]

    with pytest.raises(ValueError, match="shape of x does not match shape of y"):
        grid.sel_points(x=[0.5, 1.5], y=[0.5])
    with pytest.raises(ValueError, match="x and y must be 1d"):
        grid.sel_points(x=[x], y=[y])

    dim, as_ugrid, index, coords = grid.sel_points(x=x, y=y)
    assert dim == f"{NAME}_nFaces"
    assert not as_ugrid
    assert np.array_equal(index, [0, 3])
    assert coords["x"][0] == dim
    assert coords["y"][0] == dim
    assert np.array_equal(coords["x"][1], x)
    assert np.array_equal(coords["y"][1], y)


def test_validate_indexer():
    grid = grid2d()
    with pytest.raises(ValueError, match="slice stop should be larger than"):
        grid._validate_indexer(slice(2, 0))
    with pytest.raises(ValueError, match="step should be None"):
        grid._validate_indexer(slice(None, 2, 1))
    with pytest.raises(ValueError, match="step should be None"):
        grid._validate_indexer(slice(0, None, 1))

    expected = np.arange(0.0, 2.0, 0.5)
    assert np.allclose(grid._validate_indexer(slice(0, 2, 0.5)), expected)
    assert grid._validate_indexer(slice(None, 2)) == slice(None, 2)
    assert grid._validate_indexer(slice(0, None)) == slice(0, None)

    with pytest.raises(TypeError, match="Invalid indexer type"):
        grid._validate_indexer((0, 1, 2))

    # list
    actual = grid._validate_indexer([0.0, 1.0, 2.0])
    assert isinstance(actual, np.ndarray)
    assert np.allclose(actual, [0.0, 1.0, 2.0])

    # numpy array
    actual = grid._validate_indexer(np.array([0.0, 1.0, 2.0]))
    assert isinstance(actual, np.ndarray)
    assert np.allclose(actual, [0.0, 1.0, 2.0])

    # xarray DataArray
    indexer = xr.DataArray([0.0, 1.0, 2.0], {"x": [0, 1, 2]}, ["x"])
    actual = grid._validate_indexer(indexer)
    assert isinstance(actual, np.ndarray)
    assert np.allclose(actual, [0.0, 1.0, 2.0])

    # float
    actual = grid._validate_indexer(1.0)
    assert isinstance(actual, np.ndarray)
    assert np.allclose(actual, [1.0])

    # int
    actual = grid._validate_indexer(1)
    assert isinstance(actual, np.ndarray)
    assert np.allclose(actual, [1])


def test_sel():
    grid = grid2d()

    dim, as_ugrid, index, coords = grid.sel(x=slice(0.0, 2.0), y=slice(0.0, 1.0))
    assert dim == f"{NAME}_nFaces"
    assert as_ugrid
    assert np.allclose(index, [0, 1])
    assert coords == {}

    # _, _, index, _ = grid.sel(x=slice(None, None), y=slice(None, 1.0))
    # assert np.allclose(index, [0, 1])

    _, _, index, _ = grid.sel(x=slice(0.0, 1.0), y=slice(0.0, 2.0))
    assert np.allclose(index, [0, 2])
    assert coords == {}

    # _, _, index, _ = grid.sel(x=slice(None, 1.0), y=slice(None, None))
    # assert np.allclose(index, [0, 2])


def test_topology_subset():
    grid = grid2d()
    edge_indices = np.array([1])
    actual = grid.topology_subset(edge_indices)
    assert np.array_equal(actual.face_node_connectivity, [[0, 1, 3, 2]])
    assert np.array_equal(actual.node_x, [1.0, 2.0, 1.0, 2.0])
    assert np.array_equal(actual.node_y, [0.0, 0.0, 1.0, 1.0])

    edge_indices = np.array([False, True, False, False])
    actual = grid.topology_subset(edge_indices)
    assert np.array_equal(actual.face_node_connectivity, [[0, 1, 3, 2]])
    assert np.array_equal(actual.node_x, [1.0, 2.0, 1.0, 2.0])
    assert np.array_equal(actual.node_y, [0.0, 0.0, 1.0, 1.0])

    # Entire mesh
    edge_indices = np.array([0, 1, 2, 3])
    actual = grid.topology_subset(edge_indices)
    assert actual is grid


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


def test_mesh():
    grid = grid2d()
    assert isinstance(grid.mesh, mk.Mesh2d)


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
        xugrid.ugrid.grid_from_geodataframe([])

    with pytest.raises(ValueError, match="geodataframe contains no geometry"):
        xugrid.ugrid.grid_from_geodataframe(gpd.GeoDataFrame(geometry=[]))

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
        xugrid.ugrid.grid_from_geodataframe(gpd.GeoDataFrame(geometry=[line, polygon]))
    with pytest.raises(ValueError, match="Invalid geometry type"):
        xugrid.ugrid.grid_from_geodataframe(gpd.GeoDataFrame(geometry=points))

    grid = xugrid.ugrid.grid_from_geodataframe(gpd.GeoDataFrame(geometry=[line]))
    assert isinstance(grid, xugrid.Ugrid1d)
    grid = xugrid.ugrid.grid_from_geodataframe(gpd.GeoDataFrame(geometry=[polygon]))
    assert isinstance(grid, xugrid.Ugrid2d)
