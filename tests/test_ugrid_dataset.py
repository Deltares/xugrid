import geopandas as gpd
import numpy as np
import pandas as pd
import pygeos
import pytest
import xarray as xr

import xugrid

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
GRID = xugrid.Ugrid2d(
    node_x=VERTICES[:, 0],
    node_y=VERTICES[:, 1],
    fill_value=-1,
    face_node_connectivity=FACES,
)
DARRAY = xr.DataArray(
    data=np.ones(GRID.n_face),
    dims=[GRID.face_dimension],
)
UGRID_DS = GRID.dataset.copy()
UGRID_DS["a"] = DARRAY
UGRID_DS["b"] = DARRAY * 2


class TestUgridDataArray:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.uda = xugrid.UgridDataArray(DARRAY, GRID)

    def test_init(self):
        assert isinstance(self.uda.ugrid.obj, xr.DataArray)
        assert isinstance(self.uda.ugrid.grid, xugrid.Ugrid2d)

    def test_dunder_forward(self):
        assert isinstance(bool(self.uda[0]), bool)
        assert isinstance(int(self.uda[0]), int)
        assert isinstance(float(self.uda[0]), float)

    def test_getitem(self):
        assert isinstance(self.uda["mesh2d_face_x"], xugrid.UgridDataArray)

    def test_setitem(self):
        uda = self.uda.copy()
        # This should be forwarded to the underlying xarray object
        uda["mesh2d_face_x"] = 0.0
        assert isinstance(uda["mesh2d_face_x"], xugrid.UgridDataArray)

    def test_getattr(self):
        # Get an attribute
        assert isinstance(self.uda.dims, tuple)
        # Only DataArrays should be returned as UgridDataArrays, other object
        # should remain untouched.
        assert self.uda.dims == self.uda.ugrid.obj.dims
        assert isinstance(self.uda.data, np.ndarray)
        # DataArrays are automatically wrapped
        assert isinstance(self.uda.mesh2d_face_x, xugrid.UgridDataArray)
        # So are functions
        assert isinstance(self.uda.mean(), xugrid.UgridDataArray)

    def test_ugrid_accessor(self):
        assert isinstance(self.uda.ugrid, xugrid.ugrid_dataset.UgridAccessor)

    def test_from_structured(self):
        da = xr.DataArray([0.0, 1.0, 2.0], {"x": [5.0, 10.0, 15.0]}, ["x"])
        with pytest.raises(ValueError, match="Last two dimensions of da"):
            xugrid.UgridDataArray.from_structured(da)

        da = xr.DataArray(
            data=np.ones((2, 3, 4)),
            coords={"layer": [1, 2], "y": [5.0, 10.0, 15.0], "x": [2.0, 4.0, 6.0, 8.0]},
            dims=["layer", "y", "x"],
            name="grid",
        )
        uda = xugrid.UgridDataArray.from_structured(da)
        assert isinstance(uda, xugrid.UgridDataArray)
        assert uda.name == "grid"
        assert uda.dims == ("layer", "mesh2d_nFaces")
        assert uda.shape == (2, 12)
        assert uda.ugrid.grid.face_dimension not in uda.coords

    def test_unary_op(self):
        alltrue = self.uda.astype(bool)
        allfalse = alltrue.copy()
        allfalse[:] = False
        assert (~allfalse).all()
        assert isinstance(~allfalse, xugrid.UgridDataArray)

    def test_binary_op(self):
        alltrue = self.uda.astype(bool)
        allfalse = alltrue.copy()
        allfalse[:] = False
        assert isinstance(alltrue | allfalse, xugrid.UgridDataArray)
        assert (alltrue | allfalse).all()
        assert (alltrue ^ allfalse).all()
        assert not (alltrue & allfalse).any()
        # inplace op
        alltrue &= allfalse
        assert isinstance(alltrue, xugrid.UgridDataArray)
        assert not (alltrue).any()

    def test_math(self):
        actual = self.uda + 0
        assert isinstance(actual, xugrid.UgridDataArray)

    def test_np_ops(self):
        actual = np.abs(self.uda)
        assert isinstance(actual, xugrid.UgridDataArray)

    # Accessor tests
    def test_isel(self):
        actual = self.uda.ugrid.isel([0, 1])
        assert isinstance(actual, xugrid.UgridDataArray)
        assert actual.shape == (2,)
        assert actual.ugrid.grid.n_face == 2
        assert "mesh2d_nFaces_index" in actual.coords

    def test_sel_points(self):
        with pytest.raises(ValueError, match="x and y must be 1d"):
            self.uda.ugrid.sel_points(x=[[0.0, 1.0]], y=[[0.0, 1.0]])
        with pytest.raises(ValueError, match="shape of x does not match shape of y"):
            self.uda.ugrid.sel_points(x=[0.0], y=[0.0, 1.0])
        actual = self.uda.ugrid.sel_points(x=[0.5, 0.5], y=[0.5, 1.25])
        assert isinstance(actual, xr.DataArray)
        assert actual.shape == (2,)

    def test_sel(self):
        # Ugrid2d already tests most
        # Orthogonal points
        x = [0.4, 0.8, 1.2]
        y = [0.25, 0.75]
        actual = self.uda.ugrid.sel(x=x, y=y)
        assert isinstance(actual, xr.DataArray)
        assert actual.shape == (6,)

        actual = self.uda.ugrid.sel(x=slice(0.4, 1.3, 0.4), y=0.25)
        assert isinstance(actual, xr.DataArray)
        assert actual.shape == (3,)

        actual = self.uda.ugrid.sel(x=slice(0, 1), y=slice(0, 2))
        assert isinstance(actual, xugrid.UgridDataArray)
        assert actual.shape == (2,)
        assert actual.ugrid.grid.n_face == 2

        actual = self.uda.ugrid.sel(x=slice(0, 1), y=slice(1, None))
        assert isinstance(actual, xugrid.UgridDataArray)
        assert actual.shape == (1,)
        assert actual.ugrid.grid.n_face == 1

    def test_rasterize(self):
        actual = self.uda.ugrid.rasterize(resolution=0.5)
        x = [0.25, 0.75, 1.25, 1.75]
        y = [1.75, 1.25, 0.75, 0.25]
        assert isinstance(actual, xr.DataArray)
        assert actual.shape == (4, 4)
        assert np.allclose(actual["x"], x)
        assert np.allclose(actual["y"], y)

        da = xr.DataArray(np.empty((4, 4)), {"y": y, "x": x}, ["y", "x"])
        actual = self.uda.ugrid.rasterize_like(other=da)
        assert isinstance(actual, xr.DataArray)
        assert actual.shape == (4, 4)
        assert np.allclose(actual["x"], x)
        assert np.allclose(actual["y"], y)

    def test_crs(self):
        assert self.uda.ugrid.crs is None

    def test_to_geodataframe(self):
        with pytest.raises(ValueError, match="unable to convert unnamed"):
            self.uda.ugrid.to_geodataframe("mesh2d_nFaces")
        uda2 = self.uda.copy()
        uda2.ugrid.obj.name = "test"
        gdf = uda2.ugrid.to_geodataframe("mesh2d_nFaces")
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert (gdf.geometry.geom_type == "Polygon").all()

    def test_binary_dilation(self):
        a = self.uda > 0
        actual = a.ugrid.binary_dilation()
        assert isinstance(actual, xugrid.UgridDataArray)

    def test_binary_erosion(self):
        a = self.uda > 0
        actual = a.ugrid.binary_erosion()
        assert isinstance(actual, xugrid.UgridDataArray)

    def test_connected_components(self):
        actual = self.uda.ugrid.connected_components()
        assert isinstance(actual, xugrid.UgridDataArray)
        assert np.allclose(actual, 0)

    def test_reverse_cuthill_mckee(self):
        actual = self.uda.ugrid.reverse_cuthill_mckee()
        assert isinstance(actual, xugrid.UgridDataArray)

    def test_laplace_interpolate(self):
        uda2 = self.uda.copy()
        uda2.obj[:-2] = np.nan
        actual = uda2.ugrid.laplace_interpolate(direct_solve=True)
        assert isinstance(actual, xugrid.UgridDataArray)
        assert np.allclose(actual, 1.0)

    def test_to_dataset(self):
        uda2 = self.uda.copy()
        uda2.ugrid.obj.name = "test"
        actual = uda2.to_dataset()
        assert isinstance(actual, xugrid.UgridDataset)

    def test_to_netcdf(self, tmp_path):
        uda2 = self.uda.copy()
        uda2.ugrid.obj.name = "test"
        path = tmp_path / "uda-test.nc"
        uda2.ugrid.to_netcdf(path)
        assert path.exists()

    def test_to_zarr(self, tmp_path):
        uda2 = self.uda.copy()
        uda2.ugrid.obj.name = "test"
        path = tmp_path / "uda-test.zarr"
        uda2.ugrid.to_zarr(path)
        assert path.exists()


class TestUgridDataset:
    @pytest.fixture(autouse=True)
    def setup(self):
        ds = xr.Dataset()
        ds["a"] = DARRAY
        ds["b"] = DARRAY * 2
        self.uds = xugrid.UgridDataset(ds, GRID)

    def test_init(self):
        assert isinstance(self.uds.ugrid.obj, xr.Dataset)
        assert isinstance(self.uds.ugrid.grid, xugrid.Ugrid2d)
        assert "mesh2d_face_x" in self.uds.ugrid.obj
        # Try alternative initialization
        uds = xugrid.UgridDataset(grid=GRID)
        assert isinstance(uds, xugrid.UgridDataset)
        uds["a"] = DARRAY
        assert "a" in uds.ugrid.obj

    def test_init_from_dataset_only(self):
        uds = xugrid.UgridDataset(UGRID_DS)
        assert isinstance(uds, xugrid.UgridDataset)
        assert "a" in uds.ugrid.obj
        assert "b" in uds.ugrid.obj
        assert "mesh2d_face_nodes" in uds.ugrid.grid.dataset
        assert "mesh2d_face_nodes" not in uds.ugrid.obj

    def test_getitem(self):
        assert "a" in self.uds
        assert "b" in self.uds
        assert "mesh2d_face_x" in self.uds
        assert isinstance(self.uds["a"], xugrid.UgridDataArray)
        assert isinstance(self.uds[["a", "b"]], xugrid.UgridDataset)

    def test_setitem(self):
        uds = self.uds.copy()
        # Set a UgridDataArray
        uds["b"] = self.uds["a"]
        assert (uds["b"].data == 1.0).all()
        # Set a scalar
        uds["a"] = 3.0
        assert (uds["a"].data == 3.0).all()

    def test_getattr(self):
        assert tuple(self.uds.dims) == ("mesh2d_nFaces",)
        assert isinstance(self.uds.a, xugrid.UgridDataArray)
        assert isinstance(self.uds.mean(), xugrid.UgridDataset)

    def test_unary_op(self):
        alltrue = self.uds.astype(bool)
        assert isinstance(~alltrue, xugrid.UgridDataset)

    def test_binary_op(self):
        alltrue = self.uds.astype(bool)
        assert isinstance(alltrue ^ alltrue, xugrid.UgridDataset)
        # inplace op
        alltrue &= alltrue
        assert isinstance(alltrue, xugrid.UgridDataset)

    def test_math(self):
        actual = self.uds + 0
        assert isinstance(actual, xugrid.UgridDataset)

    def test_ugrid_accessor(self):
        assert isinstance(self.uds.ugrid, xugrid.ugrid_dataset.UgridAccessor)

    def test_from_geodataframe(self):
        xy = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 0.0],
            ]
        )
        polygon = pygeos.creation.polygons(xy)
        df = pd.DataFrame({"a": [1.0], "b": [2.0]})
        gdf = gpd.GeoDataFrame(df, geometry=[polygon])
        uds = xugrid.UgridDataset.from_geodataframe(gdf)
        assert isinstance(uds, xugrid.UgridDataset)
        assert "a" in uds
        assert "b" in uds

    # Accessor tests
    def test_isel(self):
        actual = self.uds.ugrid.isel([0, 1])
        assert isinstance(actual, xugrid.UgridDataset)
        assert actual.ugrid.grid.n_face == 2
        assert "mesh2d_nFaces_index" in actual.coords
        assert actual["a"].shape == (2,)
        assert actual["b"].shape == (2,)

    def test_sel_points(self):
        with pytest.raises(ValueError, match="x and y must be 1d"):
            self.uds.ugrid.sel_points(x=[[0.0, 1.0]], y=[[0.0, 1.0]])
        with pytest.raises(ValueError, match="shape of x does not match shape of y"):
            self.uds.ugrid.sel_points(x=[0.0], y=[0.0, 1.0])
        actual = self.uds.ugrid.sel_points(x=[0.5, 0.5], y=[0.5, 1.25])
        assert isinstance(actual, xr.Dataset)
        assert actual["a"].shape == (2,)
        assert actual["b"].shape == (2,)

    def test_sel(self):
        # Ugrid2d already tests most
        # Orthogonal points
        x = [0.4, 0.8, 1.2]
        y = [0.25, 0.75]
        actual = self.uds.ugrid.sel(x=x, y=y)
        assert isinstance(actual, xr.Dataset)
        assert actual["a"].shape == (6,)
        assert actual["b"].shape == (6,)

        actual = self.uds.ugrid.sel(x=slice(0.4, 1.3, 0.4), y=0.25)
        assert isinstance(actual, xr.Dataset)
        assert actual["a"].shape == (3,)
        assert actual["b"].shape == (3,)

        actual = self.uds.ugrid.sel(x=slice(0, 1), y=slice(0, 2))
        assert isinstance(actual, xugrid.UgridDataset)
        assert actual["a"].shape == (2,)
        assert actual["b"].shape == (2,)
        assert actual.ugrid.grid.n_face == 2

        actual = self.uds.ugrid.sel(x=slice(0, 1), y=slice(1, None))
        assert isinstance(actual, xugrid.UgridDataset)
        assert actual["a"].shape == (1,)
        assert actual["b"].shape == (1,)
        assert actual.ugrid.grid.n_face == 1

    def test_crs(self):
        assert self.uds.ugrid.crs is None

    def test_to_geodataframe(self):
        gdf = self.uds.ugrid.to_geodataframe("mesh2d_nFaces")
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert (gdf.geometry.geom_type == "Polygon").all()

    def test_connected_components(self):
        actual = self.uds.ugrid.connected_components()
        assert isinstance(actual, xugrid.UgridDataArray)
        assert np.allclose(actual, 0)

    def test_reverse_cuthill_mckee(self):
        actual = self.uds.ugrid.reverse_cuthill_mckee()
        assert isinstance(actual, xugrid.UgridDataset)

    def test_laplace_interpolate(self):
        with pytest.raises(NotImplementedError):
            self.uds.ugrid.laplace_interpolate(direct_solve=True)


def test_to_dataset():
    uds = xugrid.UgridDataset(UGRID_DS)
    assert uds.ugrid.to_dataset() == UGRID_DS


def test_open_dataset(tmp_path):
    path = tmp_path / "ugrid-dataset.nc"
    uds = xugrid.UgridDataset(UGRID_DS)
    uds.ugrid.to_netcdf(path)

    back = xugrid.open_dataset(path)
    assert isinstance(back, xugrid.UgridDataset)
    assert "b" in back
    assert "mesh2d_face_nodes" in back.ugrid.grid.dataset
    assert "mesh2d_face_nodes" not in back.ugrid.obj


def test_open_dataarray_roundtrip(tmp_path):
    path = tmp_path / "ugrid-dataset.nc"
    uds = xugrid.UgridDataset(UGRID_DS)
    uds.ugrid.to_netcdf(path)
    with pytest.raises(ValueError, match="Given file dataset contains more than one"):
        xugrid.open_dataarray(path)

    path = tmp_path / "ugrid-dataarray.nc"
    uds["a"].ugrid.to_netcdf(path)
    # TODO: remove topology_variable from dataset as well!
    # back = xugrid.open_dataarray(path)
    # assert isinstance(back, xugrid.UgridDataArray)
    # assert back.name == "a"


def test_open_mfdataset(tmp_path):
    path1 = tmp_path / "ugrid-dataset_1.nc"
    path2 = tmp_path / "ugrid-dataset_2.nc"
    uds = xugrid.UgridDataset(UGRID_DS)
    uda1 = uds["a"].expand_dims(dim="layer")
    uda2 = uds["a"].expand_dims(dim="layer")
    uda1 = uda1.assign_coords(layer=[1])
    uda2 = uda1.assign_coords(layer=[2])
    uda1.ugrid.to_netcdf(path1)
    uda2.ugrid.to_netcdf(path2)
    back = xugrid.open_mfdataset([path1, path2])
    assert isinstance(back, xugrid.UgridDataset)
    assert "a" in back
    assert tuple(back["a"].dims) == ("layer", "mesh2d_nFaces")

    with pytest.raises(ValueError, match="data_vars kwargs is not supported"):
        back = xugrid.open_mfdataset([path1, path2], data_vars="all")


def test_zarr_roundtrip(tmp_path):
    path = tmp_path / "ugrid-dataset.zarr"
    uds = xugrid.UgridDataset(UGRID_DS)
    uds.ugrid.to_zarr(path)

    back = xugrid.open_zarr(path)
    assert isinstance(back, xugrid.UgridDataset)
    assert "a" in back
    assert "b" in back
    assert "mesh2d_face_nodes" in back.ugrid.grid.dataset
    assert "mesh2d_face_nodes" not in back.ugrid.obj


def test_func_like():
    uds = xugrid.UgridDataset(UGRID_DS)

    fullda = xugrid.full_like(uds["a"], 2)
    assert isinstance(fullda, xugrid.UgridDataArray)
    assert (fullda == 2).all()
    # Topology should be untouched
    assert fullda.ugrid.grid.dataset == uds.ugrid.grid.dataset

    fullds = xugrid.full_like(uds, 2)
    assert isinstance(fullds, xugrid.UgridDataset)
    assert (fullds["a"] == 2).all()
    assert (fullds["b"] == 2).all()
    assert fullds.ugrid.grid.dataset == uds.ugrid.grid.dataset

    fullda = xugrid.zeros_like(uds["a"])
    assert isinstance(fullda, xugrid.UgridDataArray)
    assert (fullda == 0).all()

    fullda = xugrid.ones_like(uds["a"])
    assert isinstance(fullda, xugrid.UgridDataArray)
    assert (fullda == 1).all()
