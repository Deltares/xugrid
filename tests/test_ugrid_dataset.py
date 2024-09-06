import warnings

import dask
import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import pytest
import shapely
import xarray as xr

import xugrid


def GRID():
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
    return xugrid.Ugrid2d(
        node_x=VERTICES[:, 0],
        node_y=VERTICES[:, 1],
        fill_value=-1,
        face_node_connectivity=FACES,
    )


def DARRAY():
    return xr.DataArray(
        data=np.ones(GRID().n_face),
        dims=[GRID().face_dimension],
    )


def UGRID_DS():
    ds = GRID().to_dataset()
    ds["a"] = DARRAY()
    ds["b"] = DARRAY() * 2
    return ds


def ugrid1d_ds():
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
    )
    ds = grid.to_dataset()
    ds["a1d"] = xr.DataArray([1.0, 2.0, 3.0], dims=[grid.node_dimension])
    ds["b1d"] = xr.DataArray([1.0, 2.0], dims=[grid.edge_dimension])
    return xugrid.UgridDataset(ds)


def test_properties():
    uda = xugrid.UgridDataArray(DARRAY(), GRID())
    uds = xugrid.UgridDataset(UGRID_DS())

    for item in (uda, uda.ugrid, uds, uds.ugrid):
        assert isinstance(getattr(item, "grid"), xugrid.Ugrid2d)
        grids = getattr(item, "grids")
        assert isinstance(grids, list)
        assert isinstance(grids[0], xugrid.Ugrid2d)

    assert isinstance(uda.obj, xr.DataArray)
    assert isinstance(uda.ugrid.obj, xr.DataArray)
    assert isinstance(uds.obj, xr.Dataset)
    assert isinstance(uds.ugrid.obj, xr.Dataset)


def test_xarray_property_setter():
    uda = xugrid.UgridDataArray(DARRAY(), GRID())
    uda.name = "new_name"
    assert uda.name == "new_name"


def test_init_errors():
    with pytest.raises(TypeError, match="obj must be xarray.DataArray"):
        xugrid.UgridDataArray(0, GRID())
    with pytest.raises(TypeError, match="grid must be Ugrid1d or Ugrid2d"):
        xugrid.UgridDataArray(DARRAY(), 0)

    with pytest.raises(ValueError, match="At least either obj or grids is required"):
        xugrid.UgridDataset()
    with pytest.raises(TypeError, match="obj must be xarray.Dataset"):
        xugrid.UgridDataset(0, GRID())
    with pytest.raises(TypeError, match="grid must be Ugrid1d or Ugrid2d"):
        xugrid.UgridDataset(xr.Dataset(), 0)


class TestUgridDataArray:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.uda = xugrid.UgridDataArray(DARRAY(), GRID())

    def test_properties(self):
        assert self.uda.ugrid.name == "mesh2d"
        assert self.uda.ugrid.names == ["mesh2d"]
        assert self.uda.ugrid.topology == {"mesh2d": self.uda.ugrid.grid}

    def test_init(self):
        assert isinstance(self.uda.ugrid.obj, xr.DataArray)
        assert isinstance(self.uda.ugrid.grid, xugrid.Ugrid2d)
        assert self.uda.grid.face_dimension in self.uda.coords

    def test_from_data(self):
        grid = self.uda.ugrid.grid
        uda = xugrid.UgridDataArray.from_data(np.zeros(grid.n_node), grid, facet="node")
        assert isinstance(uda, xugrid.UgridDataArray)

    def test_reinit_error(self):
        # Should not be able to initialize using a UgridDataArray.
        with pytest.raises(TypeError, match="obj must be xarray.DataArray"):
            xugrid.UgridDataArray(self.uda, GRID())

    def test_dunder_forward(self):
        assert isinstance(bool(self.uda[0]), bool)
        assert isinstance(int(self.uda[0]), int)
        assert isinstance(float(self.uda[0]), float)

    def test_repr(self):
        assert self.uda.__repr__() == self.uda.obj.__repr__()

    def test_getattr(self):
        # Get an attribute
        assert isinstance(self.uda.dims, tuple)
        # Only DataArrays should be returned as UgridDataArrays, other object
        # should remain untouched.
        assert self.uda.dims == self.uda.ugrid.obj.dims
        assert isinstance(self.uda.data, np.ndarray)
        # So are functions
        assert isinstance(self.uda.isnull(), xugrid.UgridDataArray)
        # obj should be accessible
        assert isinstance(self.uda.obj, xr.DataArray)

    def test_ugrid_accessor(self):
        assert isinstance(self.uda.ugrid, xugrid.UgridDataArrayAccessor)

    def test_rename(self):
        renamed = self.uda.ugrid.rename("renamed")
        assert "renamed_nFaces" in renamed.dims

    def test_from_structured(self):
        da = xr.DataArray([0.0, 1.0, 2.0], {"x": [5.0, 10.0, 15.0]}, ["x"])
        with pytest.raises(ValueError, match="Last two dimensions of da"):
            xugrid.UgridDataArray.from_structured(da)

        da = xr.DataArray(
            data=np.arange(2 * 3 * 4).reshape((2, 3, 4)),
            coords={"layer": [1, 2], "y": [5.0, 10.0, 15.0], "x": [2.0, 4.0, 6.0, 8.0]},
            dims=["layer", "y", "x"],
            name="grid",
        )
        uda = xugrid.UgridDataArray.from_structured(da)
        assert isinstance(uda, xugrid.UgridDataArray)
        assert uda.name == "grid"
        assert uda.dims == ("layer", "mesh2d_nFaces")
        assert uda.shape == (2, 12)
        assert np.allclose(uda.ugrid.sel(x=2.0, y=5.0), [[0], [12]])
        # Check whether flipping the y-axis doesn't cause any problems
        flipped = da.isel(y=slice(None, None, -1))
        uda = xugrid.UgridDataArray.from_structured(flipped)
        assert np.allclose(uda.ugrid.sel(x=2.0, y=5.0), [[0], [12]])

    def test_from_structured_multicoord(self):
        da = xr.DataArray(
            data=[[0, 1], [2, 3]],
            coords={
                "yc": (("y", "x"), [[12.0, 11.0], [12.0, 11.0]]),
                "xc": (("y", "x"), [[10.0, 12.0], [10.0, 12.0]]),
            },
            dims=("y", "x"),
        )
        uda = xugrid.UgridDataArray.from_structured(da)
        assert isinstance(uda, xugrid.UgridDataArray)
        assert np.array_equal(np.unique(uda.ugrid.grid.node_x), [-0.5, 0.5, 1.5])
        assert np.array_equal(uda.data, [0, 1, 2, 3])

        uda = xugrid.UgridDataArray.from_structured(da, x="xc", y="yc")
        assert isinstance(uda, xugrid.UgridDataArray)
        assert np.array_equal(np.unique(uda.ugrid.grid.node_x), [9.0, 11.0, 13.0])
        assert np.array_equal(uda.data, [0, 1, 2, 3])

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
        actual = self.uda.isel({GRID().face_dimension: [0, 1]})
        assert isinstance(actual, xugrid.UgridDataArray)
        assert actual.shape == (2,)
        assert actual.ugrid.grid.n_face == 2

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

    def test_intersect_line(self):
        p0 = (0.0, 0.0)
        p1 = (2.0, 2.0)
        actual = self.uda.ugrid.intersect_line(start=p0, end=p1)
        sqrt2 = np.sqrt(2.0)
        assert isinstance(actual, xr.DataArray)
        assert actual.dims == ("mesh2d_nFaces",)
        assert np.allclose(actual["mesh2d_x"], [0.5, 1.25])
        assert np.allclose(actual["mesh2d_y"], [0.5, 1.25])
        assert np.allclose(actual["mesh2d_s"], [0.5 * sqrt2, 1.25 * sqrt2])

    def test_intersect_linestring(self):
        linestring = shapely.geometry.LineString(
            [
                [0.5, 0.5],
                [1.5, 0.5],
                [1.5, 1.5],
            ]
        )
        actual = self.uda.ugrid.intersect_linestring(linestring)
        assert isinstance(actual, xr.DataArray)
        assert actual.dims == ("mesh2d_nFaces",)
        assert np.allclose(actual["mesh2d_x"], [0.75, 1.25, 1.5, 1.5])
        assert np.allclose(actual["mesh2d_y"], [0.5, 0.5, 0.75, 1.25])
        assert np.allclose(actual["mesh2d_s"], [0.25, 0.75, 1.25, 1.75])

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

    def test_partitioning(self):
        partitions = self.uda.ugrid.partition(n_part=2)
        assert len(partitions) == 2
        for partition in partitions:
            assert isinstance(partition, xugrid.UgridDataArray)
            assert partition.name == self.uda.name

    def test_reindex_like(self):
        back = self.uda.ugrid.reindex_like(self.uda)
        assert isinstance(back, xugrid.UgridDataArray)
        back = self.uda.ugrid.reindex_like(self.uda.ugrid.grid)
        assert isinstance(back, xugrid.UgridDataArray)

    def test_crs(self):
        uda = self.uda
        crs = uda.ugrid.crs
        assert crs == {"mesh2d": None}

        uda.ugrid.set_crs(epsg=28992)
        assert uda.ugrid.crs == {"mesh2d": pyproj.CRS.from_epsg(28992)}

        result = uda.ugrid.to_crs(epsg=32631)
        assert uda is not result
        assert np.allclose(uda.values, result.values)
        assert uda.ugrid.crs == {"mesh2d": pyproj.CRS.from_epsg(28992)}
        assert result.ugrid.crs == {"mesh2d": pyproj.CRS.from_epsg(32631)}
        assert not np.array_equal(result.ugrid.grid.node_x, uda.ugrid.grid.node_x)
        assert not np.array_equal(result.ugrid.grid.node_y, uda.ugrid.grid.node_y)

    def test_is_geographic(self):
        uda = self.uda
        assert uda.grid.is_geographic is False

        uda.ugrid.set_crs(epsg=4326)
        assert uda.grid.is_geographic is True

        result = uda.ugrid.to_crs(epsg=28992)
        assert result.grid.is_geographic is False

    def test_to_geodataframe(self):
        with pytest.raises(ValueError, match="unable to convert unnamed"):
            self.uda.ugrid.to_geodataframe()
        uda2 = self.uda.copy()
        uda2.ugrid.obj.name = "test"
        uda2.ugrid.set_crs(epsg=28992)
        gdf = uda2.ugrid.to_geodataframe("mesh2d")
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert (gdf.geometry.geom_type == "Polygon").all()
        assert gdf.crs.to_epsg() == 28992

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

    def test_broadcasted_laplace_interpolate(self):
        uda2 = self.uda.copy()
        uda2.obj[:-2] = np.nan
        multiplier = xr.DataArray(
            np.ones((3, 2)),
            coords={"time": [0, 1, 2], "layer": [1, 2]},
            dims=("time", "layer"),
        )
        nd_uda2 = uda2 * multiplier
        actual = nd_uda2.ugrid.laplace_interpolate(direct_solve=True)
        assert isinstance(actual, xugrid.UgridDataArray)
        assert np.allclose(actual, 1.0)
        assert set(actual.dims) == set(nd_uda2.dims)

        # Test delayed evaluation too.
        nd_uda2 = uda2 * multiplier.chunk({"time": 1})
        actual = nd_uda2.ugrid.laplace_interpolate(direct_solve=True)
        assert isinstance(actual, xugrid.UgridDataArray)
        assert set(actual.dims) == set(nd_uda2.dims)
        assert isinstance(actual.data, dask.array.Array)

    def test_to_dataset(self):
        uda2 = self.uda.copy()
        uda2.ugrid.obj.name = "test"
        actual = uda2.to_dataset()
        assert isinstance(actual, xugrid.UgridDataset)

    def test_ugrid_to_dataset(self):
        uda2 = self.uda.copy()
        uda2.ugrid.obj.name = "test"
        ds = uda2.ugrid.to_dataset(optional_attributes=True)
        assert "mesh2d_edge_nodes" in ds
        assert "mesh2d_face_nodes" in ds
        assert "mesh2d_face_edges" in ds
        assert "mesh2d_face_faces" in ds
        assert "mesh2d_edge_faces" in ds
        assert "mesh2d_boundary_nodes" in ds
        assert "mesh2d_face_x" in ds
        assert "mesh2d_face_y" in ds
        assert "mesh2d_edge_x" in ds
        assert "mesh2d_edge_y" in ds

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

    def test_bounds(self):
        assert self.uda.ugrid.bounds == {"mesh2d": (0.0, 0.0, 2.0, 2.0)}

    def test_total_bounds(self):
        assert self.uda.ugrid.total_bounds == (0.0, 0.0, 2.0, 2.0)

    def test_assign_coords(self):
        with pytest.raises(
            ValueError,
            match="cannot add coordinates with new dimensions to a DataArray",
        ):
            self.uda.ugrid.assign_edge_coords()
        with pytest.raises(
            ValueError,
            match="cannot add coordinates with new dimensions to a DataArray",
        ):
            self.uda.ugrid.assign_node_coords()

        with_coords = self.uda.ugrid.assign_face_coords()
        assert "mesh2d_face_x" in with_coords.coords
        assert "mesh2d_face_y" in with_coords.coords

    def test_plot_with_chunks(self, tmp_path):
        time = xr.DataArray([0.0, 1.0, 2.0], coords={"time": [0, 1, 2]})
        uda = (self.uda * time).transpose()
        uda.name = "test"

        path = tmp_path / "test.nc"
        uda.ugrid.to_netcdf(path)
        back = xugrid.open_dataarray(path, chunks={"time": 1})
        primitive = back.isel(time=0).ugrid.plot()
        assert primitive is not None

    def test_plot_contourf_with_chunks(self, tmp_path):
        time = xr.DataArray([0.0, 1.0, 2.0], coords={"time": [0, 1, 2]})
        uda = (self.uda * time).transpose()
        uda.name = "test"

        path = tmp_path / "test.nc"
        uda.ugrid.to_netcdf(path)
        back = xugrid.open_dataarray(path, chunks={"time": 1})
        primitive = back.isel(time=0).ugrid.plot.contourf()
        assert primitive is not None


class TestUgridDataset:
    @pytest.fixture(autouse=True)
    def setup(self):
        ds = xr.Dataset()
        ds["a"] = DARRAY()
        ds["b"] = DARRAY() * 2
        self.uds = xugrid.UgridDataset(ds, GRID())

    def test_properties(self):
        assert self.uds.ugrid.name == "mesh2d"
        assert self.uds.ugrid.names == ["mesh2d"]
        assert self.uds.ugrid.topology == {"mesh2d": self.uds.ugrid.grid}

    def test_init(self):
        assert isinstance(self.uds.ugrid.obj, xr.Dataset)
        assert isinstance(self.uds.ugrid.grids[0], xugrid.Ugrid2d)
        # Try alternative initialization
        uds = xugrid.UgridDataset(grids=GRID())
        assert isinstance(uds, xugrid.UgridDataset)
        uds = xugrid.UgridDataset(grids=[GRID()])
        assert isinstance(uds, xugrid.UgridDataset)
        uds["a"] = DARRAY()
        assert "a" in uds.ugrid.obj

    def test_reinit_error(self):
        with pytest.raises(TypeError, match="obj must be xarray.Dataset"):
            xugrid.UgridDataset(self.uds, GRID())

    def test_init_from_dataset_only(self):
        uds = xugrid.UgridDataset(UGRID_DS())
        assert isinstance(uds, xugrid.UgridDataset)
        assert "a" in uds.ugrid.obj
        assert "b" in uds.ugrid.obj
        assert "mesh2d_face_nodes" in uds.ugrid.grids[0].to_dataset()
        assert "mesh2d_face_nodes" not in uds.ugrid.obj

    def test_repr(self):
        assert self.uds.__repr__() == self.uds.obj.__repr__()

    def test_getitem(self):
        assert "a" in self.uds
        assert "b" in self.uds
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
        assert isinstance(self.uds.notnull(), xugrid.UgridDataset)
        # obj should be accessible
        assert isinstance(self.uds.obj, xr.Dataset)

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
        assert isinstance(self.uds.ugrid, xugrid.UgridDatasetAccessor)

    def test_rename(self):
        renamed = self.uds.ugrid.rename("renamed")
        assert "renamed_nFaces" in renamed.dims

        renamed = self.uds.ugrid.rename({"mesh2d": "renamed"})
        assert "renamed_nFaces" in renamed.dims

        # This name doesn't exist, shouldn't change
        renamed = self.uds.ugrid.rename({"mesh1d": "renamed"})
        assert "mesh2d_nFaces" in renamed.dims

        with pytest.raises(TypeError):
            self.uds.ugrid.rename(["mesh1d", "mesh2d"])

    def test_from_geodataframe(self):
        xy = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 0.0],
            ]
        )
        polygon = shapely.polygons(xy)
        df = pd.DataFrame({"a": [1.0], "b": [2.0]})
        gdf = gpd.GeoDataFrame(df, geometry=[polygon])
        uds = xugrid.UgridDataset.from_geodataframe(gdf)
        assert isinstance(uds, xugrid.UgridDataset)
        assert "a" in uds
        assert "b" in uds

    # Accessor tests
    def test_isel(self):
        actual = self.uds.isel({GRID().face_dimension: [0, 1]})
        assert isinstance(actual, xugrid.UgridDataset)
        assert actual.ugrid.grids[0].n_face == 2
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
        assert actual.ugrid.grids[0].n_face == 2

        actual = self.uds.ugrid.sel(x=slice(0, 1), y=slice(1, None))
        assert isinstance(actual, xugrid.UgridDataset)
        assert actual["a"].shape == (1,)
        assert actual["b"].shape == (1,)
        assert actual.ugrid.grids[0].n_face == 1

    def test_rasterize(self):
        actual = self.uds.ugrid.rasterize(resolution=0.5)
        x = [0.25, 0.75, 1.25, 1.75]
        y = [1.75, 1.25, 0.75, 0.25]
        assert isinstance(actual, xr.Dataset)
        assert actual["a"].shape == (4, 4)
        assert actual["b"].shape == (4, 4)
        assert np.allclose(actual["x"], x)
        assert np.allclose(actual["y"], y)

        da = xr.DataArray(np.empty((4, 4)), {"y": y, "x": x}, ["y", "x"])
        actual = self.uds.ugrid.rasterize_like(other=da)
        assert isinstance(actual, xr.Dataset)
        assert actual["a"].shape == (4, 4)
        assert actual["b"].shape == (4, 4)
        assert np.allclose(actual["x"], x)
        assert np.allclose(actual["y"], y)

    def test_intersect_line(self):
        p0 = (0.0, 0.0)
        p1 = (2.0, 2.0)
        actual = self.uds.ugrid.intersect_line(start=p0, end=p1)
        sqrt2 = np.sqrt(2.0)
        assert isinstance(actual, xr.Dataset)
        assert actual.dims == {"mesh2d_nFaces": 2}
        assert np.allclose(actual["mesh2d_x"], [0.5, 1.25])
        assert np.allclose(actual["mesh2d_y"], [0.5, 1.25])
        assert np.allclose(actual["mesh2d_s"], [0.5 * sqrt2, 1.25 * sqrt2])
        assert "a" in actual
        assert "b" in actual

    def test_intersect_linestring(self):
        linestring = shapely.geometry.LineString(
            [
                [0.5, 0.5],
                [1.5, 0.5],
                [1.5, 1.5],
            ]
        )
        actual = self.uds.ugrid.intersect_linestring(linestring)
        assert isinstance(actual, xr.Dataset)
        assert actual.dims == {"mesh2d_nFaces": 4}
        assert np.allclose(actual["mesh2d_x"], [0.75, 1.25, 1.5, 1.5])
        assert np.allclose(actual["mesh2d_y"], [0.5, 0.5, 0.75, 1.25])
        assert np.allclose(actual["mesh2d_s"], [0.25, 0.75, 1.25, 1.75])
        assert "a" in actual
        assert "b" in actual

    def test_partitioning(self):
        partitions = self.uds.ugrid.partition(n_part=2)
        assert len(partitions) == 2
        for partition in partitions:
            assert isinstance(partition, xugrid.UgridDataset)
            assert "a" in partition
            assert "b" in partition

    def test_reindex_like(self):
        back = self.uds.ugrid.reindex_like(self.uds)
        assert isinstance(back, xugrid.UgridDataset)
        back = self.uds.ugrid.reindex_like(self.uds.ugrid.grid)
        assert isinstance(back, xugrid.UgridDataset)

    def test_crs(self):
        uds = self.uds
        crs = uds.ugrid.crs
        assert crs == {"mesh2d": None}

        with pytest.raises(ValueError, match="grid not found"):
            uds.ugrid.set_crs(epsg=28992, topology="grid")

        uds.ugrid.set_crs(epsg=28992, topology="mesh2d")
        assert uds.ugrid.crs == {"mesh2d": pyproj.CRS.from_epsg(28992)}

        uds.ugrid.set_crs(epsg=28992)
        assert uds.ugrid.crs == {"mesh2d": pyproj.CRS.from_epsg(28992)}

        with pytest.raises(ValueError, match="grid not found"):
            uds.ugrid.to_crs(epsg=32631, topology="grid")

        result = uds.ugrid.to_crs(epsg=32631, topology="mesh2d")
        assert uds is not result
        assert uds.ugrid.crs == {"mesh2d": pyproj.CRS.from_epsg(28992)}
        assert result.ugrid.crs == {"mesh2d": pyproj.CRS.from_epsg(32631)}

    def test_assign_coords(self):
        with_coords = (
            self.uds.ugrid.assign_edge_coords()
            .ugrid.assign_node_coords()
            .ugrid.assign_face_coords()
        )
        assert "mesh2d_node_x" in with_coords.coords
        assert "mesh2d_node_y" in with_coords.coords
        assert "mesh2d_edge_x" in with_coords.coords
        assert "mesh2d_edge_y" in with_coords.coords
        assert "mesh2d_face_x" in with_coords.coords
        assert "mesh2d_face_y" in with_coords.coords

    def test_to_geodataframe(self):
        gdf = self.uds.ugrid.to_geodataframe()
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert (gdf.geometry.geom_type == "Polygon").all()

        # Now test with an empty dataset
        grid = self.uds.grid
        empty = xugrid.UgridDataset(grid.to_dataset())
        empty.ugrid.set_crs(epsg=28992)
        gdf = empty.ugrid.to_geodataframe()
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert gdf.crs.to_epsg() == 28992

    def test_bounds(self):
        assert self.uds.ugrid.bounds == {"mesh2d": (0.0, 0.0, 2.0, 2.0)}

    def test_total_bounds(self):
        assert self.uds.ugrid.total_bounds == (0.0, 0.0, 2.0, 2.0)


class TestMultiTopologyUgridDataset:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.uds = xugrid.UgridDataset(grids=GRID())
        uda = xugrid.UgridDataArray(DARRAY(), GRID())
        self.uds["a"] = uda

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
        )
        self.uds["b"] = xugrid.UgridDataArray(
            xr.DataArray(np.ones(grid.n_node), dims=[grid.node_dimension]),
            grid,
        )

    def test_grid_membership(self):
        assert len(self.uds.grids) == 2

    def test_grid_accessor__error(self):
        with pytest.raises(TypeError):
            self.uds.ugrid.grid

        with pytest.raises(TypeError):
            self.uds.grid

    def test_multi_topology_sel(self):
        result = self.uds.ugrid.sel(x=slice(-10, 10), y=slice(-10, 10))
        # Ensure both grids are still present
        assert len(result.ugrid.grids) == 2

    def test_reindex_like(self):
        back = self.uds.ugrid.reindex_like(self.uds)
        assert isinstance(back, xugrid.UgridDataset)


def test_multiple_coordinates():
    grid = GRID()
    ds = UGRID_DS()
    attrs = ds["mesh2d"].attrs
    attrs["node_coordinates"] += " mesh2d_node_lon mesh2d_node_lat"
    ds = ds.assign_coords(
        mesh2d_node_lon=(grid.node_dimension, np.arange(grid.n_node)),
        mesh2d_node_lat=(grid.node_dimension, np.arange(grid.n_node)),
    )
    ds["mesh2d_node_lon"].attrs["standard_name"] = "longitude"
    ds["mesh2d_node_lat"].attrs["standard_name"] = "latitude"
    assert ds.ugrid_roles.coordinates == {
        "mesh2d": {
            "node_coordinates": (
                ["mesh2d_node_x", "mesh2d_node_lon"],
                ["mesh2d_node_y", "mesh2d_node_lat"],
            )
        }
    }

    # Make sure everything goes right when subsetting: tests whether all
    # attributes and grid indexes are propagated to the new grid object.
    uds = xugrid.UgridDataset(ds)
    subset = uds.isel({grid.face_dimension: [0, 1]})
    assert isinstance(subset, xugrid.UgridDataset)
    subset_ds = uds.ugrid.to_dataset()
    assert "mesh2d_node_x" in subset_ds
    assert "mesh2d_node_y" in subset_ds
    assert "mesh2d_node_lon" in subset_ds
    assert "mesh2d_node_lat" in subset_ds
    assert subset_ds["mesh2d"].attrs["node_coordinates"] == attrs["node_coordinates"]


def test_ugrid_to_dataset():
    uds = xugrid.UgridDataset(UGRID_DS())
    assert uds.ugrid.to_dataset() == UGRID_DS()

    ds = uds.ugrid.to_dataset(optional_attributes=True)
    assert "mesh2d_edge_nodes" in ds
    assert "mesh2d_face_nodes" in ds
    assert "mesh2d_face_edges" in ds
    assert "mesh2d_face_faces" in ds
    assert "mesh2d_edge_faces" in ds
    assert "mesh2d_boundary_nodes" in ds
    assert "mesh2d_face_x" in ds
    assert "mesh2d_face_y" in ds
    assert "mesh2d_edge_x" in ds
    assert "mesh2d_edge_y" in ds


def test_open_dataset(tmp_path):
    path = tmp_path / "ugrid-dataset.nc"
    uds = xugrid.UgridDataset(UGRID_DS())
    uds.ugrid.to_netcdf(path)

    back = xugrid.open_dataset(path)
    assert isinstance(back, xugrid.UgridDataset)
    assert "b" in back
    assert "mesh2d_face_nodes" in back.ugrid.grids[0].to_dataset()
    assert "mesh2d_face_nodes" not in back.ugrid.obj


def test_open_dataset_cast_invalid(tmp_path):
    grid = GRID()
    vorgrid = grid.tesselate_centroidal_voronoi()
    path = tmp_path / "voronoi-grid.nc"
    vorgrid.to_dataset().to_netcdf(path)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        xugrid.open_dataset(path)


def test_open_dataarray_roundtrip(tmp_path):
    path = tmp_path / "ugrid-dataset.nc"
    uds = xugrid.UgridDataset(UGRID_DS())
    uds.ugrid.to_netcdf(path)
    with pytest.raises(ValueError, match="Given file dataset contains more than one"):
        xugrid.open_dataarray(path)

    path = tmp_path / "ugrid-dataarray.nc"
    uds["a"].ugrid.to_netcdf(path)
    back = xugrid.open_dataarray(path)
    assert isinstance(back, xugrid.UgridDataArray)
    assert back.name == "a"


def test_open_mfdataset(tmp_path):
    path1 = tmp_path / "ugrid-dataset_1.nc"
    path2 = tmp_path / "ugrid-dataset_2.nc"
    uds = xugrid.UgridDataset(UGRID_DS())
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


def test_keep_attrs():
    uds = xugrid.UgridDataset(UGRID_DS())
    uds.attrs["date_created"] = "today"
    ds = uds.ugrid.to_dataset()
    assert ds.attrs["date_created"] == "today"

    uds = ugrid1d_ds()
    uds.attrs["date_created"] = "today"
    ds = uds.ugrid.to_dataset()
    assert ds.attrs["date_created"] == "today"


def test_zarr_roundtrip(tmp_path):
    path = tmp_path / "ugrid-dataset.zarr"
    uds = xugrid.UgridDataset(UGRID_DS())
    uds.ugrid.to_zarr(path)

    back = xugrid.open_zarr(path)
    assert isinstance(back, xugrid.UgridDataset)
    assert "a" in back
    assert "b" in back
    assert "mesh2d_face_nodes" in back.ugrid.grids[0].to_dataset()
    assert "mesh2d_face_nodes" not in back.ugrid.obj


def test_func_like():
    uds = xugrid.UgridDataset(UGRID_DS())

    fullda = xugrid.full_like(uds["a"], 2)
    assert isinstance(fullda, xugrid.UgridDataArray)
    assert (fullda == 2).all()
    # Topology should be untouched
    assert fullda.ugrid.grid.to_dataset() == uds.ugrid.grids[0].to_dataset()

    fullds = xugrid.full_like(uds, 2)
    assert isinstance(fullds, xugrid.UgridDataset)
    assert (fullds["a"] == 2).all()
    assert (fullds["b"] == 2).all()
    assert fullds.ugrid.grids[0].to_dataset == uds.ugrid.grids[0].to_dataset()

    fullda = xugrid.zeros_like(uds["a"])
    assert isinstance(fullda, xugrid.UgridDataArray)
    assert (fullda == 0).all()

    fullda = xugrid.ones_like(uds["a"])
    assert isinstance(fullda, xugrid.UgridDataArray)
    assert (fullda == 1).all()


def test_concat():
    uds = xugrid.UgridDataset(UGRID_DS())
    uda = uds["a"]
    uda1 = uda.assign_coords(layer=1)
    uda2 = uda.assign_coords(layer=2)
    result = xugrid.concat([uda1, uda2], dim="layer")
    assert result.dims == ("layer", "mesh2d_nFaces")
    assert np.array_equal(result["layer"], [1, 2])

    uds1d = ugrid1d_ds()
    uda3 = uds1d["a1d"].assign_coords(layer=2)
    with pytest.raises(ValueError, match="All UgridDataArrays must have the same grid"):
        xugrid.concat([uda1, uda3], dim="layer")

    # test issue 206 resolved
    # https://github.com/Deltares/xugrid/issues/206
    result = xugrid.concat([uda1, uda2.copy()], dim="foo")
    assert len(result.grids) == 1


def test_multiple_topology_errors():
    # Create a dataset with two UGRID topologies:
    uds = ugrid1d_ds()
    uds["a"] = xugrid.UgridDataset(UGRID_DS())["a"]

    with pytest.raises(TypeError, match="Can only access grid topology"):
        uds.ugrid.grid

    with pytest.raises(TypeError, match="Can only access grid name"):
        uds.ugrid.name

    with pytest.raises(TypeError, match="Can only rename with a single name"):
        uds.ugrid.rename("renamed")


def test_merge():
    uds2d = xugrid.UgridDataset(UGRID_DS())
    uds1d = ugrid1d_ds()
    merged = xugrid.merge([uds2d, uds1d])
    assert isinstance(merged, xugrid.UgridDataset)
    assert len(merged.grids) == 2


def get_ugrid_fillvaluem999_startindex1_ds():
    """
    Return a minimal dataset with a specific fill value.

    This is a very minimal but comparable dataset to Grevelingen_0002_map.nc
    (FM output).

    * It contains both triangles and squares.
    * The fillvalue of the connectivity arrays is -999
    * The start_index of the connectivity arrays is 1
    """

    ds2 = xr.Dataset()

    mesh2d_attrs = {
        "cf_role": "mesh_topology",
        "topology_dimension": 2,
        "node_dimension": "nmesh2d_node",
        "edge_dimension": "nmesh2d_edge",
        "face_dimension": "nmesh2d_face",
        "max_face_nodes_dimension": "max_nmesh2d_face_nodes",
        "face_node_connectivity": "mesh2d_face_nodes",
        "edge_node_connectivity": "mesh2d_edge_nodes",
        "node_coordinates": "mesh2d_node_x mesh2d_node_y",
        "name": "mesh2d",
    }
    ds2["mesh2d"] = xr.DataArray(np.array(0, dtype=int), attrs=mesh2d_attrs)

    node_x = np.array(
        [
            48231.65428822,
            48264.81400401,
            48350.0,
            48450.0,
            48235.96727817,
            48287.80875187,
            48306.85396605,
            48400.0,
            48500.0,
            48273.38390534,
            48450.0,
            48350.77800678,
            48342.73889736,
        ]
    )
    node_x_attrs = {
        "units": "m",
        "standard_name": "projection_x_coordinate",
        "long_name": "x-coordinate of mesh nodes",
        "mesh": "mesh2d",
        "location": "node",
    }
    ds2["mesh2d_node_x"] = xr.DataArray(
        node_x, dims=(mesh2d_attrs["node_dimension"]), attrs=node_x_attrs
    )

    node_y = np.array(
        [
            419541.94243367,
            419605.7455447,
            419605.94894,
            419605.94894,
            419454.05602638,
            419488.96173469,
            419552.39342675,
            419519.3464,
            419519.3464,
            419418.34074563,
            419432.74386,
            419447.63359856,
            419378.75075546,
        ]
    )
    node_y_attrs = {
        "units": "m",
        "standard_name": "projection_y_coordinate",
        "long_name": "y-coordinate of mesh nodes",
        "mesh": "mesh2d",
        "location": "node",
    }
    ds2["mesh2d_node_y"] = xr.DataArray(
        node_y, dims=(mesh2d_attrs["node_dimension"]), attrs=node_y_attrs
    )

    fnc = np.array(
        [
            [1, 7, 2, -999],
            [3, 2, 7, -999],
            [8, 4, 3, -999],
            [1, 6, 7, -999],
            [8, 3, 7, -999],
            [9, 4, 8, -999],
            [5, 10, 6, -999],
            [8, 7, 6, 12],
            [11, 9, 8, -999],
            [10, 12, 6, -999],
            [11, 8, 12, -999],
            [10, 13, 12, -999],
            [11, 12, 13, -999],
        ],
        dtype=int,
    )
    fnc_attrs = {
        "_FillValue": -999,
        "cf_role": "face_node_connectivity",
        "start_index": 1,
        "coordinates": "mesh2d_face_x mesh2d_face_y",
    }
    ds2["mesh2d_face_nodes"] = xr.DataArray(
        fnc,
        dims=(mesh2d_attrs["face_dimension"], mesh2d_attrs["max_face_nodes_dimension"]),
        attrs=fnc_attrs,
    )

    enc = np.array(
        [
            [1, 2],
            [1, 6],
            [1, 7],
            [2, 3],
            [2, 7],
            [3, 4],
            [3, 7],
            [3, 8],
            [4, 8],
            [4, 9],
            [5, 6],
            [5, 10],
            [6, 7],
            [6, 10],
            [6, 12],
            [7, 8],
            [8, 9],
            [8, 11],
            [8, 12],
            [9, 11],
            [10, 12],
            [10, 13],
            [11, 12],
            [11, 13],
            [12, 13],
        ],
        dtype=int,
    )
    enc_attrs = {
        "cf_role": "edge_node_connectivity",
        "mesh": "mesh2d",
        "location": "edge",
        "long_name": "Mapping from every edge to the two nodes that it connects",
        "start_index": 1,
        "_FillValue": -999,
    }
    ds2["mesh2d_edge_nodes"] = xr.DataArray(
        enc, dims=(mesh2d_attrs["edge_dimension"], "two"), attrs=enc_attrs
    )

    # add dummy face variable in order to have a face dimension in the uds
    facevar = np.ones(shape=(ds2.sizes[mesh2d_attrs["face_dimension"]]))
    ds2["mesh2d_facevar"] = xr.DataArray(facevar, dims=(mesh2d_attrs["face_dimension"]))

    # add dummy nodevar to plot and trigger triangulation procedure
    nodevar = np.ones(shape=(ds2.sizes[mesh2d_attrs["node_dimension"]]))
    ds2["mesh2d_nodevar"] = xr.DataArray(nodevar, dims=(mesh2d_attrs["node_dimension"]))
    return ds2


def get_ugrid_fillvaluem999_startindex1_uds():
    # upon loading a dataset from a file, xarray decodes it, so we also do it here
    ds2 = get_ugrid_fillvaluem999_startindex1_ds()
    ds2_enc = xr.decode_cf(ds2)
    uds = xugrid.UgridDataset(ds2_enc)
    return uds


def test_fm_fillvalue_startindex_isel():
    """
    FM data has 1-based starting index and _FillValue -999, this raises several
    issues. Since it is not possible to generate a Ugrid2d with these
    attributes, we are testing with raw data
    """

    # xugrid 0.5.0 warns "RuntimeWarning: invalid value encountered in cast: cast = data.astype(dtype, copy=True)"
    uds = get_ugrid_fillvaluem999_startindex1_uds()

    # xugrid 0.6.0 raises "ValueError: Invalid edge_node_connectivity"
    uds.isel({uds.grid.face_dimension: [1]})

    # Check internal fill value. Should be FILL_VALUE
    grid = uds.ugrid.grid
    assert (grid.face_node_connectivity != -999).all()
    gridds = grid.to_dataset()
    # Should be set back to the origina fill value.
    assert (gridds["mesh2d_face_nodes"] != xugrid.constants.FILL_VALUE).all()

    # And similarly for the UgridAccessors.
    ds = uds.ugrid.to_dataset()
    assert (ds["mesh2d_face_nodes"] != xugrid.constants.FILL_VALUE).all()

    ds_uda = uds["mesh2d_facevar"].ugrid.to_dataset()
    assert (ds_uda["mesh2d_face_nodes"] != xugrid.constants.FILL_VALUE).all()


def test_fm_facenodeconnectivity_fillvalue():
    """
    FM data has 1-based starting index and _FillValue -999, this raises several
    issues. Since it is not possible to generate a Ugrid2d with these
    attributes, we are testing with raw data
    """

    # xugrid 0.5.0 warns "RuntimeWarning: invalid value encountered in cast: cast = data.astype(dtype, copy=True)"
    uds = get_ugrid_fillvaluem999_startindex1_uds()

    # xugrid 0.6.0 has -2 values in the array
    assert (uds.grid.face_node_connectivity != -2).all()


def test_periodic_conversion():
    vertices = np.array(
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
    faces = np.array(
        [
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [4, 5, 9, 8],
            [5, 6, 10, 9],
            [6, 7, 11, 10],
        ]
    )
    grid = xugrid.Ugrid2d(*vertices.T, -1, faces)
    da = xr.DataArray([0, 1, 2, 3, 4, 5], dims=(grid.face_dimension,))
    uda = xugrid.UgridDataArray(da, grid)
    periodic = uda.ugrid.to_periodic()
    back = periodic.ugrid.to_nonperiodic(xmax=3.0)
    assert isinstance(periodic, xugrid.UgridDataArray)
    assert isinstance(back, xugrid.UgridDataArray)
    back_grid = back.ugrid.grid
    assert back_grid.n_face == grid.n_face
    assert back_grid.n_edge == grid.n_edge
    assert back_grid.n_node == grid.n_node

    # Also test a multi-topology dataset The 1D grid should be skipped: it
    # doesn't implement anything for these conversions, but should simply be
    # added as-is to the result.
    uds = ugrid1d_ds()
    uds["a2d"] = uda
    periodic_ds = uds.ugrid.to_periodic()
    back_ds = periodic_ds.ugrid.to_nonperiodic(xmax=3.0)
    assert isinstance(periodic_ds, xugrid.UgridDataset)
    assert isinstance(back_ds, xugrid.UgridDataset)
    assert "a1d" in back_ds
    assert "a2d" in back_ds


def test_laplace_interpolate_facets():
    grid = GRID()
    node_uda = xugrid.UgridDataArray(
        xr.DataArray(np.ones(grid.n_node), dims=(grid.node_dimension,)),
        grid=grid,
    )
    edge_uda = xugrid.UgridDataArray(
        xr.DataArray(np.ones(grid.n_edge), dims=(grid.edge_dimension,)),
        grid=grid,
    )
    face_uda = xugrid.UgridDataArray(
        xr.DataArray(np.ones(grid.n_face), dims=(grid.face_dimension,)),
        grid=grid,
    )
    node_uda[:-1] = np.nan
    edge_uda[:-1] = np.nan
    face_uda[:-1] = np.nan

    for uda in (node_uda, face_uda):
        actual = uda.ugrid.laplace_interpolate(direct_solve=True)
        assert isinstance(actual, xugrid.UgridDataArray)
        assert np.allclose(actual, 1.0)

    msg = "Laplace interpolation along edges is not allowed."
    with pytest.raises(ValueError, match=msg):
        edge_uda.ugrid.laplace_interpolate(direct_solve=True)

    for uda in (node_uda, edge_uda, face_uda):
        actual = uda.ugrid.interpolate_na()
        assert isinstance(actual, xugrid.UgridDataArray)
        assert np.allclose(actual, 1.0)


def test_laplace_interpolate_1d():
    uda = ugrid1d_ds()["a1d"]
    uda[:] = 1.0
    uda[1] = np.nan
    actual = uda.ugrid.laplace_interpolate(direct_solve=True)
    assert isinstance(actual, xugrid.UgridDataArray)
    assert np.allclose(actual, 1.0)


def test_interpolate_na_1d():
    uda = ugrid1d_ds()["a1d"]
    with pytest.raises(ValueError, match='"abc" is not a valid interpolator.'):
        uda.ugrid.interpolate_na(method="abc")

    # Node data
    uda = ugrid1d_ds()["a1d"]
    uda[:] = 1.0
    uda[1] = np.nan
    actual = uda.ugrid.interpolate_na()
    assert isinstance(actual, xugrid.UgridDataArray)
    assert np.allclose(actual, 1.0)

    # Edge data
    uda = ugrid1d_ds()["b1d"]
    uda[:] = 1.0
    uda[1] = np.nan
    actual = uda.ugrid.interpolate_na()
    assert isinstance(actual, xugrid.UgridDataArray)
    assert np.allclose(actual, 1.0)

    # Check max_distance
    actual = uda.ugrid.interpolate_na(max_distance=0.5)
    assert np.isnan(actual[1])


def test_ugriddataset_wrap_twice(tmp_path):
    """
    in issue https://github.com/Deltares/xugrid/issues/208 wrapping a ds
    twice with UgridDataset resulted in "ValueError: connectivity contains negative values",
    because the original connectivity array in the xarray dataset was altered.
    This tests ensures that future changes will not cause this issue again.
    """
    ds_raw = get_ugrid_fillvaluem999_startindex1_ds()
    file_nc = tmp_path / "ugrid_fillvaluem999_startindex1.nc"
    ds_raw.to_netcdf(file_nc)
    ds = xr.open_dataset(file_nc)

    _ = xugrid.UgridDataset(ds)
    _ = xugrid.UgridDataset(ds)
