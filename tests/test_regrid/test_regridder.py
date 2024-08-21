import numpy as np
import pandas as pd
import pytest
import xarray as xr

import xugrid as xu
from xugrid import (
    BarycentricInterpolator,
    CentroidLocatorRegridder,
    OverlapRegridder,
    RelativeOverlapRegridder,
)


@pytest.mark.parametrize(
    "cls",
    [
        CentroidLocatorRegridder,
        OverlapRegridder,
        RelativeOverlapRegridder,
        BarycentricInterpolator,
    ],
)
def test_check_source_target_types(disk, cls):
    with pytest.raises(TypeError):
        cls(source=disk, target=1.0)
    with pytest.raises(TypeError):
        cls(source=1.0, target=disk)


@pytest.mark.parametrize(
    "regridder_class",
    [
        CentroidLocatorRegridder,
        OverlapRegridder,
        RelativeOverlapRegridder,
        BarycentricInterpolator,
    ],
)
def test_structured_to_unstructured(
    regridder_class,
    disk,
    quads_structured,
):
    regridder = regridder_class(quads_structured, disk)
    actual = regridder.regrid(quads_structured)
    assert isinstance(actual, xu.UgridDataArray)


@pytest.mark.parametrize(
    "regridder_class",
    [
        CentroidLocatorRegridder,
        OverlapRegridder,
        RelativeOverlapRegridder,
        BarycentricInterpolator,
    ],
)
def test_weights_as_dataframe(
    regridder_class,
    disk,
    quads_structured,
):
    regridder = regridder_class(quads_structured, disk)
    df = regridder.weights_as_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert "source_index" in df
    assert "target_index" in df
    assert "weight" in df

    regridder._weights = None
    with pytest.raises(ValueError):
        regridder.weights_as_dataframe()


def test_centroid_locator_regridder_structured(
    grid_data_a, grid_data_a_layered, grid_data_b, expected_results_centroid
):
    regridder = CentroidLocatorRegridder(source=grid_data_a, target=grid_data_b)
    result = regridder.regrid(grid_data_a)
    assert (result.fillna(0.0) == expected_results_centroid.fillna(0.0)).any()

    # With broadcasting
    regridder = CentroidLocatorRegridder(source=grid_data_a_layered, target=grid_data_b)
    broadcasted = regridder.regrid(grid_data_a_layered)
    assert broadcasted.dims == ("layer", "y", "x")
    assert (
        broadcasted.fillna(0.0).isel(layer=0) == expected_results_centroid.fillna(0.0)
    ).any()


def test_centroid_locator_regridder(disk, quads_1):
    square = quads_1
    regridder = CentroidLocatorRegridder(source=disk, target=square)
    result = regridder.regrid(disk)
    assert isinstance(result, xu.UgridDataArray)
    assert result.notnull().any()
    assert result.min() >= disk.min()
    assert result.max() <= disk.max()
    assert result.grid.n_face == square.grid.n_face

    # other way around
    regridder = CentroidLocatorRegridder(source=result, target=disk)
    back = regridder.regrid(result)
    assert isinstance(back, xu.UgridDataArray)
    assert back.notnull().any()
    assert back.min() >= disk.min()
    assert back.max() <= disk.max()
    assert back.grid.n_face == disk.grid.n_face

    # With broadcasting
    obj = xu.UgridDataArray(
        xr.DataArray(np.ones(5), dims=["layer"]) * result.obj,
        result.grid,
    )
    broadcasted = regridder.regrid(obj)
    assert broadcasted.dims == ("layer", disk.grid.face_dimension)


def test_overlap_regridder_structured(
    grid_data_a, grid_data_a_layered, grid_data_b, expected_results_overlap
):
    regridder = OverlapRegridder(source=grid_data_a, target=grid_data_b)
    result = regridder.regrid(grid_data_a)
    assert (result == expected_results_overlap).any()

    # With broadcasting
    regridder = OverlapRegridder(source=grid_data_a_layered, target=grid_data_b)
    broadcasted = regridder.regrid(grid_data_a_layered)
    assert broadcasted.dims == ("layer", "y", "x")
    assert (broadcasted.isel(layer=0) == expected_results_overlap).any()

    # Test if "mode" method doesn't repeat first values again
    # https://github.com/Deltares/xugrid/issues/236
    grid_data_adapted = grid_data_a.copy()
    grid_data_adapted[0, 0] = 99
    regridder = OverlapRegridder(
        source=grid_data_adapted, target=grid_data_a, method="mode"
    )
    result = regridder.regrid(grid_data_adapted)
    assert not np.all(result == 99.0)


def test_overlap_regridder(disk, quads_1):
    square = quads_1
    regridder = OverlapRegridder(disk, square, method="mean")
    result = regridder.regrid(disk)
    assert result.notnull().any()
    assert result.min() >= disk.min()
    assert result.max() <= disk.max()

    # With broadcasting
    obj = xu.UgridDataArray(
        xr.DataArray(np.ones(5), dims=["layer"]) * disk.obj,
        grid=disk.grid,
    )
    broadcasted = regridder.regrid(obj)
    assert broadcasted.dims == ("layer", square.grid.face_dimension)
    assert broadcasted.shape == (5, 100)


def test_linear_interpolator_structured(
    grid_data_a, grid_data_a_layered, grid_data_b, expected_results_linear
):
    regridder = BarycentricInterpolator(source=grid_data_a, target=grid_data_b)
    result = regridder.regrid(grid_data_a)
    assert (result.fillna(0.0) == expected_results_linear.fillna(0.0)).any()

    # With broadcasting
    regridder = BarycentricInterpolator(source=grid_data_a_layered, target=grid_data_b)
    broadcasted = regridder.regrid(grid_data_a_layered)
    assert broadcasted.dims == ("layer", "y", "x")
    assert (
        broadcasted.fillna(0.0).isel(layer=0) == expected_results_linear.fillna(0.0)
    ).any()


def test_barycentric_interpolator(disk, quads_0_25):
    square = quads_0_25
    regridder = BarycentricInterpolator(source=disk, target=square)
    result = regridder.regrid(disk)
    assert result.notnull().any()
    assert result.min() >= disk.min()
    assert result.max() <= disk.max()

    # With broadcasting
    obj = xu.UgridDataArray(
        xr.DataArray(np.ones(5), dims=["layer"]) * disk.obj,
        grid=disk.grid,
    )
    broadcasted = regridder.regrid(obj)
    assert broadcasted.dims == ("layer", square.grid.face_dimension)
    assert broadcasted.shape == (5, 1600)


@pytest.mark.parametrize(
    "cls",
    [
        CentroidLocatorRegridder,
        OverlapRegridder,
        RelativeOverlapRegridder,
        BarycentricInterpolator,
    ],
)
def test_regridder_from_weights(cls, disk, quads_1):
    square = quads_1
    regridder = cls(source=disk, target=square)
    result = regridder.regrid(disk)
    weights = regridder.weights
    new_regridder = cls.from_weights(weights, target=square)
    new_result = new_regridder.regrid(disk)
    assert new_result.equals(result)


@pytest.mark.parametrize(
    "cls",
    [
        CentroidLocatorRegridder,
        OverlapRegridder,
        RelativeOverlapRegridder,
        BarycentricInterpolator,
    ],
)
def test_regridder_from_weights_layered(cls, disk, disk_layered, quads_1):
    square = quads_1
    regridder = cls(source=disk, target=square)
    result = regridder.regrid(disk)
    weights = regridder.weights
    new_regridder = cls.from_weights(weights, target=square)
    new_result = new_regridder.regrid(disk_layered)
    assert np.array_equal(new_result.sel(layer=1).values, result.values, equal_nan=True)


@pytest.mark.parametrize(
    "cls",
    [
        CentroidLocatorRegridder,
        OverlapRegridder,
        RelativeOverlapRegridder,
        BarycentricInterpolator,
    ],
)
def test_regridder_from_dataset(cls, disk, quads_1):
    square = quads_1
    regridder = cls(source=disk, target=square)
    result = regridder.regrid(disk)
    dataset = regridder.to_dataset()
    new_regridder = cls.from_dataset(dataset)
    new_result = new_regridder.regrid(disk)
    assert np.array_equal(new_result.values, result.values, equal_nan=True)


def test_regridder_daks_arrays(
    grid_data_dask_source,
    grid_data_dask_source_layered,
    grid_data_dask_target,
    grid_data_dask_expected,
    grid_data_dask_expected_layered,
):
    regridder = CentroidLocatorRegridder(
        source=grid_data_dask_source, target=grid_data_dask_target
    )
    result = regridder.regrid(grid_data_dask_source)
    assert result.equals(grid_data_dask_expected)

    # with broadcasting
    regridder = CentroidLocatorRegridder(
        source=grid_data_dask_source_layered, target=grid_data_dask_target
    )
    result = regridder.regrid(grid_data_dask_source_layered)
    assert result.isel(layer=0).equals(grid_data_dask_expected_layered.isel(layer=0))


def test_create_percentile_method():
    with pytest.raises(ValueError):
        OverlapRegridder.create_percentile_method(percentile=-1)

    with pytest.raises(ValueError):
        OverlapRegridder.create_percentile_method(percentile=101)

    median = OverlapRegridder.create_percentile_method(percentile=50)
    values = np.array([0, 1, 2, 3, 4], dtype=float)
    weights = np.ones_like(values)
    workspace = np.zeros_like(values)
    assert median(values, weights, workspace) == 2


def test_directional_dependence():
    # Increasing / decreasing x or y direction shouldn't matter for the result.
    da = xr.DataArray(
        data=[[1.0, 2.0], [3.0, 4.0]],
        coords={"y": [17.5, 12.5], "x": [2.5, 7.5]},
        dims=("y", "x"),
    )
    target_da = xr.DataArray(
        data=[[np.nan, np.nan], [np.nan, np.nan]],
        coords={"y": [10.0, 20.0], "x": [0.0, 10.0]},
        dims=("y", "x"),
    )
    target_uda = xu.UgridDataArray.from_structured(target_da)

    flip = slice(None, None, -1)
    flipy = da.isel(y=flip)
    flipx = da.isel(x=flip)
    flipxy = da.isel(x=flip, y=flip)
    uda = xu.UgridDataArray.from_structured(da)
    uda_flipxy = xu.UgridDataArray.from_structured(flipxy)

    # Structured target: test whether the result is the same regardless of source
    # orientation.
    result = []
    for source in [da, flipy, flipx, flipxy, uda, uda_flipxy]:
        regridder = xu.OverlapRegridder(source, target=target_da)
        result.append(regridder.regrid(source))
    first = result.pop(0)
    assert all(first.identical(item) for item in result)

    # Unstructured target: test whether the result is the same regardless of
    # source orientation.
    result = []
    for source in [da, flipy, flipx, flipxy, uda, uda_flipxy]:
        regridder = xu.OverlapRegridder(source, target=target_uda)
        result.append(regridder.regrid(source))
    first = result.pop(0)
    assert all(first.identical(item) for item in result)


def test_barycentric_concave():
    vertices = np.array(
        [
            [0.0, 0.0],
            [3.0, 0.0],
            [1.0, 1.0],
            [0.0, 2.0],
            [3.0, 2.0],
        ]
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [2, 4, 3],
        ]
    )
    grid = xu.Ugrid2d(*vertices.T, -1, faces)

    dx = 0.1
    x = np.arange(0.0, 3.0, dx) + 0.5 * dx
    y = np.arange(0.0, 2.0, dx) + 0.5 * dx
    other = xr.DataArray(
        data=np.ones((y.size, x.size)), coords={"y": y, "x": x}, dims=("y", "x")
    )

    uda = xu.UgridDataArray(
        obj=xr.DataArray([2.0, 0.5, 2.0], dims=[grid.face_dimension]),
        grid=grid,
    )
    regridder = xu.BarycentricInterpolator(source=uda, target=other)
    result = regridder.regrid(uda)
    assert result.min() >= 0.5
    assert result.max() <= 2.0
