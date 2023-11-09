import numpy as np
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
    assert result.notna().any()
    assert result.min() >= disk.min()
    assert result.max() <= disk.max()
    assert result.grid.n_face == square.grid.n_face

    # other way around
    regridder = CentroidLocatorRegridder(source=result, target=disk)
    back = regridder.regrid(result)
    assert isinstance(back, xu.UgridDataArray)
    assert back.notna().any()
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


def test_overlap_regridder(disk, quads_1):
    square = quads_1
    regridder = OverlapRegridder(disk, square, method="mean")
    result = regridder.regrid(disk)
    assert result.notna().any()
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
    assert result.notna().any()
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
