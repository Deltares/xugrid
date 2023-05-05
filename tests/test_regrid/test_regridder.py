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


@pytest.fixture(scope="function")
def disk():
    return xu.data.disk()["face_z"]


def quads(dx):
    xmin, ymin, xmax, ymax = xu.data.disk().ugrid.total_bounds
    x = np.arange(xmin, xmax, dx) + 0.5 * dx
    y = np.arange(ymin, ymax, dx) + 0.5 * dx

    da = xr.DataArray(
        data=np.full((y.size, x.size), np.nan),
        coords={"y": y, "x": x},
        dims=[
            "y",
            "x",
        ],
    )
    return xu.UgridDataArray.from_structured(da)


@pytest.fixture
def grid_a():
    return xr.DataArray(
        data=np.arange(9).reshape((3, 3)),
        dims=["y", "x"],
        coords={
            "y": np.array([150, 100, 50]),
            "x": np.array([50, 100, 150]),
            "dx": 50.0,
            "dy": -50.0,
        },
    )


@pytest.fixture
def grid_aa():
    return xr.DataArray(
        data=np.arange(18).reshape((2, 3, 3)),
        dims=["layer", "y", "x"],
        coords={
            "layer": np.arange(2) + 1,
            "y": np.array([150, 100, 50]),
            "x": np.array([50, 100, 150]),
            "dx": 50.0,
            "dy": -50.0,
        },
    )


@pytest.fixture
def grid_b():
    return xr.DataArray(
        data=np.zeros(16).reshape((4, 4)),
        dims=["y", "x"],
        coords={
            "y": np.array([175, 125, 75, 25]),
            "x": np.array([25, 75, 125, 175]),
            "dx": 50.0,
            "dy": -50.0,
        },
    )


@pytest.fixture
def expected_results_centroid():
    return xr.DataArray(
        data=np.array(
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                0,
                1,
                np.nan,
                np.nan,
                3,
                4,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ]
        ).reshape((4, 4)),
        dims=["y", "x"],
        coords={
            "y": np.array([175, 125, 75, 25]),
            "x": np.array([25, 75, 125, 175]),
            "dx": 50.0,
            "dy": -50.0,
        },
    )


@pytest.fixture
def expected_results_overlap():
    # --------
    # target | source            | ntarget | sum(source)/ntaget
    # 0      | 0                 | 1       | 0.0
    # 1      | 0 1               | 2       | 0.5
    # 2      |   1 2             | 2       | 1.5
    # 3      |     2             | 1       | 2.0
    # 4      | 0     3           | 2       | 1.5
    # 5      | 0 1   3 4         | 4       | 2.0
    # 6      |   1 2   4 5       | 4       | 3.0
    # 7      |     2     5       | 2       | 3.5
    # 8      |       3     6     | 2       | 4.5
    # 9      |       3 4   6 7   | 4       | 5.0
    # 10     |         4 5   7 8 | 4       | 6.0
    # 11     |           5     8 | 2       | 6.5
    # 12     |             6     | 1       | 6.0
    # 13     |             6 7   | 2       | 6.5
    # 14     |               7 8 | 2       | 7.5
    # 15     |                 8 | 1       | 8.0
    # --------
    return xr.DataArray(
        data=np.array(
            [
                0.0,
                0.5,
                1.5,
                2.0,
                1.5,
                2.0,
                3.0,
                3.5,
                4.5,
                5.0,
                6.0,
                6.5,
                6.0,
                6.5,
                7.5,
                8.0,
            ]
        ).reshape((4, 4)),
        dims=["y", "x"],
        coords={
            "y": np.array([175, 125, 75, 25]),
            "x": np.array([25, 75, 125, 175]),
            "dx": 50.0,
            "dy": -50.0,
        },
    )


@pytest.fixture
def expected_results_linear():
    # --------
    # target | source            | weights            | sum(source x weight)
    # 5      | 0 1   3 4         | 0.25 0.25 0.25 0.25    | 2.0
    # 6      |   1 2   4 5       | 0.25 0.25 0.25 0.25    | 3.0
    # 9      |       3 4   6 7   | 0.25 0.25 0.25 0.25    | 5.0
    # 10     |         4 5   7 8 | 0.25 0.25 0.25 0.25    | 6.0
    # --------
    return xr.DataArray(
        data=np.array(
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                2.0,
                3.0,
                np.nan,
                np.nan,
                5.0,
                6.0,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ]
        ).reshape((4, 4)),
        dims=["y", "x"],
        coords={
            "y": np.array([175, 125, 75, 25]),
            "x": np.array([25, 75, 125, 175]),
            "dx": 50.0,
            "dy": -50.0,
        },
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


def test_centroid_locator_regridder_structured(
    grid_a, grid_aa, grid_b, expected_results_centroid
):
    regridder = CentroidLocatorRegridder(source=grid_a, target=grid_b)
    result = regridder.regrid(grid_a)
    assert (result.fillna(0.0) == expected_results_centroid.fillna(0.0)).any()

    # With broadcasting
    regridder = CentroidLocatorRegridder(source=grid_aa, target=grid_b)
    broadcasted = regridder.regrid(grid_aa)
    assert broadcasted.dims == ("layer", "y", "x")
    assert (
        broadcasted.fillna(0.0).isel(layer=0) == expected_results_centroid.fillna(0.0)
    ).any()


def test_centroid_locator_regridder(disk):
    square = quads(1.0)
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
    grid_a, grid_aa, grid_b, expected_results_overlap
):
    regridder = OverlapRegridder(source=grid_a, target=grid_b)
    result = regridder.regrid(grid_a)
    assert (result == expected_results_overlap).any()

    # With broadcasting
    regridder = OverlapRegridder(source=grid_aa, target=grid_b)
    broadcasted = regridder.regrid(grid_aa)
    assert broadcasted.dims == ("layer", "y", "x")
    assert (broadcasted.isel(layer=0) == expected_results_overlap).any()


def test_overlap_regridder(disk):
    square = quads(1.0)
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


def test_lineair_interpolator_structured(
    grid_a, grid_aa, grid_b, expected_results_linear
):
    regridder = BarycentricInterpolator(source=grid_a, target=grid_b)
    result = regridder.regrid(grid_a)
    assert (result.fillna(0.0) == expected_results_linear.fillna(0.0)).any()

    # With broadcasting
    regridder = BarycentricInterpolator(source=grid_aa, target=grid_b)
    broadcasted = regridder.regrid(grid_aa)
    assert broadcasted.dims == ("layer", "y", "x")
    assert (
        broadcasted.fillna(0.0).isel(layer=0) == expected_results_linear.fillna(0.0)
    ).any()


def test_barycentric_interpolator(disk):
    square = quads(0.25)
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
def test_regridder_from_weights(cls, disk):
    square = quads(1.0)
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
def test_regridder_from_dataset(cls, disk):
    square = quads(1.0)
    regridder = cls(source=disk, target=square)
    result = regridder.regrid(disk)
    dataset = regridder.to_dataset()
    new_regridder = cls.from_dataset(dataset)
    new_result = new_regridder.regrid(disk)
    assert new_result.equals(result)
