import numpy as np
import pytest
import xarray as xr

import xugrid as xu
from xugrid import BarycentricInterpolator, CentroidLocatorRegridder, OverlapRegridder


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


@pytest.mark.parametrize(
    "cls", [CentroidLocatorRegridder, OverlapRegridder, BarycentricInterpolator]
)
def test_check_source_target_types(disk, cls):
    with pytest.raises(TypeError):
        cls(source=disk, target=1.0)
    with pytest.raises(TypeError):
        cls(source=1.0, target=disk)


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
