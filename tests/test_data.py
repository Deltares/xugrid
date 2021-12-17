import xarray as xr

import xugrid


def test_adh_san_diego():
    ds = xugrid.data.adh_san_diego()
    assert isinstance(ds, xugrid.UgridDataset)
    ds = xugrid.data.adh_san_diego(xarray=True)
    assert isinstance(ds, xr.Dataset)


def test_disk():
    ds = xugrid.data.disk()
    assert isinstance(ds, xugrid.UgridDataset)


def test_xoxo():
    grid = xugrid.data.xoxo()
    assert isinstance(grid, xugrid.Ugrid2d)


def test_elevation_nl():
    ds = xugrid.data.elevation_nl()
    assert isinstance(ds, xugrid.UgridDataArray)
    ds = xugrid.data.elevation_nl(xarray=True)
    assert isinstance(ds, xr.Dataset)
