import pytest
import xarray as xr

import xugrid


def test_generate_disk():
    with pytest.raises(ValueError, match="partitions should be >= 3"):
        xugrid.data.generate_disk(2, 2)

    nodes, faces = xugrid.data.generate_disk(4, 1)
    assert nodes.shape == (5, 2)
    assert faces.shape == (4, 3)
    _, faces = xugrid.data.generate_disk(4, 2)
    assert faces.shape == (16, 3)


def test_adh_san_diego():
    ds = xugrid.data.adh_san_diego()
    assert xugrid.is_ugrid_dataset(ds)
    ds = xugrid.data.adh_san_diego(xarray=True)
    assert isinstance(ds, xr.Dataset)


def test_disk():
    ds = xugrid.data.disk()
    assert xugrid.is_ugrid_dataset(ds)


def test_xoxo():
    grid = xugrid.data.xoxo()
    assert isinstance(grid, xugrid.Ugrid2d)


def test_elevation_nl():
    ds = xugrid.data.elevation_nl()
    assert xugrid.is_ugrid_dataarray(ds)
    ds = xugrid.data.elevation_nl(xarray=True)
    assert isinstance(ds, xr.Dataset)


def test_provinces_nl():
    import geopandas as gpd

    gdf = xugrid.data.provinces_nl()
    assert isinstance(gdf, gpd.GeoDataFrame)
