import xugrid


def test_adh_san_diego():
    ds = xugrid.data.adh_san_diego()
    assert isinstance(ds, xugrid.UgridDataset)


def test_disk():
    ds = xugrid.data.disk()
    assert isinstance(ds, xugrid.UgridDataset)


def test_xoxo():
    grid = xugrid.data.xoxo()
    assert isinstance(grid, xugrid.Ugrid2d)
