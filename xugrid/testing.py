import xarray as xr


def is_ugrid_dataarray(obj):
    return isinstance(obj, xr.DataArray) and obj.ugrid.is_indexed


def is_ugrid_dataset(obj):
    return isinstance(obj, xr.Dataset) and obj.ugrid.is_indexed
