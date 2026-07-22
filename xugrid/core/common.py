import xarray as xr

from xugrid.core.constructors import dataset

DATAARRAY_NAME = "__xarray_dataarray_name__"
DATAARRAY_VARIABLE = "__xarray_dataarray_variable__"


def _dataset_helper(ds: xr.Dataset):
    n_topology = len(ds.ugrid_roles.topology)
    if n_topology == 0:
        raise ValueError(
            "The file or object does not contain UGRID conventions data. "
            "One or more UGRID topologies are required. Perhaps you wrote "
            "the file using `data.to_netcdf()` instead of `data.ugrid.to_netcdf()`?"
        )
    return dataset(ds)


def open_dataset(*args, **kwargs):
    ds = xr.open_dataset(*args, **kwargs)
    return _dataset_helper(ds)


def load_dataset(*args, **kwargs):
    ds = xr.load_dataset(*args, **kwargs)
    return _dataset_helper(ds)


def _dataarray_helper(ds: xr.Dataset):
    ds = _dataset_helper(ds)
    if len(ds.data_vars) != 1:
        raise ValueError(
            "The file or object contains more than one data "
            "variable. Please read with xarray.open_dataset and "
            "then select the variable you want."
        )
    else:
        (da,) = ds.data_vars.values()

    # Reset names if they were changed during saving
    # to ensure that we can 'roundtrip' perfectly
    if DATAARRAY_NAME in ds.attrs:
        da.name = ds.attrs[DATAARRAY_NAME]
        del ds.attrs[DATAARRAY_NAME]

    if da.name == DATAARRAY_VARIABLE:
        da.name = None

    return da


def load_dataarray(*args, **kwargs):
    ds = xr.load_dataset(*args, **kwargs)
    return _dataarray_helper(ds)


def open_dataarray(*args, **kwargs):
    ds = xr.open_dataset(*args, **kwargs)
    return _dataarray_helper(ds)


def open_mfdataset(*args, **kwargs):
    if "data_vars" in kwargs:
        raise ValueError("data_vars kwargs is not supported in xugrid.open_mfdataset")
    kwargs["data_vars"] = "minimal"
    ds = xr.open_mfdataset(*args, **kwargs)
    return dataset(ds)


def open_zarr(*args, **kwargs):
    ds = xr.open_zarr(*args, **kwargs)
    return dataset(ds)


load_dataset.__doc__ = xr.load_dataset.__doc__
open_dataset.__doc__ = xr.open_dataset.__doc__
load_dataarray.__doc__ = xr.load_dataarray.__doc__
open_dataarray.__doc__ = xr.open_dataarray.__doc__
open_mfdataset.__doc__ = xr.open_mfdataset.__doc__
open_zarr.__doc__ = xr.open_zarr.__doc__
