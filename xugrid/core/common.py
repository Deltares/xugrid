from functools import wraps

import xarray as xr

from xugrid.core.utils import unique_grids
from xugrid.core.wrap import UgridDataArray, UgridDataset

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
    return UgridDataset(ds)


def open_dataset(*args, **kwargs):
    ds = xr.open_dataset(*args, **kwargs)
    return _dataset_helper(ds)


def load_dataset(*args, **kwargs):
    ds = xr.load_dataset(*args, **kwargs)
    return _dataset_helper(ds)


def _dataarray_helper(ds: xr.Dataset):
    dataset = _dataset_helper(ds)
    if len(dataset.data_vars) != 1:
        raise ValueError(
            "The file or object contains more than one data "
            "variable. Please read with xarray.open_dataset and "
            "then select the variable you want."
        )
    else:
        (data_array,) = dataset.data_vars.values()

    data_array.set_close(dataset._close)

    # Reset names if they were changed during saving
    # to ensure that we can 'roundtrip' perfectly
    if DATAARRAY_NAME in dataset.attrs:
        data_array.name = dataset.attrs[DATAARRAY_NAME]
        del dataset.attrs[DATAARRAY_NAME]

    if data_array.name == DATAARRAY_VARIABLE:
        data_array.name = None

    return UgridDataArray(data_array, dataset.grid)


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
    return UgridDataset(ds)


def open_zarr(*args, **kwargs):
    ds = xr.open_zarr(*args, **kwargs)
    return UgridDataset(ds)


load_dataset.__doc__ = xr.load_dataset.__doc__
open_dataset.__doc__ = xr.open_dataset.__doc__
load_dataarray.__doc__ = xr.load_dataarray.__doc__
open_dataarray.__doc__ = xr.open_dataarray.__doc__
open_mfdataset.__doc__ = xr.open_mfdataset.__doc__
open_zarr.__doc__ = xr.open_zarr.__doc__


# Other utilities
# ---------------


def wrap_func_like(func):
    @wraps(func)
    def _like(other, *args, **kwargs):
        obj = func(other.obj, *args, **kwargs)
        if isinstance(obj, xr.DataArray):
            return type(other)(obj, other.grid)
        elif isinstance(obj, xr.Dataset):
            return type(other)(obj, other.grids)
        else:
            raise TypeError(
                f"Expected Dataset or DataArray, received {type(other).__name__}"
            )

    _like.__doc__ = func.__doc__
    return _like


def wrap_func_objects(func):
    @wraps(func)
    def _f(objects, *args, **kwargs):
        grids = []
        bare_objs = []
        for obj in objects:
            if isinstance(obj, UgridDataArray):
                grids.append(obj.grid)
            elif isinstance(obj, UgridDataset):
                grids.extend(obj.grids)
            else:
                raise TypeError(
                    "Can only concatenate xugrid UgridDataset and UgridDataArray "
                    f"objects, got {type(obj).__name__}"
                )

            bare_objs.append(obj.obj)

        grids = unique_grids(grids)
        result = func(bare_objs, *args, **kwargs)
        if isinstance(result, xr.DataArray):
            if len(grids) > 1:
                raise ValueError("All UgridDataArrays must have the same grid")
            return UgridDataArray(result, next(iter(grids)))
        else:
            return UgridDataset(result, grids)

    _f.__doc__ = func.__doc__
    return _f


full_like = wrap_func_like(xr.full_like)
zeros_like = wrap_func_like(xr.zeros_like)
ones_like = wrap_func_like(xr.ones_like)

concat = wrap_func_objects(xr.concat)
merge = wrap_func_objects(xr.merge)
