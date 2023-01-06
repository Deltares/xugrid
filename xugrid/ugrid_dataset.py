from functools import wraps
from typing import Sequence, Tuple, Type, Union

import xarray as xr

# from .plot.pyvista import to_pyvista_grid
from xugrid.ops import UgridDataArrayOps, UgridDatasetOps
from xugrid.ugrid import AbstractUgrid, grid_from_dataset
from xugrid.wrap import DataArrayForwardMixin, DatasetForwardMixin

UgridType = Type[AbstractUgrid]


class UgridDataset(DatasetForwardMixin, UgridDatasetOps):
    def __init__(
        self,
        obj: xr.Dataset = None,
        grids: Union[UgridType, Sequence[UgridType]] = None,
    ):
        if obj is None and grids is None:
            raise ValueError("At least either obj or grids is required")

        if obj is None:
            ds = xr.Dataset()
        else:
            if not isinstance(obj, xr.Dataset):
                raise TypeError(
                    "obj must be xarray.Dataset. Received instead: "
                    f"{type(obj).__name__}"
                )
            connectivity_vars = [
                name
                for v in obj.ugrid_roles.connectivity.values()
                for name in v.values()
            ]
            ds = obj.drop_vars(obj.ugrid_roles.topology + connectivity_vars)

        if grids is None:
            topologies = obj.ugrid_roles.topology
            grids = [grid_from_dataset(obj, topology) for topology in topologies]
        else:
            # Make sure it's a new list
            if isinstance(grids, (list, tuple, set)):
                grids = [grid for grid in grids]
            else:  # not iterable
                grids = [grids]
            # Now typecheck
            for grid in grids:
                if not isinstance(grid, AbstractUgrid):
                    raise TypeError(
                        "grid must be Ugrid1d or Ugrid2d. "
                        f"Received instead: {type(grid).__name__}"
                    )

        self.grids = grids
        self.obj = ds


class UgridDataArray(DataArrayForwardMixin, UgridDataArrayOps):
    def __init__(self, obj: xr.DataArray, grid: UgridType):
        if not isinstance(obj, xr.DataArray):
            raise TypeError(
                "obj must be xarray.DataArray. Received instead: "
                f"{type(obj).__name__}"
            )
        if not isinstance(grid, AbstractUgrid):
            raise TypeError(
                "grid must be Ugrid1d or Ugrid2d. Received instead: "
                f"{type(grid).__name__}"
            )
        self.grid = grid
        self.obj = obj


# Wrapped IO methods
# ------------------


def open_dataset(*args, **kwargs):
    ds = xr.open_dataset(*args, **kwargs)
    return UgridDataset(ds)


def open_dataarray(*args, **kwargs):
    ds = xr.open_dataset(*args, **kwargs)
    dataset = UgridDataset(ds)

    if len(dataset.data_vars) != 1:
        raise ValueError(
            "Given file dataset contains more than one data "
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

    return data_array


def open_mfdataset(*args, **kwargs):
    if "data_vars" in kwargs:
        raise ValueError("data_vars kwargs is not supported in xugrid.open_mfdataset")
    kwargs["data_vars"] = "minimal"
    ds = xr.open_mfdataset(*args, **kwargs)
    return UgridDataset(ds)


def open_zarr(*args, **kwargs):
    ds = xr.open_zarr(*args, **kwargs)
    return UgridDataset(ds)


open_dataset.__doc__ = xr.open_dataset.__doc__
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

        grids = set(grids)
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
