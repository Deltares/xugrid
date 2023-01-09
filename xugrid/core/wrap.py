"""
Wrap in advance instead of overloading __getattr__.

This allows for tab completion and documentation.
"""

import abc
import types
from functools import wraps
from typing import Sequence, Union

import xarray as xr

import xugrid
from xugrid.conversion import grid_from_dataset
from xugrid.ugrid.ugrid2d import Ugrid2d
from xugrid.ugrid.ugridbase import AbstractUgrid, UgridType

# Import entire module here for circular import of UgridDatasetAccessor and
# UgridDataArrayAccessor. Note: can only be used in functions (since that code
# is run at runtime).


class AbstractForwardMixin(abc.ABC):
    """
    Serves as a common identifier for UgridDataset, UgridDataArray.
    """


def ugrid_aligns(obj, grid):
    """
    Check whether the xarray object dimensions still align with the grid.
    """
    griddims = grid.dimensions
    shared_dims = set(griddims).intersection(obj.dims)
    if shared_dims:
        for dim in shared_dims:
            ugridsize = griddims[dim]
            objsize = obj[dim].size
            if ugridsize != objsize:
                raise ValueError(
                    f"conflicting sizes for dimension '{dim}': "
                    f"length {ugridsize} in UGRID topology and "
                    f"length {objsize} in xarray dimension"
                )
        return True
    else:
        return False


def maybe_xugrid(obj, topology):
    # Topology can either be a sequence of grids or a grid.
    if isinstance(topology, (list, set, tuple)):
        grids = {dim: grid for grid in topology for dim in grid.dimensions}
    else:
        grids = {dim: topology for dim in topology.dimensions}
    item_grids = list(set(grids[dim] for dim in obj.dims if dim in grids))

    if isinstance(obj, xr.DataArray):
        if len(item_grids) == 0:
            return obj
        if len(item_grids) > 1:
            raise RuntimeError("This shouldn't happen. Please open an issue.")
        grid = item_grids[0]
        if ugrid_aligns(obj, grid):
            return UgridDataArray(obj, grid)

    elif isinstance(obj, xr.Dataset):
        if ugrid_aligns(obj, item_grids):
            return UgridDataset(obj, grid)

    return obj


def maybe_xarray(arg):
    if isinstance(arg, (UgridDataArray, UgridDataset)):
        return arg.obj
    else:
        return arg


def wraps_xarray(method):
    @wraps(method)
    def wrapped(*args, **kwargs):
        self = args[0]
        args = [maybe_xarray(arg) for arg in args]
        kwargs = {k: maybe_xarray(v) for k, v in kwargs.items()}
        result = method(*args, **kwargs)

        # Sidestep staticmethods, classmethods
        if isinstance(self, (UgridDataArray, UgridDataset)):
            return maybe_xugrid(result, self.grids)
        else:
            return result

    return wrapped


def wrap_accessor(accessor):
    # TODO: This will not add dynamic accessors, those most be included at
    # runtime instead?

    def wrapped(*args, **kwargs):
        args = [maybe_xarray(arg) for arg in args]
        kwargs = {k: maybe_xarray(v) for k, v in kwargs.items()}
        result = accessor(*args, **kwargs)
        return result

    return wrapped


def wrap(
    target_class_dict,
    source_class,
):
    FuncType = (types.FunctionType, types.MethodType)

    class Empty:
        pass

    keep = {
        "__eq__",
        "__ge__",
        "__gt__",
        "__le__",
        "__lt__",
        "__ne__",
        "__reduce__",
        "__reduce_ex__",
    }
    remove = {
        "__getatrr__",
        "__slots__",
        "__annotations__",
    }
    attr_names = (set(dir(source_class)) - set(dir(Empty)) - remove) | keep
    all_attrs = {k: getattr(source_class, k) for k in attr_names}

    methods = {k: v for k, v in all_attrs.items() if isinstance(v, FuncType)}
    for name, method in methods.items():
        wrapped = wraps_xarray(method)
        setattr(wrapped, "__doc__", method.__doc__)
        target_class_dict[name] = wrapped

    properties = {k: v for k, v in all_attrs.items() if isinstance(v, property)}
    for name, prop in properties.items():
        wrapped = property(wraps_xarray(prop.__get__))
        setattr(wrapped, "__doc__", prop.__doc__)
        target_class_dict[name] = wrapped

    accessors = {k: v for k, v in all_attrs.items() if isinstance(v, type)}
    for name, accessor in accessors.items():
        wrapped = property(wrap_accessor(accessor))
        setattr(wrapped, "__doc__", accessor.__doc__)
        target_class_dict[name] = wrapped

    return


class DataArrayForwardMixin(AbstractForwardMixin):

    wrap(
        target_class_dict=vars(),
        source_class=xr.DataArray,
    )


class DatasetForwardMixin(AbstractForwardMixin):

    wrap(
        target_class_dict=vars(),
        source_class=xr.Dataset,
    )


class UgridDataArray(DataArrayForwardMixin):
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

    @property
    def ugrid(self):
        """
        UGRID Accessor. This "accessor" makes operations using the UGRID
        topology available.
        """
        return xugrid.accessor.dataarray_accessor.UgridDataArrayAccessor(
            self.obj, self.grid
        )

    @staticmethod
    def from_structured(da: xr.DataArray):
        """
        Create a UgridDataArray from a (structured) xarray DataArray.

        The spatial dimensions are flattened into a single UGRID face dimension.

        Parameters
        ----------
        da: xr.DataArray
            Last two dimensions must be ``("y", "x")``.

        Returns
        -------
        unstructured: UgridDataArray
        """
        if da.dims[-2:] != ("y", "x"):
            raise ValueError('Last two dimensions of da must be ("y", "x")')
        grid = Ugrid2d.from_structured(da)
        dims = da.dims[:-2]
        coords = {k: da.coords[k] for k in dims}
        face_da = xr.DataArray(
            da.data.reshape(*da.shape[:-2], -1),
            coords=coords,
            dims=[*dims, grid.face_dimension],
            name=da.name,
        )
        return UgridDataArray(face_da, grid)


class UgridDataset(DatasetForwardMixin):
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

    @property
    def ugrid(self):
        """
        UGRID Accessor. This "accessor" makes operations using the UGRID
        topology available.
        """
        return xugrid.accessor.dataset_accessor.UgridDatasetAccessor(
            self.obj, self.grids
        )
