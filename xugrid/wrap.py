"""
Wrap in advance instead of overloading __getattr__.

This allows for tab completion and documentation.
"""

import abc
import types
from functools import wraps
from typing import Type

import xarray as xr

from xugrid.ugrid import AbstractUgrid
from xugrid.ugrid_dataset import UgridDataArray, UgridDataset


UgridType = Type[AbstractUgrid]


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
    

def maybe_xugrid(obj, grid):
    if isinstance(obj, xr.DataArray):
        if ugrid_aligns(obj, grid):
            return UgridDataArray(obj, grid)
    elif isinstance(obj, xr.Dataset):
        if ugrid_aligns(obj, grid):
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
        if isinstance(self, AbstractForwardMixin):
            return maybe_xugrid(result, self.grid)
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
        '__eq__',
        '__ge__',
        '__gt__',
        '__le__',
        '__lt__',
        '__ne__',
        '__reduce__',
        '__reduce_ex__',
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
    
