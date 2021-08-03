import types
from functools import wraps
from typing import Union

import numpy as np
import xarray as xr
from numpy import float64

from xugrid.ugrid import Ugrid


def dataarray_wrapper(func, grid):
    """
    runs a function, and if the result is an xarray dataset, it creates an
    UgridDataset around it
    """

    @wraps(func)
    def wrapped(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, xr.DataArray):
            return UgridDataArray(result, grid)
        else:
            return result

    return wrapped


def dataset_wrapper(func, grid):
    """
    runs a function, and if the result is an xarray dataset or an xarray
    dataArray, it creates an UgridDataset or an UgridDataArray around it
    """

    @wraps(func)
    def wrapped(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, xr.Dataset):
            return UgridDataset(result, grid)
        elif isinstance(result, xr.DataArray):
            return UgridDataArray(result, grid)
        else:
            return result

    return wrapped


class UgridDataArray:
    """
    this class wraps an xarray dataArray. It is used to work with the dataArray
    in the context of an unstructured 2d grid.
    """

    def __init__(self, obj: xr.DataArray, grid: Ugrid = None):
        self.obj = obj
        if grid is None:
            grid = Ugrid(obj)
        self.grid = grid

    def __getitem__(self, key):
        """
        forward getters to  xr.DataArray. Wrap result if necessary
        """
        result = self.obj[key]
        if isinstance(result, xr.DataArray):
            return UgridDataArray(result, self.grid)
        else:
            return result

    def __setitem__(self, key, value):
        """
        forward setters to  xr.DataArray
        """
        self.obj[key] = value

    def __getattr__(self, attr):
        """
        Appropriately wrap result if necessary.
        """
        result = getattr(self.obj, attr)
        if isinstance(result, xr.DataArray):
            return UgridDataArray(result, self.grid)
        elif isinstance(result, types.MethodType):
            return dataarray_wrapper(result, self.grid)
        else:
            return result

    @property
    def ugrid(self):
        return UgridAccessor(self.obj, self.grid)


class UgridDataset:
    """
    this class wraps an xarray Dataset. It is used to work with the Dataset in
    the context of an unstructured 2d grid.
    """

    def __init__(self, obj: xr.Dataset, grid: Ugrid = None):

        if grid is None:
            self.grid = Ugrid(obj)
        else:
            self.grid = grid
        self.ds = self.grid.remove_topology(obj)

    def __getitem__(self, key):
        result = self.ds[key]
        if isinstance(result, xr.DataArray):
            return UgridDataArray(result, self.grid)
        elif isinstance(result, xr.Dataset):
            return UgridDataset(result, self.grid)
        else:
            return result

    def __setitem__(self, key, value):
        self.ds[key] = value

    def __getattr__(self, attr):
        """
        Appropriately wrap result if necessary.
        """
        result = getattr(self.ds, attr)
        if isinstance(result, xr.DataArray):
            return UgridDataArray(result, self.grid)
        elif isinstance(result, xr.Dataset):
            return UgridDataset(result, self.grid)
        elif isinstance(result, types.MethodType):
            return dataset_wrapper(result, self.grid)
        else:
            return result

    @property
    def ugrid(self):
        return UgridAccessor(self.ds, self.grid)


class UgridAccessor:
    """
    this class implements selection logic for use with xarray dataset and
    dataarray in the context of an unstructured 2d grid.
    """

    def __init__(self, obj: Union[xr.Dataset, xr.DataArray], grid: Ugrid):
        self.obj = obj
        self.grid = grid

    def plot(self):
        """if self.grid._triangulation is None:
            self.grid.triangulation = mtri.Triangulation(
                x=self.grid.nodes[:, 0],
                y=self.grid.nodes[:, 1],
                triangles=self.grid.faces,
            )
        plt.tripcolor(self.grid.triangulation, self.obj.values.ravel())"""
        raise NotImplementedError()

    def object_from_face_indices(self, face_indices):
        """
        returns subset of dataset or dataArray containing specific face indices
        """
        result = self.obj.isel(face=face_indices)
        result.coords["face"] = face_indices
        if isinstance(self.obj, xr.DataArray):
            return UgridDataArray(result, self.grid)
        elif isinstance(self.obj, xr.Dataset):
            return UgridDataset(result, self.grid)
        else:
            raise TypeError("illegal type in _sel_points")

    def sel_points(self, points_x, points_y):
        """
        returns subset of dataset or dataArray containing specific face
        indices. INput arguments are point coordinates.  The result dataset or
        dataArray contains only the faces containing these points.
        """
        if (points_x is None) or (points_y is None):
            raise ValueError("coordinate arrays cannot be empty")
        if points_x.shape != points_y.shape:
            raise ValueError("coordinate arrays size does not match")
        if points_x.ndim != 1:
            raise ValueError("coordinate arrays must be 1d")

        points = np.column_stack([points_x, points_y])
        face_indices = self.grid.locate_faces(points)
        return self.object_from_face_indices(face_indices)

    def _sel_slices(self, x: slice, y: slice):
        if (
            (x.start is None)
            or (x.stop is None)
            or (y.start is None)
            or (y.stop is None)
        ):
            raise Exception("slice start and stop should not be None")
        elif (x.start > x.stop) or (y.start > y.stop):
            raise Exception("slice start should be smaller than its stop")
        elif (not x.step is None) and (not y.step is None):
            xcoords = np.arange(x.start, x.stop, x.step)
            ycoords = np.arange(y.start, y.stop, y.step)
            return self.sel_points(xcoords, ycoords)
        elif (x.step is None) and (y.step is None):
            face_indices = self.grid.locate_faces_bounding_box(
                x.start, x.stop, y.start, y.stop
            )
            return self.object_from_face_indices(face_indices)
        else:
            raise ValueError(
                "slices should both have a stepsize, or neither should have a stepsize"
            )

    def sel(self, x=None, y=None):
        """
        returns subset of dataset or dataArray based on spatial selection
        """
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            xv, yv = (a.ravel() for a in np.meshgrid(x, y, indexing="ij"))
            return self.sel_points(xv, yv)
        elif (isinstance(x, float) or isinstance(x, int)) and (
            isinstance(y, float) or isinstance(y, int)
        ):
            return self.sel_points(np.array([x], float64), np.array([y], float64))
        elif isinstance(x, slice) and isinstance(y, slice):
            return self._sel_slices(x, y)
        else:
            raise ValueError("argument mismatch")
