import types
from functools import wraps
from typing import Union

import geopandas as gpd
import numpy as np
import xarray as xr

from xugrid.ugrid import Ugrid2d
from .conversion import geodataframe_to_ugrid


def xarray_wrapper(func, grid):
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

    def __init__(self, obj: xr.DataArray, grid: Ugrid2d = None):
        self.obj = obj
        if grid is None:
            grid = Ugrid2d(obj)
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
            return xarray_wrapper(result, self.grid)
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

    def __init__(self, obj: xr.Dataset, grid: Ugrid2d = None):

        if grid is None:
            self.grid = Ugrid2d(obj)
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
            return xarray_wrapper(result, self.grid)
        else:
            return result

    @property
    def ugrid(self):
        return UgridAccessor(self.ds, self.grid)

    @staticmethod
    def from_geodataframe(geodataframe) -> "UgridDataset":
        ds, grid = geodataframe_to_ugrid(geodataframe)
        return UgridDataset(ds, grid)


class UgridAccessor:
    """
    this class implements selection logic for use with xarray dataset and
    dataarray in the context of an unstructured 2d grid.
    """

    def __init__(self, obj: Union[xr.Dataset, xr.DataArray], grid: Ugrid2d):
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
        grid = self.grid.topology_subset(face_indices)
        result.coords["face"] = face_indices
        if isinstance(self.obj, xr.DataArray):
            return UgridDataArray(result, grid)
        elif isinstance(self.obj, xr.Dataset):
            return UgridDataset(result, grid)
        else:
            raise TypeError("illegal type in _sel_points")

    def sel_points(self, points_x, points_y):
        """
        returns subset of dataset or dataArray containing specific face
        indices. Input arguments are point coordinates. The result dataset or
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
        result = self.obj.isel(face=face_indices)
        result.coords["face"] = face_indices
        return result

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
            return self.sel_points(np.array([x], np.float64), np.array([y], np.float64))
        elif isinstance(x, slice) and isinstance(y, slice):
            return self._sel_slices(x, y)
        else:
            raise ValueError("argument mismatch")

    def rasterize(self, cellsize: float):
        xmin = self.grid.node_x.min()
        xmax = self.grid.node_x.max()
        ymin = self.grid.node_y.min()
        ymax = self.grid.node_y.max()
        xmin = np.floor(xmin / cellsize) * cellsize
        xmax = np.ceil(xmax / cellsize) * cellsize
        ymin = np.floor(ymin / cellsize) * cellsize
        ymax = np.ceil(ymax / cellsize) * cellsize
        dx = cellsize
        dy = -cellsize
        x = np.arange(xmin + 0.5 * dx, xmax - 0.5 * dx, dx)
        y = np.arange(ymax + 0.5 * dy, ymin - 0.5 * dy, dy)
        return self.sel(x=x, y=y)

    def rasterize_like(self, other):
        x = other["x"].values
        y = other["y"].values
        return self.sel(x=x, y=y)
    
    def _dataset_obj(self) -> xr.Dataset:
        if isinstance(self.obj, xr.DataArray):
            ds = self.obj.to_dataset()
        else:
            ds = self.obj

    def to_dataset(self) -> xr.Dataset:
        """
        Converts this UgridDataArray or UgridDataset into a standard
        xarray.Dataset.

        The UGRID topology information is added as standard data variables.
        """
        ds = self._dataset_obj()
        return ds.merge(self.grid.dataset)
    
    def to_geodataframe(self, data_on: str) -> gpd.GeoDataFrame:
        """
        Parameters
        ----------
        data_on: str
            One of {node, edge, face}
        """
        ds = self._dataset_obj()
        variables = [da for da in ds.data_vars if data_on in da.dims]
        # TODO deal with time-dependent data, etc.
        # Basically requires checking which variables are static, which aren't.
        # For non-static, requires repeating all geometries.
        # Call reset_index on mult-index to generate them as regular columns.
        df = ds[variables].to_dataframe()
        geometry = self.grid.as_vector_geometry(data_on)
        return gpd.GeoDataFrame(df, geometry=geometry)
