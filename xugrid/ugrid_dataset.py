import types
from functools import wraps
from typing import Any, Callable, Union

import geopandas as gpd
import numpy as np
import scipy.sparse
import xarray as xr
from xarray.backends.api import DATAARRAY_NAME, DATAARRAY_VARIABLE
from xarray.core._typed_ops import DataArrayOpsMixin
from xarray.core.utils import UncachedAccessor

from . import connectivity
from .interpolate import laplace_interpolate
from .plot import _PlotMethods
from .ugrid import Ugrid1d, Ugrid2d, grid_from_geodataframe


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


class UgridDataArray(DataArrayOpsMixin):
    """
    this class wraps an xarray dataArray. It is used to work with the dataArray
    in the context of an unstructured 2d grid.
    """

    def __init__(self, obj: xr.DataArray, grid: Ugrid2d = None):
        self.obj = obj
        if grid is None:
            grid = Ugrid2d.from_dataset(obj)
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

    def _binary_op(
        self,
        other,
        f: Callable,
        reflexive: bool = False,
    ):
        return self.obj._binary_op(other, f, reflexive)

    def _inplace_binary_op(self, other, f: Callable):
        return self.obj._inplace_binary_op(other, f)

    @property
    def ugrid(self):
        return UgridAccessor(self.obj, self.grid)

    def to_geoseries(self):
        series = self.obj.to_series()
        geometry = self.grid.to_pygeos(self.dims[-1])
        return gpd.GeoSeries(series, geometry=geometry)

    def to_geodataframe(self, name=None, dim_order=None):
        df = self.obj.to_dataframe(name, dim_order)
        geometry = self.grid.to_pygeos(self.dims[-1])
        return gpd.GeoDataFrame(df, geometry=geometry)

    @staticmethod
    def from_geoseries(geoseries):
        if not isinstance(geoseries, gpd.GeoSeries):
            raise TypeError(f"Cannot convert a {type(geoseries)}, expected a GeoSeries")
        gdf = gpd.GeoDataFrame(geoseries)
        grid = grid_from_geodataframe(gdf)
        da = xr.DataArray.from_series(geoseries)
        return UgridDataArray(da, grid)


class UgridDataset:
    """
    this class wraps an xarray Dataset. It is used to work with the Dataset in
    the context of an unstructured 2d grid.
    """

    def __init__(self, obj: xr.Dataset = None, grid: Ugrid2d = None):
        if grid is None:
            if obj is None:
                raise ValueError("At least either obj or grid is required")
            # TODO: find out whether it's 1D or 2D topology
            self.grid = Ugrid2d.from_dataset(obj)
        else:
            self.grid = grid
        if obj is None:
            self.ds = xr.Dataset()
        else:
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
        # TODO: check with topology
        if isinstance(value, UgridDataArray):
            self.ds[key] = value.obj
        else:
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
    def from_geodataframe(geodataframe: gpd.GeoDataFrame):
        """
        Convert a geodataframe into the appropriate Ugrid topology and dataset.

        Parameters
        ----------
        geodataframe: gpd.GeoDataFrame

        Returns
        -------
        dataset: UGridDataset
        """
        grid = grid_from_geodataframe(geodataframe)
        ds = xr.Dataset.from_dataframe(geodataframe.drop("geometry", axis=1))
        return UgridDataset(ds, grid)


class UgridAccessor:
    """
    this class implements selection logic for use with xarray dataset and
    dataarray in the context of an unstructured 2d grid.
    """

    def __init__(self, obj: Union[xr.Dataset, xr.DataArray], grid: Ugrid2d):
        self.obj = obj
        self.grid = grid

    plot = UncachedAccessor(_PlotMethods)

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

    def rasterize(self, resolution: float):
        x, y, index = self.grid.rasterize(resolution)
        data = self.isel(face=index).astype(float)
        data[index == -1] = np.nan
        out = xr.DataArray(
            data=data,
            coords={"y": y, "x": x},
            dims=["y", "x"],
        )
        return out

    def rasterize_like(self, other: Union[xr.DataArray, xr.Dataset]):
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

    def to_geodataframe(self, dim: str) -> gpd.GeoDataFrame:
        """
        Parameters
        ----------
        dim: str
        """
        ds = self._dataset_obj()
        variables = [da for da in ds.data_vars if dim in da.dims]
        # TODO deal with time-dependent data, etc.
        # Basically requires checking which variables are static, which aren't.
        # For non-static, requires repeating all geometries.
        # Call reset_index on mult-index to generate them as regular columns.
        df = ds[variables].to_dataframe()
        geometry = self.grid.to_pygeos(dim)
        return gpd.GeoDataFrame(df, geometry=geometry)

    def _binary_iterate(self, iterations: int, mask, value, border_value):
        if border_value == value:
            exterior = self.grid.exterior_faces
        else:
            exterior = None
        if mask is not None:
            mask = mask.values

        obj = self.obj
        if isinstance(obj, xr.DataArray):
            output = connectivity._binary_iterate(
                self.grid.face_face_connectivity,
                obj.values,
                value,
                iterations,
                mask,
                exterior,
                border_value,
            )
            da = obj.copy(data=output)
            return UgridDataArray(da, self.grid.copy())
        elif isinstance(obj, xr.Dataset):
            raise NotImplementedError
        else:
            raise ValueError("object should be a xr.DataArray")

    def binary_dilation(
        self,
        iterations: int = 1,
        mask=None,
        border_value=False,
    ):
        return self._binary_iterate(iterations, mask, True, border_value)

    def binary_erosion(
        self,
        iterations: int = 1,
        mask=None,
        border_value=False,
    ):
        return self._binary_iterate(iterations, mask, False, border_value)

    def connected_components(self):
        _, labels = scipy.sparse.csgraph.connected_components(
            self.grid.face_face_connectivity
        )
        return UgridDataArray(
            xr.DataArray(labels, dims=[self.grid.face_dimension]),
            self.grid,
        )

    def reverse_cuthill_mckee(self):
        grid = self.grid
        attrs = grid.attrs
        # TODO: Dispatch on dimension
        face_dim = attrs.get("face_dimension", "face")
        if isinstance(self.obj, xr.Dataset):
            raise NotImplementedError
        if self.obj.dims[-1] != face_dim:
            raise NotImplementedError
        reordered_grid, reordering = self.grid.reverse_cuthill_mckee()
        reordered_data = self.obj.data[..., reordering]
        # TODO: this might not work properly if e.g. centroids are stored in obj.
        # Not all metadata would be reordered.
        return UgridDataArray(
            self.obj.copy(data=reordered_data),
            reordered_grid,
        )

    def laplace_interpolate(
        self,
        xy_weights: bool = True,
        direct_solve: bool = False,
        drop_tol: float = None,
        fill_factor: float = None,
        drop_rule: str = None,
        options: dict = None,
        tol: float = 1.0e-5,
        maxiter: int = 250,
    ):
        """
        Fill gaps in ``data`` (``np.nan`` values) using Laplace interpolation.

        This solves Laplace's equation where where there is no data, with data
        values functioning as fixed potential boundary conditions.

        Note that an iterative solver method will be required for large grids.
        In this case, some experimentation with the solver settings may be
        required to find a converging solution of sufficient accuracy. Refer to
        the documentation of :py:func:`scipy.sparse.linalg.spilu` and
        :py:func:`scipy.sparse.linalg.cg`.

        Parameters
        ----------
        xy_weights: bool, default False.
            Wether to use the inverse of the centroid to centroid distance in
            the coefficient matrix. If ``False``, defaults to uniform
            coefficients of 1 so that each face connection has equal weight.
        direct_solve: bool, optional, default ``False``
            Whether to use a direct or an iterative solver or a conjugate gradient
            solver. Direct method provides an exact answer, but are unsuitable
            for large problems.
        drop_tol: float, optional, default None.
            Drop tolerance for ``scipy.sparse.linalg.spilu`` which functions as a
            preconditioner for the conjugate gradient solver.
        fill_factor: float, optional, default None.
            Fill factor for ``scipy.sparse.linalg.spilu``.
        drop_rule: str, optional default None.
            Drop rule for ``scipy.sparse.linalg.spilu``.
        options: dict, optional, default None.
            Remaining other options for ``scipy.sparse.linalg.spilu``.
        tol: float, optional, default 1.0e-5.
            Convergence tolerance for ``scipy.sparse.linalg.cg``.
        maxiter: int, default 250.
            Maximum number of iterations for ``scipy.sparse.linalg.cg``.

        Returns
        -------
        filled: UgridDataArray of floats
        """
        grid = self.grid
        attrs = grid.mesh_topology.attrs
        da = self.obj
        if isinstance(da, xr.Dataset):
            raise NotImplementedError
        if len(da.dims) > 1 or da.dims[0] != attrs.get("face_dimension", "face"):
            raise NotImplementedError

        connectivity = grid.face_face_connectivity.copy()
        if xy_weights:
            xy = grid.centroids
            coo = connectivity.tocoo()
            i = coo.row
            j = coo.col
            connectivity.data = 1.0 / np.linalg.norm(xy[j] - xy[i], axis=1)

        filled = laplace_interpolate(
            connectivity=connectivity,
            data=da.values,
            use_weights=xy_weights,
            direct_solve=direct_solve,
            drop_tol=drop_tol,
            fill_factor=fill_factor,
            drop_rule=drop_rule,
            options=options,
            tol=tol,
            maxiter=maxiter,
        )
        da_filled = da.copy(data=filled)
        return UgridDataArray(da_filled, grid)


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
    ds = xr.open_mfdataset(*args, **kwargs)
    return UgridDataset(ds)


def open_zarr(*args, **kwargs):
    ds = xr.open_zarr(*args, **kwargs)
    return UgridDataset(ds)


open_dataset.__doc__ = xr.open_dataset.__doc__
open_dataarray.__doc__ = xr.open_dataarray.__doc__
open_mfdataset.__doc__ = xr.open_mfdataset.__doc__
open_zarr.__doc__ = xr.open_zarr.__doc__
