import types
from functools import wraps
from typing import Any, Callable, Union

import numpy as np
import scipy.sparse
import xarray as xr
from xarray.backends.api import DATAARRAY_NAME, DATAARRAY_VARIABLE
from xarray.core._typed_ops import DataArrayOpsMixin, DatasetOpsMixin
from xarray.core.utils import UncachedAccessor

from . import connectivity
from .interpolate import laplace_interpolate
from .plot.plot import _PlotMethods

# from .plot.pyvista import to_pyvista_grid
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


class DunderForwardMixin:
    """
    These methods are not present in the xarray OpsMixin classes.
    """

    def __bool__(self):
        return bool(self.obj)

    def __contains__(self, key: Any):
        return key in self.obj

    def __int__(self):
        return int(self.obj)

    def __float__(self):
        return float(self.obj)


class UgridDataArray(DataArrayOpsMixin, DunderForwardMixin):
    """
    Wraps an xarray DataArray, adding UGRID topology.
    """

    def __init__(self, obj: xr.DataArray, grid: Union[Ugrid1d, Ugrid2d] = None):
        if grid is None:
            grid = Ugrid2d.from_dataset(obj)
        self.grid = grid
        self.obj = obj.assign_coords(self.grid.topology_coords(obj))

    def __getitem__(self, key):
        """
        forward getters to xr.DataArray. Wrap result if necessary
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

    def _unary_op(
        self,
        f: Callable,
    ):
        return UgridDataArray(self.obj._unary_op(f), self.grid)

    def _binary_op(
        self,
        other,
        f: Callable,
        reflexive: bool = False,
    ):
        if isinstance(other, (UgridDataArray, UgridDataset)):
            other = other.obj
        return UgridDataArray(self.obj._binary_op(other, f, reflexive), self.grid)

    def _inplace_binary_op(self, other, f: Callable):
        if isinstance(other, (UgridDataArray, UgridDataset)):
            other = other.obj
        return UgridDataArray(self.obj._inplace_binary_op(other, f), self.grid)

    @property
    def ugrid(self):
        """
        UGRID Accessor. This "accessor" makes operations using the UGRID
        topology available.
        """
        return UgridAccessor(self.obj, self.grid)

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


class UgridDataset(DatasetOpsMixin, DunderForwardMixin):
    """
    Wraps an xarray Dataset, adding UGRID topology.
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
            ds = xr.Dataset()
        else:
            ds = self.grid.remove_topology(obj)
        self.obj = ds.assign_coords(self.grid.topology_coords(ds))

    def __contains__(self, key: Any):
        return key in self.obj

    def __getitem__(self, key):
        result = self.obj[key]
        if isinstance(result, xr.DataArray):
            return UgridDataArray(result, self.grid)
        elif isinstance(result, xr.Dataset):
            return UgridDataset(result, self.grid)
        else:
            return result

    def __setitem__(self, key, value):
        # TODO: check with topology
        if isinstance(value, UgridDataArray):
            self.obj[key] = value.obj
        else:
            self.obj[key] = value

    def __getattr__(self, attr):
        """
        Appropriately wrap result if necessary.
        """
        result = getattr(self.obj, attr)
        if isinstance(result, xr.DataArray):
            return UgridDataArray(result, self.grid)
        elif isinstance(result, xr.Dataset):
            return UgridDataset(result, self.grid)
        elif isinstance(result, types.MethodType):
            return xarray_wrapper(result, self.grid)
        else:
            return result

    def _unary_op(
        self,
        f: Callable,
    ):
        return UgridDataset(self.obj._unary_op(f), self.grid)

    def _binary_op(
        self,
        other,
        f: Callable,
        reflexive: bool = False,
    ):
        if isinstance(other, (UgridDataArray, UgridDataset)):
            other = other.obj
        return UgridDataset(self.obj._binary_op(other, f, reflexive), self.grid)

    def _inplace_binary_op(self, other, f: Callable):
        if isinstance(other, (UgridDataArray, UgridDataset)):
            other = other.obj
        return UgridDataset(self.obj._inplace_binary_op(other, f), self.grid)

    @property
    def ugrid(self):
        """
        UGRID Accessor. This "accessor" makes operations using the UGRID
        topology available.
        """
        return UgridAccessor(self.obj, self.grid)

    @staticmethod
    def from_geodataframe(geodataframe: "geopandas.GeoDataFrame"):  # type: ignore # noqa
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
    This "accessor" makes operations using the UGRID topology available via the
    ``.ugrid`` attribute for UgridDataArrays and UgridDatasets.
    """

    def __init__(self, obj: Union[xr.Dataset, xr.DataArray], grid: Ugrid2d):
        self.obj = obj
        self.grid = grid

    plot = UncachedAccessor(_PlotMethods)

    def isel(self, indexer):
        """
        Returns a new object with arrays indexed along edges or faces.

        Parameters
        ----------
        indexer: 1d array of integer or bool

        Returns
        -------
        indexed: Union[UgridDataArray, UgridDataset]
        """
        if self.grid.topology_dimension == 1:
            dim = self.grid.edge_dimension
        elif self.grid.topology_dimension == 2:
            dim = self.grid.face_dimension
        else:
            raise NotImplementedError

        result = self.obj.isel({dim: indexer})
        result = result.assign_coords({f"{dim}_index": (dim, indexer)})
        grid = self.grid.topology_subset(indexer)
        if isinstance(self.obj, xr.DataArray):
            return UgridDataArray(result, grid)
        elif isinstance(self.obj, xr.Dataset):
            return UgridDataset(result, grid)
        else:
            raise TypeError(
                f"Expected UgridDataArray or UgridDataset, got {type(result).__name__}"
            )

    def sel(self, x=None, y=None):
        """
        Returns a new object, a subselection in the UGRID x and y coordinates.

        The indexing for x and y always occurs orthogonally, i.e.:
        ``.sel(x=[0.0, 5.0], y=[10.0, 15.0])`` results in a four points. For
        vectorized indexing (equal to ``zip``ing through x and y), see
        ``.sel_points``.

        Depending on the nature of the x and y indexers, a xugrid or xarray
        object is returned:

        * slice without step: ``x=slice(-100, 100)``: returns xugrid object, a
          part of the unstructured grid.
        * slice with step: ``x=slice(-100, 100, 10)``: returns xarray object, a
          series of points (x=[-100, -90, -80, ..., 90, 100]).
        * a scalar: ``x=5.0``: returns xarray object, a point.
        * an array: ``x=[1.0, 15.0, 17.0]``: returns xarray object, a series of
          points.

        Parameters
        ----------
        x: float, 1d array, slice
        y: float, 1d array, slice

        Returns
        -------
        selection: Union[UgridDataArray, UgridDataset, xr.DataArray, xr.Dataset]
        """
        # TODO: also do vectorized indexing like xarray?
        # Might not be worth it, as orthogonal and vectorized indexing are
        # quite confusing.
        dim, ugrid, index, coords = self.grid.sel(x=x, y=y)
        result = self.obj.isel({dim: index})

        if not ugrid:
            return result.assign_coords(coords)

        grid = self.grid.topology_subset(index)
        if isinstance(self.obj, xr.DataArray):
            return UgridDataArray(result, grid)
        elif isinstance(self.obj, xr.Dataset):
            return UgridDataset(result, grid)
        else:
            raise TypeError(
                f"Expected UgridDataArray or UgridDataset, got {type(result).__name__}"
            )

    def sel_points(self, x, y):
        """
        Select points in the unstructured grid.

        Parameters
        ----------
        x: ndarray of floats with shape ``(n_points,)``
        y: ndarray of floats with shape ``(n_points,)``

        Returns
        -------
        points: Union[xr.DataArray, xr.Dataset]
        """
        if self.grid.topology_dimension != 2:
            raise NotImplementedError
        dim, _, index, coords = self.grid.sel_points(x, y)
        result = self.obj.isel({dim: index})
        return result.assign_coords(coords)

    def _raster(self, x, y, index) -> xr.DataArray:
        index = index.ravel()
        data = self.obj.isel({self.grid.face_dimension: index}).astype(float).values
        data[index == -1] = np.nan
        out = xr.DataArray(
            data=data.reshape(y.size, x.size),
            coords={"y": y, "x": x},
            dims=["y", "x"],
        )
        return out

    def rasterize(self, resolution: float):
        """
        Rasterize unstructured grid by sampling.

        Parameters
        ----------
        resolution: float
            Spacing in x and y.

        Returns
        -------
        rasterized: Union[xr.DataArray, xr.Dataset]
        """
        x, y, index = self.grid.rasterize(resolution)
        return self._raster(x, y, index)

    def rasterize_like(self, other: Union[xr.DataArray, xr.Dataset]):
        """
        Rasterize unstructured grid by sampling on the x and y coordinates
        of ``other``.

        Parameters
        ----------
        resolution: float
            Spacing in x and y.
        other: Union[xr.DataArray, xr.Dataset]
            Object to take x and y coordinates from.

        Returns
        -------
        rasterized: Union[xr.DataArray, xr.Dataset]
        """
        x, y, index = self.grid.rasterize_like(
            x=other["x"].values,
            y=other["y"].values,
        )
        return self._raster(x, y, index)

    @property
    def crs(self):
        """
        The Coordinate Reference System (CRS) represented as a ``pyproj.CRS`` object.

        Returns None if the CRS is not set.
        """
        return self.grid.crs

    def to_geodataframe(
        self, dim: str = None, name: str = None, dim_order=None
    ) -> "geopandas.GeoDataFrame":  # type: ignore # noqa
        """
        Convert data and topology of one facet (node, edge, face) of the grid
        to a geopandas GeoDataFrame. This also determines the geometry type of
        the geodataframe:

        * node: point
        * edge: line
        * face: polygon

        Parameters
        ----------
        dim: str
            node, edge, or face dimension. Inferred for DataArray.
        name: str
            Name to give to the array (required if unnamed).
        dim_order:
            Hierarchical dimension order for the resulting dataframe. Array content is
            transposed to this order and then written out as flat vectors in contiguous
            order, so the last dimension in this list will be contiguous in the resulting
            DataFrame. This has a major influence on which operations are efficient on the
            resulting dataframe.

            If provided, must include all dimensions of this DataArray. By default,
            dimensions are sorted according to the DataArray dimensions order.

        Returns
        -------
        geodataframe: gpd.GeoDataFrame
        """
        import geopandas as gpd

        if isinstance(self.obj, xr.DataArray):
            dim = self.obj.dims[-1]
            if self.obj.name is None:
                ds = self.obj.to_dataset(name=name)
            else:
                ds = self.obj.to_dataset()
        else:
            ds = self.obj
        variables = [var for var in ds.data_vars if dim in ds[var].dims]
        # TODO deal with time-dependent data, etc.
        # Basically requires checking which variables are static, which aren't.
        # For non-static, requires repeating all geometries.
        # Call reset_index on mult-index to generate them as regular columns.
        df = ds[variables].to_dataframe(dim_order=dim_order)
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
        """
        Binary dilation can be used on a boolean array to expand the "shape" of
        features.

        Compare with :py:func:`scipy.ndimage.binary_dilation`.

        Parameters
        ----------
        iterations: int, default: 1
        mask: 1d array of bool, optional
        border_value: bool, default value: False

        Returns
        -------
        dilated: UgridDataArray
        """
        return self._binary_iterate(iterations, mask, True, border_value)

    def binary_erosion(
        self,
        iterations: int = 1,
        mask=None,
        border_value=False,
    ):
        """
        Binary erosion can be used on a boolean array to shrink the "shape" of
        features.

        Compare with :py:func:`scipy.ndimage.binary_erosion`.

        Parameters
        ----------
        iterations: int, default: 1
        mask: 1d array of bool, optional
        border_value: bool, default value: False

        Returns
        -------
        eroded: UgridDataArray
        """
        return self._binary_iterate(iterations, mask, False, border_value)

    def connected_components(self):
        """
        Every edge or face is given a component number. If all are connected,
        all will have the same number.

        Wraps :py:func:`scipy.sparse.csgraph.connected_components``.

        Returns
        -------
        labelled: UgridDataArray
        """
        _, labels = scipy.sparse.csgraph.connected_components(
            self.grid.face_face_connectivity
        )
        return UgridDataArray(
            xr.DataArray(labels, dims=[self.grid.face_dimension]),
            self.grid,
        )

    def reverse_cuthill_mckee(self):
        """
        Reduces bandwith of the connectivity matrix.

        Wraps :py:func:`scipy.sparse.csgraph.reverse_cuthill_mckee`.

        Returns
        -------
        reordered: Union[UgridDataArray, UgridDataset]
        """
        grid = self.grid
        reordered_grid, reordering = self.grid.reverse_cuthill_mckee()
        reordered_data = self.obj.isel({grid.face_dimension: reordering})
        # TODO: this might not work properly if e.g. centroids are stored in obj.
        # Not all metadata would be reordered.
        if isinstance(self.obj, xr.DataArray):
            return UgridDataArray(
                reordered_data,
                reordered_grid,
            )
        elif isinstance(self.obj, xr.Dataset):
            return UgridDataset(
                reordered_data,
                reordered_grid,
            )
        else:
            raise ValueError("object should be a xr.DataArray")

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
        da = self.obj
        if isinstance(da, xr.Dataset):
            raise NotImplementedError
        if grid.topology_dimension != 2:
            raise NotImplementedError
        if len(da.dims) > 1 or da.dims[0] != grid.face_dimension:
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

    def to_dataset(self):
        """
        Converts this UgridDataArray or UgridDataset into a standard
        xarray.Dataset.

        The UGRID topology information is added as standard data variables.

        Returns
        -------
        dataset: UgridDataset
        """
        return self.grid.dataset.merge(self.obj)

    def to_netcdf(self, *args, **kwargs):
        """
        Write dataset contents to a UGRID compliant netCDF file.

        This function wraps :py:meth:`xr.Dataset.to_netcdf`; it adds the UGRID
        variables and coordinates to a standard xarray Dataset, then writes the
        result to a netCDF.

        All arguments are forwarded to :py:meth:`xr.Dataset.to_netcdf`.
        """
        self.to_dataset().to_netcdf(*args, **kwargs)

    def to_zarr(self, *args, **kwargs):
        """
        Write dataset contents to a UGRID compliant Zarr file.

        This function wraps :py:meth:`xr.Dataset.to_zarr`; it adds the UGRID
        variables and coordinates to a standard xarray Dataset, then writes the
        result to a Zarr file.

        All arguments are forwarded to :py:meth:`xr.Dataset.to_zarr`.
        """
        self.to_dataset().to_zarr(*args, **kwargs)


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
        return type(other)(obj, other.grid)

    _like.__doc__ = func.__doc__
    return _like


full_like = wrap_func_like(xr.full_like)
zeros_like = wrap_func_like(xr.zeros_like)
ones_like = wrap_func_like(xr.ones_like)
