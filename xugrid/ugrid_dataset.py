import abc
import types
from functools import wraps
from itertools import chain
from typing import Any, Callable, Dict, Sequence, Tuple, Type, Union

import numpy as np
import pandas as pd
import scipy.sparse
import xarray as xr
from xarray.backends.api import DATAARRAY_NAME, DATAARRAY_VARIABLE
from xarray.core._typed_ops import DataArrayOpsMixin, DatasetOpsMixin
from xarray.core.utils import UncachedAccessor, either_dict_or_kwargs

from . import connectivity
from .interpolate import laplace_interpolate
from .plot.plot import _PlotMethods

# from .plot.pyvista import to_pyvista_grid
from .ugrid import AbstractUgrid, Ugrid2d, grid_from_dataset, grid_from_geodataframe

UgridType = Type[AbstractUgrid]


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
            return UgridDataset(obj, grid)
    elif isinstance(obj, xr.Dataset):
        if ugrid_aligns(obj, grid):
            return UgridDataArray(obj, grid)
    return obj


def maybe_xarray(arg):
    if isinstance(arg, (UgridDataArray, UgridDataset)):
        return arg.obj
    else:
        return arg


def xarray_wrapper(func, grid):
    """
    Runs a function, and if the result is an xarray Dataset or an xarray
    DataArray, it creates an UgridDataset or an UgridDataArray around it
    if UGRID dimensions are present.
    """

    @wraps(func)
    def wrapped(*args, **kwargs):
        args = [maybe_xarray(arg) for arg in args]
        kwargs = {k: maybe_xarray(v) for k, v in kwargs.items()}
        result = func(*args, **kwargs)
        return maybe_xugrid(result, grid)

    return wrapped


def filter_indexers(indexers, grids):
    indexers = indexers.copy()
    ugrid_indexers = []
    for grid in grids:
        ugrid_dims = set(grid.dimensions).intersection(indexers)
        if ugrid_dims:
            if len(ugrid_dims) > 1:
                raise ValueError(
                    "Can only index along one UGRID dimension at a time. "
                    f"Received for topology {grid.name}: {ugrid_dims}"
                )
            dim = ugrid_dims.pop()
            ugrid_indexers.append((grid, (dim, indexers.pop(dim))))
        else:
            ugrid_indexers.append((grid, None))
    return indexers, ugrid_indexers


def ugrid_sel(obj, ugrid_indexers):
    grids = []
    for (grid, indexer_args) in ugrid_indexers:
        if indexer_args is None:
            grids.append(grid)
        else:
            obj, new_grid = grid.isel(*indexer_args, obj)
            grids.append(new_grid)
    return obj, grids


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

    def __repr__(self):
        return self.obj.__repr__()

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

    def __getitem__(self, key):
        """
        forward getters to xr.DataArray. Wrap result if necessary
        """
        result = self.obj[key]
        return maybe_xugrid(result, self.grid)

    def __setitem__(self, key, value):
        """
        forward setters to xr.DataArray
        """
        self.obj[key] = value

    def __getattr__(self, attr):
        """
        Appropriately wrap result if necessary.
        """
        if attr == "obj":
            return self.obj

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
        other = maybe_xarray(other)
        return UgridDataArray(self.obj._binary_op(other, f, reflexive), self.grid)

    def _inplace_binary_op(self, other, f: Callable):
        other = maybe_xarray(other)
        return UgridDataArray(self.obj._inplace_binary_op(other, f), self.grid)

    def isel(
        self,
        indexers=None,
        drop: bool = False,
        missing_dims="raise",
        **indexers_kwargs,
    ):
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "sel")
        obj_indexers, ugrid_indexers = filter_indexers(indexers, [self.grid])
        result = self.obj.isel(obj_indexers, drop=drop, missing_dims=missing_dims)
        result, grids = ugrid_sel(result, ugrid_indexers)
        return UgridDataArray(result, grids[0])

    def sel(
        self,
        indexers=None,
        method=None,
        tolerance=None,
        drop=False,
        **indexers_kwargs,
    ):
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "sel")
        obj_indexers, ugrid_indexers = filter_indexers(indexers, [self.grid])
        result = self.obj.sel(
            obj_indexers, method=method, tolerance=tolerance, drop=drop
        )
        result, grids = ugrid_sel(result, ugrid_indexers)
        return UgridDataArray(result, grids[0])

    @property
    def ugrid(self):
        """
        UGRID Accessor. This "accessor" makes operations using the UGRID
        topology available.
        """
        return UgridDataArrayAccessor(self.obj, self.grid)

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


UgridDataArray.isel.__doc__ = xr.DataArray.isel.__doc__
UgridDataArray.sel.__doc__ = xr.DataArray.sel.__doc__


class AbstractUgridAccessor(abc.ABC):
    @abc.abstractmethod
    def to_dataset():
        """ """

    @abc.abstractmethod
    def assign_node_coords():
        """ """

    @abc.abstractmethod
    def set_node_coords():
        """ """

    @abc.abstractproperty
    def crs():
        """ """

    @abc.abstractmethod
    def set_crs():
        """ """

    @abc.abstractmethod
    def to_crs():
        """ """

    @abc.abstractproperty
    def bounds():
        """ """

    @abc.abstractproperty
    def total_bounds():
        """ """

    @staticmethod
    def _sel(obj, grid, x, y):
        # TODO: also do vectorized indexing like xarray?
        # Might not be worth it, as orthogonal and vectorized indexing are
        # quite confusing.
        result = grid.sel(obj, x, y)
        if isinstance(result, tuple):
            return maybe_xugrid(*result)
        else:
            return result

    @staticmethod
    def _sel_points(obj, grid, x, y):
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
        if grid.topology_dimension != 2:
            raise NotImplementedError
        dim, _, index, coords = grid.sel_points(x, y)
        result = obj.isel({dim: index})
        return result.assign_coords(coords)

    def clip_box(
        self,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
    ):
        """
        Clip the DataArray or Dataset by a bounding box.

        Parameters
        ----------
        xmin: float
        ymin: float
        xmax: float
        ymax: float

        -------
        clipped:
            xugrid.UgridDataArray or xugrid.UgridDataset
        """
        return self.sel(x=slice(xmin, xmax), y=slice(ymin, ymax))

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


class UgridDataset(DatasetOpsMixin, DunderForwardMixin):
    """
    Wraps an xarray Dataset, adding UGRID topology.
    """

    def __repr__(self):
        return self.obj.__repr__()

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

    def __contains__(self, key: Any):
        return key in self.obj

    def __getitem__(self, key):
        result = self.obj[key]
        grids = {dim: grid for grid in self.grids for dim in grid.dimensions}
        item_grids = list(set(grids[dim] for dim in result.dims if dim in grids))

        # Now find out which grid to return for this key
        if isinstance(result, xr.DataArray):
            if len(item_grids) == 0:
                return result
            elif len(item_grids) > 1:
                raise RuntimeError("This shouldn't happen. Please open an issue.")
            return UgridDataArray(result, item_grids[0])
        elif isinstance(result, xr.Dataset):
            return UgridDataset(result, item_grids)
        else:
            return result

    def __setitem__(self, key, value):
        # TODO: check with topology
        if isinstance(value, UgridDataArray):
            append = True
            # Check if the dimensions occur in self.
            # if they don't, the grid should be added.
            if self.grids is not None:
                alldims = set(
                    chain.from_iterable([grid.dimensions for grid in self.grids])
                )
                matching_dims = set(value.grid.dimensions).intersection(alldims)
                if matching_dims:
                    # If they do match: the grids should match.
                    grids = {
                        dim: grid for grid in self.grids for dim in grid.dimensions
                    }
                    firstdim = next(iter(matching_dims))
                    if not grids[firstdim].equals(value.grid):
                        raise ValueError(
                            "Grids share dimension names but do not are not identical. "
                            f"Matching dimensions: {matching_dims}"
                        )
                    append = False

            self.obj[key] = value.obj
            if append:
                self.grids.append(value.grid)
        else:
            self.obj[key] = value

    def __getattr__(self, attr):
        """
        Appropriately wrap result if necessary.
        """
        if attr == "obj":
            return self.obj

        result = getattr(self.obj, attr)
        grids = {dim: grid for grid in self.grids for dim in grid.dimensions}
        if isinstance(result, xr.DataArray):
            item_grids = list(set(grids[dim] for dim in result.dims if dim in grids))
            if len(item_grids) == 0:
                return result
            if len(item_grids) > 1:
                raise RuntimeError("This shouldn't happen. Please open an issue.")
            return UgridDataArray(result, item_grids[0])
        elif isinstance(result, xr.Dataset):
            item_grids = list(set(grids[dim] for dim in result.dims if dim in grids))
            return UgridDataset(result, item_grids)
        elif isinstance(result, types.MethodType):
            return xarray_wrapper(result, self.grids)
        else:
            return result

    def _unary_op(
        self,
        f: Callable,
    ):
        return UgridDataset(self.obj._unary_op(f), self.grids)

    def _binary_op(
        self,
        other,
        f: Callable,
        reflexive: bool = False,
    ):
        other = maybe_xarray(other)
        return UgridDataset(self.obj._binary_op(other, f, reflexive), self.grids)

    def _inplace_binary_op(self, other, f: Callable):
        other = maybe_xarray(other)
        return UgridDataset(self.obj._inplace_binary_op(other, f), self.grids)

    def isel(
        self,
        indexers=None,
        drop: bool = False,
        missing_dims="warn",
        **indexers_kwargs: Any,
    ):
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "isel")
        obj_indexers, ugrid_indexers = filter_indexers(indexers, self.grids)
        result = self.obj.isel(obj_indexers, drop=drop, missing_dims=missing_dims)
        result, grids = ugrid_sel(result, ugrid_indexers)
        return UgridDataset(result, grids)

    def sel(
        self,
        indexers=None,
        method=None,
        tolerance=None,
        drop: bool = False,
        **indexers_kwargs: Any,
    ):
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "sel")
        obj_indexers, ugrid_indexers = filter_indexers(indexers, self.grids)
        result = self.obj.sel(
            obj_indexers, method=method, tolerance=tolerance, drop=drop
        )
        result, grids = ugrid_sel(result, ugrid_indexers)
        return UgridDataset(result, grids)

    @property
    def ugrid(self):
        """
        UGRID Accessor. This "accessor" makes operations using the UGRID
        topology available.
        """
        return UgridDatasetAccessor(self.obj, self.grids)

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
        return UgridDataset(ds, [grid])


# Set docstrings
UgridDataset.sel.__doc__ = xr.Dataset.sel.__doc__
UgridDataset.isel.__doc__ = xr.Dataset.isel.__doc__


class UgridDatasetAccessor(AbstractUgridAccessor):
    def __init__(self, obj: xr.Dataset, grids: Sequence[UgridType]):
        self.obj = obj
        self.grids = grids

    @property
    def grid(self) -> UgridType:
        ngrid = len(self.grids)
        if ngrid == 1:
            return self.grids[0]
        else:
            raise AttributeError(
                "Can only access grid topology via `.grid` if dataset contains "
                f"exactly one grid. Dataset contains {ngrid} grids. Use "
                "`.grids` instead."
            )

    @property
    def bounds(self) -> Dict[str, Tuple]:
        """
        Mapping from grid name to tuple containing ``minx, miny, maxx, maxy``
        values of the grid's node coordinates for every grid in the dataset.
        """
        return {grid.name: grid.bounds for grid in self.grids}

    @property
    def total_bounds(self) -> Tuple:
        """
        Returns a tuple containing ``minx, miny, maxx, maxy`` values for the
        bounds of the dataset as a whole. Currently does not check whether the
        coordinate reference systems (CRS) of the grids in the dataset match.
        """
        bounds = np.column_stack([bound for bound in self.bounds.values()])
        return (
            bounds[0].min(),
            bounds[1].min(),
            bounds[2].max(),
            bounds[3].max(),
        )

    def assign_node_coords(self) -> UgridDataset:
        """
        Assign node coordinates from the grid to the object.

        Returns a new object with all the original data in addition to the new
        node coordinates of the grid.

        Returns
        -------
        assigned: UgridDataset
        """
        result = self.obj
        for grid in self.grids:
            result = grid.assign_node_coords(result)
        return UgridDataset(result, self.grids)

    def assign_edge_coords(self) -> UgridDataset:
        """
        Assign edge coordinates from the grid to the object.

        Returns a new object with all the original data in addition to the new
        node coordinates of the grid.

        Returns
        -------
        assigned: UgridDataset
        """
        result = self.obj
        for grid in self.grids:
            result = grid.assign_edge_coords(result)
        return UgridDataset(result, self.grids)

    def assign_face_coords(self) -> UgridDataset:
        """
        Assign face coordinates from the grid to the object.

        Returns a new object with all the original data in addition to the new
        node coordinates of the grid.

        Returns
        -------
        assigned: UgridDataset
        """
        result = self.obj
        for grid in self.grids:
            if grid.topology_dimension > 1:
                result = grid.assign_face_coords(result)
        return UgridDataset(result, self.grids)

    def set_node_coords(self, node_x: str, node_y: str, topology: str = None):
        """
        Given names of x and y coordinates of the nodes of an object, set them
        as the coordinates in the grid.

        Parameters
        ----------
        node_x: str
            Name of the x coordinate of the nodes in the object.
        node_y: str
            Name of the y coordinate of the nodes in the object.
        topology: str, optional
            Name of the grid topology in which to set the node_x and node_y
            coordinates. Can be omitted if the UgridDataset contains only a
            single grid.
        """
        if topology is None:
            if len(self.grids) == 1:
                grid = self.grids[0]
            else:
                raise ValueError(
                    "topology must be specified when dataset contains multiple datasets"
                )
        else:
            grids = [grid for grid in self.grid if grid.name == topology]
            if len(grids != 1):
                raise ValueError(
                    f"Expected one grid with topology {topology}, found: {len(grids)}"
                )
            grid = grids[0]

        grid.set_node_coords(node_x, node_y, self.obj)

    def isel(self, indexers, **indexers_kwargs):
        """
        Returns a new object with arrays indexed along edges or faces.

        Parameters
        ----------
        indexer: 1d array of integer or bool

        Returns
        -------
        indexed: Union[UgridDataArray, UgridDataset]
        """
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "isel")
        alldims = set(chain.from_iterable([grid.dimensions for grid in self.grids]))
        invalid = indexers.keys() - alldims
        if invalid:
            raise ValueError(
                f"Dimensions {invalid} do not exist. Expected one of {alldims}"
            )

        if len(indexers) > 1:
            raise NotImplementedError("Can only index a single dimension at a time")
        dim, indexer = next(iter(indexers.items()))

        # Find grid to use for selection
        grids = [
            grid.isel(dim, indexer) if dim in grid.dimensions else grid
            for grid in self.grids
        ]
        result = self.obj.isel({dim: indexer})

        if isinstance(self.obj, xr.Dataset):
            return UgridDataset(result, grids)
        else:
            raise TypeError(f"Expected UgridDataset, got {type(result).__name__}")

    def sel(self, x=None, y=None):
        result = self.obj
        for grid in self.grids:
            result = self._sel(result, grid, x, y)
        return result

    def sel_points(self, x, y):
        result = self.obj
        for grid in self.grids:
            result = self._sel_points(result, grid, x, y)
        return result

    def to_dataset(self):
        """
        Converts this UgridDataArray or UgridDataset into a standard
        xarray.Dataset.

        The UGRID topology information is added as standard data variables.

        Returns
        -------
        dataset: UgridDataset
        """
        return xr.merge([grid.to_dataset(self.obj) for grid in self.grids])

    @property
    def crs(self):
        """
        The Coordinate Reference System (CRS) represented as a ``pyproj.CRS`` object.

        Returns None if the CRS is not set.

        Returns
        -------
        crs: dict
            A dictionary containing the names of the grids and their CRS.
        """
        return {grid.name: grid.crs for grid in self.grids}

    def set_crs(
        self,
        crs: Union["pyproj.CRS", str] = None,  # type: ignore # noqa
        epsg: int = None,
        allow_override: bool = False,
        topology: str = None,
    ):
        """
        Set the Coordinate Reference System (CRS) of a UGRID topology.

        NOTE: The underlying geometries are not transformed to this CRS. To
        transform the geometries to a new CRS, use the ``to_crs`` method.

        Parameters
        ----------
        crs : pyproj.CRS, optional if `epsg` is specified
            The value can be anything accepted
            by :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
            such as an authority string (eg "EPSG:4326") or a WKT string.
        epsg : int, optional if `crs` is specified
            EPSG code specifying the projection.
        allow_override : bool, default False
            If the the UGRID topology already has a CRS, allow to replace the
            existing CRS, even when both are not equal.
        topology: str, optional
            Name of the grid topology in which to set the CRS.
            Sets the CRS for all grids if left unspecified.
        """
        if topology is None:
            grids = self.grids
        else:
            names = [grid.name for grid in self.grids]
            if topology not in names:
                raise ValueError(f"{topology} not found. Expected one of: {names}")
            grids = [grid for grid in self.grids if grid.name == topology]

        for grid in grids:
            grid.set_crs(crs, epsg, allow_override)

    def to_crs(
        self,
        crs: Union["pyproj.CRS", str] = None,  # type: ignore # noqa
        epsg: int = None,
        topology: str = None,
    ) -> UgridDataset:
        """
        Transform geometries to a new coordinate reference system.
        Transform all geometries in an active geometry column to a different coordinate
        reference system. The ``crs`` attribute on the current Ugrid must
        be set. Either ``crs`` or ``epsg`` may be specified for output.

        This method will transform all points in all objects. It has no notion
        of projecting the cells. All segments joining points are assumed to be
        lines in the current projection, not geodesics. Objects crossing the
        dateline (or other projection boundary) will have undesirable behavior.

        Parameters
        ----------
        crs : pyproj.CRS, optional if `epsg` is specified
            The value can be anything accepted by
            :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
            such as an authority string (eg "EPSG:4326") or a WKT string.
        epsg : int, optional if `crs` is specified
            EPSG code specifying output projection.
        topology: str, optional
            Name of the grid topology to reproject.
            Reprojects all grids if left unspecified.
        """
        if topology is None:
            grids = [grid.to_crs(crs, epsg) for grid in self.grids]
        else:
            names = [grid.name for grid in self.grids]
            if topology not in names:
                raise ValueError(f"{topology} not found. Expected one of: {names}")
            grids = [
                grid.to_crs(crs, epsg) if grid.name == topology else grid.copy()
                for grid in self.grids
            ]

        uds = UgridDataset(self.obj, grids)
        return uds.ugrid.assign_node_coords()

    def to_geodataframe(
        self, dim_order=None
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

        ds = self.obj

        gdfs = []
        for grid in self.grids:
            if grid.topology_dimension == 1:
                dim = grid.edge_dimension
            elif grid.topology_dimension == 2:
                dim = grid.face_dimension
            else:
                raise ValueError("invalid topology dimension on grid")

            variables = [var for var in ds.data_vars if dim in ds[var].dims]
            # TODO deal with time-dependent data, etc.
            # Basically requires checking which variables are static, which aren't.
            # For non-static, requires repeating all geometries.
            # Call reset_index on multi-index to generate them as regular columns.
            gdfs.append(
                gpd.GeoDataFrame(
                    data=ds[variables].to_dataframe(dim_order=dim_order),
                    geometry=grid.to_pygeos(dim),
                )
            )

        return pd.concat(gdfs)


class UgridDataArrayAccessor(AbstractUgridAccessor):
    """
    This "accessor" makes operations using the UGRID topology available via the
    ``.ugrid`` attribute for UgridDataArrays and UgridDatasets.
    """

    def __init__(self, obj: xr.DataArray, grid: UgridType):
        self.obj = obj
        self.grid = grid

    @property
    def bounds(self) -> Dict[str, Tuple]:
        """
        Mapping from grid name to tuple containing ``minx, miny, maxx, maxy``
        values of the grid's node coordinates.
        """
        return {self.grid.name: self.grid.bounds}

    @property
    def total_bounds(self) -> Tuple:
        """
        Returns a tuple containing ``minx, miny, maxx, maxy`` values of the grid's
        node coordinates.
        """
        return next(iter(self.bounds.values()))

    plot = UncachedAccessor(_PlotMethods)

    def assign_node_coords(self) -> UgridDataArray:
        """
        Assign node coordinates from the grid to the object.

        Returns a new object with all the original data in addition to the new
        node coordinates of the grid.

        Returns
        -------
        assigned: UgridDataset
        """
        return UgridDataArray(self.grid.assign_node_coords(self.obj), self.grid)

    def assign_edge_coords(self) -> UgridDataArray:
        """
        Assign edge coordinates from the grid to the object.

        Returns a new object with all the original data in addition to the new
        node coordinates of the grid.

        Returns
        -------
        assigned: UgridDataset
        """
        return UgridDataArray(self.grid.assign_edge_coords(self.obj), self.grid)

    def assign_face_coords(self) -> UgridDataArray:
        """
        Assign face coordinates from the grid to the object.

        Returns a new object with all the original data in addition to the new
        node coordinates of the grid.

        Returns
        -------
        assigned: UgridDataset
        """
        if self.grid.topology_dimension == 1:
            raise TypeError("Cannot set face coords from a Ugrid1D topology")
        return UgridDataArray(self.grid.assign_face_coords(self.obj), self.grid)

    def set_node_coords(self, node_x: str, node_y: str):
        """
        Given names of x and y coordinates of the nodes of an object, set them
        as the coordinates in the grid.

        Parameters
        ----------
        node_x: str
            Name of the x coordinate of the nodes in the object.
        node_y: str
            Name of the y coordinate of the nodes in the object.
        """
        self.grid.set_node_coords(node_x, node_y, self.obj)

    def isel(self, indexers, **indexers_kwargs):
        """
        Returns a new object with arrays indexed along edges or faces.

        Parameters
        ----------
        indexer: 1d array of integer or bool

        Returns
        -------
        indexed: Union[UgridDataArray, UgridDataset]
        """
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "isel")
        dims = self.grid.dimensions
        invalid = indexers.keys() - set(dims)
        if invalid:
            raise ValueError(
                f"Dimensions {invalid} do not exist. Expected one of {dims}"
            )

        if len(indexers) > 1:
            raise NotImplementedError("Can only index a single dimension at a time")
        dim, indexer = next(iter(indexers.items()))

        result = self.obj.isel({dim: indexer})
        grid = self.grid.isel(dim, indexer)
        if isinstance(self.obj, xr.DataArray):
            return UgridDataArray(result, grid)
        else:
            raise TypeError(f"Expected UgridDataArray, got {type(result).__name__}")

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
        return self._sel(self.obj, self.grid, x, y)

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
        return self._sel_points(self.obj, self.grid, x, y)

    def _raster(self, x, y, index) -> xr.DataArray:
        index = index.ravel()
        # Cast to float for nodata values (NaN)
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

        Returns
        -------
        crs: dict
            A dictionary containing the name of the grid and its CRS.
        """
        return {self.grid.name: self.grid.crs}

    def set_crs(
        self,
        crs: Union["pyproj.CRS", str] = None,  # type: ignore # noqa
        epsg: int = None,
        allow_override: bool = False,
    ):
        """
        Set the Coordinate Reference System (CRS) of a UGRID topology.

        NOTE: The underlying geometries are not transformed to this CRS. To
        transform the geometries to a new CRS, use the ``to_crs`` method.

        Parameters
        ----------
        crs : pyproj.CRS, optional if `epsg` is specified
            The value can be anything accepted
            by :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
            such as an authority string (eg "EPSG:4326") or a WKT string.
        epsg : int, optional if `crs` is specified
            EPSG code specifying the projection.
        allow_override : bool, default False
            If the the UGRID topology already has a CRS, allow to replace the
            existing CRS, even when both are not equal.
        """
        self.grid.set_crs(crs, epsg, allow_override)

    def to_crs(
        self,
        crs: Union["pyproj.CRS", str] = None,  # type: ignore # noqa
        epsg: int = None,
    ) -> UgridDataArray:
        """
        Transform geometries to a new coordinate reference system.
        Transform all geometries in an active geometry column to a different coordinate
        reference system. The ``crs`` attribute on the current Ugrid must
        be set. Either ``crs`` or ``epsg`` may be specified for output.

        This method will transform all points in all objects. It has no notion
        of projecting the cells. All segments joining points are assumed to be
        lines in the current projection, not geodesics. Objects crossing the
        dateline (or other projection boundary) will have undesirable behavior.

        Parameters
        ----------
        crs : pyproj.CRS, optional if `epsg` is specified
            The value can be anything accepted by
            :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
            such as an authority string (eg "EPSG:4326") or a WKT string.
        epsg : int, optional if `crs` is specified
            EPSG code specifying output projection.
        """
        uda = UgridDataArray(self.obj, self.grid.to_crs(crs, epsg))
        if self.grid.node_dimension in self.obj.dims:
            return uda.ugrid.assign_node_coords()
        else:
            return uda

    def to_geodataframe(
        self, name: str = None, dim_order=None
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

        dim = self.obj.dims[-1]
        if name is not None:
            ds = self.obj.to_dataset(name=name)
        else:
            ds = self.obj.to_dataset()

        variables = [var for var in ds.data_vars if dim in ds[var].dims]
        # TODO deal with time-dependent data, etc.
        # Basically requires checking which variables are static, which aren't.
        # For non-static, requires repeating all geometries.
        # Call reset_index on multi-index to generate them as regular columns.
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
        return UgridDataArray(
            reordered_data,
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
        da = self.obj
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
        return self.grid.to_dataset(self.obj)


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
