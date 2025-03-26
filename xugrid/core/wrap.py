"""
Wrap in advance instead of overloading __getattr__.

This allows for tab completion and documentation.
"""

from __future__ import annotations

import types
import warnings
from collections import ChainMap
from functools import wraps
from itertools import chain
from typing import List, Sequence, Union

import xarray as xr
from numpy.typing import ArrayLike
from pandas import RangeIndex

import xugrid
from xugrid.conversion import grid_from_dataset, grid_from_geodataframe
from xugrid.core.utils import unique_grids
from xugrid.ugrid.ugrid2d import Ugrid2d
from xugrid.ugrid.ugridbase import AbstractUgrid, UgridType, align

# Import entire module here for circular import of UgridDatasetAccessor and
# UgridDataArrayAccessor. Note: can only be used in functions (since that code
# is run at runtime).


def maybe_xugrid(obj, topology, old_indexes=None):
    if not isinstance(obj, (xr.DataArray, xr.Dataset)):
        return obj

    # Topology can either be a sequence of grids or a grid.
    if isinstance(topology, (list, set, tuple)):
        grids = {dim: grid for grid in topology for dim in grid.dims}
    else:
        grids = dict.fromkeys(topology.dims, topology)

    item_grids = unique_grids([grids[dim] for dim in obj.dims if dim in grids])

    if len(item_grids) == 0:
        return obj
    else:
        result, aligned = align(obj, item_grids, old_indexes)

        if isinstance(result, xr.DataArray):
            if len(aligned) > 1:
                raise RuntimeError("This shouldn't happen. Please open an issue.")
            return UgridDataArray(result, aligned[0])

        elif isinstance(result, xr.Dataset):
            return UgridDataset(result, aligned)


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

        # Sidestep staticmethods, classmethods: in that case self will not be a
        # xugrid type.
        if isinstance(self, (UgridDataArray, UgridDataset)):
            return maybe_xugrid(result, self.grids, self.obj.indexes)
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

    # Set every method, property from the xarray object to the UgridDataArray,
    # UgridDataset. Don't set everything, as this will break the objects.
    #
    # class Empty:
    #     pass
    #
    # keep = {
    #     "__eq__",
    #     "__ge__",
    #     "__gt__",
    #     "__le__",
    #     "__lt__",
    #     "__ne__",
    #     "__repr__",
    #     "__str__",
    # }
    #
    # remove = set(dir(Empty)) - keep

    remove = {
        # These members are shared by all objects:
        "__class__",
        "__delattr__",
        "__dict__",
        "__dir__",
        "__doc__",
        "__format__",
        "__getattribute__",
        "__hash__",
        "__init__",
        "__init_subclass__",
        "__module__",
        "__new__",
        "__reduce__",
        "__reduce_ex__",
        "__setattr__",
        "__sizeof__",
        "__subclasshook__",
        "__weakref__"
        # These are additionally included in xarray:
        "__getatrr__",
        "__slots__",
        "__annotations__",
    }

    attr_names = set(dir(source_class)) - remove
    all_attrs = {k: getattr(source_class, k) for k in attr_names}

    methods = {k: v for k, v in all_attrs.items() if isinstance(v, FuncType)}
    for name, method in methods.items():
        wrapped = wraps_xarray(method)
        setattr(wrapped, "__doc__", method.__doc__)
        target_class_dict[name] = wrapped

    properties = {k: v for k, v in all_attrs.items() if isinstance(v, property)}
    for name, prop in properties.items():
        wrapped = property(
            fget=wraps_xarray(prop.__get__),
            fset=wraps_xarray(prop.__set__),
            doc=prop.__doc__,
        )
        target_class_dict[name] = wrapped

    accessors = {k: v for k, v in all_attrs.items() if isinstance(v, type)}
    for name, accessor in accessors.items():
        wrapped = property(wrap_accessor(accessor))
        setattr(wrapped, "__doc__", accessor.__doc__)
        target_class_dict[name] = wrapped

    return


class DataArrayForwardMixin:
    wrap(
        target_class_dict=vars(),
        source_class=xr.DataArray,
    )


class DatasetForwardMixin:
    wrap(
        target_class_dict=vars(),
        source_class=xr.Dataset,
    )


def assign_ugrid_coords(obj, grids):
    grid_dims = ChainMap(*(grid.sizes for grid in grids))
    ugrid_dims = set(grid_dims.keys()).intersection(obj.dims)
    ugrid_coords = {dim: RangeIndex(0, grid_dims[dim]) for dim in ugrid_dims}
    obj = obj.assign_coords(ugrid_coords)
    return obj


class UgridDataArray(DataArrayForwardMixin):
    def __init__(self, obj: xr.DataArray, grid: UgridType):
        if not isinstance(obj, xr.DataArray):
            raise TypeError(
                f"obj must be xarray.DataArray. Received instead: {type(obj).__name__}"
            )
        if not isinstance(grid, AbstractUgrid):
            raise TypeError(
                "grid must be Ugrid1d or Ugrid2d. Received instead: "
                f"{type(grid).__name__}"
            )

        self._grid = grid
        self._obj = assign_ugrid_coords(obj, [grid])
        self._obj.set_close(obj._close)

    def __getattr__(self, attr):
        result = getattr(self.obj, attr)
        return maybe_xugrid(result, [self.grid])

    @property
    def obj(self):
        return self._obj

    @property
    def grid(self):
        return self._grid

    @property
    def grids(self) -> List[UgridType]:
        return [self._grid]

    @property
    def ugrid(self):
        """
        UGRID Accessor. This "accessor" makes operations using the UGRID
        topology available.
        """
        return xugrid.core.dataarray_accessor.UgridDataArrayAccessor(
            self.obj, self.grid
        )

    @staticmethod
    def from_structured2d(
        da: xr.DataArray,
        x: str | None = None,
        y: str | None = None,
        x_bounds: xr.DataArray = None,
        y_bounds: xr.DataArray = None,
    ) -> "UgridDataArray":
        """
        Create a UgridDataArray from a (structured) xarray DataArray.

        The spatial dimensions are flattened into a single UGRID face dimension.
        By default, this method looks for:

        1. "x" and "y" dimensions
        2. "longitude" and "latitude" dimensions
        3. "axis" attributes of "X" or "Y" on coordinates
        4. "standard_name" attributes of "longitude", "latitude",
           "projection_x_coordinate", or "projection_y_coordinate" on coordinate
           variables

        Parameters
        ----------
        da : xr.DataArray
            The structured data array to convert. The last two dimensions must be
            the y and x dimensions (in that order).
        x : str, optional
            Name of the UGRID x-coordinate, or x-dimension if bounds are provided.
            Defaults to None.
        y : str, optional
            Name of the UGRID y-coordinate, or y-dimension if bounds are provided.
            Defaults to None.
        x_bounds : xr.DataArray, optional
            Bounds for x-coordinates. Required for non-monotonic coordinates.
            Defaults to None.
        y_bounds : xr.DataArray, optional
            Bounds for y-coordinates. Required for non-monotonic coordinates.
            Defaults to None.

        Returns
        -------
        UgridDataArray
            The unstructured grid data array.

        Notes
        -----
        When using bounds, they should have one of these shapes:
        * x bounds: (M, 2) or (N, M, 4)
        * y bounds: (N, 2) or (N, M, 4)
        where N is the number of rows (along y) and M is columns (along x).
        Cells with NaN bounds coordinates are omitted.

        Examples
        --------
        Basic usage with default coordinate detection:

        >>> uda = xugrid.UgridDataArray.from_structured2d(data_array)

        Specifying explicit coordinate names:

        >>> uda = xugrid.UgridDataArray.from_structured2d(
        ...     data_array,
        ...     x="longitude",
        ...     y="latitude"
        ... )

        Using bounds for curvilinear grids:

        >>> uda = xugrid.UgridDataArray.from_structured2d(
        ...     data_array,
        ...     x="x_dim",
        ...     y="y_dim",
        ...     x_bounds=x_bounds_array,
        ...     y_bounds=y_bounds_array
        ... )
        """
        if da.ndim < 2:
            raise ValueError(
                "DataArray must have at least two spatial dimensions. "
                f"Found: {da.dims}."
            )
        if x_bounds is not None and y_bounds is not None:
            if x is None or y is None:
                raise ValueError("x and y must be provided for bounds")
            yx = (y, x)
            grid, index = Ugrid2d.from_structured_bounds(
                x_bounds=x_bounds.transpose(y, x, ...).to_numpy(),
                y_bounds=y_bounds.transpose(y, x, ...).to_numpy(),
                return_index=True,
            )
        else:
            # Possibly rely on inference of x and y dims.
            grid, yx = Ugrid2d.from_structured(da, x, y, return_dims=True)
            index = slice(None, None)

        face_da = (
            da.stack(  # noqa: PD013
                {grid.face_dimension: (yx)}, create_index=False
            )
            .isel({grid.face_dimension: index})
            .drop_vars(yx, errors="ignore")
        )
        return UgridDataArray(face_da, grid)

    @staticmethod
    def from_structured(
        da: xr.DataArray,
        x: str | None = None,
        y: str | None = None,
        x_bounds: xr.DataArray = None,
        y_bounds: xr.DataArray = None,
    ) -> "UgridDataArray":
        warnings.warn(
            "UgridDataArray.from_structured is deprecated and will be removed. "
            "Use UgridDataArray.from_structured2d instead.",
            FutureWarning,
            stacklevel=2,
        )
        return UgridDataArray.from_structured2d(da, x, y, x_bounds, y_bounds)

    @staticmethod
    def from_data(data: ArrayLike, grid: UgridType, facet: str) -> UgridDataArray:
        """
        Create a UgridDataArray from a grid and a 1D array of values.

        Parameters
        ----------
        data: array like
            Values for this array. Must be a ``numpy.ndarray`` or castable to
            it.
        grid: Ugrid1d, Ugrid2d
        facet: str
            With which facet to associate the data. Options for Ugrid1d are,
            ``"node"`` or ``"edge"``. Options for Ugrid2d are ``"node"``,
            ``"edge"``, or ``"face"``.

        Returns
        -------
        uda: UgridDataArray
        """
        return grid.create_data_array(data=data, facet=facet)


class UgridDataset(DatasetForwardMixin):
    def __init__(
        self,
        obj: xr.Dataset = None,
        grids: Union[UgridType, Sequence[UgridType]] = None,
    ):
        if obj is None and grids is None:
            raise ValueError("At least either obj or grids is required")

        if obj is None:
            original = ds = xr.Dataset()
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
            original = obj

        if grids is None:
            topologies = obj.ugrid_roles.topology
            grids = [grid_from_dataset(obj, topology) for topology in topologies]
        else:
            # Make sure it's a new list
            if isinstance(grids, (list, tuple, set)):
                grids = list(grids)
            else:  # not iterable
                grids = [grids]
            # Now typecheck
            for grid in grids:
                if not isinstance(grid, AbstractUgrid):
                    raise TypeError(
                        "grid must be Ugrid1d or Ugrid2d. "
                        f"Received instead: {type(grid).__name__}"
                    )

        self._grids = grids
        self._obj = assign_ugrid_coords(ds, grids)
        # We've created a new object; the file handle will be associated with the original.
        # set_close makes sure that when close is called on the UgridDataset, that the
        # file will actually be closed (via the original).
        self._obj.set_close(original._close)

    @property
    def obj(self):
        return self._obj

    @property
    def grid(self) -> UgridType:
        # We need to do some checking. Don't duplicate that logic.
        return self.ugrid.grid

    @property
    def grids(self) -> List[UgridType]:
        return self._grids

    @property
    def ugrid(self):
        """
        UGRID Accessor. This "accessor" makes operations using the UGRID
        topology available.
        """
        return xugrid.core.dataset_accessor.UgridDatasetAccessor(self.obj, self.grids)

    def __getattr__(self, attr):
        result = getattr(self.obj, attr)
        return maybe_xugrid(result, self.grids)

    def __setitem__(self, key, value):
        # TODO: check with topology
        if isinstance(value, UgridDataArray):
            append = True
            # Check if the dimensions occur in self.
            # if they don't, the grid should be added.
            if self.grids is not None:
                alldims = set(chain.from_iterable([grid.dims for grid in self.grids]))
                matching_dims = set(value.grid.dims).intersection(alldims)
                if matching_dims:
                    append = False
                    # If they do match: the grids should match.
                    grids = {dim: grid for grid in self.grids for dim in grid.dims}
                    firstdim = next(iter(matching_dims))
                    grid_to_check = grids[firstdim]
                    if not grid_to_check.equals(value.grid):
                        raise ValueError(
                            "Grids share dimension names but do not are not identical. "
                            f"Matching dimensions: {matching_dims}"
                        )

            self.obj[key] = value.obj
            if append:
                self._grids.append(value.grid)
        else:
            self.obj[key] = value

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

    @staticmethod
    def from_structured2d(
        dataset: xr.Dataset, topology: dict | None = None
    ) -> "UgridDataset":
        """
        Create a UgridDataset from a (structured) xarray Dataset.

        The spatial dimensions are flattened into a single UGRID face dimension.
        By default, this method looks for:

        1. "x" and "y" dimensions
        2. "longitude" and "latitude" dimensions
        3. "axis" attributes of "X" or "Y" on coordinates
        4. "standard_name" attributes of "longitude", "latitude",
           "projection_x_coordinate", or "projection_y_coordinate" on coordinate
           variables

        Parameters
        ----------
        dataset : xr.Dataset
            The structured dataset to convert.
        topology : dict, optional
            Either:
            * A mapping of topology name to (x, y) coordinate names
            * A mapping of topology name to a dict containing:
            - "x": x-dimension name
            - "y": y-dimension name
            - "bounds_x": x-bounds variable name
            - "bounds_y": y-bounds variable name
            Defaults to {"mesh2d": (None, None)}.

        Returns
        -------
        UgridDataset
            The unstructured grid dataset.

        Notes
        -----
        When using bounds, they should have one of these shapes:
        * x bounds: (M, 2) or (N, M, 4)
        * y bounds: (N, 2) or (N, M, 4)
        where N is the number of rows (along y) and M is columns (along x).
        Cells with NaN bounds coordinates are omitted.

        Examples
        --------
        Basic usage with default coordinate names:

        >>> uds = xugrid.UgridDataset.from_structured2d(dataset)

        Specifying custom coordinate names:

        >>> uds = xugrid.UgridDataset.from_structured2d(
        ...     dataset,
        ...     topology={"my_mesh2d": {"x": "xc", "y": "yc"}}
        ... )

        Multiple grid topologies in a single dataset:

        >>> uds = xugrid.UgridDataset.from_structured2d(
        ...     dataset,
        ...     topology={
        ...         "mesh2d_xy": {"x": "x", "y": "y"},
        ...         "mesh2d_lonlat": {"x": "lon", "y": "lat"}
        ...     }
        ... )

        Using bounds for non-monotonic coordinates (e.g., curvilinear grids):

        >>> uds = xugrid.UgridDataset.from_structured2d(
        ...     dataset,
        ...     topology={
        ...         "my_mesh2d": {
        ...             "x": "M",
        ...             "y": "N",
        ...             "bounds_x": "grid_x",
        ...             "bounds_y": "grid_y"
        ...         }
        ...     }
        ... )
        """
        if topology is None:
            # By default, set None. This communicates to
            # Ugrid2d.from_structured to infer x and y dims.
            topology = {"mesh2d": (None, None)}

        grids = []
        dss = []
        xy_vars = set()  # store x, y, x_bounds, y_bounds to drop.
        for name, args in topology.items():
            x_bounds = None
            y_bounds = None
            if isinstance(args, dict):
                x = args.get("x")
                y = args.get("y")
                if "x_bounds" in args and "y_bounds" in args:
                    if x is None or y is None:
                        raise ValueError("x and y must be provided for bounds")
                    x_bounds = dataset[args["x_bounds"]]
                    y_bounds = dataset[args["y_bounds"]]
                    xy_vars.update((args["x_bounds"], args["y_bounds"]))
            elif isinstance(args, tuple):
                x, y = args
            else:
                raise TypeError(
                    "Expected dict or tuple in topology, received: "
                    f"{type(args).__name__}"
                )

            if x_bounds is not None and y_bounds is not None:
                stackdims = (y, x)
                grid, index = Ugrid2d.from_structured_bounds(
                    x_bounds.transpose(*stackdims, ...).to_numpy(),
                    y_bounds.transpose(*stackdims, ...).to_numpy(),
                    name=name,
                    return_index=True,
                )
            else:
                grid, stackdims = Ugrid2d.from_structured(
                    dataset, x=x, y=y, name=name, return_dims=True
                )
                index = slice(None, None)

            # Use subset to check that ALL dims of stackdims are present in the
            # variable.
            checkdims = set(stackdims)
            xy_vars.update(checkdims)
            ugrid_vars = [
                name
                for name, var in dataset.data_vars.items()
                if checkdims.issubset(var.dims) and name not in xy_vars
            ]
            dss.append(
                dataset[ugrid_vars]  # noqa: PD013
                .stack({grid.face_dimension: stackdims})
                .isel({grid.face_dimension: index})
                .drop_vars(stackdims + (grid.face_dimension,))
            )
            grids.append(grid)

        # Add the original dataset to include all non-UGRID variables.
        dss.append(dataset.drop_vars(xy_vars, errors="ignore"))
        # Then merge with compat="override". This'll pick the first available
        # variable: i.e. it will prioritize the UGRID form.
        merged = xr.merge(dss, compat="override")
        return UgridDataset(merged, grids)

    @staticmethod
    def from_structured(
        dataset: xr.Dataset, topology: dict | None = None
    ) -> "UgridDataset":
        warnings.warn(
            "UgridDataset.from_structured is deprecated and will be removed. "
            "Use UgridDataset.from_structured2d instead.",
            FutureWarning,
            stacklevel=2,
        )
        return UgridDataset.from_structured2d(dataset, topology)
