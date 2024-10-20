"""
Wrap in advance instead of overloading __getattr__.

This allows for tab completion and documentation.
"""
from __future__ import annotations

import types
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
        grids = {dim: topology for dim in topology.dims}

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
                "obj must be xarray.DataArray. Received instead: "
                f"{type(obj).__name__}"
            )
        if not isinstance(grid, AbstractUgrid):
            raise TypeError(
                "grid must be Ugrid1d or Ugrid2d. Received instead: "
                f"{type(grid).__name__}"
            )

        self._grid = grid
        self._obj = assign_ugrid_coords(obj, [grid])

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
    def from_structured(
        da: xr.DataArray,
        x: str | None = None,
        y: str | None = None,
    ) -> "UgridDataArray":
        """
        Create a UgridDataArray from a (structured) xarray DataArray.

        The spatial dimensions are flattened into a single UGRID face dimension.

        By default, this method looks for the "x" and "y" coordinates and assumes
        they are one-dimensional. To convert rotated or curvilinear coordinates,
        provide the names of the x and y coordinates.

        Parameters
        ----------
        da: xr.DataArray
            Last two dimensions must be the y and x dimension (in that order!).
        x: str, default: None
            Which coordinate to use as the UGRID x-coordinate.
        y: str, default: None
            Which coordinate to use as the UGRID y-coordinate.

        Returns
        -------
        unstructured: UgridDataArray
        """
        if da.ndim < 2:
            raise ValueError(
                "DataArray must have at least two spatial dimensions. "
                f"Found: {da.dims}."
            )
        grid, stackdims = Ugrid2d.from_structured(da, x, y, return_dims=True)
        face_da = da.stack(  # noqa: PD013
            {grid.face_dimension: stackdims}, create_index=False
        ).drop_vars(stackdims, errors="ignore")
        return UgridDataArray(face_da, grid)

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
    def from_structured(ds: xr.Dataset, topology: dict | None = None) -> "UgridDataset":
        """
        Create a UgridDataset from a (structured) xarray Dataset.

        The spatial dimensions are flattened into a single UGRID face dimension.

        By default, this method looks for the "x" and "y" coordinates and assumes
        they are one-dimensional. To convert rotated or curvilinear coordinates,
        provide the names of the x and y coordinates.

        Parameters
        ----------
        ds: xr.Dataset
        topology: dict, optional, default is None.
            Mapping of topology name to x and y coordinate variables.
            If None, searches for "x" and "y" coordinates and creates a Ugrid2d
            topology with name "mesh2d".

        Returns
        -------
        unstructured: UgridDataset
        """
        if topology is None:
            topology = {"mesh2d": ("x", "y")}

        dataset = ds
        grids = []
        for name, (x, y) in topology.items():
            stackdims, grid = Ugrid2d.from_structured(
                ds, x=x, y=y, name=name, return_dims=True
            )
            dataset = dataset.stack(*stackdims).drop_vars(stackdims, errors="ignore")  # noqa: PD013
            grids.append(grid)

        return UgridDataset(dataset, grids)
