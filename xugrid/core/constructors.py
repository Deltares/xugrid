import warnings
from typing import Sequence, Union

import xarray as xr
from numpy.typing import ArrayLike

from xugrid.conversion import grid_from_geodataframe
from xugrid.core.index import UGRID_INDEXES
from xugrid.ugrid.ugrid2d import Ugrid2d
from xugrid.ugrid.ugridbase import AbstractUgrid, UgridType


def dataarray(obj: xr.DataArray, grid: UgridType) -> xr.DataArray:
    if not isinstance(obj, xr.DataArray):
        raise TypeError(
            f"obj must be xarray.DataArray. Received instead: {type(obj).__name__}"
        )
    if not isinstance(grid, AbstractUgrid):
        raise TypeError(
            f"grid must be Ugrid1d or Ugrid2d. Received instead: {type(grid).__name__}"
        )

    if obj.ugrid.is_indexed:
        raise ValueError("obj is already UGRID indexed")

    index_cls = UGRID_INDEXES[grid.topology_dimension]
    index = index_cls.from_ugrid(grid)
    coords = xr.Coordinates.from_xindex(index)
    return obj.assign_coords(coords)


def dataarray_from_structured2d(
    da: xr.DataArray,
    x: str | None = None,
    y: str | None = None,
    x_bounds: xr.DataArray = None,
    y_bounds: xr.DataArray = None,
) -> xr.DataArray:
    """
    Create a DataArray with UGRID indexes from a (structured) xarray DataArray.

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
    xr.DataArray
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
            f"DataArray must have at least two spatial dimensions. Found: {da.dims}."
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
    return dataarray(obj=face_da, grid=grid)


def dataarray_from_data(data: ArrayLike, grid: UgridType, facet: str) -> xr.DataArray:
    """
    Create a UGRID Indexed DataArray from a grid and a 1D array of values.

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
    uda: xr.DataArray
    """
    return grid.create_data_array(data=data, facet=facet)


def dataset(
    obj: xr.Dataset | None = None,
    grids: Union[UgridType, Sequence[UgridType]] = None,
) -> xr.Dataset:
    if obj is None and grids is None:
        raise ValueError("At least either obj or grids is required")
    if obj is not None:
        if not isinstance(obj, xr.Dataset):
            raise TypeError(
                f"obj must be xarray.Dataset. Received instead: {type(obj).__name__}"
            )
        if obj.ugrid.is_indexed:
            raise ValueError("obj is already UGRID indexed")
    if grids is not None:
        # Make sure it's iterable.
        if not isinstance(grids, Sequence):
            grids = [grids]
        for grid in grids:
            if not isinstance(grid, AbstractUgrid):
                raise TypeError(
                    f"grid must be Ugrid1d or Ugrid2d. Received instead: {type(grid).__name__}"
                )

    new = obj
    topologies = obj.ugrid_roles.topology
    if grids is None:
        topology_dimensions = obj.ugrid_roles.topology_dimensions
        connectivity_vars = [
            name for v in obj.ugrid_roles.connectivity.values() for name in v.values()
        ]
        grid_mapping_vars = [
            name
            for name in obj.ugrid_roles.grid_mapping_names.values()
            if name is not None
        ]

        for topology in topologies:
            topodim = topology_dimensions[topology]
            index_cls = UGRID_INDEXES[topodim]
            variables, options = index_cls._variables_from_dataset(obj, topology)
            index = index_cls.from_variables(
                {name: obj[name] for name in variables}, options=options
            )
            coords = xr.Coordinates.from_xindex(index)
            new = new.assign_coords(coords)

        to_drop = obj.ugrid_roles.topology + connectivity_vars + grid_mapping_vars
        new = new.drop_vars(to_drop, errors="ignore").copy()
        for var in new.variables.values():
            var.attrs = var.attrs.copy()
            var.attrs.pop("grid_mapping", None)

    else:
        if len(topologies) > 0:
            raise ValueError(
                "Received both a dataset containing UGRID mesh topology variables: "
                f"({', '.join(topologies)})\n and explicit grids. Pass either a dataset "
                "with UGRID mesh topology variables or provide grids separately, not both."
            )

        for grid in grids:
            index_cls = UGRID_INDEXES[grid.topology_dimension]
            index = index_cls.from_ugrid(grid)
            coords = xr.Coordinates.from_xindex(index)
            new = new.assign_coords(coords)

    return new


class UgridDataArray:
    def __new__(cls, obj: xr.DataArray, grid: UgridType) -> xr.DataArray:
        warnings.warn(
            "UgridDataArray is deprecated as a constructor and will be removed in a future version.\n"
            "UgridDataArray is no longer a distinct type: the unstructured grid topology is now stored "
            "as an explicit xarray index.\nUse xugrid.dataarray(obj, grid) instead.",
            FutureWarning,
            stacklevel=2,
        )
        return dataarray(obj, grid)


class UgridDataset:
    def __new__(
        cls,
        obj: xr.Dataset = None,
        grids: Union[UgridType, Sequence[UgridType]] = None,
    ) -> xr.Dataset:
        warnings.warn(
            "UgridDataset is deprecated as a constructor and will be removed in a future version.\n"
            "UgridDataset is no longer a distinct type: the unstructured grid topology is now stored "
            "as an explicit xarray index.\nUse xugrid.dataset(obj, grids) instead.",
            FutureWarning,
            stacklevel=2,
        )
        return dataset(obj, grids)


def dataset_from_geodataframe(geodataframe: "geopandas.GeoDataFrame"):  # type: ignore # noqa
    """
    Convert a geodataframe into the appropriate Ugrid topology and dataset.

    Parameters
    ----------
    geodataframe: gpd.GeoDataFrame

    Returns
    -------
    dataset: xr.Dataset
    """
    grid = grid_from_geodataframe(geodataframe)
    ds = xr.Dataset.from_dataframe(geodataframe.drop("geometry", axis=1))
    return dataset(obj=ds, grids=[grid])


def dataset_from_structured2d(
    dataset: xr.Dataset, topology: dict | None = None
) -> xr.Dataset:
    """
    Create a UGRID indexed dataset from a (structured) xarray Dataset.

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
    Dataset
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
                f"Expected dict or tuple in topology, received: {type(args).__name__}"
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
    return dataset(obj=merged, grid=grids)
