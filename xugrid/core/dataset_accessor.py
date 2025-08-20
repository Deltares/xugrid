from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from xugrid.conversion import grid_from_geodataframe

# from xugrid.plot.pyvista import to_pyvista_grid
from xugrid.core.accessorbase import AbstractUgridAccessor
from xugrid.core.index import UgridIndex, UgridIndex2d
from xugrid.core.wrap import UgridDataArray, UgridDataset
from xugrid.ugrid.ugrid1d import Ugrid1d
from xugrid.ugrid.ugrid2d import Ugrid2d
from xugrid.ugrid.ugridbase import UgridType


@xr.register_dataset_accessor("ugrid")
class UgridDatasetAccessor(AbstractUgridAccessor):
    def __init__(self, obj: xr.Dataset):
        self.obj = obj

    def initialize(self):  # , grids=None):
        new = self.obj
        for topology in self.obj.ugrid_roles.topology:
            # TODO: currently hardcoded for Ugrid2d
            ugrid_vars = self.obj.ugrid_roles[topology]
            node_xs, node_ys = ugrid_vars["node_coordinates"]
            variables = (
                node_xs[0],
                node_ys[0],
                ugrid_vars["face_node_connectivity"],
            )
            new = new.set_coords(variables).set_xindex(variables, UgridIndex2d)
        return new

    @property
    def grids(self) -> list[UgridType]:
        indexes = list(self.obj.xindexes.values())
        grids = {index._ugrid for index in indexes if isinstance(index, UgridIndex)}
        if len(grids) == 0:
            raise ValueError("Dataset contains no UgridIndex")
        return list(grids)

    @property
    def grid(self) -> UgridType:
        """
        Returns the single UGRID topology in this dataset. Raises a TypeError if
        the dataset contains more than one topology.
        """
        ngrid = len(self.grids)
        if ngrid == 1:
            return self.grids[0]
        else:
            raise TypeError(
                "Can only access grid topology via `.grid` if dataset contains "
                f"exactly one grid. Dataset contains {ngrid} grids. Use "
                "`.grids` instead."
            )

    @property
    def name(self) -> str:
        """
        Returns name of the single UGRID topology in this dataset. Raises a
        TypeError if the dataset contains more than one topology.
        """
        ngrid = len(self.grids)
        if ngrid == 1:
            return self.grid.name
        else:
            raise TypeError(
                "Can only access grid name via `.name` if dataset contains "
                f"exactly one grid. Dataset contains {ngrid} grids. Use "
                "`.names` instead."
            )

    @property
    def names(self) -> List[str]:
        """Names of all the UGRID topologies in the dataset."""
        return [grid.name for grid in self.grids]

    @property
    def topology(self) -> Dict[str, UgridType]:
        """Mapping from names to UGRID topologies."""
        return {grid.name: grid for grid in self.grids}

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
        bounds = np.column_stack(list(self.bounds.values()))
        return (
            bounds[0].min(),
            bounds[1].min(),
            bounds[2].max(),
            bounds[3].max(),
        )

    def rename(self, new_name_or_name_dict: Union[str, Dict[str, str]]) -> UgridDataset:
        """
        Give a new name to the UGRID topology and update the associated
        coordinate and dimension names in the Dataset.

        Parameters
        ----------
        new_name_or_name_dict: str or dict
            If the argument is a string, the new name of the topology. This
            only works if the dataset contains a single UGRID topology. If the
            argument is a dict, it used as a mapping from old names to new
            names.
        """
        if isinstance(new_name_or_name_dict, str):
            ngrid = len(self.grids)
            if ngrid != 1:
                raise TypeError(
                    "Can only rename with a single name if dataset contains "
                    f"exactly one grid. Dataset contains {ngrid} grids. Provide "
                    "a dictionary of old name to new name instead."
                )

            name = new_name_or_name_dict
            new_grid, names = self.grid.rename(name, return_name_dict=True)
            new_grids = [new_grid]

        elif isinstance(new_name_or_name_dict, dict):
            names = {}
            new_grids = []
            for grid in self.grids:
                name = new_name_or_name_dict.get(grid.name)
                if name is None:
                    new_grid = grid
                else:
                    new_grid, name_dict = grid.rename(name, return_name_dict=True)
                    names.update(name_dict)
                new_grids.append(new_grid)

        else:
            raise TypeError(
                "new_name_or_name_dict should be str or dict, received instead: "
                f"{type(new_name_or_name_dict).__name__}"
            )

        obj = self.obj
        to_rename = tuple(obj.data_vars) + tuple(obj.coords) + tuple(obj.dims)
        new_obj = obj.rename({k: v for k, v in names.items() if k in to_rename})
        return UgridDataset(new_obj, new_grids)

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

    def sel(self, x=None, y=None):
        new_obj = self.obj
        new_grids = []
        for grid in self.grids:
            result = grid.sel(new_obj, x, y)
            if isinstance(result, tuple):
                new_obj, new_grid = result
                new_grids.append(new_grid)
            else:
                new_obj = result

        if new_grids:
            return UgridDataset(new_obj, new_grids)
        else:
            return result

    def sel_points(self, x, y, out_of_bounds="warn", fill_value=np.nan):
        """
        Select points in the unstructured grid.

        Out-of-bounds points are ignored. They may be identified via the
        ``index`` coordinate of the returned selection.

        Parameters
        ----------
        x: ndarray of floats with shape ``(n_points,)``
        y: ndarray of floats with shape ``(n_points,)``
        out_of_bounds: str, default ``"warn"``
            What to do when points are located outside of any feature:

            * raise: raise a ValueError.
            * ignore: return ``fill_value`` for the out of bounds points.
            * warn: give a warning and return NaN for the out of bounds points.
            * drop: drop the out of bounds points. They may be identified
              via the ``index`` coordinate of the returned selection.
        fill_value: scalar, DataArray, Dataset, or callable, optional, default: np.nan
            Value to assign to out-of-bounds points if out_of_bounds is warn
            or ignore. Forwarded to xarray's ``.where()`` method.

        Returns
        -------
        points: Union[xr.DataArray, xr.Dataset]
            The name of the topology is prefixed in the x, y coordinates.
        """
        result = self.obj
        for grid in self.grids:
            result = grid.sel_points(result, x, y, out_of_bounds, fill_value)
        return result

    def rasterize(self, resolution: float) -> xr.Dataset:
        """
        Rasterize all face data on 2D unstructured grids by sampling.

        Parameters
        ----------
        resolution: float
            Spacing in x and y.

        Returns
        -------
        rasterized: xr.Dataset
        """
        datasets = []
        for grid in self.grids:
            xx, yy, index = grid.rasterize(resolution, self.total_bounds)
            datasets.append(self._raster(xx, yy, index))
        return xr.merge(datasets)

    def rasterize_like(self, other: Union[xr.DataArray, xr.Dataset]) -> xr.Dataset:
        """
        Rasterize unstructured all face data on 2D unstructured grids by
        sampling on the x and y coordinates of ``other``.

        Parameters
        ----------
        resolution: float
            Spacing in x and y.
        other: Union[xr.DataArray, xr.Dataset]
            Object to take x and y coordinates from.

        Returns
        -------
        rasterized: xr.Dataset
        """
        x = other["x"].to_numpy()
        y = other["y"].to_numpy()
        datasets = []
        for grid in self.grids:
            xx, yy, index = grid.rasterize_like(x, y)
            datasets.append(self._raster(xx, yy, index))
        return xr.merge(datasets)

    def to_periodic(self):
        """
        Convert every grid to a periodic grid, where the rightmost boundary
        shares its nodes with the leftmost boundary.

        Returns
        -------
        periodic: UgridDataset
        """
        grids = []
        result = self.obj
        for grid in self.grids:
            new_grid, result = grid.to_periodic(obj=result)
            grids.append(new_grid)
        return UgridDataset(result, grids)

    def to_nonperiodic(self, xmax: float):
        """
        Convert the grid from a periodic grid (where the rightmost boundary shares its
        nodes with the leftmost boundary) to an aperiodic grid, where the leftmost nodes
        are separate from the rightmost nodes.

        Parameters
        ----------
        xmax: float
            The x-value of the newly created rightmost boundary nodes.

        Returns
        -------
        nonperiodic: UgridDataset
        """
        grids = []
        result = self.obj
        for grid in self.grids:
            new_grid, result = grid.to_nonperiodic(xmax=xmax, obj=result)
            grids.append(new_grid)
        return UgridDataset(result, grids)

    def intersect_line(
        self, start: Sequence[float], end: Sequence[float]
    ) -> xr.Dataset:
        """
        Intersect a line with the grid of this data, and fetch the values of
        the intersected faces.

        Parameters
        ----------
        obj: xr.DataArray or xr.Dataset
        start: sequence of two floats
            coordinate pair (x, y), designating the start point of the line.
        end: sequence of two floats
            coordinate pair (x, y), designating the end point of the line.

        Returns
        -------
        intersection: xr.Dataset
            The name of the topology is prefixed in the x, y and s
            (spatium=distance) coordinates.
        """
        obj = self.obj
        for grid in self.grids:
            obj = grid.intersect_line(obj, start, end)
        return obj

    def intersect_linestring(self, linestring) -> xr.Dataset:
        """
        Intersect the grid along a collection of linestrings. Returns a new Dataset
        with the values for each intersected segment.

        Parameters
        ----------
        linestring: shapely.LineString

        Returns
        -------
        intersection: xr.Dataset
            The name of the topology is prefixed in the x, y and s
            (spatium=distance) coordinates.
        """
        obj = self.obj
        for grid in self.grids:
            obj = grid.intersect_linestring(obj, linestring)
        return obj

    def to_dataset(self, optional_attributes: bool = False):
        """
        Convert this UgridDataset into a standard
        xarray.Dataset.

        The UGRID topology information is added as standard data variables.

        Parameters
        ----------
        optional_attributes: bool, default: False.
            Whether to generate the UGRID optional attributes.

        Returns
        -------
        dataset: UgridDataset
        """
        return xr.merge(
            [grid.to_dataset(self.obj, optional_attributes) for grid in self.grids]
        )

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

    def to_geodataframe(self, dim_order=None) -> "geopandas.GeoDataFrame":  # type: ignore # noqa
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

            variables = [var for var in ds.data_vars if dim in ds.variables[var].dims]
            if variables:
                data = ds[variables].to_dataframe(dim_order=dim_order)
            else:
                data = None

            # TODO deal with time-dependent data, etc.
            # Basically requires checking which variables are static, which aren't.
            # For non-static, requires repeating all geometries.
            # Call reset_index on multi-index to generate them as regular columns.
            gdfs.append(
                gpd.GeoDataFrame(
                    data=data,
                    geometry=grid.to_shapely(dim),
                    crs=grid.crs,
                )
            )

        return pd.concat(gdfs)

    def reindex_like(
        self,
        other: Union[UgridType, UgridDataArray, UgridDataset],
        tolerance: float = 0.0,
    ):
        """
        Conform this object to match the topology of another object. The
        topologies must be exactly equivalent: only the order of the nodes,
        edges, and faces may differ.

        Topologies are matched by name, and dimension names must match for
        equivalent topologies.

        Parameters
        ----------
        other: Ugrid1d, Ugrid2d, UgridDataArray, UgridDataset
        obj: DataArray or Dataset
        tolerance: float, default value 0.0.
            Maximum distance between inexact coordinate matches.

        Returns
        -------
        reindexed: UgridDataset
        """
        if isinstance(other, (Ugrid1d, Ugrid2d)):
            other_grids = [other]
        elif isinstance(other, (UgridDataArray, UgridDataset)):
            other_grids = other.ugrid.grids
        else:
            raise TypeError(
                "Expected Ugrid1d, Ugrid2d, UgridDataArray, or UgridDataset,"
                f"received instead: {type(other).__name__}"
            )
        # Convert to dict to match by name
        other_grids = {grid.name: grid for grid in other_grids}

        new_grids = []
        result = self.obj
        for grid in self.grids:
            other = other_grids.get(grid.name)
            if other:
                result = grid.reindex_like(other, obj=result, tolerance=tolerance)
                new_grids.append(other)
            else:
                new_grids.append(grid)
        return UgridDataset(result, new_grids)

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
        return ds.ugrid.initialize(grids=[grid])

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
        return merged.ugrid.initialize(merged, grid=grids)
