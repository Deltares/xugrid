from typing import Dict, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

# from xugrid.plot.pyvista import to_pyvista_grid
from xugrid.core.accessorbase import AbstractUgridAccessor
from xugrid.core.wrap import UgridDataset
from xugrid.ugrid.ugridbase import UgridType


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
            raise TypeError(
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

    def sel(self, x=None, y=None):
        result = self.obj
        for grid in self.grids:
            result = self._sel(result, grid, x, y)
        return result

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
        result = self.obj
        for grid in self.grids:
            result = grid.sel_points(result, x, y)
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
