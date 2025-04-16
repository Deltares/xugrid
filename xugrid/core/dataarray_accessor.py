from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.sparse
import xarray as xr

# from xugrid.plot.pyvista import to_pyvista_grid
from xugrid.core.accessorbase import AbstractUgridAccessor
from xugrid.core.utils import UncachedAccessor
from xugrid.core.wrap import UgridDataArray, UgridDataset
from xugrid.plot.plot import _PlotMethods
from xugrid.ugrid import connectivity
from xugrid.ugrid.interpolate import (
    interpolate_na_helper,
    laplace_interpolate,
)
from xugrid.ugrid.ugrid1d import Ugrid1d
from xugrid.ugrid.ugrid2d import Ugrid2d
from xugrid.ugrid.ugridbase import UgridType


class UgridDataArrayAccessor(AbstractUgridAccessor):
    """
    This "accessor" makes operations using the UGRID topology available via the
    ``.ugrid`` attribute for UgridDataArrays and UgridDatasets.
    """

    def __init__(self, obj: xr.DataArray, grid: UgridType):
        self.obj = obj
        self.grid = grid

    @property
    def grids(self) -> List[UgridType]:
        """
        The UGRID topology of this DataArry, as a list. Included for
        consistency with UgridDataset.
        """
        return [self.grid]

    @property
    def name(self) -> str:
        """Name of the UGRID topology of this DataArray."""
        return self.grid.name

    @property
    def names(self) -> List[str]:
        """
        Name of the UGRID topology, as a list. Included for consistency with
        UgridDataset.
        """
        return [self.grid.name]

    @property
    def topology(self) -> Dict[str, UgridType]:
        """Mapping from name to UGRID topology."""
        return {self.name: self.grid}

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

    def rename(self, name: str) -> UgridDataArray:
        """
        Give a new name to the UGRID topology and update the associated
        coordinate and dimension names in the DataArray.

        Parameters
        ----------
        name: str
            The new name of the topology.
        """
        obj = self.obj
        new_grid, name_dict = self.grid.rename(name, return_name_dict=True)
        to_rename = tuple(obj.coords) + tuple(obj.dims)
        new_obj = obj.rename({k: v for k, v in name_dict.items() if k in to_rename})
        return UgridDataArray(new_obj, new_grid)

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

    def sel(self, x=None, y=None):
        """
        Return a new object, a subselection in the UGRID x and y coordinates.

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
        result = self.grid.sel(self.obj, x, y)
        if isinstance(result, tuple):
            return UgridDataArray(*result)
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
        out_of_bounds: str, default: "warn"
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
        """
        return self.grid.sel_points(self.obj, x, y, out_of_bounds, fill_value)

    def rasterize(self, resolution: float) -> xr.DataArray:
        """
        Rasterize unstructured grid by sampling.

        Parameters
        ----------
        resolution: float
            Spacing in x and y.

        Returns
        -------
        rasterized: xr.DataArray
        """
        x, y, index = self.grid.rasterize(resolution)
        return self._raster(x, y, index)

    def rasterize_like(self, other: Union[xr.DataArray, xr.Dataset]) -> xr.DataArray:
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
        rasterized: xr.DataArray
        """
        x, y, index = self.grid.rasterize_like(
            x=other["x"].to_numpy(),
            y=other["y"].to_numpy(),
        )
        return self._raster(x, y, index)

    def to_periodic(self):
        """
        Convert this grid to a periodic grid, where the rightmost boundary
        shares its nodes with the leftmost boundary.

        Returns
        -------
        periodic: UgridDataArray
        """
        grid, obj = self.grid.to_periodic(obj=self.obj)
        return UgridDataArray(obj, grid)

    def to_nonperiodic(self, xmax: float):
        """
        Convert this grid from a periodic grid (where the rightmost boundary shares its
        nodes with the leftmost boundary) to an aperiodic grid, where the leftmost nodes
        are separate from the rightmost nodes.

        Parameters
        ----------
        xmax: float
            The x-value of the newly created rightmost boundary nodes.

        Returns
        -------
        nonperiodic: UgridDataArray
        """
        grid, obj = self.grid.to_nonperiodic(xmax=xmax, obj=self.obj)
        return UgridDataArray(obj, grid)

    def _to_facet(self, facet: str, contributors_dim: str):
        """
        Map the data from one facet to another.

        Parameters
        ----------
        facet: str
            node, edge, face
        contributors_dim: str
            how to name the dimension for the contributors (e.g. three nodes
            per triangle face, two nodes per edge, etc.).
        """
        grid = self.grid
        obj = self.obj

        gridfacets = grid.facets
        if facet not in gridfacets:
            raise ValueError(
                f"Cannot map to {facet} for a {type(grid).__name__} topology."
            )

        source_dim = set(grid.dimensions).intersection(obj.dims).pop()
        target_dim = getattr(grid, f"{facet}_dimension")
        if source_dim == target_dim:
            raise ValueError(
                f"No conversion needed, data is already {facet}-associated."
            )

        # Find out on which facet we're currently located
        source = {v: k for k, v in gridfacets.items()}[source_dim]
        connectivity = grid.format_connectivity_as_dense(
            getattr(grid, f"{facet}_{source}_connectivity")
        )
        indexer = xr.DataArray(connectivity, dims=(target_dim, contributors_dim))
        # Ensure the source dimension is not chunked for efficient indexing.
        obj = obj.chunk({source_dim: -1})
        # Set the fill values (-1) to NaN
        mapped = obj.isel({source_dim: indexer}).where(connectivity != -1)
        return UgridDataArray(mapped, grid)

    def to_node(self, dim="contributors"):
        """
        Map data to nodes.

        Creates a new dimension representing the contributing source elements
        for each node, as multiple faces/edges can connect to a single node.

        Parameters
        ----------
        dim : str, optional

        Returns
        -------
        mapped: UgridDataArray
            A new UgridDataArray with data mapped to the nodes of the grid.

        Examples
        --------
        Compute the mean elevation based on the surrounding faces for each node:

        >>> node_elevation = face_elevation.to_node().mean("contributors")
        """
        return self._to_facet("node", dim)

    def to_edge(self, dim="contributors"):
        """
        Map data to edges.

        Creates a new dimension representing the contributing source elements
        for each node, as two nodes or two faces are connected to an edge.

        Parameters
        ----------
        dim : str, optional

        Returns
        -------
        mapped: UgridDataArray
            A new UgridDataArray with data mapped to the edges of the grid.

        Examples
        --------
        Compute the mean elevation based on the nodes for each edge:

        >>> edge_elevation = node_elevation.to_edge().mean("contributors")
        """
        return self._to_facet("edge", dim)

    def to_face(self, dim="contributors"):
        """
        Map data to faces.

        Creates a new dimension representing the contributing source elements
        for each node, as two edges or multiple nodes are connected to a face.

        Parameters
        ----------
        dim : str, optional

        Returns
        -------
        mapped: UgridDataArray
            A new UgridDataArray with data mapped to the faces of the grid.

        Examples
        --------
        Compute the mean elevation based on the nodes for each face:

        >>> face_elevation = node_elevation.to_face().mean("contributors")
        """
        return self._to_facet("face", dim)

    def intersect_line(
        self, start: Sequence[float], end: Sequence[float]
    ) -> xr.DataArray:
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
        intersection: xr.DataArray
            The length along the line is returned as the "s" coordinate.
        """
        return self.grid.intersect_line(self.obj, start, end)

    def intersect_linestring(self, linestring) -> xr.DataArray:
        """
        Intersect the grid along a collection of linestrings. Returns a new DataArray
        with the values for each intersected segment.

        Parameters
        ----------
        linestring: shapely.LineString

        Returns
        -------
        intersection: xr.DataArray
            The length along the linestring is returned as the "s" coordinate.
        """
        return self.grid.intersect_linestring(self.obj, linestring)

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

        variables = [var for var in ds.data_vars if dim in ds.variables[var].dims]
        # TODO deal with time-dependent data, etc.
        # Basically requires checking which variables are static, which aren't.
        # For non-static, requires repeating all geometries.
        # Call reset_index on multi-index to generate them as regular columns.
        df = ds[variables].to_dataframe(dim_order=dim_order)
        geometry = self.grid.to_shapely(dim)
        return gpd.GeoDataFrame(df, geometry=geometry, crs=self.grid.crs)

    def reindex_like(
        self,
        other: Union[UgridType, UgridDataArray, UgridDataset],
        tolerance: float = 0.0,
    ):
        """
        Conform this object to match the topology of another object. The
        topologies must be exactly equivalent: only the order of the nodes,
        edges, and faces may differ.

        Dimension names must match for equivalent topologies.

        Parameters
        ----------
        other: Ugrid1d, Ugrid2d, UgridDataArray, UgridDataset
        obj: DataArray or Dataset
        tolerance: float, default value 0.0.
            Maximum distance between inexact coordinate matches.

        Returns
        -------
        reindexed: UgridDataArray
        """
        if isinstance(other, (Ugrid1d, Ugrid2d)):
            other_grid = other
        elif isinstance(other, (UgridDataArray, UgridDataset)):
            other_grid = other.ugrid.grid
        else:
            raise TypeError(
                "Expected Ugrid1d, Ugrid2d, UgridDataArray, or UgridDataset,"
                f"received instead: {type(other).__name__}"
            )
        new_obj = self.grid.reindex_like(other_grid, obj=self.obj, tolerance=tolerance)
        return UgridDataArray(new_obj, other_grid)

    def _binary_iterate(self, iterations: int, mask, value, border_value):
        if border_value == value:
            exterior = self.grid.exterior_faces
        else:
            exterior = None
        if mask is not None:
            mask = mask.to_numpy()

        obj = self.obj
        if isinstance(obj, xr.DataArray):
            output = connectivity._binary_iterate(
                self.grid.face_face_connectivity,
                obj.to_numpy(),
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

    def interpolate_na(
        self,
        method: str = "nearest",
        max_distance: Optional[float] = None,
    ):
        """
        Fill in NaNs by interpolating.

        This function automatically finds the UGRID dimension and broadcasts
        over the other dimensions.

        Parameters
        ----------
        method: str, default is "nearest"
            Currently the only supported method.
        max_distance: nonnegative float, optional.
            Use ``None`` for no maximum distance.

        Returns
        -------
        filled: UgridDataArray of floats
        """

        if method != "nearest":
            raise ValueError(f'"{method}" is not a valid interpolator.')

        if max_distance is None:
            max_distance = np.inf

        grid = self.grid
        da = self.obj
        ugrid_dim = grid.find_ugrid_dim(da)

        da_filled = interpolate_na_helper(
            da,
            ugrid_dim=ugrid_dim,
            func=grid._nearest_interpolate,
            kwargs={
                "ugrid_dim": ugrid_dim,
                "max_distance": max_distance,
            },
        )
        return UgridDataArray(da_filled, grid)

    def laplace_interpolate(
        self,
        xy_weights: bool = True,
        direct_solve: bool = False,
        delta=0.0,
        relax=0.0,
        rtol=1.0e-5,
        atol=0.0,
        maxiter: int = 500,
    ):
        """
        Fill in NaNs by using Laplace interpolation.

        This function automatically finds the UGRID dimension and broadcasts
        over the other dimensions.

        This solves Laplace's equation where where there is no data, with data
        values functioning as fixed potential boundary conditions.

        Note that an iterative solver method will be required for large grids.
        In this case, some experimentation with the solver settings may be
        required to find a converging solution of sufficient accuracy. Refer to
        the documentation of :py:func:`scipy.sparse.linalg.spilu` and
        :py:func:`scipy.sparse.linalg.cg`.

        Data can be interpolated from nodes or faces. Direct interpolation of edge
        associated data is not allowed. Instead, create node associated data first,
        then translate that data to the edges.

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
        delta: float, default 0.0.
            ILU0 preconditioner non-diagonally dominant correction.
        relax: float, default 0.0.
            Modified ILU0 preconditioner relaxation factor.
        rtol: float, optional, default 1.0e-5.
            Convergence tolerance for ``scipy.sparse.linalg.cg``.
        atol: float, optional, default 0.0.
            Convergence tolerance for ``scipy.sparse.linalg.cg``.
        maxiter: int, default 500.
            Maximum number of iterations for ``scipy.sparse.linalg.cg``.

        Returns
        -------
        filled: UgridDataArray of floats
        """
        grid = self.grid
        da = self.obj

        grid = self.grid
        da = self.obj
        ugrid_dim = grid.find_ugrid_dim(da)
        if ugrid_dim == grid.edge_dimension:
            raise ValueError("Laplace interpolation along edges is not allowed.")

        connectivity = grid.get_connectivity_matrix(ugrid_dim, xy_weights=xy_weights)
        da_filled = interpolate_na_helper(
            da,
            ugrid_dim,
            func=laplace_interpolate,
            kwargs={
                "connectivity": connectivity,
                "use_weights": xy_weights,
                "direct_solve": direct_solve,
                "delta": delta,
                "relax": relax,
                "rtol": rtol,
                "atol": atol,
                "maxiter": maxiter,
            },
        )
        return UgridDataArray(da_filled, grid)

    def to_dataset(self, optional_attributes: bool = False):
        """
        Convert this UgridDataArray or UgridDataset into a standard
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
        return self.grid.to_dataset(self.obj, optional_attributes)
