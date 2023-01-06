import abc


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
