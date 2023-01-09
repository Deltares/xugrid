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
        from xugrid.core.wrap import maybe_xugrid

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
