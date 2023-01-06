from xarray.core.utils import either_dict_or_kwargs


class UgridDataArrayOps:
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