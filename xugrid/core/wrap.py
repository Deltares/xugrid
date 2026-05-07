from typing import Sequence, Union

import xarray as xr

from xugrid.core.index import UGRID_INDEXES, UgridIndex
from xugrid.ugrid.ugridbase import AbstractUgrid, UgridType


def _has_ugrid_index(obj):
    return any(isinstance(v, UgridIndex) for v in obj.xindexes.values())


def is_ugrid_dataarray(obj) -> bool:
    """Return True if *obj* is an xr.DataArray backed by a UgridIndex."""
    return isinstance(obj, xr.DataArray) and _has_ugrid_index(obj)


def is_ugrid_dataset(obj) -> bool:
    """Return True if *obj* is an xr.Dataset backed by a UgridIndex."""
    return isinstance(obj, xr.Dataset) and _has_ugrid_index(obj)


class UgridDataArray:
    def __new__(cls, obj: xr.DataArray, grid: UgridType):
        if not isinstance(obj, xr.DataArray):
            raise TypeError(
                f"obj must be xarray.DataArray. Received instead: {type(obj).__name__}"
            )
        if not isinstance(grid, AbstractUgrid):
            raise TypeError(
                "grid must be Ugrid1d or Ugrid2d. Received instead: "
                f"{type(grid).__name__}"
            )

        from xugrid.core.index import UgridIndex

        # Strip any existing UgridIndex before attaching the new one
        existing = [k for k, v in obj.xindexes.items() if isinstance(v, UgridIndex)]
        if existing:
            obj = obj.drop_indexes(existing).drop_vars(existing)

        index_cls = UGRID_INDEXES[grid.topology_dimension]
        index = index_cls.from_ugrid(grid)
        coords = xr.Coordinates.from_xindex(index)
        return obj.assign_coords(coords)

    @staticmethod
    def from_data(data, grid: UgridType, facet: str = "face"):
        return grid.create_data_array(data=data, facet=facet)

    @staticmethod
    def from_structured2d(*args, **kwargs):
        from xugrid.core.dataarray_accessor import UgridDataArrayAccessor

        return UgridDataArrayAccessor.from_structured2d(*args, **kwargs)

    @staticmethod
    def from_structured(da, x=None, y=None, x_bounds=None, y_bounds=None):
        import warnings

        from xugrid.core.dataarray_accessor import UgridDataArrayAccessor

        warnings.warn(
            "UgridDataArray.from_structured is deprecated and will be removed. "
            "Use UgridDataArray.from_structured2d instead.",
            FutureWarning,
            stacklevel=2,
        )
        return UgridDataArrayAccessor.from_structured2d(da, x, y, x_bounds, y_bounds)


class UgridDataset:
    def __new__(
        cls,
        obj: xr.Dataset = None,
        grids: Union[UgridType, Sequence[UgridType]] = None,
    ):
        if obj is None and grids is None:
            raise ValueError("At least either obj or grids is required")

        if not isinstance(grids, (list, tuple, set, type(None))):
            grids = [grids]

        if grids is not None:
            for grid in grids:
                if not isinstance(grid, AbstractUgrid):
                    raise TypeError(
                        "grid must be Ugrid1d or Ugrid2d. "
                        f"Received instead: {type(grid).__name__}"
                    )

        if obj is not None:
            if not isinstance(obj, xr.Dataset):
                raise TypeError(
                    f"obj must be xarray.Dataset. Received instead: {type(obj).__name__}"
                )
            if grids is not None:
                # Strip any existing UgridIndexes before re-attaching
                from xugrid.core.index import UgridIndex

                existing = [k for k, v in obj.xindexes.items() if isinstance(v, UgridIndex)]
                ds = obj.drop_indexes(existing).drop_vars(existing) if existing else obj
                for grid in grids:
                    index_cls = UGRID_INDEXES[grid.topology_dimension]
                    index = index_cls.from_ugrid(grid)
                    coords = xr.Coordinates.from_xindex(index)
                    ds = ds.assign_coords(coords)
                return ds
            else:
                return obj.ugrid.from_dataset()

        # grids only: create an empty Dataset with each grid's index attached
        ds = xr.Dataset()
        for grid in grids:
            index_cls = UGRID_INDEXES[grid.topology_dimension]
            index = index_cls.from_ugrid(grid)
            coords = xr.Coordinates.from_xindex(index)
            ds = ds.assign_coords(coords)
        return ds

    @staticmethod
    def from_geodataframe(*args, **kwargs):
        from xugrid.core.dataset_accessor import UgridDatasetAccessor

        return UgridDatasetAccessor.from_geodataframe(*args, **kwargs)

    @staticmethod
    def from_structured2d(*args, **kwargs):
        from xugrid.core.dataset_accessor import UgridDatasetAccessor

        return UgridDatasetAccessor.from_structured2d(*args, **kwargs)

    @staticmethod
    def from_structured(dataset, topology=None):
        import warnings

        from xugrid.core.dataset_accessor import UgridDatasetAccessor

        warnings.warn(
            "UgridDataset.from_structured is deprecated and will be removed. "
            "Use UgridDataset.from_structured2d instead.",
            FutureWarning,
            stacklevel=2,
        )
        return UgridDatasetAccessor.from_structured2d(dataset, topology)
