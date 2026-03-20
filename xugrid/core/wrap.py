import warnings
from typing import Sequence, Union

import xarray as xr

from xugrid.core.index import UGRID_INDEXES
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

    index_cls = UGRID_INDEXES[grid.topology_dimension]
    index = index_cls.from_ugrid(grid)
    coords = xr.Coordinates.from_xindex(index)
    return obj.assign_coords(coords)


def dataset(
    obj: xr.Dataset | None = None,
    grids: Union[UgridType, Sequence[UgridType]] = None,
) -> xr.Dataset:
    if obj is None and grids is None:
        raise ValueError("At least either obj or grids is required")

    if obj is not None:
        return obj.ugrid.from_dataset()


class UgridDataArray:
    def __new__(cls, obj: xr.DataArray, grid: UgridType) -> xr.DataArray:
        warnings.warn(
            "UgridDataArray is deprecated as a constructor and will be removed in a future version. "
            "UgridDataArray is no longer a distinct type: the unstructured grid topology is now stored "
            "as an explicit xarray index. Use xugrid.dataarray(obj, grid) instead.",
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
            "UgridDataset is deprecated as a constructor and will be removed in a future version. "
            "UgridDataset is no longer a distinct type: the unstructured grid topology is now stored "
            "as an explicit xarray index. Use xugrid.dataset(obj, grids) instead.",
            FutureWarning,
            stacklevel=2,
        )
        return dataset(obj, grids)


#
#  TODO: xarray Dataset can be initialized with just coords, not data vars;
#        presumably providing just topologies is the equivalent.
#        if obj is not None:
#            if not isinstance(obj, xr.Dataset):
#                raise TypeError(
#                    "obj must be xarray.Dataset. Received instead: "
#                    f"{type(obj).__name__}"
#                )
#            connectivity_vars = [
#                name
#                for v in obj.ugrid_roles.connectivity.values()
#                for name in v.values()
#            ]
#            ds = obj.drop_vars(obj.ugrid_roles.topology + connectivity_vars)
#
#        if grids is None:
#            topologies = obj.ugrid_roles.topology
#            grids = [grid_from_dataset(obj, topology) for topology in topologies]
#        else:
#            # Make sure it's a new list
#            if isinstance(grids, (list, tuple, set)):
#                grids = list(grids)
#            else:  # not iterable
#                grids = [grids]
#            # Now typecheck
#            for grid in grids:
#                if not isinstance(grid, AbstractUgrid):
#                    raise TypeError(
#                        "grid must be Ugrid1d or Ugrid2d. "
#                        f"Received instead: {type(grid).__name__}"
#                    )
