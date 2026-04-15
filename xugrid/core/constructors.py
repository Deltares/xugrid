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

        # Make sure it's iterable.
        if not isinstance(grids, Sequence):
            grids = [grids]

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
