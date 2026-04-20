from xugrid import data
from xugrid.core.common import (
    load_dataarray,
    load_dataset,
    open_dataarray,
    open_dataset,
    open_mfdataset,
    open_zarr,
)
from xugrid.core.constructors import (
    UgridDataArray,
    UgridDataset,
    dataarray,
    dataarray_from_data,
    dataarray_from_structured2d,
    dataset,
    dataset_from_geodataframe,
    dataset_from_structured2d,
)
from xugrid.core.dataarray_accessor import UgridDataArrayAccessor
from xugrid.core.dataset_accessor import UgridDatasetAccessor
from xugrid.plot import plot
from xugrid.regrid.gridder import NetworkGridder
from xugrid.regrid.regridder import (
    BarycentricInterpolator,
    CentroidLocatorRegridder,
    OverlapRegridder,
    RelativeOverlapRegridder,
)
from xugrid.ugrid.burn import burn_vector_geometry, earcut_triangulate_polygons
from xugrid.ugrid.conventions import UgridRolesAccessor
from xugrid.ugrid.partitioning import merge_partitions
from xugrid.ugrid.polygonize import polygonize
from xugrid.ugrid.snapping import (
    create_snap_to_grid_dataframe,
    snap_nodes,
    snap_to_grid,
)
from xugrid.ugrid.ugrid1d import Ugrid1d
from xugrid.ugrid.ugrid2d import Ugrid2d

__version__ = "0.15.2"

__all__ = (
    "data",
    "load_dataarray",
    "load_dataset",
    "open_dataarray",
    "open_dataset",
    "open_mfdataset",
    "open_zarr",
    "UgridDataArrayAccessor",
    "UgridDatasetAccessor",
    "UgridDataArray",
    "UgridDataset",
    "dataarray",
    "dataarray_from_data",
    "dataarray_from_structured2d",
    "dataset",
    "dataset_from_structured2d",
    "dataset_from_geodataframe",
    "plot",
    "BarycentricInterpolator",
    "CentroidLocatorRegridder",
    "OverlapRegridder",
    "RelativeOverlapRegridder",
    "burn_vector_geometry",
    "earcut_triangulate_polygons",
    "NetworkGridder",
    "UgridRolesAccessor",
    "merge_partitions",
    "polygonize",
    "snap_nodes",
    "snap_to_grid",
    "create_snap_to_grid_dataframe",
    "Ugrid1d",
    "Ugrid2d",
)
