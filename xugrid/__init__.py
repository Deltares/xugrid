import pkg_resources

from xugrid import data
from xugrid.core.common import (
    concat,
    full_like,
    merge,
    ones_like,
    open_dataarray,
    open_dataset,
    open_mfdataset,
    open_zarr,
    zeros_like,
)
from xugrid.core.dataarray_accessor import UgridDataArrayAccessor
from xugrid.core.dataset_accessor import UgridDatasetAccessor
from xugrid.core.wrap import UgridDataArray, UgridDataset
from xugrid.plot import plot
from xugrid.regrid.regridder import (
    BarycentricInterpolator,
    CentroidLocatorRegridder,
    OverlapRegridder,
    RelativeOverlapRegridder,
)
from xugrid.ugrid.conventions import UgridRolesAccessor
from xugrid.ugrid.partitioning import merge_partitions
from xugrid.ugrid.snapping import snap_to_grid
from xugrid.ugrid.ugrid1d import Ugrid1d
from xugrid.ugrid.ugrid2d import Ugrid2d

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except pkg_resources.DistributionNotFound:
    # package is not installed
    pass
