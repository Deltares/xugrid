import pkg_resources

from . import data
from .ugrid import Ugrid1d, Ugrid2d
from .ugrid_dataset import (
    UgridAccessor,
    UgridDataArray,
    UgridDataset,
    open_dataarray,
    open_dataset,
    open_mfdataset,
    open_zarr,
)

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except pkg_resources.DistributionNotFound:
    # package is not installed
    pass
