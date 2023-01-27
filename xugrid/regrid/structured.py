"""
This module contains the logic for regridding from a structured form to another
structured form. All coordinates are assumed to be fully orthogonal to each
other.

While the unstructured logic would work for structured data as well, it is much
less efficient than utilizing the structure of the coordinates.
"""
import numpy as np

from xugrid.regrid.overlap_1d import overlap_1d
from xugrid.regrid.utils import broadcast


class StructuredGrid1d:
    """
    e.g. z -> z; so also works for unstructured

    Parameters
    ----------
    bounds: (n, 2)
    """

    def __init__(self, bounds):
        self.bounds = bounds

    @property
    def size(self):
        return len(self.bounds)

    def overlap(self, other, relative: bool):
        source_index, target_index, weights = overlap_1d(self.bounds, other.bounds)
        if relative:
            weights /= self.length()[source_index]
        return source_index, target_index, weights

    def length(self):
        return abs(np.diff(self.bounds, axis=1))


class StructuredGrid2d:
    """
    e.g. (x,y) -> (x,y)

    Parameters
    ----------
    xbounds: (nx, 2)
    ybounds: (ny, 2)
    """

    def __init__(
        self,
        xbounds,
        ybounds,
    ):
        self.xbounds = StructuredGrid1d(xbounds)
        self.ybounds = StructuredGrid1d(ybounds)

    @property
    def shape(self):
        return (self.ybounds.size, self.xbounds.size)

    def overlap(self, other, relative: bool):
        source_index_x, target_index_x, weights_x = self.xbounds.overlap(
            other.xbounds, relative
        )
        source_index_y, target_index_y, weights_y = self.ybounds.overlap(
            other.ybounds, relative
        )
        return broadcast(
            self.shape,
            other.shape,
            (source_index_y, source_index_x),
            (target_index_y, target_index_x),
            (weights_y, weights_x),
        )
