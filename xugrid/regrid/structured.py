"""
This module contains the logic for regridding from a structured form to another
structured form. All coordinates are assumed to be fully orthogonal to each
other.

While the unstructured logic would work for structured data as well, it is much
less efficient than utilizing the structure of the coordinates.
"""
from typing import Union

import numpy as np
import xarray as xr

from xugrid.regrid.overlap_1d import overlap_1d, overlap_1d_nd
from xugrid.regrid.unstructured import UnstructuredGrid2d
from xugrid.regrid.utils import broadcast
from xugrid.ugrid.ugrid2d import Ugrid2d

# from xugrid import Ugrid2d


class StructuredGrid1d:
    """
    e.g. z -> z; so also works for unstructured

    Parameters
    ----------
    bounds: (n, 2)
    """

    def __init__(self, obj: Union[xr.DataArray, xr.Dataset], name: str):
        bounds_name_left = f"{name}bounds_left"  # e.g. xbounds
        bounds_name_right = f"{name}bounds_right"  # e.g. xbounds
        size_name = f"d{name}"  # e.g. dx

        index = obj.indexes[name]
        # take care of potentially decreasing coordinate values
        if index.is_monotonic_decreasing:
            midpoints = index.values[::-1]
            flipped = True
            side = "right"
        elif index.is_monotonic_increasing:
            midpoints = index.values
            flipped = False
            side = "left"
        else:
            raise ValueError(f"{name} is not monotonic for array {obj.name}")

        if bounds_name_left in obj.coords:
            start = obj[bounds_name_left].values
            end = obj[bounds_name_right].values
            bounds = np.column_stack((start, end))
        else:
            if size_name in obj.coords:
                # works for scalar size and array size
                size = np.abs(obj[size_name].values)
            else:
                # no bounds defined, no dx defined
                # make an estimate of cell size
                size = np.diff(midpoints)
                # Check if equidistant
                atolx = 1.0e-4 * size[0]
                if not np.allclose(size, size[0], atolx):
                    raise ValueError(
                        f"DataArray has to be equidistant along {name}, or "
                        f'explicit bounds must be given as "{name}bounds", or '
                        f'cellsizes must be as "d{name}"'
                    )

            start = midpoints - 0.5 * size
            end = midpoints + 0.5 * size
            bounds = np.column_stack((start, end))

        self.name = name
        self.midpoints = midpoints
        self.bounds = bounds
        self.flipped = flipped
        self.side = side
        self.grid = obj

    @property
    def ndim(self):
        return 1

    @property
    def dims(self):
        return (self.name,)

    @property
    def size(self):
        return len(self.bounds)

    @property
    def length(self):
        return abs(np.diff(self.bounds, axis=1))

    def flip_if_needed(self, index):
        if self.flipped:
            return self.size - index - 1
        else:
            return index

    def valid_nodes_within_bounds(self, other):
        """
        Retruns nodes when midpoints are within bounding box of overlaying grid.
        In cases that midpoints (and bounding boxes) are flipped, computed indexes
        are fliped as well.

        Args:
            self (StructuredGrid1d): source grid
            other (StructuredGrid1d): target grid

        Returns:
            valid_self_index (np.array): valid source indexes
            valid_other_index (np.array): valid target indexes
        """
        start = np.searchsorted(self.bounds[:, 0], other.midpoints, side=self.side)
        end = np.searchsorted(self.bounds[:, 1], other.midpoints, side=self.side)
        valid = (
            (start == (end + 1))
            & (other.midpoints > self.bounds[0, 0])
            & (other.midpoints < self.bounds[-1, 1])
        )
        valid_other_index = np.arange(other.size)[valid]
        valid_self_index = end[valid]
        valid_self_index = self.flip_if_needed(valid_self_index)
        valid_other_index = other.flip_if_needed(valid_other_index)
        return valid_self_index, valid_other_index

    def valid_nodes_within_bounds_and_extend(self, other):
        """
        returns all valid nodes for linear interpolation. In addition to valid_nodes_within_bounds()
        is checked if target midpoints are not outside outer source boundary midpoints. In that case
        there is no interpolation possible.

        Args:
            self (StructuredGrid1d): source grid
            other (StructuredGrid1d): target grid

        Returns:
            valid_self_index (np.array): valid source indexes
            valid_other_index (np.array): valid target indexes
        """
        source_index, target_index = self.valid_nodes_within_bounds(other)
        valid = (other.midpoints[target_index] > self.midpoints[0]) & (
            (other.midpoints[target_index] < self.midpoints[-1])
        )
        return source_index[valid], target_index[valid]

    def overlap_1d_structured(self, other):
        """
        Returns source and target nodes and overlapping length. It utilises overlap_1d()
        and does an aditional flip in cases of reversed midpoints

        Args:
            self (StructuredGrid1d): source grid
            other (StructuredGrid1d): target grid

        Returns:
            valid_self_index (np.array): valid source indexes
            valid_other_index (np.array): valid target indexes
            weights (np.array): length of overlap
        """
        source_index, target_index, weights = overlap_1d(self.bounds, other.bounds)
        source_index = self.flip_if_needed(source_index)
        target_index = other.flip_if_needed(target_index)
        return source_index, target_index, weights

    def centroids_to_linear_sets(
        self,
        other,
        source_index: np.array,
        target_index: np.array,
        weights: np.array,
        neighbour: np.array,
    ):
        """
        Returns for every target node an pair of connected source nodes based on
        centroids connection inputs

        Args:
            self (StructuredGrid1d): source grid
            other (StructuredGrid1d): target grid
            source_index (np.array): source index (centroids)
            target_index (np.array): target index (centroids)
            weights (np.array): weights (centroids)

        Returns:biliniar

            source_index (np.array): source index (linear)
            target_index (np.array): target index (linear)
            weights (np.array): weights (linear)
        """
        # if source_index is flipped, source-index is decreasing and neighbour need to be flipped
        if self.flipped:
            neighbour = -neighbour
        source_index = np.column_stack((source_index, source_index + neighbour)).ravel()
        target_index = np.repeat(target_index, 2)
        weights = np.column_stack((weights, 1.0 - weights)).ravel()

        # correct for possibility of out of bound due to column-stack source_index + 1 and -1
        valid = np.logical_and(source_index <= self.size - 1, source_index >= 0)
        return source_index[valid], target_index[valid], weights[valid]

    def get_midpoint_index(self, array_index):
        """
        Returns midpoint array indexes for given array_index.

        Args:
            array_index (np.array): array_index

        Returns:biliniar
            midpoint_index (np.array): midpoint_index

        """
        if self.flipped:
            return self.size - array_index - 1
        else:
            return array_index

    def compute_distance_to_centroids(self, other, source_index, target_index):
        """
        computes linear weights bases on centroid indexes.

        Args:
            self (StructuredGrid1d): source grid
            other (StructuredGrid1d): target grid
            other (_type_): _description_
            source_index (np.array): source index (centroids)
            target_index (np.array): target index (centroids)

        Raises:
            ValueError: for not enought midpoints

        Returns:
            weights (np.array): weights
        """

        source_midpoint_index = self.get_midpoint_index(source_index)
        target_midpoints_index = other.get_midpoint_index(target_index)
        neighbour = np.ones(target_midpoints_index.size, dtype=int)
        # cases where midpoint target < midpoint source
        condition = (
            other.midpoints[target_midpoints_index]
            < self.midpoints[source_midpoint_index]
        )
        neighbour[condition] = -neighbour[condition]

        if not self.midpoints.size > 2:
            raise ValueError(
                "source index must larger than 2. Cannot interpolate with one point"
            )
        weights = (
            other.midpoints[target_midpoints_index]
            - self.midpoints[source_midpoint_index]
        ) / (
            self.midpoints[source_midpoint_index + neighbour]
            - self.midpoints[source_midpoint_index]
        )
        weights[weights < 0.0] = 0.0
        weights[weights > 1.0] = 1.0

        return weights, neighbour

    def sorted_output(
        self, source_index: np.array, target_index: np.array, weights: np.array
    ):
        """
        Returns sorted input based on target index. The regridder needs input sorted on
        row index of WeightMatrixCOO (target index)

        Args:
            source_index (np.array): source index
            target_index (np.array): target index
            weights (np.array): weights

        Returns:
            source_index (np.array): source index
            target_index (np.array): target index
            weights (np.array): weights
        """
        sorter_target = np.argsort(target_index)
        return (
            source_index[sorter_target],
            target_index[sorter_target],
            weights[sorter_target],
        )

    def overlap(self, other: "StructuredGrid1d", relative: bool):
        """
        Returns source and target indexes and overlapping length

        Args:
            self (StructuredGrid1d): source grid
            other (StructuredGrid1d): target grid
            relative (bool): overlapping length target relative to source length

        Returns:
            source_index (np.array): source indexes
            target_index (np.array): target indexes
            weights (np.array): overlapping length
        """
        source_index, target_index, weights = self.overlap_1d_structured(other)
        if relative:
            weights /= self.length()[source_index]
        return self.sorted_output(source_index, target_index, weights)

    def locate_centroids(self, other: "StructuredGrid1d"):
        """
        Returns source and target indexes based on nearest neighbor of centroids

        Args:
            self (StructuredGrid1d): source grid
            other (StructuredGrid1d): target grid

        Returns:
            source_index (np.array): source indexes
            target_index (np.array): target indexes
            weights (np.array): array of ones
        """
        source_index, target_index = self.valid_nodes_within_bounds(other)
        weights = np.ones(source_index.size, dtype=float)
        return self.sorted_output(source_index, target_index, weights)

    def linear_weights(self, other: "StructuredGrid1d"):
        """
        Returns linear source and target indexes and corresponding weights

        Args:
            self (StructuredGrid1d): source grid
            other (StructuredGrid1d): target grid

        Raises:
            ValueError: when number of nodes is to small to compute linear weights

        Returns:
            source_index (np.array): overlaying self indexes
            target_index (np.array): overlaying other indexes
            weights (np.array): array linear weights
        """

        source_index, target_index = self.valid_nodes_within_bounds_and_extend(other)
        weights, neighbour = self.compute_distance_to_centroids(
            other, source_index, target_index
        )
        source_index, target_index, weights = self.centroids_to_linear_sets(
            other,
            source_index,
            target_index,
            weights,
            neighbour,
        )
        return self.sorted_output(source_index, target_index, weights)


class StructuredGrid2d(StructuredGrid1d):
    """
    e.g. (x,y) -> (x,y)
    """

    def __init__(
        self,
        obj: Union[xr.DataArray, xr.Dataset],
        name_x: str,
        name_y: str,
    ):
        self.xbounds = StructuredGrid1d(obj, name_x)
        self.ybounds = StructuredGrid1d(obj, name_y)

    @property
    def ndim(self):
        return 2

    @property
    def dims(self):
        return self.ybounds.dims + self.xbounds.dims  # ("y", "x")

    @property
    def size(self):
        return self.ybounds.size * self.xbounds.size

    @property
    def shape(self):
        return (self.ybounds.size, self.xbounds.size)

    @property
    def area(self):
        return np.multiply.outer(self.ybounds.length, self.xbounds.length)

    def convert_to(self, matched_type):
        if isinstance(self, matched_type):
            return self
        elif isinstance(self, UnstructuredGrid2d):
            return Ugrid2d.from_structured(self.xbounds, self.ybounds)
        else:
            raise TypeError(
                f"Cannot convert StructuredGrid2d to {matched_type.__name__}"
            )

    def broadcast_sorted(
        self,
        other,
        source_index_y: np.array,
        source_index_x: np.array,
        target_index_y: np.array,
        target_index_x: np.array,
        weights_y: np.array,
        weights_x: np.array,
    ):
        source_index, target_index, weights = broadcast(
            self.shape,
            other.shape,
            (source_index_y, source_index_x),
            (target_index_y, target_index_x),
            (weights_y, weights_x),
        )
        sorter = np.argsort(target_index)
        return source_index[sorter], target_index[sorter], weights[sorter]

    def overlap(self, other, relative: bool):
        """
        Returns
        -------
        source_index: 1d np.ndarray of int
        target_index: 1d np.ndarray of int
        weights: 1d np.ndarray of float
        """
        source_index_x, target_index_x, weights_x = self.xbounds.overlap(
            other.xbounds, relative
        )
        source_index_y, target_index_y, weights_y = self.ybounds.overlap(
            other.ybounds, relative
        )
        return self.broadcast_sorted(
            other,
            source_index_y,
            source_index_x,
            target_index_y,
            target_index_x,
            weights_y,
            weights_x,
        )

    def locate_centroids(self, other):
        """
        Returns
        -------
        source_index: 1d np.ndarray of int
        target_index: 1d np.ndarray of int
        weights: 1d np.ndarray of float
        """
        source_index_x, target_index_x, weights_x = self.xbounds.locate_centroids(
            other.xbounds
        )
        source_index_y, target_index_y, weights_y = self.ybounds.locate_centroids(
            other.ybounds
        )
        return self.broadcast_sorted(
            other,
            source_index_y,
            source_index_x,
            target_index_y,
            target_index_x,
            weights_y,
            weights_x,
        )

    def linear_weights(self, other):
        """
        Returns
        -------
        source_index: 1d np.ndarray of int
        target_index: 1d np.ndarray of int
        weights: 1d np.ndarray of float
        """
        source_index_x, target_index_x, weights_x = self.xbounds.linear_weights(
            other.xbounds
        )
        source_index_y, target_index_y, weights_y = self.ybounds.linear_weights(
            other.ybounds
        )
        return self.broadcast_sorted(
            other,
            source_index_y,
            source_index_x,
            target_index_y,
            target_index_x,
            weights_y,
            weights_x,
        )


class StructuredGrid3d:
    """
    e.g. (x,y,z) -> (x,y,z)

    A voxel model (GeoTOP)
    """

    def __init__(
        self,
        obj: Union[xr.DataArray, xr.Dataset],
        name_x: str,
        name_y: str,
        name_z: str,
    ):
        self.xbounds = StructuredGrid1d(obj, name_x)
        self.ybounds = StructuredGrid1d(obj, name_y)
        self.zbounds = StructuredGrid1d(obj, name_z)

    @property
    def shape(self):
        return (self.zbound.size, self.ybounds.size, self.xbounds.size)

    @property
    def volume(self):
        return np.multiply.outer(self.zbounds.length, self.area)

    def broadcast_sorted(
        self,
        other,
        source_index_z: np.array,
        source_index_y: np.array,
        source_index_x: np.array,
        target_index_z: np.array,
        target_index_y: np.array,
        target_index_x: np.array,
        weights_z: np.array,
        weights_y: np.array,
        weights_x: np.array,
    ):
        source_index, target_index, weights = broadcast(
            self.shape,
            other.shape,
            (source_index_z, source_index_y, source_index_x),
            (target_index_z, target_index_y, target_index_x),
            (weights_z, weights_y, weights_x),
        )
        sorter = np.argsort(target_index)
        return source_index[sorter], target_index[sorter], weights[sorter]

    def overlap(self, other, relative: bool):
        """
        Returns
        -------
        source_index: 1d np.ndarray of int
        target_index: 1d np.ndarray of int
        weights: 1d np.ndarray of float
        """
        source_index_x, target_index_x, weights_x = self.xbounds.overlap(
            other.xbounds, relative
        )
        source_index_y, target_index_y, weights_y = self.ybounds.overlap(
            other.ybounds, relative
        )
        source_index_z, target_index_z, weights_z = self.zbounds.overlap(
            other.zbounds, relative
        )
        return self.broadcast_sorted(
            other,
            source_index_z,
            source_index_y,
            source_index_x,
            target_index_z,
            target_index_y,
            target_index_x,
            weights_z,
            weights_y,
            weights_x,
        )

    def locate_centroids(self, other):
        source_index_x, target_index_x, weights_x = self.xbounds.locate_centroids(
            other.xbounds
        )
        source_index_y, target_index_y, weights_y = self.ybounds.locate_centroids(
            other.ybounds
        )
        source_index_z, target_index_z, weights_z = self.zbounds.locate_centroids(
            other.zbounds
        )
        return self.broadcast_sorted(
            other,
            source_index_z,
            source_index_y,
            source_index_x,
            target_index_z,
            target_index_y,
            target_index_x,
            weights_z,
            weights_y,
            weights_x,
        )

    def linear_weights(self, other):
        source_index_x, target_index_x, weights_x = self.xbounds.linear_weights(
            other.xbounds
        )
        source_index_y, target_index_y, weights_y = self.ybounds.linear_weights(
            other.ybounds
        )
        source_index_z, target_index_z, weights_z = self.zbounds.linear_weights(
            other.zbounds
        )
        return self.broadcast_sorted(
            other,
            source_index_z,
            source_index_y,
            source_index_x,
            target_index_z,
            target_index_y,
            target_index_x,
            weights_z,
            weights_y,
            weights_x,
        )


class ExplicitStructuredGrid3d:
    """
    e.g. (x,y,z) -> (x,y,z)

    A layered model (E.g. REGIS)
    """

    def __init__(
        self,
        obj: Union[xr.DataArray, xr.Dataset],
    ):
        # zbounds is a 3D array with dimensions (nlayer, y.size * x.size, 2)
        self.xbounds = StructuredGrid1d(obj, "x")
        self.ybounds = StructuredGrid1d(obj, "y")
        self.zbounds = StructuredGrid1d(obj, "z")

    @property
    def shape(self):
        raise NotImplementedError

    @property
    def volume(self):
        return np.multiply.outer(self.zbounds.length, self.area)

    def overlap(self, other, relative: bool):
        """
        Returns
        -------
        source_index: 1d np.ndarray of int
        target_index: 1d np.ndarray of int
        weights: 1d np.ndarray of float
        """
        source_index_x, target_index_x, weights_x = self.xbounds.overlap(
            other.xbounds, relative
        )
        source_index_y, target_index_y, weights_y = self.ybounds.overlap(
            other.ybounds, relative
        )
        source_index_yx, target_index_yx, weights_yx = broadcast(
            self.shape,
            other.shape,
            (source_index_y, source_index_x),
            (target_index_y, target_index_x),
            (weights_y, weights_x),
        )

        if isinstance(other, StructuredGrid3d):
            zbounds = other.zbounds[np.newaxis, ...]
            target_index = np.zeros(zbounds.shape[1], dtype=int)
        elif isinstance(other, ExplicitStructuredGrid3d):
            zbounds = other.zbounds
            target_index = target_index_yx
        else:
            raise TypeError

        source_index_zyx, target_index_zyx, weights_z = overlap_1d_nd(
            self.zbounds,
            zbounds,
            source_index_yx,
            target_index,
        )
        # TODO: check array dims
        weights_zyx = weights_z * weights_yx
        return source_index_zyx, target_index_zyx, weights_zyx
