"""
This module contains the logic for regridding from a structured form to another
structured form. All coordinates are assumed to be fully orthogonal to each
other.

While the unstructured logic would work for structured data as well, it is much
less efficient than utilizing the structure of the coordinates.
"""

from typing import Any, Tuple, Union

import numpy as np
import xarray as xr

from xugrid.constants import FloatArray, IntArray
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
        bounds_name = f"{name}bounds"  # e.g. xbounds
        size_name = f"d{name}"  # e.g. dx

        index = obj.indexes[name]
        # take care of potentially decreasing coordinate values
        if index.is_monotonic_decreasing:
            midpoints = index.to_numpy()[::-1]
            flipped = True
            side = "right"
        elif index.is_monotonic_increasing:
            midpoints = index.to_numpy()
            flipped = False
            side = "left"
        else:
            raise ValueError(f"{name} is not monotonic for array {obj.name}")

        if bounds_name in obj.coords:
            bounds = obj[bounds_name].to_numpy()
            size = bounds[:, 1] - bounds[:, 0]
        else:
            if size_name in obj.coords:
                # works for scalar size and array size
                size = obj[size_name].to_numpy()
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
                size = np.full_like(midpoints, size[0])

            abs_size = np.abs(size)
            start = midpoints - 0.5 * abs_size
            end = midpoints + 0.5 * abs_size
            bounds = np.column_stack((start, end))

        self.name = name
        self.midpoints = midpoints
        self.bounds = bounds
        self.flipped = flipped
        self.side = side
        self.dname = size_name
        self.dvalue = size
        self.index = index.to_numpy()

    @property
    def coords(self) -> dict:
        coords = {self.name: self.index}
        if self.dvalue.ndim == 0:
            coords[self.dname] = self.dvalue
        else:
            coords[self.dname] = (self.name, self.dvalue)
        return coords

    @property
    def ndim(self) -> int:
        return 1

    @property
    def dims(self) -> Tuple[str]:
        return (self.name,)

    @property
    def size(self) -> int:
        return len(self.bounds)

    @property
    def length(self) -> FloatArray:
        return np.squeeze(abs(np.diff(self.bounds, axis=1)))

    @property
    def directional_bounds(self):
        # Only flip bounds if needed
        if self.flipped:
            return self.bounds[::-1, :].copy()
        else:
            return self.bounds

    def flip_if_needed(self, index: IntArray) -> IntArray:
        if self.flipped:
            return self.size - index - 1
        else:
            return index

    def valid_nodes_within_bounds(
        self, other: "StructuredGrid1d"
    ) -> Tuple[IntArray, IntArray]:
        """
        Return nodes when midpoints are within bounding box of overlaying grid.

        In cases that midpoints (and bounding boxes) are flipped, computed indexes
        are fliped as well.

        Parameters
        ----------
        other: StructuredGrid1d
            The target grid topology

        Returns
        -------
        valid_self_index: np.array
            valid source indexes
        valid_other_index: np.array
            valid target indexes
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

    def valid_nodes_within_bounds_and_extend(
        self, other: "StructuredGrid1d"
    ) -> Tuple[IntArray, IntArray]:
        """
        Return all valid nodes for linear interpolation.
        In addition to valid_nodes_within_bounds() is checked if target
        midpoints are not outside outer source boundary midpoints. In that case
        there is no interpolation possible.

        Parameters
        ----------
        other: StructuredGrid1d
            The target grid.

        Returns
        -------
        valid_self_index: np.array
            valid source indexes
        valid_other_index: np.array
            valid target indexes
        """
        source_index, target_index = self.valid_nodes_within_bounds(other)
        return source_index, target_index
        valid = (other.midpoints[target_index] > self.midpoints[0]) & (
            other.midpoints[target_index] < self.midpoints[-1]
        )
        return source_index[valid], target_index[valid]

    def overlap_1d_structured(
        self, other: "StructuredGrid1d"
    ) -> Tuple[IntArray, IntArray, FloatArray]:
        """
        Return source and target nodes and overlapping length. It utilises overlap_1d()
        and does an aditional flip in cases of reversed midpoints

        Parameters
        ----------
        other: StructuredGrid1d
            The target grid.

        Returns
        -------
        valid_self_index: np.array
            valid source indexes
        valid_other_index: np.array
            valid target indexes
        weights: np.array
            length of overlap
        """
        source_index, target_index, weights = overlap_1d(self.bounds, other.bounds)
        source_index = self.flip_if_needed(source_index)
        target_index = other.flip_if_needed(target_index)
        return source_index, target_index, weights

    def centroids_to_linear_sets(
        self,
        source_index: np.array,
        target_index: np.array,
        weights: np.array,
        neighbour: np.array,
    ) -> Tuple[IntArray, IntArray, FloatArray]:
        """
        Return for every target node an pair of connected source nodes based
        on centroids connection inputs.

        Parameters
        ----------
        source_index: np.array
        target_index: np.array
        weights: np.array

        Returns
        -------
        valid_self_index: np.array
        valid_other_index: np.array
        weights: np.array
            lineair interpolation weights.
        """
        # if source_index is flipped, source-index is decreasing and neighbour
        # need to be flipped
        if self.flipped:
            neighbour = -neighbour
        source_index = np.column_stack((source_index, source_index + neighbour)).ravel()
        target_index = np.repeat(target_index, 2)
        weights = np.column_stack((weights, 1.0 - weights)).ravel()

        # correct for possibility of out of bound due to column-stack
        # source_index + 1 and -1
        valid = np.logical_and(source_index <= self.size - 1, source_index >= 0)
        return source_index[valid], target_index[valid], weights[valid]

    def maybe_reverse_index(self, index: IntArray) -> IntArray:
        """
        Flips the index if needed for descending coordinates.

        Parameters
        ----------
        index: np.ndarray

        Returns
        -------
        checked_index: np.ndarray
        """
        if self.flipped:
            return self.size - index - 1
        else:
            return index

    def compute_linear_weights_to_centroids(
        self, other: "StructuredGrid1d", source_index: IntArray, target_index: IntArray
    ) -> Tuple[FloatArray, IntArray]:
        """
        Compute linear weights bases on centroid indexes.

        Parameters
        ----------
        other: StructuredGrid1d
        source_index: np.array
        target_index: np.array

        Raises
        ------
        ValueError
            When the coordinate contains only a single point.

        Returns
        -------
        weights: np.array
        neighbor: np.narray
        """
        if self.midpoints.size < 2:
            raise ValueError(
                f"Coordinate {self.name} has size: {self.midpoints.size}. "
                "At least two points are required for interpolation."
            )

        source_midpoint_index = self.maybe_reverse_index(source_index)
        target_midpoint_index = other.maybe_reverse_index(target_index)
        # cases where midpoint target <= midpoint source: set neighbor to -1
        neighbor = np.where(
            other.midpoints[target_midpoint_index]
            <= self.midpoints[source_midpoint_index],
            -1,
            1,
        )

        # When we exceed the original domain, it should still interpolate
        # within the bounds.
        # Make sure neighbor falls in [0, n)
        neighbor_index = np.clip(
            source_midpoint_index + neighbor, 0, self.midpoints.size - 1
        )
        # Update neighbor since we return it
        neighbor = neighbor_index - source_midpoint_index

        # If neighbor is 0, we end up computing zero distance, since we're
        # comparing a midpoint to iself. Instead, set a weight of 1.0 on one,
        # (and impliclity 0 in the other). Similarly, if source and target
        # midpoints coincide, the distance may end up 0.
        length = (
            other.midpoints[target_midpoint_index]
            - self.midpoints[source_midpoint_index]
        )
        total_length = (
            self.midpoints[neighbor_index] - self.midpoints[source_midpoint_index]
        )
        # Do not divide by zero.
        # We will overwrite the value anyway at neighbor == 0.
        total_length[total_length == 0] = 1
        weights = 1 - (length / total_length)
        weights[neighbor == 0] = 0.0
        condition = np.logical_and(weights < 0.0, weights > 1.0)
        if condition.any():
            raise ValueError(
                f"Computed invalid weights for dimensions: {self.name} at coords: {self.midpoints[condition]}"
            )
        return weights, neighbor

    def sorted_output(
        self, source_index: IntArray, target_index: IntArray, weights: FloatArray
    ) -> Tuple[IntArray, IntArray, FloatArray]:
        """
        Return sorted input based on target index. The regridder needs input
        sorted on row index of WeightMatrixCOO (target index).

        Parameters
        ----------
        source_index: np.array
        target_index: np.array
        weights: np.array

        Returns
        -------
        source_index: np.array
        target_index: np.array
        weights: np.array
        """
        sorter_target = np.argsort(target_index)
        return (
            source_index[sorter_target],
            target_index[sorter_target],
            weights[sorter_target],
        )

    def overlap(
        self, other: "StructuredGrid1d", relative: bool
    ) -> Tuple[IntArray, IntArray, FloatArray]:
        """
        Return source and target indexes and overlapping length

        Parameters
        ----------
        other: StructuredGrid1d
        relative: bool
            Overlapping length target relative to source length

        Returns
        -------
        source_index: np.array
        target_index: np.array
        weights: np.array
            Overlapping length
        """
        source_index, target_index, weights = self.overlap_1d_structured(other)
        if relative:
            weights /= self.length[source_index]
        return self.sorted_output(source_index, target_index, weights)

    def locate_centroids(
        self,
        other: "StructuredGrid1d",
    ) -> Tuple[IntArray, IntArray, FloatArray]:
        """
        Return source and target indexes based on nearest neighbor of
        centroids.

        Parameters
        ----------
        other: StructuredGrid1d

        Returns
        -------
        source_index: np.array
        target_index: np.array
        weights: np.array
        """
        source_index, target_index = self.valid_nodes_within_bounds(other)
        weights = np.ones(source_index.size, dtype=float)
        return self.sorted_output(source_index, target_index, weights)

    def linear_weights(
        self, other: "StructuredGrid1d"
    ) -> Tuple[IntArray, IntArray, FloatArray]:
        """
        Return linear source and target indexes and corresponding weights.

        Parameters
        ----------
        other: StructuredGrid1d

        Returns
        -------
        source_index: np.array
        target_index: np.array
        weights: np.array
        """
        source_index, target_index = self.valid_nodes_within_bounds_and_extend(other)
        weights, neighbour = self.compute_linear_weights_to_centroids(
            other, source_index, target_index
        )
        source_index, target_index, weights = self.centroids_to_linear_sets(
            source_index,
            target_index,
            weights,
            neighbour,
        )
        return self.sorted_output(source_index, target_index, weights)

    def to_dataset(self, name: str) -> xr.DataArray:
        export_name = name + "_" + self.name
        return xr.DataArray(
            name=name,
            data=np.nan,
            dims=[export_name, export_name + "nbounds"],
            coords={
                export_name: self.midpoints,
                export_name + "bounds": (
                    [export_name, export_name + "nbounds"],
                    self.bounds,
                ),
                export_name + "nbounds": np.arange(2),
            },
        )


class StructuredGrid2d(StructuredGrid1d):
    """Represent e.g. raster data topology."""

    def __init__(
        self,
        obj: Union[xr.DataArray, xr.Dataset],
        name_x: str,
        name_y: str,
    ):
        self.xbounds = StructuredGrid1d(obj, name_x)
        self.ybounds = StructuredGrid1d(obj, name_y)

    @property
    def coords(self) -> dict:
        return self.ybounds.coords | self.xbounds.coords

    @property
    def ndim(self) -> int:
        return 2

    @property
    def dims(self) -> Tuple[str, str]:
        return self.ybounds.dims + self.xbounds.dims  # ("y", "x")

    @property
    def size(self) -> int:
        return self.ybounds.size * self.xbounds.size

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.ybounds.size, self.xbounds.size)

    @property
    def area(self) -> FloatArray:
        return np.multiply.outer(self.ybounds.length, self.xbounds.length)

    def convert_to(self, matched_type: Any) -> Any:
        if matched_type == StructuredGrid2d:
            return self
        elif matched_type == UnstructuredGrid2d:
            ugrid2d = Ugrid2d.from_structured_bounds(
                self.xbounds.directional_bounds,
                self.ybounds.directional_bounds,
            )
            return UnstructuredGrid2d(ugrid2d)
        else:
            raise TypeError(
                f"Cannot convert StructuredGrid2d to {matched_type.__name__}"
            )

    def broadcast_sorted(
        self,
        other: Any,
        source_index_y: IntArray,
        source_index_x: IntArray,
        target_index_y: IntArray,
        target_index_x: IntArray,
        weights_y: FloatArray,
        weights_x: FloatArray,
    ) -> Tuple[IntArray, IntArray, FloatArray]:
        source_index, target_index, weights = broadcast(
            self.shape,
            other.shape,
            (source_index_y, source_index_x),
            (target_index_y, target_index_x),
            (weights_y, weights_x),
        )
        sorter = np.argsort(target_index)
        return source_index[sorter], target_index[sorter], weights[sorter]

    def overlap(self, other, relative: bool) -> Tuple[IntArray, IntArray, FloatArray]:
        """
        Compute (relative) overlap with other.

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

    def locate_centroids(
        self, other, tolerance
    ) -> Tuple[IntArray, IntArray, FloatArray]:
        """
        Locate centroids of other in self.

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

    def linear_weights(self, other) -> Tuple[IntArray, IntArray, FloatArray]:
        """
        Compute linear interpolation weights with other.

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

    def to_dataset(self, name: str) -> xr.Dataset:
        ds_x = self.xbounds.to_dataset(name)
        ds_y = self.ybounds.to_dataset(name)
        ds = xr.merge([ds_x, ds_y])
        ds[name + "_type"] = xr.DataArray(-1, attrs={"type": "StructuredGrid2d"})
        return ds


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
        Compute (relative) overlap with other.

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
        Compute (relative) overlap with other.

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
