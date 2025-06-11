"""
This module contains the logic for regridding from a structured form to another
structured form. All coordinates are assumed to be fully orthogonal to each
other.

While the unstructured logic would work for structured data as well, it is much
less efficient than utilizing the structure of the coordinates.
"""
from __future__ import annotations

import abc
from functools import singledispatchmethod
from typing import Any, Tuple, Union

import numpy as np
import xarray as xr

from xugrid.constants import FloatArray, IntArray
from xugrid.regrid.grid.basegrid import Grid
from xugrid.regrid.grid.unstructured import UnstructuredGrid2d
from xugrid.regrid.utils.array import broadcast
from xugrid.regrid.utils.overlap_1d import overlap_1d, overlap_1d_nd
from xugrid.ugrid.ugrid2d import Ugrid2d


class StructuredGrid(Grid, abc.ABC):
    def flip_if_needed(self, index: IntArray) -> IntArray:
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


class StructuredGrid1d(StructuredGrid):
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
    def coordinates(self) -> FloatArray:
        return self.midpoints

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

    def locate_points(
        self,
        points: np.ndarray,
    ) -> Tuple[IntArray, IntArray]:
        start = np.searchsorted(self.bounds[:, 0], points, side=self.side)
        end = np.searchsorted(self.bounds[:, 1], points, side=self.side)
        valid = (
            (start == (end + 1))
            & (points > self.bounds[0, 0])
            & (points < self.bounds[-1, 1])
        )
        other_index = np.arange(len(points))[valid]
        self_index = end[valid]
        self_index = self.flip_if_needed(self_index)
        return self_index, other_index

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

    def centroids_to_linear_pairs(
        self,
        source_index: np.array,
        target_index: np.array,
        weights: np.array,
        neighbor: np.array,
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
        # if source_index is flipped, source-index is decreasing and neighbor
        # need to be flipped
        if self.flipped:
            neighbor = -neighbor
        source_index = np.column_stack((source_index, source_index + neighbor)).ravel()
        target_index = np.repeat(target_index, 2)
        weights = np.column_stack((weights, 1.0 - weights)).ravel()

        # correct for possibility of out of bound due to column-stack
        # source_index + 1 and -1
        valid = np.logical_and(source_index <= self.size - 1, source_index >= 0)
        return source_index[valid], target_index[valid], weights[valid]

    def find_neighbor(
        self, points, source_index: IntArray, target_index: IntArray
    ) -> IntArray:
        if self.midpoints.size < 2:
            raise ValueError(
                f"Coordinate {self.name} has length: {len(self.midpoints)}. "
                "At least two points are required for interpolation."
            )

        # cases where midpoint target <= midpoint source: set neighbor to -1
        neighbor = np.where(
            points[target_index] <= self.midpoints[source_index],
            -1,
            1,
        )

        # When we exceed the original domain, it should still interpolate
        # within the bounds.
        # Make sure neighbor falls in [0, n)
        neighbor_index = np.clip(source_index + neighbor, 0, self.midpoints.size - 1)
        # Update neighbor since we return it
        neighbor = neighbor_index - source_index
        return neighbor, neighbor_index

    def compute_linear_weights_to_points(
        self, points, source_index: IntArray, target_index: IntArray
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
        neighbor, neighbor_index = self.find_neighbor(
            points, source_index, target_index
        )

        # If neighbor is 0, we end up computing zero distance, since we're
        # comparing a midpoint to iself. Instead, set a weight of 1.0 on one,
        # (and impliclity 0 in the other). Similarly, if source and target
        # midpoints coincide, the distance may end up 0.
        length = points.coordinates[target_index] - self.midpoints[source_index]
        total_length = self.midpoints[neighbor_index] - self.midpoints[source_index]
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

    def locate_inside(
        self,
        other: "StructuredGrid1d",
    ) -> Tuple[IntArray, IntArray, FloatArray]:
        """
        Return source and target indexes based on nearest neighbor of
        centroids.

        Parameters
        ----------
        other: Grid

        Returns
        -------
        source_index: np.array
        target_index: np.array
        weights: np.array
        """
        source_index, target_index = self.locate_points(other.coordinates)
        target_index = other.flip_if_needed(target_index)
        weights = np.ones(source_index.size, dtype=float)
        return self.sorted_output(source_index, target_index, weights)

    def linear_weights(
        self, points: FloatArray
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
        source_index, target_index = self.locate_points(points)
        weights, neighbor = self.compute_linear_weights_to_points(
            points, source_index, target_index
        )
        return self.centroids_to_linear_pairs(
            source_index,
            target_index,
            weights,
            neighbor,
        )

    def locate_nearest(
        self,
        points: FloatArray,
    ) -> Tuple[IntArray, IntArray, FloatArray]:
        source_index, target_index = self.locate_points(points)
        neighbor, neighbor_index = self.find_neighbor(source_index)
        if self.flipped:
            neighbor = -neighbor
        distance_first = np.linalg.norm(self.midpoints[source_index] - points, axis=1)
        distance_second = np.linalg.norm(
            self.midpoints[neighbor_index] - points, axis=1
        )
        first_nearest = distance_first <= distance_second
        source_index = np.where(first_nearest, source_index, neighbor_index)
        distance = np.where(first_nearest, distance_first, distance_second)
        return self.sorted_output(source_index, target_index, distance)


@singledispatchmethod
def _1d_barycentric(
    self, other: "StructuredGrid1d"
) -> Tuple[IntArray, IntArray, FloatArray]:
    raise NotImplementedError(
        f"barycentric method not supported for {type(self).__name__} -> {type(other).__name__}"
    )


@_1d_barycentric.register
def _(self, other: StructuredGrid1d) -> Tuple[IntArray, IntArray, FloatArray]:
    source_index, target_index, weights = self.linear_weights(other.coordinates)
    target_index = other.flip_if_needed(target_index)
    return self.sorted_output(source_index, target_index, weights)


@_1d_barycentric.register
def _(self, other: FloatArray) -> Tuple[IntArray, IntArray, FloatArray]:
    source_index, target_index, weights = self.linear_weights(other)
    return self.sorted_output(source_index, target_index, weights)


StructuredGrid1d.barycentric = _1d_barycentric


class StructuredGrid2d(StructuredGrid):
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
    def coordinates(self) -> np.ndarray:
        yy, xx = np.meshgrid(
            self.ybounds.coordinates, self.xbounds.coordinates, indexing="ij"
        )
        return np.column_stack((xx.ravel(), yy.ravel()))

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
        return self.sorted_output(source_index, target_index, weights)

    def to_dataset(self, name: str) -> xr.Dataset:
        ds_x = self.xbounds.to_dataset(name)
        ds_y = self.ybounds.to_dataset(name)
        ds = xr.merge([ds_x, ds_y])
        ds[name + "_type"] = xr.DataArray(-1, attrs={"type": "StructuredGrid2d"})
        return ds


@singledispatchmethod
def _2d_locate_inside(self, other) -> Tuple[IntArray, IntArray, FloatArray]:
    raise NotImplementedError(
        f"locate_inside method not supported for {type(self).__name__} -> {type(other).__name__}"
    )


@singledispatchmethod
def _2d_locate_nearest(self, other) -> Tuple[IntArray, IntArray, FloatArray]:
    raise NotImplementedError(
        f"locate_nearest method not supported for {type(self).__name__} -> {type(other).__name__}"
    )


@singledispatchmethod
def _2d_overlap(self, other, relative: bool) -> Tuple[IntArray, IntArray, FloatArray]:
    """
    Compute (relative) overlap with other.

    Returns
    -------
    source_index: 1d np.ndarray of int
    target_index: 1d np.ndarray of int
    weights: 1d np.ndarray of float
    """
    raise NotImplementedError(
        f"overlap method not supported for {type(self).__name__} -> {type(other).__name__}"
    )


@singledispatchmethod
def _2d_barycentric(self, other) -> Tuple[IntArray, IntArray, FloatArray]:
    """
    Compute linear interpolation weights with other.

    Returns
    -------
    source_index: 1d np.ndarray of int
    target_index: 1d np.ndarray of int
    weights: 1d np.ndarray of float
    """
    raise NotImplementedError(
        f"barycentric method not supported for {type(self).__name__} -> {type(other).__name__}"
    )


@_2d_locate_inside.register
def _(self, other: "StructuredGrid2d") -> Tuple[IntArray, IntArray, FloatArray]:
    source_index_x, target_index_x, weights_x = self.xbounds.locate_inside(
        other.xbounds
    )
    source_index_y, target_index_y, weights_y = self.ybounds.locate_inside(
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


@_2d_locate_inside.register
def _(self, other: "UnstructuredGrid2d") -> Tuple[IntArray, IntArray, FloatArray]:
    x, y = other.coordinates.T
    source_index_x, target_index = self.xbounds.locate_points(x)
    source_index_y, _ = self.ybounds.locate_points(y)
    weights = np.ones(source_index_x.size, dtype=float)
    source_index = np.ravel_multi_index((source_index_y, source_index_x), self.shape)
    return self.sorted_output(source_index, target_index, weights)


@_2d_locate_nearest.register
def _(self, other: "StructuredGrid2d") -> Tuple[IntArray, IntArray, FloatArray]:
    source_index_x, target_index_x, weights_x = self.xbounds.locate_nearest(
        other.xbounds.coordinates
    )
    source_index_y, target_index_y, weights_y = self.ybounds.locate_nearest(
        other.ybounds.coordinates
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


@_2d_locate_nearest.register
def _(self, other: "UnstructuredGrid2d") -> Tuple[IntArray, IntArray, FloatArray]:
    x, y = other.coordinates.T
    source_index_x, target_index, _ = self.xbounds.locate_nearest(x)
    source_index_y, _, _ = self.xbounds.locate_nearest(y)
    weights = np.ones(source_index_x.size, dtype=float)
    source_index = np.ravel_multi_index((source_index_y, source_index_x), self.shape)
    return self.sorted_output(source_index, target_index, weights)


@_2d_overlap.register
def _(
    self, other: "StructuredGrid2d", relative: bool
) -> Tuple[IntArray, IntArray, FloatArray]:
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


@_2d_overlap.register
def _(
    self, other: "UnstructuredGrid2d", relative: bool
) -> Tuple[IntArray, IntArray, FloatArray]:
    converted = self.convert_to(other)
    return converted.overlap(other, relative=relative)


@_2d_barycentric.register
def _(self, other: "StructuredGrid2d") -> Tuple[IntArray, IntArray, FloatArray]:
    source_index_x, target_index_x, weights_x = self.xbounds.barycentric(other.xbounds)
    source_index_y, target_index_y, weights_y = self.ybounds.barycentric(other.ybounds)
    return self.broadcast_sorted(
        other,
        source_index_y,
        source_index_x,
        target_index_y,
        target_index_x,
        weights_y,
        weights_x,
    )


@_2d_barycentric.register
def _(self, other: "UnstructuredGrid2d") -> Tuple[IntArray, IntArray, FloatArray]:
    x, y = other.coordinates.T
    source_index_x, target_index, weights_x = self.xbounds.linear_weights(x)
    source_index_y, _, weights_y = self.ybounds.linear_weights(y)
    source_index = np.ravel_multi_index((source_index_y, source_index_x), self.shape)
    weights = weights_x * weights_y
    return self.sorted_output(source_index, target_index, weights)


StructuredGrid2d.locate_inside = _2d_locate_inside
StructuredGrid2d.locate_nearest = _2d_locate_nearest
StructuredGrid2d.overlap = _2d_overlap
StructuredGrid2d.barycentric = _2d_barycentric


class StructuredGrid3d(StructuredGrid):
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
        raise NotImplementedError()
        self.xbounds = StructuredGrid1d(obj, name_x)
        self.ybounds = StructuredGrid1d(obj, name_y)
        self.zbounds = StructuredGrid1d(obj, name_z)

    @property
    def coordinates(self):
        zz, yy, xx = np.meshgrid(
            self.zbounds.coordinates,
            self.ybounds.coordinates,
            self.xbounds.coordinates,
            indexing="ij",
        )
        return np.column_stack(((zz.ravel(), yy.ravel(), xx.ravel())))

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

    def locate_inside(self, other):
        source_index_x, target_index_x, weights_x = self.xbounds.locate_inside(
            other.xbounds
        )
        source_index_y, target_index_y, weights_y = self.ybounds.locate_inside(
            other.ybounds
        )
        source_index_z, target_index_z, weights_z = self.zbounds.locate_inside(
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
        # TODO?
        self.zbounds = StructuredGrid1d(obj, "z")

    @property
    def shape(self):
        raise (self.zbounds.shape)

    @property
    def area(self) -> FloatArray:
        return np.multiply.outer(self.ybounds.length, self.xbounds.length)

    @property
    def volume(self):
        return np.multiply.outer(self.zbounds.length, self.area)


@singledispatchmethod
def _3d_overlap(self, other, relative: bool):
    """
    Compute (relative) overlap with other.

    Returns
    -------
    source_index: 1d np.ndarray of int
    target_index: 1d np.ndarray of int
    weights: 1d np.ndarray of float
    """
    raise NotImplementedError()


@_3d_overlap.register
def _(self, other: StructuredGrid3d, relative: bool):
    source_index_x, target_index_x, weights_x = self.xbounds.overlap(
        other.xbounds, relative
    )
    source_index_y, target_index_y, weights_y = self.ybounds.overlap(
        other.ybounds, relative
    )
    source_index_z, target_index_z, weights_z = self.zbounds.overlap(
        other.zbounds, relative
    )
    source_index, target_index, weights = broadcast(
        self.shape,
        other.shape,
        (source_index_z, source_index_y, source_index_x),
        (target_index_z, target_index_y, target_index_x),
        (weights_z, weights_y, weights_x),
    )
    return self.sorted_output(source_index, target_index, weights)


@_3d_overlap.register
def _(self, other: "ExplicitStructuredGrid3d", relative: bool):
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

    zbounds = other.zbounds
    target_index = target_index_yx

    source_index, target_index, weights_z = overlap_1d_nd(
        self.zbounds,
        zbounds,
        source_index_yx,
        target_index,
    )
    # TODO: check array dims
    weights_zyx = weights_z * weights_yx
    return self.sorted_output(source_index, target_index, weights_zyx)


StructuredGrid3d.overlap = _3d_overlap
