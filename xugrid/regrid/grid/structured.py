"""
This module contains the logic for regridding from a structured form to another
structured form. All coordinates are assumed to be fully orthogonal to each
other.

While the unstructured logic would work for structured data as well, it is much
less efficient than utilizing the structure of the coordinates. These classes use
singledispatch methods to achieve some sense of multiple (or rather: dual)
dispatch, where it specializes on the type of self (standard single dispatch) and
the type of other (thereby effective multiple dispatch).

Note: This is equivalent to a big if-else instance function.
"""
from __future__ import annotations

from functools import singledispatchmethod
from typing import Any, Optional, Tuple, Union

import numpy as np
import xarray as xr
from scipy.spatial import KDTree

from xugrid.constants import FloatArray, IntArray
from xugrid.regrid.grid.basegrid import Grid
from xugrid.regrid.grid.unstructured import UnstructuredGrid2d
from xugrid.regrid.utils.array import broadcast
from xugrid.regrid.utils.overlap_1d import overlap_1d, overlap_1d_nd
from xugrid.ugrid.ugrid2d import Ugrid2d

# Define a sentinel index value to keep shapes constant and allow broadcasting.
OUT_OF_BOUNDS = -1


class StructuredGrid1d(Grid):
    """
    e.g. z -> z; so should also work the constant depth layers of unstructured2d (x, y).

    Parameters
    ----------
    bounds: (n, 2)
    """

    def __init__(self, obj: Union[xr.DataArray, xr.Dataset], name: str):
        bounds_name = f"{name}bounds"  # e.g. xbounds
        size_name = f"d{name}"  # e.g. dx

        index = obj.indexes[name]
        midpoints = index.to_numpy()
        # Note: the regridder should ensure all axes are increasing!
        if not index.is_monotonic_increasing:
            raise ValueError(f"{name} is not monotonic for array {obj.name}")

        if bounds_name in obj.coords:
            bounds = obj[bounds_name].to_numpy()
            length = bounds[:, 1] - bounds[:, 0]
        else:
            if size_name in obj.coords:
                # works for scalar size and array size
                length = obj[size_name].to_numpy()
            else:
                # no bounds defined, no dx defined
                # make an estimate of cell size
                length = np.diff(midpoints)
                # Check if equidistant
                atolx = 1.0e-4 * length[0]
                if not np.allclose(length, length[0], atolx):
                    raise ValueError(
                        f"DataArray has to be equidistant along {name}, or "
                        f'explicit bounds must be given as "{name}bounds", or '
                        f'cellsizes must be as "d{name}"'
                    )
                length = np.full_like(midpoints, length[0])

            abs_length = np.abs(length)
            start = midpoints - 0.5 * abs_length
            end = midpoints + 0.5 * abs_length
            bounds = np.column_stack((start, end))

        self.name = name
        self.midpoints = midpoints
        self.bounds = bounds
        self.dname = size_name
        self.dvalue = length
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
    def facet(self) -> str:
        return "face"

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
    ) -> IntArray:
        start = np.searchsorted(self.bounds[:, 0], points, side="left")
        end = np.searchsorted(self.bounds[:, 1], points, side="left")
        in_bounds = (
            (start == (end + 1))
            & (points > self.bounds[0, 0])
            & (points < self.bounds[-1, 1])
        )
        out_of_bounds = ~in_bounds
        self_index = end
        self_index[out_of_bounds] = OUT_OF_BOUNDS
        return self_index

    def overlap_1d_structured(
        self, other: "StructuredGrid1d"
    ) -> Tuple[IntArray, IntArray, FloatArray]:
        """
        Return source and target nodes and overlapping length via overlap_1d().

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
        return source_index, target_index, weights

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
        points: FloatArray,
    ) -> Tuple[IntArray, IntArray, FloatArray]:
        """
        Return source and target indexes based on source cells containing
        the points.

        Parameters
        ----------
        other: Grid

        Returns
        -------
        source_index: np.array
        target_index: np.array
        weights: np.array
        """
        source_index = self.locate_points(points)
        target_index = np.arange(len(points))
        weights = np.ones_like(source_index, dtype=float)
        return self.sorted_output(source_index, target_index, weights)

    def _find_neighbor(
        self, points, source_index: IntArray, target_index: IntArray
    ) -> IntArray:
        """
        Find the relative neighbor, either +1 or -1 depending on whether
        the point is to the left or right of the midpoint.

        When there is no neighbor, return 0 (points to self).
        """
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

        # Make sure neighbor falls in [0, n)
        n = self.midpoints.size - 1
        # Left side
        neighbor[
            (source_index == 0) | (source_index == n) | (source_index == OUT_OF_BOUNDS)
        ] = 0
        return neighbor

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
        source_index = self.locate_points(points)
        target_index = np.arange(len(points))
        neighbor = self.find_neighbor(points, source_index, target_index)
        neighbor_index = source_index + neighbor

        # If neighbor is 0, we end up computing zero distance, since we're
        # comparing a midpoint to iself. Instead, set a weight of 1.0 on one,
        # (and impliclity 0 in the other). Similarly, if source and target
        # midpoints coincide, the distance may end up 0.
        length = points[target_index] - self.midpoints[source_index]
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

        # Create pairs of weights and source for each target.
        source_index = np.column_stack((source_index, source_index + neighbor)).ravel()
        target_index = np.repeat(target_index, 2)
        weights = np.column_stack((weights, 1.0 - weights)).ravel()

        # correct for possibility of out of bound due to column-stack
        # source_index + 1 and -1
        out_of_bounds = ~np.logical_and(
            source_index <= self.size - 1, source_index >= 0
        )
        source_index[out_of_bounds] = OUT_OF_BOUNDS
        weights[out_of_bounds] = 0.0
        return source_index, target_index, weights

    def locate_nearest(
        self,
        points: FloatArray,
    ) -> Tuple[IntArray, IntArray, FloatArray]:
        # This might be of dubious value since we need Euclidian distance,
        # and this measures only along one axis.
        source_index = self.locate_points(points)
        target_index = np.arange(len(points))
        neighbor = self.find_neighbor(points, source_index, target_index)
        neighbor_index = source_index + neighbor
        distance_first = self.midpoints[source_index] - points
        distance_second = self.midpoints[neighbor_index] - points
        first_nearest = distance_first <= distance_second
        source_index = np.where(first_nearest, source_index, neighbor_index)
        distance = np.where(first_nearest, distance_first, distance_second)
        return self.sorted_output(source_index, target_index, distance)

    def barycentric(
        self,
        points: FloatArray,
    ) -> Tuple[IntArray, IntArray, FloatArray]:
        source_index, target_index, weights = self.linear_weights(points)
        return self.sorted_output(source_index, target_index, weights)


class StructuredGrid2d(Grid):
    """Represent e.g. raster data topology."""

    def __init__(
        self,
        obj: Union[xr.DataArray, xr.Dataset],
        name_x: str,
        name_y: str,
    ):
        self.xbounds = StructuredGrid1d(obj, name_x)
        self.ybounds = StructuredGrid1d(obj, name_y)
        self.facet = "face"
        self._kdtree = None

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
    def facet(self) -> str:
        return "face"

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

    @property
    def kdtree(self):
        if self._kdtree is None:
            self._kdtree = KDTree(self.coordinates)
        return self._kdtree

    def convert_to(self, matched_type: Any) -> Any:
        if matched_type == StructuredGrid2d:
            return self
        elif matched_type == UnstructuredGrid2d:
            ugrid2d = Ugrid2d.from_structured_bounds(
                self.xbounds.bounds,
                self.ybounds.bounds,
            )
            return UnstructuredGrid2d(ugrid2d, dim=ugrid2d.face_dimension)
        else:
            raise TypeError(
                f"Cannot convert StructuredGrid2d to {matched_type.__name__}"
            )

    def locate_points(self, points, tolerance) -> IntArray:
        x, y = points.T
        index_x = self.xbounds.locate_points(x)
        index_y = self.ybounds.locate_points(y)
        out_x = index_x == OUT_OF_BOUNDS
        out_y = index_y == OUT_OF_BOUNDS
        index_x[out_x] = 0
        index_y[out_y] = 0
        source_index = np.ravel_multi_index((index_y, index_x), self.shape)
        source_index[out_y | out_x] = OUT_OF_BOUNDS
        return source_index

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
        valid = (source_index_x != OUT_OF_BOUNDS) & (source_index_y != OUT_OF_BOUNDS)
        source_index, target_index, weights = broadcast(
            self.shape,
            other.shape,
            (source_index_y[valid], source_index_x[valid]),
            (target_index_y[valid], target_index_x[valid]),
            (weights_y[valid], weights_x[valid]),
        )
        return self.sorted_output(source_index, target_index, weights)

    def ravel_sorted(
        self,
        source_index_x: IntArray,
        source_index_y: IntArray,
        target_index_x: IntArray,
        target_index_y: IntArray,
        weights_y: Optional[FloatArray] = None,
        weights_x: Optional[FloatArray] = None,
    ) -> Tuple[IntArray, IntArray, FloatArray]:
        valid = (source_index_x != OUT_OF_BOUNDS) & (source_index_y != OUT_OF_BOUNDS)
        source_index = np.ravel_multi_index(
            (source_index_y[valid], source_index_x[valid]), self.shape
        )
        target_index = target_index_x[valid]
        if weights_x is not None and weights_y is not None:
            weights = weights_x[valid] * weights_y[valid]
        else:
            weights = np.ones_like(source_index, dtype=float)
        return self.sorted_output(source_index, target_index, weights)

    def to_dataset(self, name: str) -> xr.Dataset:
        ds_x = self.xbounds.to_dataset(name)
        ds_y = self.ybounds.to_dataset(name)
        ds = xr.merge([ds_x, ds_y])
        ds[name + "_type"] = xr.DataArray(-1, attrs={"type": "StructuredGrid2d"})
        return ds


# Note: the dispatch methods are defined outside of the class, since they need
# type of self.


@singledispatchmethod
def _2d_locate_inside(self, other, tolerance) -> Tuple[IntArray, IntArray, FloatArray]:
    raise NotImplementedError(
        f"locate_inside method not supported for {type(self).__name__} -> {type(other).__name__}"
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
def _2d_barycentric(self, other, tolerance) -> Tuple[IntArray, IntArray, FloatArray]:
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
def _(
    self, other: StructuredGrid2d, tolerance
) -> Tuple[IntArray, IntArray, FloatArray]:
    source_index_x, target_index_x, weights_x = self.xbounds.locate_inside(
        other.xbounds.coordinates
    )
    source_index_y, target_index_y, weights_y = self.ybounds.locate_inside(
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


@_2d_locate_inside.register
def _(
    self, other: "UnstructuredGrid2d", tolerance
) -> Tuple[IntArray, IntArray, FloatArray]:
    x, y = other.coordinates.T
    source_index_x, target_index_x, weights_x = self.xbounds.locate_inside(x)
    source_index_y, target_index_y, weights_y = self.ybounds.locate_inside(y)
    return self.ravel_sorted(
        source_index_x,
        source_index_y,
        target_index_x,
        target_index_y,
        weights_y,
        weights_x,
    )


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
def _(
    self, other: "StructuredGrid2d", tolerance
) -> Tuple[IntArray, IntArray, FloatArray]:
    source_index_x, target_index_x, weights_x = self.xbounds.barycentric(
        other.xbounds.coordinates
    )
    source_index_y, target_index_y, weights_y = self.ybounds.barycentric(
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


@_2d_barycentric.register
def _(
    self, other: "UnstructuredGrid2d", tolerance
) -> Tuple[IntArray, IntArray, FloatArray]:
    x, y = other.coordinates.T
    source_index_x, target_index_x, weights_x = self.xbounds.linear_weights(x)
    source_index_y, target_index_y, weights_y = self.ybounds.linear_weights(y)
    return self.ravel_sorted(
        source_index_x,
        source_index_y,
        target_index_x,
        target_index_y,
        weights_x,
        weights_y,
    )


StructuredGrid2d.locate_inside = _2d_locate_inside
StructuredGrid2d.overlap = _2d_overlap
StructuredGrid2d.barycentric = _2d_barycentric


class StructuredGrid3d(Grid):
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
