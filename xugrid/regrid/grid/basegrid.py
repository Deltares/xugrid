import abc
from typing import Optional, Tuple

import numpy as np

from xugrid.constants import FloatArray, IntArray


class Grid(abc.ABC):
    @property
    def facet(self) -> str:
        pass

    @property
    def coords(self) -> dict:
        pass

    @property
    @abc.abstractmethod
    def coordinates(self) -> FloatArray:
        pass

    @property
    @abc.abstractmethod
    def ndim(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def dims(self) -> Tuple[str]:
        pass

    @property
    @abc.abstractmethod
    def size(self) -> int:
        pass

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

    def locate_nearest(self, other, tolerance: Optional[float] = None):
        points = other.coordinates
        _, source_index = self.kdtree.query(points, workers=-1)
        inside = self.locate_points(points, tolerance=tolerance) != -1
        source_index = source_index[inside]
        target_index = np.arange(other.size, dtype=source_index.dtype)[inside]
        weight_values = np.ones_like(source_index, dtype=float)
        return source_index, target_index, weight_values

    def inverse_distance(
        self,
        other,
        tolerance,
        max_distance,
        min_points,
        max_points,
        power,
        smoothing,
    ):
        points = other.coordinates
        # Do not interpolate points that are beyond the boundaries
        # of the faces.
        inside = self.locate_points(points, tolerance) != -1
        # Find k nearest points.
        distance, indices = self.kdtree.query(
            x=points,
            k=max_points,
            distance_upper_bound=max_distance,
            workers=-1,
        )

        # Eliminate entries that have insufficient neighbors.
        n_found = np.isfinite(distance).sum(axis=1)
        distance[n_found < min_points] = np.inf

        # Exact match (distance of zero), avoid the singularity.
        if smoothing == 0:
            exact_match = distance < 1e-12
            # Eliminate the entire row
            distance[exact_match.any(axis=1)] = np.inf
            # Then restore the exact match.
            distance[exact_match] = 1.0

        keep = (inside[:, None] & np.isfinite(distance)).ravel()
        # Generate target index and weights.
        target_index = np.repeat(np.arange(len(points)), max_points)[keep]
        source_index = indices.ravel()[keep]
        weights = 1.0 / (distance.ravel()[keep] + smoothing) ** power
        return self.sorted_output(source_index, target_index, weights)
