import abc
from typing import Tuple

import numpy as np

from xugrid.constants import FloatArray, IntArray


class Grid(abc.ABC):
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
