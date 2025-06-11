from typing import Callable, Optional, Union

import xarray as xr

import xugrid
from xugrid.core.sparse import MatrixCSR
from xugrid.regrid.base_regridder import BaseRegridder
from xugrid.regrid.grid.network import Network1d
from xugrid.regrid.grid.structured import StructuredGrid2d
from xugrid.regrid.grid.unstructured import UnstructuredGrid2d
from xugrid.regrid.utils import reduce


def convert_to_match(source, target):
    PROMOTIONS = {
        frozenset({StructuredGrid2d}): UnstructuredGrid2d,
        frozenset({UnstructuredGrid2d}): UnstructuredGrid2d,
    }
    types = set({type(target)})
    matched_type = PROMOTIONS[frozenset(types)]
    return source, target.convert_to(matched_type)


class NetworkGridder(BaseRegridder):
    """
    Network gridder for 2D unstructured grids.

    Parameters
    ----------
    grid: Ugrid1d
        The grid to be used for the regridding.
    """

    _JIT_FUNCTIONS = {
        k: BaseRegridder.make_regrid(f)
        for k, f in reduce.ABSOLUTE_OVERLAP_METHODS.items()
    }

    def __init__(
        self,
        source: "xugrid.Ugrid1d",
        target: "xugrid.Ugrid2d",
        target_dim: Optional[str] = None,
        method: Union[str, Callable] = "mean",
    ):
        self._source = Network1d(source)
        self._target = self.setup_grid(target)
        self._target_dim = target_dim
        self._weights = None
        self._compute_weights(self._source, self._target, relative=False)
        self._setup_regrid(method)
        return

    @property
    def weights(self):
        return self.to_dataset()

    @weights.setter
    def weights(self, weights: MatrixCSR):
        if not isinstance(weights, MatrixCSR):
            raise TypeError(f"Expected MatrixCSR, received: {type(weights).__name__}")
        self._weights = weights
        return

    @classmethod
    def _weights_from_dataset(cls, dataset: xr.Dataset) -> MatrixCSR:
        return cls._csr_from_dataset(dataset)

    def _compute_weights(self, source, target, relative: bool) -> None:
        source, target = convert_to_match(source, target)
        source_index, target_index, weight_values = target.intersection_length(
            source, relative=relative
        )
        self._weights = MatrixCSR.from_triplet(
            target_index, source_index, weight_values, n=target.size, m=source.size
        )
        return

    @classmethod
    def from_weights(
        cls,
        weights: xr.Dataset,
        target: Union["xugrid.Ugrid2d", xr.DataArray, xr.Dataset],
        method: Union[str, Callable] = "mean",
    ):
        instance = super().from_weights(weights, target)
        instance._setup_regrid(method)
        return instance
