import abc
from typing import Callable, Optional, Union

import xarray as xr

import xugrid
from xugrid.core.sparse import MatrixCOO, MatrixCSR
from xugrid.regrid import reduce
from xugrid.regrid.base_regridder import BaseRegridder


class BaseOverlapRegridder(BaseRegridder, abc.ABC):
    def _compute_weights(self, source, target, relative: bool) -> None:
        source_index, target_index, weight_values = source.overlap(
            target, relative=relative
        )
        self._weights = MatrixCSR.from_triplet(
            target_index, source_index, weight_values, n=target.size, m=source.size
        )
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
    def _weights_from_dataset(cls, dataset: xr.Dataset) -> MatrixCOO:
        return cls._csr_from_dataset(dataset)


class OverlapRegridder(BaseOverlapRegridder):
    """
    The OverlapRegridder regrids by computing which target faces overlap with
    which source faces. It stores the area of overlap, which can be used in
    multiple ways to aggregate the values associated with the source faces.

    Currently supported aggregation methods are:

    * ``"mean"``
    * ``"harmonic_mean"``
    * ``"geometric_mean"``
    * ``"sum"``
    * ``"minimum"``
    * ``"maximum"``
    * ``"mode"``
    * ``"median"``
    * ``"max_overlap"``
    * percentiles 5, 10, 25, 50, 75, 90, 95: as ``"p5"``, ``"p10"``, etc.

    Custom aggregation functions are also supported, if they can be compiled by
    Numba. See the User Guide.

    Any percentile method can be created via:
    ``method = OverlapRegridder.create_percentile_methode(percentile)``
    See the examples.

    Parameters
    ----------
    source: Ugrid2d, UgridDataArray
    target: Ugrid2d, UgridDataArray
    method: str, function, optional
        Default value is ``"mean"``.

    Examples
    --------
    Create an OverlapRegridder to regrid with mean:

    >>> regridder = OverlapRegridder(source_grid, target_grid, method="mean")
    >>> regridder.regrid(source_data)

    Setup a custom percentile method and apply it:

    >>> p33_3 = OverlapRegridder.create_percentile_method(33.3)
    >>> regridder = OverlapRegridder(source_grid, target_grid, method=p33_3)
    >>> regridder.regrid(source_data)
    """

    _JIT_FUNCTIONS = {
        k: BaseOverlapRegridder.make_regrid(f)
        for k, f in reduce.ABSOLUTE_OVERLAP_METHODS.items()
    }

    def __init__(
        self,
        source: xugrid.UgridDataArray,
        target: xugrid.UgridDataArray,
        target_dim: Optional[str] = None,
        method: Union[str, Callable] = "mean",
    ):
        super().__init__(source=source, target=target, target_dim=target_dim)
        self._setup_regrid(method)

    def _compute_weights(
        self, source, target, tolerance: Optional[float] = None
    ) -> None:
        super()._compute_weights(source, target, relative=False)

    @staticmethod
    def create_percentile_method(percentile: float) -> Callable:
        return reduce.create_percentile_method(percentile)

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


class RelativeOverlapRegridder(BaseOverlapRegridder):
    """
    The RelativeOverlapRegridder regrids by computing which target faces
    overlap with which source faces. It stores the area of overlap, which can
    be used in multiple ways to aggregate the values associated with the source
    faces. Unlike the OverlapRegridder, the intersection area is divided by the
    total area of the source face. This is required for e.g. first-order
    conserative regridding.

    Currently supported aggregation methods are:

    * ``"max_overlap"``

    Custom aggregation functions are also supported, if they can be compiled by
    Numba. See the User Guide.

    Parameters
    ----------
    source: Ugrid2d, UgridDataArray
    target: Ugrid2d, UgridDataArray
    method: str, function, optional
        Default value is "first_order_conservative".
    """

    _JIT_FUNCTIONS = {
        k: BaseOverlapRegridder.make_regrid(f)
        for k, f in reduce.RELATIVE_OVERLAP_METHODS.items()
    }

    def __init__(
        self,
        source: xugrid.UgridDataArray,
        target: xugrid.UgridDataArray,
        target_dim: Optional[str] = None,
        method: Union[str, Callable] = "first_order_conservative",
    ):
        super().__init__(
            source=source, target=target, target_dim=target_dim, tolerance=None
        )
        self._setup_regrid(method)

    def _compute_weights(
        self, source, target, tolerance: Optional[float] = None
    ) -> None:
        super()._compute_weights(source, target, relative=True)

    @classmethod
    def from_weights(
        cls,
        weights: MatrixCSR,
        target: "xugrid.Ugrid2d",
        method: Union[str, Callable] = "first_order_conservative",
    ):
        instance = super().from_weights(weights, target)
        instance._setup_regrid(method)
        return instance
