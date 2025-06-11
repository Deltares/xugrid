from typing import Optional

import xarray as xr

import xugrid
from xugrid.core.sparse import MatrixCOO, MatrixCSR
from xugrid.regrid.base_regridder import BaseRegridder
from xugrid.regrid.utils import reduce


class BarycentricInterpolator(BaseRegridder):
    """
    The BaryCentricInterpolator searches the centroid of every face of the
    target grid in the source grid. It finds by which source faces the centroid
    is surrounded (via its centroidal voronoi tesselation), and computes
    barycentric weights which can be used for to interpolate smoothly between
    the values associated with the source faces.

    Parameters
    ----------
    source: Ugrid2d, UgridDataArray
        Source grid to regrid from.
    target: Ugrid2d, UgridDataArray
        Target grid to regrid to.
    tolerance: float, optional
        The tolerance used to determine whether a point is on an edge. This
        accounts for the inherent inexactness of floating point calculations.
        If None, an appropriate tolerance is automatically estimated based on
        the geometry size. Consider adjusting this value if edge detection
        results are unsatisfactory.
    """

    _JIT_FUNCTIONS = {"mean": BaseRegridder.make_regrid(reduce.mean)}

    def __init__(
        self,
        source,
        target,
        target_dim: Optional[str] = None,
        tolerance: Optional[float] = None,
    ):
        super().__init__(source, target, target_dim, tolerance)
        # Since the weights for a target face sum up to 1.0, a weight mean is
        # appropriate, and takes care of NaN values in the source data.
        self._setup_regrid("mean")

    def _compute_weights(
        self,
        source,
        target,
        tolerance: Optional[float] = None,
    ):
        source_index, target_index, weights = source.barycentric(target, tolerance)
        self._weights = MatrixCSR.from_triplet(
            target_index, source_index, weights, n=target.size, m=source.size
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
    def from_weights(cls, weights: MatrixCSR, target: Optional["xugrid.Ugrid2d"]):
        instance = super().from_weights(weights, target)
        instance._setup_regrid("mean")
        return instance

    @classmethod
    def _weights_from_dataset(cls, dataset: xr.Dataset) -> MatrixCOO:
        return cls._csr_from_dataset(dataset)
