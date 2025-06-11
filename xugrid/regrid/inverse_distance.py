from typing import Optional

import xugrid
from xugrid.core.sparse import MatrixCSR
from xugrid.regrid.base_regridder import BaseRegridder


class InverseDistanceRegridder(BaseRegridder):
    def __init__(
        self,
        source: xugrid.UgridDataArray,
        target: xugrid.UgridDataArray,
        target_dim: Optional[str] = None,
        max_distance: Optional[float] = None,
        min_points: int = 0,
        max_points: int = 12,
        power: float = 2.0,
        smoothing: float = 0.0,
        tolerance: Optional[float] = None,
    ):
        super().__init__(source, target, target_dim, tolerance)

    def _compute_weights(
        self,
        source,
        target,
        tolerance: Optional[float] = None,
    ):
        source_index, target_index, weights = source.inverse_distance(target, tolerance)
        self._weights = MatrixCSR.from_triplet(
            target_index, source_index, weights, n=target.size, m=source.size
        )
        return
