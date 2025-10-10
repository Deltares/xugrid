from typing import Optional

from xugrid.core.sparse import MatrixCOO
from xugrid.regrid.base_regridder import BasePointRegridder


class NearestRegridder(BasePointRegridder):
    """
    The NearestRegridder regrids by searching the source grid for the
    nearest centroid or node of the target grid.

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

    def _compute_weights(self, source, target, tolerance: Optional[float] = None):
        source_index, target_index, weight_values = source.locate_nearest(
            target, tolerance
        )
        self._weights = MatrixCOO.from_triplet(
            target_index,
            source_index,
            weight_values,
            n=target.size,
            m=source.size,
        )
        return
