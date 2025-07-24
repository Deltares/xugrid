from typing import Optional

from xugrid.core.sparse import MatrixCOO
from xugrid.regrid.base_regridder import BasePointRegridder
from xugrid.regrid.grid.unstructured import UnstructuredGrid2d


class LocatorRegridder(BasePointRegridder):
    """
    The CentroidLocatorRegridded regrids by searching the source grid for the
    centroids of the target grid.

    If a centroid is exactly located on an edge between two faces, the value of
    either face may be used.

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
        if isinstance(source, UnstructuredGrid2d):
            if source.facet == "node":
                raise ValueError(
                    "Cannot regrid node-associated data with the LocatorRegridder. "
                    "This regridder locates points within grid faces, which "
                    "is incompatible with node data. Try using a different "
                    "regridder or convert your data to face-associated values."
                )

        source_index, target_index, weight_values = source.locate_inside(
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
