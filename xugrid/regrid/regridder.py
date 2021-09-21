from typing import Union

import imod
import numpy as np
import pandas as pd
import pyproj
import xarray as xr
from numba_celltree import CellTree2d

from ..ugrid_dataset import UgridDataArray, UgridDataset


def structured_xy_weights():
    pass


def unstructured_xy_weights():
    pass


def area_weighted_mean(
    da: xr.DataArray,
    destination_index: np.ndarray,
    source_index: np.ndarray,
    weights: np.ndarray,
):
    """
    Area weighted mean.

    Parameters
    ----------
    da: xr.DataArray
        Contains source data.
    destination_index: np.ndarray
        In which destination the overlap is located.
    source_index: np.ndarray
        In which source cell the overlap is located.
    weights: np.ndarray
        Area of each overlap.

    Returns
    -------
    destination_index: np.ndarray
    values: np.ndarray
    """
    values = da.data.ravel()[source_index]
    df = pd.DataFrame(
        {"dst": destination_index, "area": weights, "av": weights * values}
    )
    aggregated = df.groupby("dst").sum("sum", min_count=1)
    out = aggregated["av"] / aggregated["area"]
    return out.index.values, out.values


class Regridder:
    """
    Regridder to reproject and/or regrid structured and unstructured grids, up
    to three dimensions..  When no ``crs_source`` and ``crs_destination`` are
    provided, it is assumed that ``source`` and ``destination`` share the same
    coordinate system.

    Note that an area weighted regridding method only makes sense for projected
    (Cartesian!) coordinate systems.

    Parameters
    ----------
    source: xr.DataArray
        Source topology.
    destination: xr.DataArray
        Destination topology.
    crs_source: optional, default: None
    crs_destination: optional, default: None
    """

    def __init__(
        self,
        source: Union[xr.Dataset, xr.DataArray, UgridDataset, UgridDataArray],
        destination: Union[xr.Dataset, xr.DataArray, UgridDataset, UgridDataArray],
        crs_source=None,
        crs_destination=None,
    ):
        src = imod.util.ugrid2d_topology(source)
        dst = imod.util.ugrid2d_topology(destination)
        src_yy = src["node_y"].values
        src_xx = src["node_x"].values
        if crs_source and crs_destination:
            transformer = pyproj.Transformer.from_crs(
                crs_from=crs_source, crs_to=crs_destination, always_xy=True
            )
            src_xx, src_yy = transformer.transform(xx=src_xx, yy=src_yy)
        elif crs_source ^ crs_destination:
            raise ValueError("Received only one of (crs_source, crs_destination)")

        src_vertices = np.column_stack([src_xx, src_yy])
        src_faces = src["face_nodes"].values.astype(int)
        dst_vertices = np.column_stack((dst["node_x"].values, dst["node_y"].values))
        dst_faces = dst["face_nodes"].values
        celltree = CellTree2d(src_vertices, src_faces, fill_value=-1)

        self.source = source.copy()
        self.destination = destination.copy()
        (
            self.destination_index,
            self.source_index,
            self.weights,
        ) = celltree.intersect_faces(
            dst_vertices,
            dst_faces,
            fill_value=-1,
        )

    def regrid(self, da: xr.DataArray, fill_value=np.nan):
        """
        Parameters
        ----------
        da: xr.DataArray
            Data to regrid.
        fill_value: optional, default: np.nan
            Default value of the output grid, e.g. where no overlap occurs.

        Returns
        -------
        regridded: xr.DataArray
            Data of da, regridded using an area weighted mean.
        """
        src = self.source
        if not (np.allclose(da["y"], src["y"]) and np.allclose(da["x"], src["x"])):
            raise ValueError("da does not match source")
        index, values = area_weighted_mean(
            da,
            self.destination_index,
            self.source_index,
            self.weights,
        )
        data = np.full(self.destination.shape, fill_value)
        data.ravel()[index] = values
        out = self.destination.copy(data=data)
        out.name = da.name
        return out
