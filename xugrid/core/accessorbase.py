import abc
from typing import Tuple, Union

import numpy as np
import xarray as xr

import xugrid


class AbstractUgridAccessor(abc.ABC):
    @abc.abstractmethod
    def to_dataset():
        """ """

    @abc.abstractmethod
    def assign_node_coords():
        """ """

    @abc.abstractmethod
    def set_node_coords():
        """ """

    @abc.abstractproperty
    def crs():
        """ """

    @abc.abstractmethod
    def set_crs():
        """ """

    @abc.abstractmethod
    def to_crs():
        """ """

    @abc.abstractmethod
    def sel_points():
        """ """

    @abc.abstractmethod
    def intersect_line():
        """ """

    @abc.abstractmethod
    def intersect_linestring():
        """ """

    @abc.abstractproperty
    def bounds():
        """ """

    @abc.abstractproperty
    def total_bounds():
        """ """

    @abc.abstractproperty
    def name():
        """ """

    @abc.abstractproperty
    def names():
        """ """

    @abc.abstractproperty
    def topology():
        """ """

    @staticmethod
    def _sel(obj, grid, x, y):
        from xugrid.core.wrap import maybe_xugrid

        # TODO: also do vectorized indexing like xarray?
        # Might not be worth it, as orthogonal and vectorized indexing are
        # quite confusing.
        result = grid.sel(obj, x, y)
        if isinstance(result, tuple):
            return maybe_xugrid(*result)
        else:
            return result

    @staticmethod
    def _raster_xy(bounds: Tuple[float, float, float, float], resolution: float):
        xmin, ymin, xmax, ymax = bounds
        d = abs(resolution)
        xmin = np.floor(xmin / d) * d
        xmax = np.ceil(xmax / d) * d
        ymin = np.floor(ymin / d) * d
        ymax = np.ceil(ymax / d) * d
        x = np.arange(xmin + 0.5 * d, xmax, d)
        y = np.arange(ymax - 0.5 * d, ymin, -d)
        return x, y

    def _raster(self, x, y, index) -> xr.DataArray:
        index = index.ravel()
        indexer = xr.DataArray(
            data=index.reshape(y.size, x.size),
            coords={"y": y, "x": x},
            dims=["y", "x"],
        )
        out = self.obj.isel({self.grid.face_dimension: indexer}).where(indexer != -1)
        return out

    def clip_box(
        self,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
    ):
        """
        Clip the DataArray or Dataset by a bounding box.

        Parameters
        ----------
        xmin: float
        ymin: float
        xmax: float
        ymax: float

        -------
        clipped:
            xugrid.UgridDataArray or xugrid.UgridDataset
        """
        return self.sel(x=slice(xmin, xmax), y=slice(ymin, ymax))

    def partition_by_label(
        self, labels: np.ndarray
    ) -> Union["xugrid.UgridDataArray", "xugrid.UgridDataset"]:
        """
        Partition a grid by labels.

        Parameters
        ----------
        labels: np.ndarray of integers labeling each face.

        Returns
        -------
        partitioned: list of partitions
        """
        from xugrid.ugrid import partitioning

        return partitioning.partition_by_label(self.grid, self.obj, labels)

    def partition(
        self, n_part: int
    ) -> Union["xugrid.UgridDataArray", "xugrid.UgridDataset"]:
        """
        Partition a grid into a given number of parts.

        Parameters
        ----------
        n_part: integer
            The number of parts to partition the mesh.

        Returns
        -------
        partitioned: list of partitions
        """
        labels = self.grid.label_partitions(n_part)
        return self.partition_by_label(labels)

    def to_netcdf(self, *args, **kwargs):
        """
        Write dataset contents to a UGRID compliant netCDF file.

        This function wraps :py:meth:`xr.Dataset.to_netcdf`; it adds the UGRID
        variables and coordinates to a standard xarray Dataset, then writes the
        result to a netCDF.

        All arguments are forwarded to :py:meth:`xr.Dataset.to_netcdf`.
        """
        self.to_dataset().to_netcdf(*args, **kwargs)

    def to_zarr(self, *args, **kwargs):
        """
        Write dataset contents to a UGRID compliant Zarr file.

        This function wraps :py:meth:`xr.Dataset.to_zarr`; it adds the UGRID
        variables and coordinates to a standard xarray Dataset, then writes the
        result to a Zarr file.

        All arguments are forwarded to :py:meth:`xr.Dataset.to_zarr`.
        """
        self.to_dataset().to_zarr(*args, **kwargs)
