"""This module is heavily inspired by xemsf.frontend.py"""

import abc
import warnings
from typing import Callable, Optional, Tuple, Union

import numba
import numpy as np
import pandas as pd
import xarray as xr

import xugrid

# dask as optional dependency
try:
    import dask.array

    DaskArray = dask.array.Array
except ImportError:
    DaskArray = ()

from xugrid.constants import FloatArray
from xugrid.core.sparse import MatrixCOO, MatrixCSR, row_slice
from xugrid.regrid.grid.structured import StructuredGrid2d
from xugrid.regrid.grid.unstructured import UnstructuredGrid2d


class BaseRegridder(abc.ABC):
    _JIT_FUNCTIONS = {}

    def __init__(
        self,
        source,
        target,
        target_dim: Optional[str] = None,
        tolerance: Optional[float] = None,
        **kwargs,
    ):
        self._source, _ = self.setup_grid(source, None)
        self._target, self._target_flipper = self.setup_grid(target, target_dim)
        self._weights = None
        self._compute_weights(self._source, self._target, tolerance, **kwargs)
        return

    @staticmethod
    def _monotonic_increasing_indexer(
        obj: Union[xr.DataArray, xr.Dataset], name_x: str, name_y: str
    ) -> dict[str, slice]:
        x = obj.indexes[name_x]
        y = obj.indexes[name_y]

        if not (x.is_monotonic_increasing or x.is_monotonic_decreasing):
            raise ValueError(f"x-coordinate {name_x} is not monotonic")
        if not (y.is_monotonic_increasing or y.is_monotonic_decreasing):
            raise ValueError(f"x-coordinate {name_y} is not monotonic")

        flipper = {"x": slice(None, None), "y": slice(None, None)}
        if x.is_monotonic_decreasing:
            flipper["x"] = slice(None, None, -1)
        if y.is_monotonic_decreasing:
            flipper["y"] = slice(None, None, -1)
        return flipper

    @classmethod
    def setup_grid(cls, obj, dim, **kwargs):
        flipper = None
        if isinstance(obj, xugrid.Ugrid2d):
            grid = obj
            if dim is None:
                warnings.warn(
                    "In the future, passing a Ugrid2d target requires an explicit target_dim.",
                    FutureWarning,
                    stacklevel=2,
                )
                dim = grid.face_dimension
            return UnstructuredGrid2d(grid, dim), flipper

        elif isinstance(obj, (xugrid.UgridDataArray, xugrid.UgridDataset)):
            # TODO: Make error more meaningful.
            #  Can only infer from a UgridDataset if:
            #
            # * it has a single topology
            # * it has a single UGRID dim in its data.
            #
            grid = obj.ugrid.grid
            if dim is None:
                candidates = set(obj.dims).intersection(grid.dims)
                if len(candidates) > 1:
                    # TODO:
                    raise ValueError(
                        f"Could not derive a single target dimension from multiple candidates: {candidates}"
                    )
                dim = candidates.pop()
            return UnstructuredGrid2d(grid, dim), flipper

        elif isinstance(obj, (xr.DataArray, xr.Dataset)):
            # Make sure x and y are increasing.
            name_x = kwargs.get("name_x", "x")
            name_y = kwargs.get("name_y", "y")
            flipper = cls._monotonic_increasing_indexer(obj, name_x, name_y)
            return StructuredGrid2d(
                obj.isel(flipper), name_y=name_y, name_x=name_x
            ), flipper

        else:
            raise TypeError()

    @staticmethod
    def make_regrid(func):
        """
        Use a closure to capture func, so numba can compile it efficiently without
        function call overhead.
        """
        f = numba.njit(func, inline="always")

        def _regrid(source: FloatArray, A: MatrixCSR, size: int):
            # Pre-allocate the output array
            n_extra = source.shape[0]
            out = np.full((n_extra, size), np.nan)
            # Pre-allocate workspace arrays. Every reduction algorithm should use
            # no more than the size of indices. Every thread gets it own workspace
            # row!
            n_work = np.diff(A.indptr).max()
            workspace = np.empty((n_extra, 2, n_work), dtype=np.float64)
            for extra_index in numba.prange(n_extra):
                source_flat = source[extra_index]
                for target_index in range(A.n):
                    slice = row_slice(A, target_index)
                    indices = A.indices[slice]
                    weights = A.data[slice]

                    # Copy the source data for this row to values.
                    n_value = len(indices)
                    values = workspace[extra_index, 0, :n_value]
                    for i, index in enumerate(indices):
                        values[i] = source_flat[index]

                    if len(indices) > 0:
                        out[extra_index, target_index] = f(
                            values, weights, workspace[extra_index, 1, :n_value]
                        )
            return out

        return numba.njit(_regrid, parallel=True, cache=True)

    @property
    @abc.abstractmethod
    def weights(self):
        pass

    @abc.abstractmethod
    def _compute_weights(self, source, target, tolerance: Optional[float] = None):
        pass

    def _setup_regrid(self, func) -> Callable:
        if isinstance(func, str):
            functions = self._JIT_FUNCTIONS
            try:
                self._regrid = functions[func]
            except KeyError as e:
                raise ValueError(
                    "Invalid regridding method. Available methods are: {}".format(
                        functions.keys()
                    )
                ) from e
        elif callable(func):
            self._regrid = self.make_regrid(func)
        else:
            raise TypeError(
                f"method must be string or callable, received: {type(func).__name}"
            )
        return

    def _regrid_array(self, source):
        source_grid = self._source
        first_dims_shape = source.shape[: -source_grid.ndim]

        # The regridding can be mapped over additional dimensions, e.g. for
        # every time slice. This is the `extra_index` iteration in _regrid().
        # But it should work consistently even if no additional present: in
        # that case we create a 1-sized additional dimension in front, so the
        # `extra_index` iteration always applies.
        if source.ndim == source_grid.ndim:
            source = source[np.newaxis]

        # All additional dimension are flattened into one, in front.
        # E.g.:
        #
        #   * ("dummy", "face") -> ("dummy", "face")
        #   * ("time", "layer", "y", "x") -> ("stacked_time_layer", "stacked_y_x")
        #   * ("time", "layer", "face") -> ("stacked_time_layer", "face")
        #
        # Source is always 2D after this step, sized: (n_extra, size).
        source = source.reshape((-1, source_grid.size))

        size = self._target.size

        if isinstance(source, DaskArray):
            # It's possible that the topology dimensions are chunked (e.g. from
            # reading multiple partitions). The regrid operation does not
            # support this, since we might need multiple source chunks for a
            # single target chunk, which destroys the 1:1 relation between
            # chunks. Here we ensure that the topology dimensions are contained
            # in a single contiguous chunk.
            contiguous_chunks = (source.chunks[0], (source.shape[-1],))
            source = source.rechunk(contiguous_chunks)
            chunks = source.chunks[:-1] + (self._target.size,)
            out = dask.array.map_blocks(
                self._regrid,  # func
                source,  # *args
                self._weights,  # *args
                size,  # *args
                dtype=np.float64,
                chunks=chunks,
                meta=np.array((), dtype=source.dtype),
            )
        elif isinstance(source, np.ndarray):
            out = self._regrid(source, self._weights, size)
        else:
            raise TypeError(
                "Expected dask.array.Array or numpy.ndarray. Received: "
                f"{type(source).__name__}"
            )
        # E.g.: sizes of ("time", "layer") + ("y", "x")
        out_shape = first_dims_shape + self._target.shape
        return out.reshape(out_shape)

    def regrid_dataarray(self, source: xr.DataArray, source_dims: Tuple[str]):
        # Do not set vectorize=True: numba will run the for loop more
        # efficiently, and guarantees a single large allocation.
        out = xr.apply_ufunc(
            self._regrid_array,
            source,
            input_core_dims=[source_dims],
            exclude_dims=set(source_dims),
            output_core_dims=[self._target.dims],
            dask="allowed",
            keep_attrs=True,
            output_dtypes=np.float64,  # [source.dtype],
        )
        return out

    def regrid(
        self, data: Union[xr.DataArray, "xugrid.UgridDataArray"]
    ) -> "xugrid.UgridDataArray":
        """
        Regrid the data from a DataArray from its old grid topology to the new
        target topology.

        Automatically regrids over additional dimensions (e.g. time).

        Supports lazy evaluation for dask arrays inside the DataArray.

        Parameters
        ----------
        data: UgridDataArray or xarray.DataArray

        Returns
        -------
        regridded: UgridDataArray or xarray.DataArray
        """

        # FIXME: this should work:
        # source_dims = self._source.dims
        #
        # But it causes problems with initializing a regridder
        # from_dataset, because the name has been changed to
        # __source_nFace.
        if isinstance(data, (xr.DataArray, xr.Dataset)):
            flipper = self._monotonic_increasing_indexer(data, name_x="x", name_y="y")
            obj = data.isel(flipper)
            source_dims = ("y", "x")
        elif isinstance(data, (xugrid.UgridDataArray, xugrid.UgridDataset)):
            obj = data.ugrid.obj
            source_dims = self._source.dims
        else:
            raise TypeError(
                f"Expected DataArray or UgridDataAray, received: {type(data).__name__}"
            )

        missing_dims = set(source_dims).difference(data.dims)
        if missing_dims:
            raise ValueError(
                f"data does not contain regridder source dimensions: {missing_dims}"
            )

        regridded = self.regrid_dataarray(obj, source_dims)

        if isinstance(self._target, StructuredGrid2d):
            regridded = regridded.assign_coords(coords=self._target.coords)
            return regridded.isel(self._target_flipper)
        else:
            return xugrid.UgridDataArray(
                regridded,
                self._target.ugrid_topology,
            )

    def to_dataset(self) -> xr.Dataset:
        """Store the computed weights and target in a dataset for re-use."""
        weights_ds = xr.Dataset(
            {f"__regrid_{k}": v for k, v in zip(self._weights._fields, self._weights)}
        )
        source_ds = self._source.to_dataset("__source")
        target_ds = self._target.to_dataset("__target")
        return xr.merge((weights_ds, source_ds, target_ds))

    def weights_as_dataframe(self) -> pd.DataFrame:
        """
        Return the weights as a three column dataframe:

        * source index
        * target index
        * weight

        Returns
        -------
        weights: pd.DataFrame
        """
        matrix = self._weights
        if matrix is None:
            raise ValueError("Weights have not been computed yet.")
        if isinstance(matrix, MatrixCSR):
            matrix = matrix.to_coo()
        return pd.DataFrame(
            {
                "target_index": matrix.row,
                "source_index": matrix.col,
                "weight": matrix.data,
            }
        )

    @staticmethod
    def _csr_from_dataset(dataset: xr.Dataset) -> MatrixCSR:
        """
        Create a compressed sparse row matrix from the dataset variables.

        Variables n and nnz are expected to be scalar variables.
        """
        return MatrixCSR(
            dataset["__regrid_data"].to_numpy(),
            dataset["__regrid_indices"].to_numpy(),
            dataset["__regrid_indptr"].to_numpy(),
            dataset["__regrid_n"].item(),
            dataset["__regrid_m"].item(),
            dataset["__regrid_nnz"].item(),
        )

    @staticmethod
    def _coo_from_dataset(dataset: xr.Dataset) -> MatrixCOO:
        """
        Create a coordinate/triplet sparse row matrix from the dataset variables.

        Variables n and nnz are expected to be scalar variables.
        """
        return MatrixCOO(
            dataset["__regrid_data"].to_numpy(),
            dataset["__regrid_row"].to_numpy(),
            dataset["__regrid_col"].to_numpy(),
            dataset["__regrid_n"].item(),
            dataset["__regrid_m"].item(),
            dataset["__regrid_nnz"].item(),
        )

    @abc.abstractmethod
    def _weights_from_dataset(cls, dataset: xr.Dataset) -> Union[MatrixCOO, MatrixCSR]:
        """Return either COO or CSR weights."""

    @classmethod
    def from_weights(
        cls, weights, target: Union["xugrid.Ugrid2d", xr.DataArray, xr.Dataset]
    ):
        instance = cls.__new__(cls)
        instance._weights = cls._weights_from_dataset(weights)
        instance._target = cls.setup_grid(target)
        unstructured = weights["__source_type"].attrs["type"] == "UnstructuredGrid2d"
        if unstructured:
            instance._source = cls.setup_grid(
                xugrid.Ugrid2d.from_dataset(weights, "__source")
            )
        else:
            instance._source = cls.setup_grid(
                weights, name_x="__source_x", name_y="__source_y"
            )
        return instance

    @classmethod
    def from_dataset(cls, dataset: xr.Dataset):
        """
        Reconstruct the regridder from a dataset with source, target indices
        and weights.
        """
        unstructured = dataset["__target_type"].attrs["type"] == "UnstructuredGrid2d"
        if unstructured:
            target = xugrid.Ugrid2d.from_dataset(dataset, "__target")

        # weights = cls._weights_from_dataset(dataset)
        return cls.from_weights(dataset, target)


class BasePointRegridder(BaseRegridder, abc.ABC):
    """
    Base class for regridders that search points (nodes, centroids) that provide
    a 1:1 mapping.
    """

    @staticmethod
    @numba.njit(parallel=True, cache=True)
    def _regrid(source: FloatArray, A: MatrixCOO, size: int):
        n_extra = source.shape[0]
        out = np.full((n_extra, size), np.nan)
        for extra_index in numba.prange(n_extra):
            source_flat = source[extra_index]
            for target_index, source_index in zip(A.row, A.col):
                out[extra_index, target_index] = source_flat[source_index]
        return out

    @property
    def weights(self):
        return self.to_dataset()

    @weights.setter
    def weights(self, weights: MatrixCOO, target: "xugrid.Ugrid2d"):
        if not isinstance(weights, MatrixCOO):
            raise TypeError(f"Expected MatrixCOO, received: {type(weights).__name__}")
        self._weights = weights
        return

    @classmethod
    def _weights_from_dataset(cls, dataset: xr.Dataset) -> MatrixCOO:
        return cls._coo_from_dataset(dataset)
