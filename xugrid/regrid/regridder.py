"""This module is heavily inspired by xemsf.frontend.py"""

import abc
from typing import Callable, Optional, Tuple, Union

import numba
import numpy as np
import pandas as pd
import xarray as xr

import xugrid as xu

# dask as optional dependency
try:
    import dask.array

    DaskArray = dask.array.Array
except ImportError:
    DaskArray = ()

import xugrid
from xugrid.constants import FloatArray
from xugrid.core.sparse import MatrixCOO, MatrixCSR, row_slice
from xugrid.core.wrap import UgridDataArray
from xugrid.regrid import reduce
from xugrid.regrid.structured import StructuredGrid2d
from xugrid.regrid.unstructured import UnstructuredGrid2d


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


def setup_grid(obj, **kwargs):
    if isinstance(obj, (xu.Ugrid2d, xu.UgridDataArray, xu.UgridDataset)):
        return UnstructuredGrid2d(obj)
    elif isinstance(obj, (xr.DataArray, xr.Dataset)):
        return StructuredGrid2d(
            obj, name_y=kwargs.get("name_y", "y"), name_x=kwargs.get("name_x", "x")
        )
    else:
        raise TypeError()


def convert_to_match(source, target):
    PROMOTIONS = {
        frozenset({StructuredGrid2d}): StructuredGrid2d,
        frozenset({StructuredGrid2d, UnstructuredGrid2d}): UnstructuredGrid2d,
        frozenset({UnstructuredGrid2d, UnstructuredGrid2d}): UnstructuredGrid2d,
        #    {StructuredGrid3d, ExplicitStructuredGrid3d}: ExplicitStructuredGrid3d,
        #    {LayeredUnstructuredGrid2d, StructuredGrid2d}: StructuredGrid2d,
        #    {LayeredUnstructuredGrid2d, StructuredGrid2d}: StructuredGrid2d,
        #    {StructuredGrid3d, StructuredGrid2d}: StructuredGrid2d,
        #    # etc.
    }
    types = set({type(source), type(target)})
    matched_type = PROMOTIONS[frozenset(types)]
    return source.convert_to(matched_type), target.convert_to(matched_type)


class BaseRegridder(abc.ABC):
    _JIT_FUNCTIONS = {}

    def __init__(
        self,
        source: "xugrid.Ugrid2d",
        target: "xugrid.Ugrid2d",
        tolerance: Optional[float] = None,
    ):
        self._source = setup_grid(source)
        self._target = setup_grid(target)
        self._weights = None

        self._compute_weights(self._source, self._target, tolerance)
        return

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
            self._regrid = make_regrid(func)
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
            output_dtypes=[source.dtype],
        )
        return out

    def regrid(self, data: Union[xr.DataArray, UgridDataArray]) -> UgridDataArray:
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
        if isinstance(data, xr.DataArray):
            obj = data
            source_dims = ("y", "x")
        elif isinstance(data, UgridDataArray):
            obj = data.ugrid.obj
            source_dims = (data.ugrid.grid.core_dimension,)
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
            return regridded
        else:
            return UgridDataArray(
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
        instance._target = setup_grid(target)
        unstructured = weights["__source_type"].attrs["type"] == "UnstructuredGrid2d"
        if unstructured:
            instance._source = setup_grid(xu.Ugrid2d.from_dataset(weights, "__source"))
        else:
            instance._source = setup_grid(
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
            target = xu.Ugrid2d.from_dataset(dataset, "__target")

        # weights = cls._weights_from_dataset(dataset)
        return cls.from_weights(dataset, target)


class CentroidLocatorRegridder(BaseRegridder):
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
        source, target = convert_to_match(source, target)
        source_index, target_index, weight_values = source.locate_centroids(
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


class BaseOverlapRegridder(BaseRegridder, abc.ABC):
    def _compute_weights(self, source, target, relative: bool) -> None:
        source, target = convert_to_match(source, target)
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
        k: make_regrid(f) for k, f in reduce.ABSOLUTE_OVERLAP_METHODS.items()
    }

    def __init__(
        self,
        source: UgridDataArray,
        target: UgridDataArray,
        method: Union[str, Callable] = "mean",
    ):
        super().__init__(source=source, target=target)
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
        k: make_regrid(f) for k, f in reduce.RELATIVE_OVERLAP_METHODS.items()
    }

    def __init__(
        self,
        source: UgridDataArray,
        target: UgridDataArray,
        method: Union[str, Callable] = "first_order_conservative",
    ):
        super().__init__(source=source, target=target, tolerance=None)
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

    _JIT_FUNCTIONS = {"mean": make_regrid(reduce.mean)}

    def __init__(
        self,
        source: UgridDataArray,
        target: UgridDataArray,
        tolerance: Optional[float] = None,
    ):
        super().__init__(source, target, tolerance)
        # Since the weights for a target face sum up to 1.0, a weight mean is
        # appropriate, and takes care of NaN values in the source data.
        self._setup_regrid("mean")

    def _compute_weights(
        self,
        source,
        target,
        tolerance: Optional[float] = None,
    ):
        source, target = convert_to_match(source, target)
        if isinstance(source, StructuredGrid2d):
            source_index, target_index, weights = source.linear_weights(target)
        else:
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
