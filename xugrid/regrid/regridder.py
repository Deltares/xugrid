"""
This module is heavily inspired by xemsf.frontend.py
"""
import abc
from typing import Callable, Optional, Tuple, Union

import numba
import numpy as np
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
from xugrid.core.wrap import UgridDataArray
from xugrid.regrid import reduce
from xugrid.regrid.structured import StructuredGrid2d
from xugrid.regrid.unstructured import UnstructuredGrid2d
from xugrid.regrid.weight_matrix import (
    WeightMatrixCOO,
    WeightMatrixCSR,
    nzrange,
    weight_matrix_coo,
    weight_matrix_csr,
)


def make_regrid(func):
    """
    Uses a closure to capture func, so numba can compile it efficiently without
    function call overhead.
    """
    f = numba.njit(func, inline="always")

    def _regrid(source: FloatArray, A: WeightMatrixCSR, size: int):
        n_extra = source.shape[0]
        out = np.full((n_extra, size), np.nan)
        for extra_index in numba.prange(n_extra):
            source_flat = source[extra_index]
            for target_index in range(A.n):
                indices, weights = nzrange(A, target_index)
                if len(indices) > 0:
                    out[extra_index, target_index] = f(source_flat, indices, weights)
        return out

    return numba.njit(_regrid, parallel=True, cache=True)


def setup_grid(obj):
    if isinstance(obj, (xu.Ugrid2d, xu.UgridDataArray, xu.UgridDataset)):
        return UnstructuredGrid2d(obj)
    elif isinstance(obj, (xr.DataArray, xr.Dataset)):
        return StructuredGrid2d(obj, name_y="y", name_x="x")
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
    ):
        self._source = setup_grid(source)
        self._target = setup_grid(target)
        self._compute_weights(self._source, self._target)
        return

    @abc.abstractproperty
    def weights(self):
        """ """

    @abc.abstractmethod
    def _compute_weights(self, source, target):
        """ """

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
        if hasattr(self, "_source"):
            source_grid = self._source
        else:
            source_grid = source
        first_dims_shape = source.shape[: -source_grid.ndim]

        # The regridding can be mapped over additional dimensions (e.g. for every time slice).
        # This is the `extra_index` iteration in _regrid().
        # But it should work consistently even if no additional present: in that case we create
        # a 1-sized additional dimension in front, so the `extra_index` iteration always applies.
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
            chunks = source.chunks[: -source_grid.ndim] + (self._target.shape,)
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

    def regrid(self, object) -> UgridDataArray:
        """
        Regrid the data from a DataArray from its old grid topology to the new
        target topology.

        Automatically regrids over additional dimensions (e.g. time).

        Supports lazy evaluation for dask arrays inside the DataArray.

        Parameters
        ----------
        object: UgridDataArray or xarray.DataArray

        Returns
        -------
        regridded: UgridDataArray or xarray.DataArray
        """

        if type(self._target) is StructuredGrid2d:
            source_dims = ("y", "x")
            regridded = self.regrid_dataarray(object, source_dims)
            regridded = regridded.assign_coords(
                coords={
                    "y": np.flip(self._target.ybounds.midpoints),
                    "x": self._target.xbounds.midpoints,
                }
            )
            return regridded
        else:
            source_dims = (object.ugrid.grid.face_dimension,)
            regridded = self.regrid_dataarray(object.ugrid.obj, source_dims)
            return UgridDataArray(
                regridded,
                self._target.ugrid_topology,
            )

    def to_dataset(self) -> xr.Dataset:
        """
        Store the computed weights and target in a dataset for re-use.
        """
        weights_ds = xr.Dataset(
            {f"__regrid_{k}": v for k, v in zip(self._weights._fields, self._weights)}
        )
        source_ds = self._source.to_dataset("source")
        target_ds = self._target.to_dataset("target")
        return xr.merge((weights_ds, source_ds, target_ds))

    @staticmethod
    def _csr_from_dataset(dataset: xr.Dataset) -> WeightMatrixCSR:
        return WeightMatrixCSR(
            dataset["__regrid_data"].values,
            dataset["__regrid_indices"].values,
            dataset["__regrid_indptr"].values,
            dataset["__regrid_n"].values,
            dataset["__regrid_nnz"].values,
        )

    @staticmethod
    def _coo_from_dataset(dataset: xr.Dataset) -> WeightMatrixCOO:
        return WeightMatrixCOO(
            dataset["__regrid_data"].values,
            dataset["__regrid_row"].values,
            dataset["__regrid_col"].values,
            dataset["__regrid_nnz"].values,
        )

    @abc.abstractclassmethod
    def _weights_from_dataset(
        cls, dataset: xr.Dataset
    ) -> Union[WeightMatrixCOO, WeightMatrixCSR]:
        """
        Return either COO or CSR weights.
        """

    @classmethod
    def from_weights(cls, weights, target: "xugrid.Ugrid2d"):
        instance = cls.__new__(cls)
        instance._weights = weights
        instance._target = UnstructuredGrid2d(target)
        return instance

    @classmethod
    def from_dataset(cls, dataset: xr.Dataset):
        """
        Reconstruct the regridder from a dataset with source, target indices
        and weights.
        """
        target = xu.Ugrid2d.from_dataset(dataset)
        weights = cls._weights_from_dataset(dataset)
        return cls.from_weights(weights, target)


class CentroidLocatorRegridder(BaseRegridder):
    """
    The CentroidLocatorRegridded regrids by searching the source grid for the
    centroids of the target grid.

    If a centroid is exactly located on an edge between two faces, the value of
    either face may be used.

    Parameters
    ----------
    source: Ugrid2d, UgridDataArray
    target: Ugrid2d, UgridDataArray
    weights: Optional[WeightMatrixCOO]
    """

    def _compute_weights(self, source, target):
        source, target = convert_to_match(source, target)
        source_index, target_index, weight_values = source.locate_centroids(target)
        self._weights = weight_matrix_coo(source_index, target_index, weight_values)
        return

    @staticmethod
    @numba.njit(parallel=True, cache=True)
    def _regrid(source: FloatArray, A: WeightMatrixCOO, size: int):
        n_extra = source.shape[0]
        out = np.full((n_extra, size), np.nan)
        for extra_index in numba.prange(n_extra):
            source_flat = source[extra_index]
            for target_index, source_index in zip(A.row, A.col):
                out[extra_index, target_index] = source_flat[source_index]
        return out

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights: WeightMatrixCOO, target: "xugrid.Ugrid2d"):
        if not isinstance(weights, WeightMatrixCOO):
            raise TypeError(
                f"Expected WeightMatrixCOO, received: {type(weights).__name__}"
            )
        self._weights = weights
        return

    @classmethod
    def _weights_from_dataset(cls, dataset: xr.Dataset) -> WeightMatrixCOO:
        return cls._coo_from_dataset(dataset)


class BaseOverlapRegridder(BaseRegridder, abc.ABC):
    def _compute_weights(self, source, target, relative: bool) -> None:
        source, target = convert_to_match(source, target)
        source_index, target_index, weight_values = source.overlap(
            target, relative=relative
        )
        self._weights = weight_matrix_csr(source_index, target_index, weight_values)
        return

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights: WeightMatrixCSR):
        if not isinstance(weights, WeightMatrixCSR):
            raise TypeError(
                f"Expected WeightMatrixCSR, received: {type(weights).__name__}"
            )
        self._weights = weights
        return

    @classmethod
    def _weights_from_dataset(cls, dataset: xr.Dataset) -> WeightMatrixCOO:
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

    Custom aggregation functions are also supported, if they can be compiled by
    Numba. See the User Guide.

    Parameters
    ----------
    source: Ugrid2d, UgridDataArray
    target: Ugrid2d, UgridDataArray
    method: str, function, optional
        Default value is ``"mean"``.
    """

    _JIT_FUNCTIONS = {
        k: make_regrid(f) for k, f in reduce.ASBOLUTE_OVERLAP_METHODS.items()
    }

    def __init__(
        self,
        source: UgridDataArray,
        target: UgridDataArray,
        method: Union[str, Callable] = "mean",
    ):
        super().__init__(source=source, target=target)
        self._setup_regrid(method)

    def _compute_weights(self, source, target) -> None:
        super()._compute_weights(source, target, relative=False)

    @classmethod
    def from_weights(
        cls,
        weights: WeightMatrixCSR,
        target: "xugrid.Ugrid2d",
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
        super().__init__(source=source, target=target)
        self._setup_regrid(method)

    def _compute_weights(self, source, target) -> None:
        super()._compute_weights(source, target, relative=True)

    @classmethod
    def from_weights(
        cls,
        weights: WeightMatrixCSR,
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
    target: Ugrid2d, UgridDataArray

    """

    _JIT_FUNCTIONS = {"mean": make_regrid(reduce.mean)}

    def __init__(
        self,
        source: UgridDataArray,
        target: UgridDataArray,
    ):
        super().__init__(source, target)
        # Since the weights for a target face sum up to 1.0, a weight mean is
        # appropriate, and takes care of NaN values in the source data.
        self._setup_regrid("mean")

    def _compute_weights(self, source, target):
        source, target = convert_to_match(source, target)
        if type(source) == StructuredGrid2d:
            source_index, target_index, weights = source.linear_weights(target)
        else:
            source_index, target_index, weights = source.barycentric(target)
        self._weights = weight_matrix_csr(source_index, target_index, weights)
        return

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights: WeightMatrixCSR):
        if not isinstance(weights, WeightMatrixCSR):
            raise TypeError(
                f"Expected WeightMatrixCSR, received: {type(weights).__name__}"
            )
        self._weights = weights
        return

    @classmethod
    def from_weights(cls, weights: WeightMatrixCSR, target: Optional["xugrid.Ugrid2d"]):
        instance = super().from_weights(weights, target)
        instance._setup_regrid("mean")
        return instance

    @classmethod
    def _weights_from_dataset(cls, dataset: xr.Dataset) -> WeightMatrixCOO:
        return cls._csr_from_dataset(dataset)
