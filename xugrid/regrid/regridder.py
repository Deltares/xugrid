"""
This module is heavily inspired by xemsf.frontend.py
"""
import abc
from itertools import chain
from typing import Callable, NamedTuple, Optional, Tuple, Union

import numba
import numpy as np
import xarray as xr

# dask as optional dependency
try:
    import dask.array

    DaskArray = dask.array.Array
except ImportError:
    DaskArray = ()

from xugrid.constants import FloatArray, IntArray
from xugrid.core.wrap import UgridDataArray, UgridDataset
from xugrid.regrid import reduce
from xugrid.regrid.unstructured import UnstructuredGrid2d
from xugrid.regrid.weight_matrix import WeightMatrixCSR, create_weight_matrix, nzrange
from xugrid.ugrid.ugrid2d import Ugrid2d


def get_method(method, methods):
    try:
        return methods[method]
    except KeyError as e:
        raise ValueError(
            "Invalid regridding method. Available methods are: {}".format(
                methods.keys()
            )
        ) from e


def _prepend(ds: xr.Dataset, prefix: str):
    vars = ds.data_vars
    dims = ds.dims
    name_dict = {v: f"{prefix}{v}" for v in chain(vars, dims)}
    return ds.rename(name_dict)


def _get_grid_variables(ds: xr.Dataset, prefix: str):
    ds = ds[[var for var in ds.data_vars if var.startswith(prefix)]]
    name_dict = {
        v: v.replace(prefix, "") if v.startswith(prefix) else v
        for v in chain(ds.data_vars, ds.dims)
    }
    return ds.rename(name_dict)


def get_grid(arg):
    if isinstance(arg, (UgridDataArray, UgridDataset)):
        return arg.grid
    elif isinstance(arg, Ugrid2d):
        return arg
    else:
        options = {"Ugrid2d", "UgridDataArray", "UgridDataset"}
        raise TypeError(f"Expected one of {options}, received: {type(arg).__name__}")
    

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
                    out[extra_index, target_index] = f(
                        source_flat, indices, weights
                    )
        return out
    
    return _regrid


class BaseRegridder(abc.ABC):
    _JIT_FUNCTIONS = {}

    def __init__(
        self,
        source: Ugrid2d,
        target: Ugrid2d,
        weights: Optional[Tuple] = None,
    ):
        source = get_grid(source)
        target = get_grid(target)
        self.source_grid = source
        self.target_grid = target
        if weights is None:
            self.compute_weights()
        else:
            self.source_index, self.target_index, self.weights = weights
            
    @property
    def regrid_weights(self):
        return (
            self.source_index,
            self.target_index,
            self.weights,
        )

    @abc.abstractmethod
    def to_dataset(self):
        """
        Store the computed weights in a dataset for re-use.
        """

    def _setup_regrid(self, func) -> Callable:
        if isinstance(func, str):
            self._regrid = get_method(func, self._JIT_FUNCTIONS)
        elif callable(func):
            self._regrid = make_regrid(func) 
        else:
            raise TypeError(
                f"method must be string or callable, received: {type(func).__name}")
        return

    def regrid_array(self, source):
        first_dims = source.shape[:-1]
        last_dims = source.shape[-1:]

        if last_dims != self.source.shape:
            raise ValueError(
                "Shape of last source dimensions does not match regridder "
                f"shape: {last_dims} versus {self.source.shape}"
            )

        if source.ndim == 1:
            source = source[np.newaxis]
        elif source.ndim > 2:
            source = source.reshape((-1,) + last_dims)

        size = self.target.size
        out_shape = first_dims + self.target.shape

        if isinstance(source, DaskArray):
            chunks = source.chunks[:-1] + (self.target.shape,)
            out = dask.array.map_blocks(
                self._regrid,  # func
                source,  # *args
                self.csr_weights,  # *args
                size,  # *args
                dtype=np.float64,
                chunks=chunks,
                meta=np.array((), dtype=source.dtype),
            )
        elif isinstance(source, np.ndarray):
            out = self._regrid(source, self.csr_weights, size)
        else:
            raise TypeError(
                "Expected dask.array.Array or numpy.ndarray. Received: "
                f"{type(source).__name__}"
            )

        return out.reshape(out_shape)

    def regrid_dataarray(self, source):
        source_dims = (source.ugrid.grid.face_dimension,)
        # Do not set vectorize=True: numba will run the for loop more
        # efficiently, and guarantees a single large allocation.
        out = xr.apply_ufunc(
            self.regrid_array,
            source.ugrid.obj,
            input_core_dims=[source_dims],
            exclude_dims=set(source_dims),
            output_core_dims=[self.target.dims],
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
        object: UgridDataArray

        Returns
        -------
        regridded: UgridDataArray
        """
        regridded = self.regrid_dataarray(object)
        return UgridDataArray(
            regridded,
            self.target.grid,
        )

    @classmethod
    def from_dataset(cls, dataset: xr.Dataset):
        """
        Reconstruct the regridder from a dataset with source, target indices
        and weights.
        """
        source = Ugrid2d.from_dataset(_get_grid_variables(dataset, "__source__"))
        target = Ugrid2d.from_dataset(_get_grid_variables(dataset, "__target__"))
        weights = (
            dataset["source_index"].values,
            dataset["target_index"].values,
            dataset["weights"].values,
        )
        return cls(
            source,
            target,
            weights,
        )


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

    """

    def compute_weights(self):
        tree = self.source_grid.celltree
        self.source_index = tree.locate_points(self.target_grid.centroids)
        self.weights = xr.DataArray(
            data=np.where(self.source_index != -1, 1.0, np.nan),
            dims=[self.target_grid.face_dimension],
        )
        return

    def regrid(self, obj: UgridDataArray) -> UgridDataArray:
        """
        Regrid an object to the target grid topology.

        Parameters
        ----------
        obj: UgridDataArray

        Returns
        -------
        regridded: UgridDataArray
            The data regridded to the target grid. The target grid has been set
            as the face dimension.
        """
        grid = obj.ugrid.grid
        facedim = grid.face_dimension
        da = obj.obj.isel({facedim: self.source_index})
        da.name = obj.name
        uda = UgridDataArray(da, self.target_grid)
        uda = uda.rename({facedim: self.target_grid.face_dimension})
        uda = uda * self.weights.values
        return uda

    def to_dataset(self) -> xr.Dataset:
        source_ds = _prepend(self.source_grid.to_dataset(), "__source__")
        target_ds = _prepend(self.target_grid.to_dataset(), "__target__")
        regrid_ds = xr.Dataset(
            {
                "source_index": self.source_index,
                "target_index": np.nan,
                "weights": self.weights,
            },
        )
        return xr.merge((source_ds, target_ds, regrid_ds))
    

class BaseOverlapRegridder(BaseRegridder, abc.ABC):
    def __init__(
        self,
        source: UgridDataArray,
        target: UgridDataArray,
        method: Union[str, Callable],
        weights: Optional[Union[Tuple, WeightMatrixCSR]] = None,
    ):
        source = get_grid(source)
        target = get_grid(target)
        self.source = UnstructuredGrid2d(source)
        self.target = UnstructuredGrid2d(target)
        self._setup_regrid(method)
        if weights is None:
            self.compute_weights()
        elif isinstance(weights, WeightMatrixCSR):
            self.source_index = None
            self.target_index = None
            self.weights = None
            self.csr_weights = weights
        else:
            self.source_index, self.target_index, self.weights = weights

    def _compute_weights(self, relative: bool):
        self.source_index, self.target_index, self.weights = self.source.overlap(
            self.target, relative=relative
        )
        self.csr_weights = create_weight_matrix(
            self.target_index, self.source_index, self.weights
        )
        return
    
    @property
    def regrid_weights(self):
        return self.csr_weights

    @classmethod
    def to_dataset(self, dataset: xr.Dataset):
        return


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
    _JIT_FUNCTIONS = {k: numba.njit(make_regrid(f), parallel=True, cache=True) for k, f in reduce.ASBOLUTE_OVERLAP_METHODS.items()}
    
    def __init__(
        self,
        source: UgridDataArray,
        target: UgridDataArray,
        method: Union[str, Callable] = "mean",
        weights: Optional[Tuple] = None,
    ):
        super().__init__(source=source, target=target, method=method, weights=weights)
        
    def compute_weights(self) -> None:
        self._compute_weights(relative=False)


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
    """
    _JIT_FUNCTIONS = {k: numba.njit(make_regrid(f), parallel=True, cache=True) for k, f in reduce.RELATIVE_OVERLAP_METHODS.items()}

    def compute_weights(self) -> None:
        self._compute_weights(relative=True)


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
    _JIT_FUNCTIONS = {"mean": numba.njit(make_regrid(reduce.mean), parallel=True, cache=True)}

    def __init__(
        self,
        source: UgridDataArray,
        target: UgridDataArray,
        weights: Optional[Union[Tuple, WeightMatrixCSR]] = None,
    ):
        source = get_grid(source)
        target = get_grid(target)
        self.source = UnstructuredGrid2d(source)
        self.target = UnstructuredGrid2d(target)
        # Since the weights for a target face sum up to 1.0, a weight mean is
        # appropriate, and takes care of NaN values in the source data.
        self._setup_regrid("mean")

        if weights is None:
            self.compute_weights()
        elif isinstance(weights, WeightMatrixCSR):
            self.source_index = None
            self.target_index = None
            self.weights = None
            self.csr_weights = weights
        else:
            self.source_index, self.target_index, self.weights = weights

    def compute_weights(self):
        (
            self.source_index,
            self.target_index,
            self.weights,
        ) = self.source.barycentric(self.target)
        self.csr_weights = create_weight_matrix(
            self.target_index, self.source_index, self.weights
        )
        return

    @property
    def regrid_weights(self):
        return self.csr_weights

    @classmethod
    def to_dataset(self, dataset: xr.Dataset):
        return
