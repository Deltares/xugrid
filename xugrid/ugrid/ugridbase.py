import abc
import copy
import warnings
from itertools import chain
from typing import Dict, Literal, Set, Tuple, Type, Union, cast

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike
from scipy.sparse import csr_matrix

from xugrid.constants import FILL_VALUE, BoolArray, FloatArray, IntArray
from xugrid.ugrid import connectivity, conventions


def as_pandas_index(index: Union[BoolArray, IntArray, pd.Index], n: int):
    if isinstance(index, np.ndarray):
        if index.size > n:
            raise ValueError(
                f"index size {index.size} is larger than dimension size: {n}"
            )
        if np.issubdtype(index.dtype, np.bool_):
            # Significantly quicker if all true.
            if index.all():
                pd_index = pd.RangeIndex(0, n)
            else:
                pd_index = pd.Index((np.arange(n)[index]))
        elif np.issubdtype(index.dtype, np.integer):
            pd_index = pd.Index(index)
        else:
            raise TypeError(f"index should be bool or integer. Received: {index.dtype}")

    elif isinstance(index, pd.Index):
        pd_index = index

    else:
        raise TypeError(
            "index should be pandas Index or numpy array. Received: "
            f"{type(index).__name__}"
        )

    if not pd_index.is_unique:
        raise ValueError(
            "index contains repeated values; only subsets will result "
            "in valid UGRID topology."
        )
    #    # TODO?
    #    # Uniqueness is required, but sorting arguably not.
    #    if not pd_index.is_monotonic_increasing:
    #        raise NotImplementedError("UGRID indexes must be sorted and unique.")

    return pd_index


def align(obj, grids, old_indexes):
    """
    Check which indexes have changed. Index on those new values.
    If none are changed, return (obj, grids) as is.
    """
    if old_indexes is None:
        return obj, grids

    ugrid_dims = set(chain.from_iterable(grid.dims for grid in grids)).intersection(
        old_indexes
    )
    new_indexes = {
        k: index
        for k, index in obj.indexes.items()
        if (k in ugrid_dims) and (not index.equals(old_indexes[k]))
    }
    if not new_indexes:
        return obj, grids

    # Group the indexers by grid
    new_grids = []
    for grid in grids:
        ugrid_dims = grid.dims.intersection(new_indexes)
        ugrid_indexes = {dim: new_indexes[dim] for dim in ugrid_dims}
        newgrid, indexers = grid.isel(indexers=ugrid_indexes, return_index=True)
        indexers = {
            k: v for k, v in indexers.items() if k in obj.dims and k not in new_indexes
        }
        obj = obj.isel(indexers)
        new_grids.append(newgrid)
    return obj, new_grids


class AbstractUgrid(abc.ABC):
    @property
    @abc.abstractmethod
    def topology_dimension(self):
        pass

    @property
    @abc.abstractmethod
    def core_dimension(self):
        pass

    @property
    @abc.abstractmethod
    def dims(self) -> Set[str]:
        pass

    @property
    def dimensions(self) -> Dict[str, Dict[str, str]]:
        """
        Mapping from UGRID dimension names to lengths.

        This property will be changed to return a type more consistent with
        DataArray.dims in the future, i.e. a set of dimension names.
        """

        warnings.warn(
            ".dimensions will is replaced by .dims and its return type is a set "
            "of dimension names in future. To access a mapping of names to "
            "lengths, use .sizes instead.",
            FutureWarning,
        )
        return self.sizes

    @property
    @abc.abstractmethod
    def sizes(self):
        pass

    @property
    @abc.abstractmethod
    def mesh(self):
        pass

    @property
    @abc.abstractmethod
    def meshkernel(self):
        pass

    @staticmethod
    @abc.abstractmethod
    def from_dataset(self):
        pass

    @abc.abstractmethod
    def to_dataset(self) -> xr.Dataset:
        pass

    @abc.abstractmethod
    def topology_subset(self):
        pass

    @abc.abstractmethod
    def clip_box(self):
        pass

    @abc.abstractmethod
    def sel_points(self, obj, x, y, out_bounds, fill_value):
        pass

    @abc.abstractmethod
    def intersect_line(self):
        pass

    @abc.abstractmethod
    def intersect_linestring(self):
        pass

    @abc.abstractmethod
    def sel(self):
        pass

    @abc.abstractmethod
    def _clear_geometry_properties(self):
        pass

    @staticmethod
    @abc.abstractmethod
    def merge_partitions():
        pass

    @abc.abstractmethod
    def reindex_like(self):
        pass

    @abc.abstractmethod
    def get_connectivity_matrix(self, dim: str, xy_weights: bool) -> csr_matrix:
        pass

    @abc.abstractmethod
    def create_data_array(self, data: ArrayLike, facet: str):
        pass

    @abc.abstractmethod
    def get_coordinates(self, dim: str):
        pass

    def _create_data_array(self, data: ArrayLike, dimension: str):
        from xugrid import UgridDataArray

        data = np.array(data)
        if data.ndim != 1:
            raise ValueError(
                "Can only create DataArrays from 1D arrays. "
                f"Data has {data.ndim} dimensions."
            )
        len_data = len(data)
        len_grid = self.sizes[dimension]
        if len_data != len_grid:
            raise ValueError(
                f"Conflicting sizes for dimension {dimension}: length "
                f"{len_data} on the data, but length {len_grid} on the grid."
            )

        da = xr.DataArray(data=data, dims=(dimension,))

        # TODO: is there a better way to do this to satisfy mypy?
        grid = cast(UgridType, self)
        return UgridDataArray(da, grid)

    def _initialize_indexes_attrs(self, name, dataset, indexes, attrs):
        defaults = conventions.default_topology_attrs(name, self.topology_dimension)

        if dataset is None:
            if attrs is None:
                x, y = defaults["node_coordinates"].split()
                indexes = {"node_x": x, "node_y": y}
            else:
                if indexes is None:
                    raise ValueError("indexes must be provided for attrs")
                defaults.update(attrs)

            self._indexes = indexes
            self._attrs = defaults

        else:
            if attrs is not None:
                raise ValueError("Provide either dataset or attrs, not both.")
            if indexes is None:
                raise ValueError("indexes must be provided for dataset")

            derived_dims = dataset.ugrid_roles.dimensions[name]
            self._indexes = indexes
            self._attrs = {**defaults, **derived_dims, **dataset[name].attrs}

        # Ensure the name is always in sync.
        self._attrs["name"] = name
        return

    def rename(self, name: str, return_name_dict: bool = False):
        """
        Create a new grid with all variables named according to the default
        naming conventions.
        """
        # Get the old and the new names. Their keys are the same.
        old_attrs = self._attrs
        new_attrs = conventions.default_topology_attrs(name, self.topology_dimension)

        # The attrs will have some roles joined together, e.g. node_coordinates
        # will contain x and y as "mesh2d_node_x mesh2d_node_y".
        name_dict = {self.name: name}
        skip = ("cf_role", "long_name", "topology_dimension")
        for key, value in old_attrs.items():
            if key in new_attrs and key not in skip:
                split_new = new_attrs[key].split()
                split_old = value.split()
                if len(split_new) != len(split_old):
                    raise ValueError(
                        f"Number of entries does not match on {key}: "
                        f"{split_new} versus {split_old}"
                    )
                for name_key, name_value in zip(split_old, split_new):
                    name_dict[name_key] = name_value

        new = self.copy()
        new.name = name
        new._attrs = new_attrs
        new._indexes = {k: name_dict[v] for k, v in new._indexes.items()}
        if new._dataset is not None:
            to_rename = (
                tuple(new._dataset.data_vars)
                + tuple(new._dataset.coords)
                + tuple(new._dataset.dims)
            )
            new._dataset = new._dataset.rename(
                {k: v for k, v in name_dict.items() if k in to_rename}
            )

        if return_name_dict:
            return new, name_dict
        else:
            return new

    def _propagate_properties(self, other) -> None:
        other.start_index = self.start_index
        other.fill_value = self.fill_value

    @staticmethod
    def _single_topology(dataset: xr.Dataset):
        topologies = dataset.ugrid_roles.topology
        n_topology = len(topologies)
        if n_topology == 0:
            raise ValueError("Dataset contains no UGRID topology variable.")
        elif n_topology > 1:
            raise ValueError(
                f"Dataset contains {n_topology} topology variables, "
                "please specify the topology variable name to use."
            )
        return topologies[0]

    def _filtered_attrs(self, dataset: xr.Dataset):
        """Remove names that are not present in the dataset."""
        topodim = self.topology_dimension
        attrs = self._attrs.copy()

        ugrid_dims = conventions._DIM_NAMES[topodim] + tuple(
            [dims[0] for dims in conventions._CONNECTIVITY_DIMS.values()]
        )
        for key in ugrid_dims:
            if key in attrs and attrs[key] not in dataset.dims:
                attrs.pop(key)

        for key in conventions._CONNECTIVITY_NAMES[topodim]:
            if key in attrs and attrs[key] not in dataset:
                attrs.pop(key)

        for coord in conventions._COORD_NAMES[topodim]:
            if coord in attrs:
                names = attrs[coord].split(" ")
                present = [name for name in names if name in dataset]
                if present:
                    attrs[coord] = " ".join(present)
                else:
                    attrs.pop(coord)

        return attrs

    def __repr__(self):
        if self._dataset:
            return self._dataset.__repr__()
        else:
            return self.to_dataset().__repr__()

    def equals(self, other) -> bool:
        if other is self:
            return True
        elif isinstance(other, type(self)):
            xr_self = self.to_dataset()
            xr_other = other.to_dataset()
            return xr_self.identical(xr_other)
        return False

    def copy(self):
        """Create a deepcopy."""
        return copy.deepcopy(self)

    @property
    def fill_value(self) -> int:
        """Fill value for UGRID connectivity arrays."""
        return self._fill_value

    @fill_value.setter
    def fill_value(self, value: int):
        self._fill_value = value

    @property
    def start_index(self) -> int:
        """Start index for UGRID connectivity arrays."""
        return self._start_index

    @start_index.setter
    def start_index(self, value: Literal[0, 1]):
        if value not in (0, 1):
            raise ValueError(f"start_index must be 0 or 1, received: {value}")
        self._start_index = value

    @property
    def attrs(self):
        return copy.deepcopy(self._attrs)

    @property
    def node_dimension(self):
        """Name of node dimension"""
        return self._attrs["node_dimension"]

    @property
    def edge_dimension(self):
        """Name of edge dimension"""
        return self._attrs["edge_dimension"]

    @property
    def max_connectivity_dimensions(self) -> tuple[str]:
        return ()

    @property
    def max_connectivity_sizes(self) -> dict[str, int]:
        return {}

    @property
    def node_coordinates(self) -> FloatArray:
        """Coordinates (x, y) of the nodes (vertices)"""
        return np.column_stack([self.node_x, self.node_y])

    @property
    def n_node(self) -> int:
        """Number of nodes (vertices) in the UGRID topology"""
        return self.node_x.size

    @property
    def n_edge(self) -> int:
        """Number of edges in the UGRID topology"""
        return self.edge_node_connectivity.shape[0]

    @property
    def edge_x(self):
        """x-coordinate of every edge in the UGRID topology"""
        if self._edge_x is None:
            self._edge_x = self.node_x[self.edge_node_connectivity].mean(axis=1)
        return self._edge_x

    @property
    def edge_y(self):
        """y-coordinate of every edge in the UGRID topology"""
        if self._edge_y is None:
            self._edge_y = self.node_y[self.edge_node_connectivity].mean(axis=1)
        return self._edge_y

    @property
    def edge_coordinates(self) -> FloatArray:
        """Centroid (x,y) coordinates of every edge in the UGRID topology"""
        return np.column_stack([self.edge_x, self.edge_y])

    @property
    def edge_node_coordinates(self) -> FloatArray:
        """Node coordinates for every edge, shape: ``n_edge, 2, 2``."""
        return self.node_coordinates[self.edge_node_connectivity]

    @property
    @abc.abstractmethod
    def coords(self):
        pass

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Returns a tuple with the node bounds: xmin, ymin, xmax, ymax"""
        if any(
            [
                self._xmin is None,
                self._ymin is None,
                self._xmax is None,
                self._ymax is None,
            ]
        ):
            self._xmin = self.node_x.min()
            self._ymin = self.node_y.min()
            self._xmax = self.node_x.max()
            self._ymax = self.node_y.max()
        return (
            self._xmin,
            self._ymin,
            self._xmax,
            self._ymax,
        )

    @property
    def edge_bounds(self) -> FloatArray:
        """
        Returns a numpy array with columns ``minx, miny, maxx, maxy``,
        describing the bounds of every edge in the grid.

        Returns
        -------
        edge_bounds: np.ndarray of shape (n_edge, 4)
        """
        x = self.node_x[self.edge_node_connectivity]
        y = self.node_y[self.edge_node_connectivity]
        return np.column_stack(
            [
                x.min(axis=1),
                y.min(axis=1),
                x.max(axis=1),
                y.max(axis=1),
            ]
        )

    @staticmethod
    def _prepare_connectivity(
        da: xr.DataArray, fill_value: Union[int, float], dtype: type
    ) -> xr.DataArray:
        """
        Undo the work xarray does when it encounters a _FillValue for UGRID
        connectivity arrays. Set an external unified value back (across all
        connectivities!), and cast back to the desired dtype.
        """
        data = da.to_numpy().copy()
        # If xarray detects a _FillValue, it converts the array to floats and
        # replaces the fill value by NaN, and moves the _FillValue to
        # da.encoding.
        if "_FillValue" in da.attrs:
            is_fill = data == da.attrs["_FillValue"]
        else:
            is_fill = np.isnan(data)
        # Set the fill_value before casting: otherwise the cast may fail.
        data[is_fill] = fill_value
        cast = data.astype(dtype, copy=False)
        not_fill = ~is_fill
        if (cast[not_fill] < 0).any():
            raise ValueError("connectivity contains negative values")
        return da.copy(data=cast)

    def _adjust(self, connectivity: IntArray) -> IntArray:
        """Adjust connectivity for desired fill_value and start_index."""
        c = connectivity.copy()
        if self.start_index == 0 and self.fill_value == FILL_VALUE:
            return c
        is_fill = c == FILL_VALUE
        if self.start_index:
            c[~is_fill] += self.start_index
        if self.fill_value != FILL_VALUE:
            c[is_fill] = self.fill_value
        return c

    def _precheck(self, multi_index):
        dim, index = multi_index.popitem()
        for check_dim, check_index in multi_index.items():
            if not index.equals(check_index):
                raise ValueError(
                    f"UGRID dimensions do not align: {dim} versus {check_dim}"
                )
        return index

    def _postcheck(self, indexers, finalized_indexers):
        for dim, indexer in indexers.items():
            if dim != self.core_dimension:
                if not indexer.equals(finalized_indexers[dim]):
                    raise ValueError(
                        f"This subset selection of UGRID dimension {dim} results "
                        "in an invalid topology "
                    )
        return

    def find_ugrid_dim(self, obj: Union[xr.DataArray, xr.Dataset]):
        """Find the UGRID dimension that is present in the object."""
        ugrid_dims = self.dims.intersection(obj.dims)
        if len(ugrid_dims) != 1:
            raise ValueError(
                "UgridDataArray should contain exactly one of the UGRID "
                f"dimensions: {self.dims}"
            )
        return ugrid_dims.pop()

    def set_node_coords(
        self,
        node_x: str,
        node_y: str,
        obj: Union[xr.DataArray, xr.Dataset],
        projected: bool = True,
    ):
        """
        Given names of x and y coordinates of the nodes of an object, set them
        as the coordinates in the grid.

        Parameters
        ----------
        node_x: str
            Name of the x coordinate of the nodes in the object.
        node_y: str
            Name of the y coordinate of the nodes in the object.
        """
        if " " in node_x or " " in node_y:
            raise ValueError("coordinate names may not contain spaces")

        x = obj[node_x].to_numpy()
        y = obj[node_y].to_numpy()

        if (x.ndim != 1) or (x.size != self.n_node):
            raise ValueError(
                "shape of node_x does not match n_node of grid: "
                f"{x.shape} versus {self.n_node}"
            )
        if (y.ndim != 1) or (y.size != self.n_node):
            raise ValueError(
                "shape of node_y does not match n_node of grid: "
                f"{y.shape} versus {self.n_node}"
            )

        # Remove them, then append at the end.
        node_coords = [
            coord
            for coord in self._attrs["node_coordinates"].split(" ")
            if coord not in (node_x, node_y)
        ]
        node_coords.extend((node_x, node_y))

        self._clear_geometry_properties()
        self.node_x = np.ascontiguousarray(x)
        self.node_y = np.ascontiguousarray(y)
        self._attrs["node_coordinates"] = " ".join(node_coords)
        self._indexes["node_x"] = node_x
        self._indexes["node_y"] = node_y
        self.projected = projected

    def assign_node_coords(
        self,
        obj: Union[xr.DataArray, xr.Dataset],
    ) -> Union[xr.DataArray, xr.Dataset]:
        """
        Assign node coordinates from the grid to the object.

        Returns a new object with all the original data in addition to the new
        node coordinates of the grid.

        Parameters
        ----------
        obj: xr.DataArray or xr.Dataset

        Returns
        -------
        assigned (same type as obj)
        """
        xname = self._indexes["node_x"]
        yname = self._indexes["node_y"]
        x_attrs = conventions.DEFAULT_ATTRS["node_x"][self.projected]
        y_attrs = conventions.DEFAULT_ATTRS["node_y"][self.projected]
        coords = {
            xname: xr.DataArray(
                data=self.node_x,
                dims=(self.node_dimension,),
                attrs=x_attrs,
            ),
            yname: xr.DataArray(
                data=self.node_y,
                dims=(self.node_dimension,),
                attrs=y_attrs,
            ),
        }
        return obj.assign_coords(coords)

    def assign_edge_coords(
        self,
        obj: Union[xr.DataArray, xr.Dataset],
    ) -> Union[xr.DataArray, xr.Dataset]:
        """
        Assign node coordinates from the grid to the object.

        Returns a new object with all the original data in addition to the new
        node coordinates of the grid.

        Parameters
        ----------
        obj: xr.DataArray or xr.Dataset

        Returns
        -------
        assigned (same type as obj)
        """
        xname = self._indexes.get("edge_x", f"{self.name}_edge_x")
        yname = self._indexes.get("edge_y", f"{self.name}_edge_y")
        x_attrs = conventions.DEFAULT_ATTRS["edge_x"][self.projected]
        y_attrs = conventions.DEFAULT_ATTRS["edge_y"][self.projected]
        coords = {
            xname: xr.DataArray(
                data=self.edge_x,
                dims=(self.edge_dimension,),
                attrs=x_attrs,
            ),
            yname: xr.DataArray(
                data=self.edge_y,
                dims=(self.edge_dimension,),
                attrs=y_attrs,
            ),
        }
        return obj.assign_coords(coords)

    @property
    def node_edge_connectivity(self) -> csr_matrix:
        """
        Node to edge connectivity.

        Returns
        -------
        connectivity: csr_matrix
        """
        if self._node_edge_connectivity is None:
            self._node_edge_connectivity = connectivity.invert_dense_to_sparse(
                self.edge_node_connectivity
            )
        return self._node_edge_connectivity

    @property
    def node_node_connectivity(self) -> csr_matrix:
        """
        Node to node connectivity.

        The connectivity is represented as an adjacency matrix in CSR format,
        with the row and column indices as a (0-based) node index. The data of
        the matrix contains the edge index as every connection is formed by an
        edge.

        Returns
        -------
        connectivity: csr_matrix
        """
        if self._node_node_connectivity is None:
            self._node_node_connectivity = connectivity.node_node_connectivity(
                self.edge_node_connectivity
            )
        return self._node_node_connectivity

    @property
    def directed_node_node_connectivity(self) -> csr_matrix:
        """
        Directed node to node connectivity.

        The connectivity is represented as an adjacency matrix in CSR format,
        with the row and column indices as a (0-based) node index. The data of
        the matrix contains the edge index as every connection is formed by an
        edge.

        Returns
        -------
        connectivity: csr_matrix
        """
        return connectivity.directed_node_node_connectivity(self.edge_node_connectivity)

    @staticmethod
    def _connectivity_weights(connectivity: csr_matrix, coordinates: FloatArray):
        xy = coordinates
        coo = connectivity.tocoo()
        i = coo.row
        j = coo.col
        distance = np.linalg.norm(xy[j] - xy[i], axis=1)
        # The inverse distance is a measure of the strength of the connection.
        # Normalize so the weights are around 1.0
        return distance.mean() / distance

    def set_crs(
        self,
        crs: Union["pyproj.CRS", str] = None,  # type: ignore # noqa
        epsg: int = None,
        allow_override: bool = False,
    ):
        """
        Set the Coordinate Reference System (CRS) of a UGRID topology.

        NOTE: The underlying geometries are not transformed to this CRS. To
        transform the geometries to a new CRS, use the ``to_crs`` method.

        Parameters
        ----------
        crs : pyproj.CRS, optional if `epsg` is specified
            The value can be anything accepted
            by :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
            such as an authority string (eg "EPSG:4326") or a WKT string.
        epsg : int, optional if `crs` is specified
            EPSG code specifying the projection.
        allow_override : bool, default False
            If the the UGRID topology already has a CRS, allow to replace the
            existing CRS, even when both are not equal.
        """
        import pyproj

        if crs is not None:
            crs = pyproj.CRS.from_user_input(crs)
        elif epsg is not None:
            crs = pyproj.CRS.from_epsg(epsg)
        else:
            raise ValueError("Must pass either crs or epsg.")

        if not allow_override and self.crs is not None and not self.crs == crs:
            raise ValueError(
                "The Ugrid already has a CRS which is not equal to the passed "
                "CRS. Specify 'allow_override=True' to allow replacing the existing "
                "CRS without doing any transformation. If you actually want to "
                "transform the geometries, use '.to_crs' instead."
            )
        self.crs = crs

    def to_crs(
        self,
        crs: Union["pyproj.CRS", str] = None,  # type: ignore # noqa
        epsg: int = None,
    ):
        """
        Transform geometries to a new coordinate reference system.
        Transform all geometries in an active geometry column to a different coordinate
        reference system. The ``crs`` attribute on the current Ugrid must
        be set. Either ``crs`` or ``epsg`` may be specified for output.

        This method will transform all points in all objects. It has no notion
        of projecting the cells. All segments joining points are assumed to be
        lines in the current projection, not geodesics. Objects crossing the
        dateline (or other projection boundary) will have undesirable behavior.

        Parameters
        ----------
        crs : pyproj.CRS, optional if `epsg` is specified
            The value can be anything accepted by
            :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
            such as an authority string (eg "EPSG:4326") or a WKT string.
        epsg : int, optional if `crs` is specified
            EPSG code specifying output projection.
        """
        import pyproj

        if self.crs is None:
            raise ValueError(
                "Cannot transform naive geometries.  "
                "Please set a crs on the object first."
            )
        if crs is not None:
            crs = pyproj.CRS.from_user_input(crs)
        elif epsg is not None:
            crs = pyproj.CRS.from_epsg(epsg)
        else:
            raise ValueError("Must pass either crs or epsg.")

        grid = self.copy()
        if self.crs.is_exact_same(crs):
            return grid

        transformer = pyproj.Transformer.from_crs(
            crs_from=self.crs, crs_to=crs, always_xy=True
        )
        node_x, node_y = transformer.transform(xx=grid.node_x, yy=grid.node_y)
        grid.node_x = node_x
        grid.node_y = node_y
        grid._clear_geometry_properties()
        grid._dataset = None
        grid.crs = crs

        return grid

    @property
    def is_geographic(self):
        if self.crs is None:
            return False
        return self.crs.is_geographic

    def plot(self, **kwargs):
        """
        Plot the edges of the mesh.

        Parameters
        ----------
        **kwargs : optional
            Additional keyword arguments to ``matplotlib.pyplot.line``.
        """
        from xugrid.plot import line

        return line(self, **kwargs)


UgridType = Type[AbstractUgrid]
