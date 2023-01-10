import abc
import copy
from itertools import chain
from typing import Tuple, Type, Union

import numpy as np
import pandas as pd
import xarray as xr
from scipy.sparse import csr_matrix

from xugrid.constants import BoolArray, FloatArray, IntArray
from xugrid.ugrid import connectivity, conventions


def as_pandas_index(index, n: int):
    if isinstance(index, np.ndarray):
        if index.size > n:
            raise ValueError(
                f"index size {index.size} is larger than dimension size: {n}"
            )
        if np.issubdtype(index.dtype, np.bool_):
            pd_index = np.arange(np.arange(n)[index])
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
    if not pd_index.is_monotonic_increasing:
        raise NotImplementedError("UGRID indexes must be sorted and unique.")

    return pd_index


def align(obj, grids, old_indexes):
    """
    Check which indexes have changed. Index on those new values.
    If none are changed, return (obj, grids) as is.
    """
    if old_indexes is None:
        return obj, grids

    ugrid_dims = set(chain.from_iterable(grid.dimensions for grid in grids)).intersection(old_indexes)
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
        ugrid_dims = set(grid.dimensions).intersection(new_indexes)
        ugrid_indexes = {dim: new_indexes[dim] for dim in ugrid_dims}
        newgrid, indexers = grid.isel(indexers=ugrid_indexes, return_index=True)
        indexers = {
            k: v for k, v in indexers.items() if k in obj.dims and k not in new_indexes
        }
        obj = obj.isel(indexers)
        new_grids.append(newgrid)
    return obj, new_grids


class AbstractUgrid(abc.ABC):
    @abc.abstractproperty
    def topology_dimension():
        """ """

    @abc.abstractproperty
    def dimensions():
        """ """

    @abc.abstractstaticmethod
    def from_dataset():
        """ """

    @abc.abstractmethod
    def to_dataset():
        """ """

    @abc.abstractmethod
    def topology_subset():
        """ """

    @abc.abstractmethod
    def clip_box():
        """ """

    @abc.abstractmethod
    def isel():
        """ """

    @abc.abstractmethod
    def sel():
        """ """

    @abc.abstractmethod
    def _clear_geometry_properties():
        """ """

    def _initialize_indexes_attrs(self, name, dataset, indexes, attrs):
        defaults = conventions.default_topology_attrs(name, self.topology_dimension)
        if dataset is None:
            if attrs is not None:
                defaults.update(attrs)
            x, y = defaults["node_coordinates"].split(" ")
            self._indexes = {"node_x": x, "node_y": y}
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

    @staticmethod
    def _single_topology(dataset: xr.Dataset):
        topologies = dataset.ugrid_roles.topology
        n_topology = len(topologies)
        if n_topology == 0:
            raise ValueError("Dataset contains no UGRID topology variable.")
        elif n_topology > 1:
            raise ValueError(
                "Dataset contains {n_topology} topology variables, "
                "please specify the topology variable name to use."
            )
        return topologies[0]

    def _filtered_attrs(self, dataset: xr.Dataset):
        """
        Removes names that are not present in the dataset.
        """
        topodim = self.topology_dimension
        attrs = self._attrs.copy()

        for key in conventions._DIM_NAMES[topodim]:
            if key in attrs:
                if attrs[key] not in dataset.dims:
                    attrs.pop(key)

        for key in conventions._CONNECTIVITY_NAMES[topodim]:
            if key in attrs:
                if attrs[key] not in dataset:
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

    def equals(self, other):
        if not isinstance(other, type(self)):
            return False
        # TODO: check values, etc.
        return True

    def copy(self):
        """Creates deepcopy"""
        return copy.deepcopy(self)

    @property
    def node_dimension(self):
        """Name of node dimension"""
        return self._attrs["node_dimension"]

    @property
    def edge_dimension(self):
        """Name of edge dimension"""
        return self._attrs["edge_dimension"]

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
        da: xr.DataArray, fill_value: Union[float, int], dtype: type
    ) -> xr.DataArray:
        start_index = da.attrs.get("start_index", 0)
        if start_index not in (0, 1):
            raise ValueError(f"start_index should be 0 or 1, received: {start_index}")

        data = da.values
        if "_FillValue" in da.attrs:
            is_fill = data == da.attrs["_FillValue"]
        else:
            is_fill = np.isnan(data)

        cast = data.astype(dtype, copy=True)
        if start_index:
            cast -= start_index
        cast[is_fill] = fill_value
        if (cast[~is_fill] < 0).any():
            raise ValueError("connectivity contains negative values")
        return da.copy(data=cast)

    @staticmethod
    def _check_index_identity(index: Union[BoolArray, IntArray], n: int) -> bool:
        """
        Check whether index would return all values.
        """
        if isinstance(np.ndarray):
            if np.issubdtype(index.dtype, np.bool_):
                return index.all()
            elif np.issubdtype(index.dtype, np.integer):
                index = pd.Index(index)
            else:
                raise TypeError(
                    f"index should be bool or integer. Received: {index.dtype}"
                )
        elif not isinstance(index, pd.Index):
            raise TypeError(
                "index should be pandas Index or numpy arrray. Received: "
                f"{type(index).__name__}"
            )

        if isinstance(index, pd.Index):
            if not index.is_unique():
                raise ValueError(
                    "index contains repeated values. Only subsets will result "
                    "in valid UGRID topology."
                )
            return index.size == n

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

        x = obj[node_x].values
        y = obj[node_y].values

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
                self.edge_node_connectivity, self.fill_value
            )
        return self._node_edge_connectivity

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

    def plot(self, **kwargs):
        """
        Plots the edges of the mesh.

        Parameters
        ----------
        **kwargs : optional
            Additional keyword arguments to ``matplotlib.pyplot.line``.
        """
        from ..plot import line

        return line(self, **kwargs)


UgridType = Type[AbstractUgrid]
