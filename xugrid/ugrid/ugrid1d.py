from itertools import chain
from typing import Any, Dict, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from xugrid import conversion
from xugrid.constants import BoolArray, FloatArray, FloatDType, IntArray, IntDType
from xugrid.core.utils import either_dict_or_kwargs
from xugrid.ugrid import connectivity, conventions
from xugrid.ugrid.ugridbase import AbstractUgrid, as_pandas_index


class Ugrid1d(AbstractUgrid):
    """
    This class stores the topological data of a "1-D unstructured grid": a
    collection of connected line elements, such as a river network.

    Parameters
    ----------
    node_x: ndarray of floats
    node_y: ndarray of floats
    fill_value: int
    edge_node_connectivity: ndarray of integers
    name: string, optional
        Network name. Defaults to "network1d".
    dataset: xr.Dataset, optional
    indexes: Dict[str, str], optional
        When a dataset is provided, a mapping from the UGRID role to the dataset
        variable name. E.g. {"face_x": "mesh2d_face_lon"}.
    projected: bool, optional
        Whether node_x and node_y are longitude and latitude or projected x and
        y coordinates. Used to write the appropriate standard_name in the
        coordinate attributes.
    crs: Any, optional
        Coordinate Reference System of the geometry objects. Can be anything accepted by
        :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
        such as an authority string (eg "EPSG:4326") or a WKT string.
    attrs: Dict[str, str], optional
        UGRID topology attributes. Should not be provided together with
        dataset: if other names are required, update the dataset instead.
        A name entry is ignored, as name is given explicitly.
    """

    def __init__(
        self,
        node_x: FloatArray,
        node_y: FloatArray,
        fill_value: int,
        edge_node_connectivity: IntArray = None,
        name: str = "network1d",
        dataset: xr.Dataset = None,
        indexes: Dict[str, str] = None,
        projected: bool = True,
        crs: Any = None,
        attrs: Dict[str, str] = None,
    ):
        self.node_x = np.ascontiguousarray(node_x)
        self.node_y = np.ascontiguousarray(node_y)
        self.fill_value = fill_value
        self.edge_node_connectivity = edge_node_connectivity
        self.name = name
        self.projected = projected

        self._initialize_indexes_attrs(name, dataset, indexes, attrs)
        self._dataset = dataset

        # Optional attributes, deferred initialization
        # Meshkernel
        self._mesh = None
        self._meshkernel = None
        # Celltree
        self._celltree = None
        # Bounds
        self._xmin = None
        self._xmax = None
        self._ymin = None
        self._ymax = None
        # Edges
        self._edge_x = None
        self._edge_y = None
        # Connectivity
        self._node_node_connectivity = None
        self._node_edge_connectivity = None
        # crs
        if crs is None:
            self.crs = None
        else:
            import pyproj

            self.crs = pyproj.CRS.from_user_input(crs)

    @classmethod
    def from_dataset(cls, dataset: xr.Dataset, topology: str = None):
        """
        Extract the 1D UGRID topology information from an xarray Dataset.

        Parameters
        ----------
        dataset: xr.Dataset
            Dataset containing topology information stored according to UGRID conventions.

        Returns
        -------
        grid: Ugrid1dAdapter
        """
        ds = dataset
        if not isinstance(ds, xr.Dataset):
            raise TypeError(
                "Ugrid should be initialized with xarray.Dataset. "
                f"Received instead: {type(ds)}"
            )
        if topology is None:
            topology = cls._single_topology(ds)

        indexes = {}

        # Collect names
        connectivity = ds.ugrid_roles.connectivity[topology]
        coordinates = ds.ugrid_roles.coordinates[topology]
        ugrid_vars = (
            [topology]
            + list(connectivity.values())
            + list(chain.from_iterable(chain.from_iterable(coordinates.values())))
        )
        fill_value = -1

        # Take the first coordinates by default.
        # They can be reset with .set_node_coords()
        x_index = coordinates["node_coordinates"][0][0]
        y_index = coordinates["node_coordinates"][1][0]
        node_x_coordinates = ds[x_index].astype(FloatDType).values
        node_y_coordinates = ds[y_index].astype(FloatDType).values

        edge_nodes = connectivity["edge_node_connectivity"]
        edge_node_connectivity = cls._prepare_connectivity(
            ds[edge_nodes], fill_value, dtype=IntDType
        ).values

        indexes["node_x"] = x_index
        indexes["node_y"] = y_index
        projected = False  # TODO

        return cls(
            node_x_coordinates,
            node_y_coordinates,
            fill_value,
            edge_node_connectivity,
            name=topology,
            dataset=dataset[ugrid_vars],
            indexes=indexes,
            projected=projected,
            crs=None,
        )

    def _clear_geometry_properties(self):
        """Clear all properties that may have been invalidated"""
        # Meshkernel
        self._mesh = None
        self._meshkernel = None
        # Celltree
        self._celltree = None
        # Bounds
        self._xmin = None
        self._xmax = None
        self._ymin = None
        self._ymax = None
        # Edges
        self._edge_x = None
        self._edge_y = None

    @classmethod
    def from_meshkernel(
        cls,
        mesh,
        name: str = "network1d",
        projected: bool = True,
        crs: Any = None,
    ):
        """
        Create a 1D UGRID topology from a MeshKernel Mesh1d object.

        Parameters
        ----------
        mesh: MeshKernel.Mesh2d
        name: str
            Mesh name. Defaults to "network1d".
        projected: bool
            Whether node_x and node_y are longitude and latitude or projected x and
            y coordinates. Used to write the appropriate standard_name in the
            coordinate attributes.
        crs: Any, optional
            Coordinate Reference System of the geometry objects. Can be anything accepted by
            :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
            such as an authority string (eg "EPSG:4326") or a WKT string.

        Returns
        -------
        grid: Ugrid1d
        """
        return cls(
            mesh.node_x,
            mesh.node_y,
            fill_value=-1,
            edge_node_connectivity=mesh.edge_nodes.reshape((-1, 2)),
            name=name,
            projected=projected,
            crs=crs,
        )

    def to_dataset(
        self, other: xr.Dataset = None, optional_attributes: bool = False
    ) -> xr.Dataset:
        node_x = self._indexes["node_x"]
        node_y = self._indexes["node_y"]
        edge_nodes = self._attrs["edge_node_connectivity"]
        edge_nodes_attrs = conventions.DEFAULT_ATTRS["edge_node_connectivity"]

        data_vars = {
            self.name: 0,
            edge_nodes: xr.DataArray(
                data=self.edge_node_connectivity,
                attrs=edge_nodes_attrs,
                dims=(self.edge_dimension, "two"),
            ),
        }

        attrs = {"Conventions": "CF-1.9 UGRID-1.0"}
        if other is not None:
            attrs.update(other.attrs)

        dataset = xr.Dataset(data_vars, attrs=attrs)
        if self._dataset:
            dataset.update(self._dataset)
        if other is not None:
            dataset = dataset.merge(other)
        if node_x not in dataset or node_y not in dataset:
            dataset = self.assign_node_coords(dataset)
        if optional_attributes:
            dataset = self.assign_edge_coords(dataset)

        dataset[self.name].attrs = self._filtered_attrs(dataset)
        return dataset

    @property
    def topology_dimension(self):
        """Highest dimensionality of the geometric elements: 1"""
        return 1

    @property
    def core_dimension(self):
        return self.node_dimension

    @property
    def dimensions(self):
        return {self.node_dimension: self.n_node, self.edge_dimension: self.n_edge}

    # These are all optional attributes. They are not computed by default, only
    # when called upon.

    @property
    def mesh(self) -> "mk.Mesh1d":  # type: ignore # noqa
        """
        Create if needed, and return meshkernel Mesh1d object.

        Returns
        -------
        mesh: meshkernel.Mesh1d
        """
        import meshkernel as mk

        if self._mesh is None:
            edge_nodes = self.edge_node_connectivity.ravel().astype(np.int32)
            self._mesh = mk.Mesh1d(
                node_x=self.node_x,
                node_y=self.node_y,
                edge_nodes=edge_nodes,
            )
        return self._mesh

    @property
    def meshkernel(self) -> "mk.MeshKernel":  # type: ignore # noqa
        """
        Create if needed, and return meshkernel MeshKernel instance.

        Returns
        -------
        meshkernel: meshkernel.MeshKernel
        """
        import meshkernel as mk

        if self._meshkernel is None:
            self._meshkernel = mk.MeshKernel(is_geographic=False)
            self._meshkernel.mesh1d_set(self.mesh)
        return self._meshkernel

    @staticmethod
    def from_geodataframe(geodataframe: "geopandas.GeoDataFrame") -> "Ugrid1d":  # type: ignore # noqa
        """
        Convert geodataframe of linestrings into a UGRID1D topology.

        Parameters
        ----------
        geodataframe: gpd.GeoDataFrame

        Returns
        -------
        topology: Ugrid1d
        """
        x, y, edge_node_connectivity = conversion.linestrings_to_edges(
            geodataframe.geometry
        )
        fill_value = -1
        return Ugrid1d(x, y, fill_value, edge_node_connectivity, crs=geodataframe.crs)

    def to_pygeos(self, dim):
        from warnings import warn

        warn(
            ".to_pygeos has been deprecated. Use .to_shapely instead.",
            DeprecationWarning,
        )
        return self.to_shapely(dim)

    def to_shapely(self, dim):
        """
        Convert UGRID topology to shapely objects.

        * nodes: points
        * edges: linestrings

        Parameters
        ----------
        dim: str
            Node or edge dimension.

        Returns
        -------
        geometry: ndarray of shapely.Geometry
        """
        if dim == self.node_dimension:
            return conversion.nodes_to_points(
                self.node_x,
                self.node_y,
            )
        elif dim == self.edge_dimension:
            return conversion.edges_to_linestrings(
                self.node_x,
                self.node_y,
                self.edge_node_connectivity,
            )
        else:
            raise ValueError(
                f"Dimension {dim} is not a node or edge dimension of the"
                " Ugrid1d topology."
            )

    def isel(self, indexers=None, return_index=False, **indexers_kwargs):
        """
        Select based on node or edge.

        Edge selection always results in a valid UGRID topology. Node selection
        may result in invalid topologies (incomplete edges), and will error in
        such a case.

        Parameters
        ----------
        indexers: dict of str to np.ndarray of integers or bools
        return_index: bool, optional
            Whether to return node_index, edge_index.

        Returns
        -------
        obj: xr.Dataset or xr.DataArray
        grid: Ugrid2d
        indexes: dict
            Dictionary with keys node dimension, edge dimension and values
            their respective index. Only returned if return_index is True.
        """
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "isel")
        alldims = set(self.dimensions)
        invalid = indexers.keys() - alldims
        if invalid:
            raise ValueError(
                f"Dimensions {invalid} do not exist. Expected one of {alldims}"
            )

        indexers = {
            k: as_pandas_index(v, self.dimensions[k]) for k, v in indexers.items()
        }
        nodedim, edgedim = self.dimensions
        edge_index = {}
        if nodedim in indexers:
            node_index = indexers[nodedim]
            edge_index[nodedim] = np.unique(
                self.node_edge_connectivity[node_index].data
            )
        if edgedim in indexers:
            edge_index[edgedim] = indexers[edgedim]

        # Convert all to pandas index.
        edge_index = {k: as_pandas_index(v, self.n_edge) for k, v in edge_index.items()}

        # Check the indexes against each other.
        index = self._precheck(edge_index)
        grid, finalized_indexers = self.topology_subset(index, return_index=True)
        self._postcheck(indexers, finalized_indexers)

        if return_index:
            return grid, finalized_indexers
        else:
            return grid

    def _validate_indexer(self, indexer) -> Tuple[float, float]:
        if isinstance(indexer, slice):
            if indexer.step is not None:
                raise ValueError("Ugrid1d does not support steps in slices")
            if indexer.start >= indexer.stop:
                raise ValueError("slice start should be smaller than slice stop")
        else:
            raise ValueError("Ugrid1d only supports slice indexing")
        return indexer.start, indexer.stop

    def sel(self, obj, x, y) -> Tuple:
        """
        Select a selection of edges, based on edge centroids.

        Parameters
        ----------
        x: slice
        y: slice

        Returns
        -------
        dimension: str
        as_ugrid: bool
        index: 1d array of integers
        coords: dict
        """
        xmin, xmax = self._validate_indexer(x)
        ymin, ymax = self._validate_indexer(y)
        edge_index = np.nonzero(
            (self.edge_x >= xmin)
            & (self.edge_x < xmax)
            & (self.edge_y >= ymin)
            & (self.edge_y < ymax)
        )[0]
        grid, indexes = self.topology_subset(edge_index, return_index=True)
        indexes = {k: v for k, v in indexes.items() if k in obj.dims}
        new_obj = obj.isel(indexes)
        return new_obj, grid

    def topology_subset(
        self, edge_index: Union[BoolArray, IntArray], return_index: bool = False
    ):
        """
        Create a new UGRID1D topology for a subset of this topology.

        Parameters
        ----------
        edge_index: 1d array of integers or bool
            Edges of the subset.
        return_index: bool, optional
            Whether to return node_index, edge_index.

        Returns
        -------
        subset: Ugrid1d
        indexes: dict
            Dictionary with keys node dimension and edge dimension and values
            their respective index. Only returned if return_index is True.
        """
        if not isinstance(edge_index, pd.Index):
            edge_index = as_pandas_index(edge_index, self.n_edge)

        if edge_index.size == self.n_edge:
            if return_index:
                indexes = {
                    self.node_dimension: pd.RangeIndex(0, self.n_node),
                    self.edge_dimension: pd.RangeIndex(0, self.n_edge),
                }
                return self, indexes
            else:
                return self

        # N.B. edges do not contain fill values, as there are always two nodes
        # required to form an edge.
        edge_subset = self.edge_node_connectivity[edge_index]
        node_index = np.unique(edge_subset.ravel())
        new_edges = connectivity.renumber(edge_subset)
        node_x = self.node_x[node_index]
        node_y = self.node_y[node_index]
        grid = self.__class__(
            node_x,
            node_y,
            self.fill_value,
            new_edges,
            name=self.name,
            indexes=self._indexes,
            projected=self.projected,
            crs=self.crs,
            attrs=self._attrs,
        )
        if return_index:
            indexes = {
                self.node_dimension: pd.Index(node_index),
                self.edge_dimension: edge_index,
            }
            return grid, indexes
        else:
            return grid

    def clip_box(
        self,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
    ):
        return self.sel(x=slice(xmin, xmax), y=slice(ymin, ymax))

    def sel_points(obj, x, y):
        return obj

    def intersect_line(self, obj, start, stop):
        return obj

    def intersect_linestring(self, obj, linestring):
        return obj

    def to_periodic(self, obj):
        return self, obj

    def to_nonperiodic(self, xmax, obj):
        return self, obj

    def topological_sort_by_dfs(self) -> IntArray:
        """
        Returns an array of vertices in topological order.

        Returns
        -------
        sorted_vertices: np.ndarray of integer
        """
        return connectivity.topological_sort_by_dfs(
            self.directed_node_node_connectivity
        )

    def contract_vertices(self, indices: IntArray) -> "Ugrid1d":
        """
        Returns a simplified network topology by removing all nodes that are
        not listed in ``indices``.

        Parameters
        ----------
        indices: np.ndarray of integers

        Returns
        -------
        contracted: Ugrid1d
        """
        edges = connectivity.contract_vertices(
            self.directed_node_node_connectivity, indices
        )
        node_index = np.unique(edges.ravel())
        new_edges = connectivity.renumber(edges)
        return Ugrid1d(
            node_x=self.node_x[node_index],
            node_y=self.node_y[node_index],
            fill_value=self.fill_value,
            edge_node_connectivity=new_edges,
            name=self.name,
            indexes=self._indexes,
            projected=self.projected,
            crs=self.crs,
            attrs=self._attrs,
        )

    @staticmethod
    def merge_partitions(grids: Sequence["Ugrid1d"]) -> "Ugrid1d":
        """
        Merge grid partitions into a single whole.

        Duplicate edges are included only once, and removed from subsequent
        partitions before merging.

        Parameters
        ----------
        grids: sequence of Ugrid1d

        Returns
        -------
        merged: Ugrid1d
        """
        from xugrid.ugrid import partitioning

        # Grab a sample grid
        grid = next(iter(grids))
        fill_value = grid.fill_value
        node_coordinates, node_indexes, node_inverse = partitioning.merge_nodes(grids)
        new_edges, edge_indexes = partitioning.merge_edges(grids, node_inverse)
        indexes = {
            grid.node_dimension: node_indexes,
            grid.edge_dimension: edge_indexes,
        }

        merged_grid = Ugrid1d(
            *node_coordinates.T,
            fill_value,
            new_edges,
            name=grid.name,
            projected=grid.projected,
            crs=grid.crs,
            attrs=grid._attrs,
        )
        return merged_grid, indexes

    def reindex_like(self, other: "Ugrid1d", obj: Union[xr.DataArray, xr.Dataset]):
        """
        Conform a DataArray or Dataset to match the topology of another Ugrid1D
        topology. The topologies must be exactly equivalent: only the order of
        the nodes and edges may differ.

        Parameters
        ----------
        other: Ugrid1d
        obj: DataArray or Dataset

        Returns
        -------
        reindexed: DataArray or Dataset
        """
        if not isinstance(other, Ugrid1d):
            raise TypeError(f"Expected Ugrid1d, received: {type(other).__name__}")

        indexers = {
            self.node_dimension: connectivity.index_like(
                xy_a=self.node_coordinates,
                xy_b=other.node_coordinates,
            ),
            self.edge_dimension: connectivity.index_like(
                xy_a=self.edge_coordinates,
                xy_b=other.edge_coordinates,
            ),
        }
        return obj.isel(indexers, missing_dims="ignore")
