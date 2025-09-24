from itertools import chain
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
from numba_celltree import EdgeCellTree2d
from numpy.typing import ArrayLike
from scipy import sparse

import xugrid
from xugrid import conversion
from xugrid.constants import (
    FILL_VALUE,
    BoolArray,
    FloatArray,
    FloatDType,
    IntArray,
    IntDType,
    LineArray,
)
from xugrid.core.utils import either_dict_or_kwargs
from xugrid.regrid.utils.array import alt_cumsum
from xugrid.ugrid import connectivity, conventions
from xugrid.ugrid.selection_utils import section_coordinates_1d
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
    start_index: int, 0 or 1, default is 0.
        Start index of the connectivity arrays. Must match the start index
        of the provided face_node_connectivity and edge_node_connectivity.
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
        start_index: int = 0,
    ):
        self.node_x = np.ascontiguousarray(node_x)
        self.node_y = np.ascontiguousarray(node_y)
        self.fill_value = fill_value
        self.start_index = start_index
        self.edge_node_connectivity = edge_node_connectivity - self.start_index
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
        self._node_kdtree = None
        self._edge_kdtree = None
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

        # Take the first coordinates by default.
        # They can be reset with .set_node_coords()
        x_index = coordinates["node_coordinates"][0][0]
        y_index = coordinates["node_coordinates"][1][0]
        node_x_coordinates = ds[x_index].astype(FloatDType).to_numpy()
        node_y_coordinates = ds[y_index].astype(FloatDType).to_numpy()

        edge_nodes = connectivity["edge_node_connectivity"]
        fill_value = ds[edge_nodes].encoding.get("_FillValue", -1)
        start_index = ds[edge_nodes].attrs.get("start_index", 0)
        edge_node_connectivity = cls._prepare_connectivity(
            ds[edge_nodes], fill_value, dtype=IntDType
        ).to_numpy()

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
            start_index=start_index,
        )

    def _clear_geometry_properties(self):
        """Clear all properties that may have been invalidated"""
        # Meshkernel
        self._mesh = None
        self._meshkernel = None
        # Celltree
        self._celltree = None
        self._node_kdtree = None
        self._edge_kdtree = None
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
            fill_value=FILL_VALUE,
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
                data=self._adjust_connectivity(self.edge_node_connectivity),
                attrs=edge_nodes_attrs,
                dims=(self.edge_dimension, "two"),
            ),
        }

        attrs = {"Conventions": "CF-1.9 UGRID-1.0"}
        if other is not None:
            attrs.update(other.attrs)

        dataset = xr.Dataset(data_vars, attrs=attrs)
        if self._dataset:
            dataset = dataset.merge(self._dataset, compat="override")
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
        return self.edge_dimension

    @property
    def dims(self):
        """Set of UGRID dimension names: node dimension, edge dimension."""
        return {self.node_dimension, self.edge_dimension}

    @property
    def sizes(self):
        return {self.node_dimension: self.n_node, self.edge_dimension: self.n_edge}

    @property
    def coords(self):
        """Dictionary for grid coordinates."""
        return {
            self.node_dimension: self.node_coordinates,
            self.edge_dimension: self.edge_coordinates,
        }

    def get_coordinates(self, dim: str) -> FloatArray:
        """Return the coordinates for the specified UGRID dimension."""
        if dim == self.node_dimension:
            return self.node_coordinates
        elif dim == self.edge_dimension:
            return self.edge_coordinates
        else:
            raise ValueError(
                f"Expected {self.node_dimension} or {self.edge_dimension}; got: {dim}"
            )

    @property
    def facets(self) -> dict[str, str]:
        return {
            "node": self.node_dimension,
            "edge": self.edge_dimension,
        }

    def get_connectivity_matrix(self, dim: str, xy_weights: bool):
        """Return the connectivity matrix for the specified UGRID dimension."""
        if dim == self.node_dimension:
            connectivity = self.node_node_connectivity.copy()
            coordinates = self.node_coordinates
        else:
            raise ValueError(f"Expected {self.node_dimension}; got: {dim}")

        if xy_weights:
            connectivity.data = self._connectivity_weights(connectivity, coordinates)

        return connectivity

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
            if self.is_geographic:
                mk_projection = mk.ProjectionType.SPHERICAL
            else:
                mk_projection = mk.ProjectionType.CARTESIAN
            self._meshkernel = mk.MeshKernel(mk_projection)
            self._meshkernel.mesh1d_set(self.mesh)
        return self._meshkernel

    @property
    def is_cyclic(self) -> bool:
        """
        Return True if the directed node node connectivity contains cycles. If
        False, then the directed node node connectivity is a directed acyclic
        graph (DAG).

        Runs a depth-first-search.
        """
        # topological sort returns a ValueError if it detects a cycle.
        try:
            self.topological_sort_by_dfs()
            return False
        except ValueError as e:
            if str(e) == "The graph contains at least one cycle":
                return True
            raise  # Re-raise any other ValueError

    @classmethod
    def from_geodataframe(cls, geodataframe: "geopandas.GeoDataFrame") -> "Ugrid1d":  # type: ignore # noqa
        """
        Convert geodataframe of linestrings into a UGRID1D topology.

        Parameters
        ----------
        geodataframe: geopandas GeoDataFrame

        Returns
        -------
        topology: Ugrid1d
        """
        import geopandas as gpd

        if not isinstance(geodataframe, gpd.GeoDataFrame):
            raise TypeError(
                f"Expected GeoDataFrame, received: {type(geodataframe).__name__}"
            )
        return cls.from_shapely(geodataframe.geometry.to_numpy(), crs=geodataframe.crs)

    @staticmethod
    def from_shapely(geometry: LineArray, crs=None) -> "Ugrid1d":
        """
        Convert an array of shapely linestrings to UGRID1D topology.

        Parameters
        ----------
        geometry: np.ndarray of shapely linestrings
        crs: Any, optional
            Coordinate Reference System of the geometry objects. Can be anything accepted by
            :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
            such as an authority string (eg "EPSG:4326") or a WKT string.
        """

        import shapely

        if not (shapely.get_type_id(geometry) == shapely.GeometryType.LINESTRING).all():
            raise TypeError(
                "Can only create Ugrid1d from shapely LineString geometries, "
                "geometry contains other types of geometries."
            )

        x, y, edge_node_connectivity = conversion.linestrings_to_edges(geometry)
        return Ugrid1d(x, y, FILL_VALUE, edge_node_connectivity, crs=crs)

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
        alldims = self.dims
        invalid = indexers.keys() - alldims
        if invalid:
            raise ValueError(
                f"Dimensions {invalid} do not exist. Expected one of {alldims}"
            )

        indexers = {k: as_pandas_index(v, self.sizes[k]) for k, v in indexers.items()}
        nodedim = self.node_dimension
        edgedim = self.edge_dimension

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

        range_index = pd.RangeIndex(0, self.n_edge)
        if edge_index.size == self.n_edge and edge_index.equals(range_index):
            if return_index:
                indexes = {
                    self.node_dimension: pd.RangeIndex(0, self.n_node),
                    self.edge_dimension: range_index,
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
        grid = Ugrid1d(
            node_x,
            node_y,
            FILL_VALUE,
            new_edges,
            name=self.name,
            indexes=self._indexes,
            projected=self.projected,
            crs=self.crs,
            attrs=self._attrs,
        )
        self._propagate_properties(grid)
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

    @property
    def celltree(self) -> EdgeCellTree2d:
        """
        Initializes the celltree if needed, and returns celltree.

        A celltree is a search structure for spatial lookups in unstructured grids.
        """
        if self._celltree is None:
            self._celltree = EdgeCellTree2d(
                self.node_coordinates,
                self.edge_node_connectivity,
            )
        return self._celltree

    @staticmethod
    def _section_coordinates(
        edges: FloatArray, xy: FloatArray, dim: str, index: IntArray, name: str
    ):
        return section_coordinates_1d(edges, xy, dim, index, name)

    def to_periodic(self, obj):
        return self, obj

    def to_nonperiodic(self, xmax, obj):
        return self, obj

    def topological_sort_by_dfs(self) -> IntArray:
        """
        Return an array of vertices in topological order.

        Returns
        -------
        sorted_vertices: np.ndarray of integer
        """
        return connectivity.topological_sort_by_dfs(
            self.directed_node_node_connectivity
        )

    def remove_self_loops(self) -> "Ugrid1d":
        """
        Remove degenerate edges: those that join a node to itself, also called
        a self-loop. These edges have a length of exactly zero.

        Returns
        -------
        grid: Ugrid1d
        """
        a, b = self.edge_node_connectivity.T
        not_self_loop = a != b
        edge_subset = self.edge_node_connectivity[not_self_loop]
        valid = np.bincount(edge_subset.ravel()) > 0
        new_edges = connectivity.renumber(edge_subset)
        return Ugrid1d(
            node_x=self.node_x[valid],
            node_y=self.node_y[valid],
            fill_value=self.fill_value,
            edge_node_connectivity=new_edges,
            name=self.name,
            indexes=self._indexes,
            projected=self.projected,
            crs=self.crs,
            attrs=self._attrs,
        )

    def contract_vertices(self, indices: IntArray) -> "Ugrid1d":
        """
        Return a simplified network topology by removing all nodes that are
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

    def refine_by_vertices(
        self,
        vertices: FloatArray,
        return_index: bool = False,
        tolerance: Optional[float] = None,
    ) -> "Ugrid1d":
        """
        Refine Ugrid1d with extra vertices to be inserted and returns new grid.
        Vertices need to be located on existing grid edges, if not, a ValueError
        will be returned.

        Parameters
        ----------
        vertices: np.ndarray of floats
            Coordinates of vertices to be inserted in the grid. Must have shape
            (N, 2).
        return_index: bool, optional
            If set to to True, the index of the new vertices in the grid will be
            returned. Defaults to False.
        tolerance: float, optional
            The tolerance used to determine whether a point is on an edge. This
            accounts for the inherent inexactness of floating point calculations.
            If None, an appropriate tolerance is automatically estimated based on
            the geometry size. Consider adjusting this value if edge detection
            results are unsatisfactory.

        Returns
        -------
        grid: Ugrid1d
            Refined grid with new vertices.

        Examples
        --------
        Let's first create a simple grid with 3 nodes and 2 edges:

        >>> import numpy as np
        >>> import xugrid as xu
        >>> node_xy = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 5.0]])
        >>> edge_nodes = np.array([[0, 1],[1, 2]])
        >>> grid = xu.Ugrid1d(*node_xy.T, -1, edge_nodes)

        Now refine the grid by adding new vertices:

        >>> vertices = np.array([[2.0, 2.0], [7.0, 5.0]])
        >>> new = grid.refine_by_vertices(vertices)
        >>> print(new.node_coordinates)

        To return the indices of the inserted vertices:

        >>> new, new_vertices_index = grid.refine_by_vertices(vertices, return_index=True)
        >>> print(new_vertices_index)
        >>> print(new.node_coordinates[new_vertices_index])

        """
        edge_index = self.celltree.locate_points(vertices, tolerance)
        invalid = edge_index == -1
        if invalid.any():
            raise ValueError(
                f"The following vertices are not located on any edge:\n{vertices[invalid]}"
            )

        # Do not insert vertices that are already present in the grid.
        node_xy = self.node_coordinates
        combined = np.concatenate((node_xy, vertices))
        _, index, inverse = np.unique(
            combined, return_index=True, return_inverse=True, axis=0
        )
        index_to_vertices = index[inverse][self.n_node :]
        not_duplicated = index_to_vertices >= self.n_node
        new_vertices = vertices[not_duplicated]
        edge_index = edge_index[not_duplicated]

        first_node = self.edge_node_connectivity[edge_index, 0]
        distance = np.linalg.norm(new_vertices - node_xy[first_node], axis=1)
        grid_edge_index = np.arange(self.n_edge)
        repeats = np.bincount(np.concatenate((grid_edge_index, edge_index)))
        new_edges = np.repeat(self.edge_node_connectivity, repeats, axis=0)
        order = np.lexsort((distance, edge_index))

        # Index for new vertices
        node_index = np.arange(self.n_node, self.n_node + len(edge_index))[order]

        # For the new edges, modify:
        #
        # * all but the last entry per edge of the second column
        # * all but the first entry per edge of the first column
        #
        i = np.arange(len(new_edges))
        mask0 = np.repeat(alt_cumsum(repeats), repeats)
        mask1 = np.repeat(np.cumsum(repeats), repeats) - 1
        new_edges[i > mask0, 0] = node_index
        new_edges[i < mask1, 1] = node_index

        grid = Ugrid1d(
            np.concatenate((self.node_x, new_vertices[:, 0])),
            np.concatenate((self.node_y, new_vertices[:, 1])),
            self.fill_value,
            new_edges,
            name=self.name,
            projected=self.projected,
            crs=self.crs,
        )
        self._propagate_properties(grid)
        if return_index:
            return grid, node_index
        else:
            return grid

    def _nearest_interpolate(
        self,
        data: FloatArray,
        ugrid_dim: str,
        max_distance: float,
    ) -> FloatArray:
        isnull = np.isnan(data)
        if isnull.all():
            raise ValueError("All values are NA.")

        edge_length = self.edge_length
        if ugrid_dim == self.node_dimension:
            # Set the edge length as the graph weights.
            # We can do this easily since the data of the node_node_connectivity CSR matrix
            # contains the edge through which the connection is made.
            connectivity = self.node_node_connectivity.copy()
            connectivity.data = edge_length[connectivity.data]
        elif ugrid_dim == self.edge_dimension:
            # Convert to COO-form so we can index with rows and columns.
            connectivity = self.edge_edge_connectivity.tocoo()
            # Compute distance from edge centroid to centroid along the edges.
            # I.e. half of both edge lengths
            connectivity.data = 0.5 * (
                edge_length[connectivity.row] + edge_length[connectivity.col]
            )
        else:
            raise ValueError(
                f"Expected {self.node_dimension} or {self.edge_dimension}, "
                f"received instead: {ugrid_dim}"
            )
        _, _, index = sparse.csgraph.dijkstra(
            csgraph=connectivity,
            indices=np.flatnonzero(~isnull),
            return_predecessors=True,
            limit=max_distance,
            min_only=True,
        )
        # dijkstra returns -9999 when no path could be found.
        # There, we'll keep the NaN value.
        found = index != -9999
        index = index[found]
        out = data.copy()
        out[found] = data[index]
        return out

    @staticmethod
    def merge_partitions(
        grids: Sequence["Ugrid1d"],
    ) -> tuple["Ugrid1d", dict[str, np.array]]:
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
            indexes=grid._indexes,
            projected=grid.projected,
            crs=grid.crs,
            attrs=grid._attrs,
        )
        return merged_grid, indexes

    def reindex_like(
        self,
        other: "Ugrid1d",
        obj: Union[xr.DataArray, xr.Dataset],
        tolerance: float = 0.0,
    ):
        """
        Conform a DataArray or Dataset to match the topology of another Ugrid1D
        topology. The topologies must be exactly equivalent: only the order of
        the nodes and edges may differ.

        Parameters
        ----------
        other: Ugrid1d
        obj: DataArray or Dataset
        tolerance: float, default value 0.0.
            Maximum distance between inexact coordinate matches.

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
                tolerance=tolerance,
            ),
            self.edge_dimension: connectivity.index_like(
                xy_a=self.edge_coordinates,
                xy_b=other.edge_coordinates,
                tolerance=tolerance,
            ),
        }
        return obj.isel(indexers, missing_dims="ignore")

    def create_data_array(self, data: ArrayLike, facet: str) -> "xugrid.UgridDataArray":
        """
        Create a UgridDataArray from this grid and a 1D array of values.

        Parameters
        ----------
        data: array like
            Values for this array. Must be a ``numpy.ndarray`` or castable to
            it.
        grid: Ugrid1d, Ugrid2d
        facet: str
            With which facet to associate the data. Options for Ugrid1d are,
            ``"node"`` or ``"edge"``. Options for Ugrid2d are ``"node"``,
            ``"edge"``, or ``"face"``.

        Returns
        -------
        uda: UgridDataArray
        """
        if facet == "node":
            dimension = self.node_dimension
        elif facet == "edge":
            dimension = self.edge_dimension
        else:
            raise ValueError(f"Invalid facet: {facet}. Must be one of: node, edge.")
        return self._create_data_array(data, dimension)
