import abc
import copy
from typing import Any, Tuple, Union

import geopandas as gpd
import meshkernel as mk
import numpy as np
import pyproj
import shapely.geometry as sg
import xarray as xr
from meshkernel.meshkernel import MeshKernel
from meshkernel.py_structures import Mesh2d
from numba_celltree import CellTree2d
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse.csr import csr_matrix

from . import connectivity, conversion
from . import meshkernel_utils as mku
from . import ugrid_io
from .typing import FloatArray, IntArray, IntDType
from .voronoi import voronoi_topology


class AbstractUgrid(abc.ABC):
    @abc.abstractproperty
    def topology_dimension(self):
        return

    @abc.abstractstaticmethod
    def from_dataset():
        return

    @abc.abstractmethod
    def topology_dataset(self):
        return

    @abc.abstractmethod
    def topology_subset(self):
        return

    @abc.abstractmethod
    def _clear_geometry_properties(self):
        return

    def copy(self):
        return copy.deepcopy(self)

    @property
    def node_coordinates(self) -> FloatArray:
        return np.column_stack([self.node_x, self.node_y])

    @property
    def edge_x(self):
        if self._edge_x is None:
            self._edge_x = self.node_x[self.edge_node_connectivity].mean(axis=1)
        return self._edge_x

    @property
    def edge_y(self):
        if self._edge_y is None:
            self._edge_y = self.node_y[self.edge_node_connectivity].mean(axis=1)
        return self._edge_y

    @property
    def edge_coordinates(self) -> FloatArray:
        return np.column_stack([self.edge_x, self.edge_y])

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
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
    def node_edge_connectivity(self) -> csr_matrix:
        if self._node_edge_connectivity is None:
            self._node_edge_connectivity = connectivity.invert_dense_to_sparse(
                self.edge_node_connectivity, self.fill_value
            )
        return self._node_edge_connectivity

    def set_crs(self, crs=None, epsg=None, allow_override=False):
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

    def to_crs(
        self,
        crs: Union[pyproj.CRS, str] = None,
        epsg: int = None,
        inplace: bool = False,
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
        inplace : bool, optional, default: False
            Whether to return a new Ugrid or do the transformation in place.
        """
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

        if inplace:
            grid = self
        else:
            grid = self.copy()

        transformer = pyproj.Transformer.from_crs(
            crs=self.crs, crs_to=crs, always_xy=True
        )
        node_x, node_y = transformer.transform(xx=grid.node_x, yy=grid.node_y)
        grid.node_x = node_x
        grid.node_y = node_y
        grid._clear_geometry_properties()
        grid.crs = crs

        if not inplace:
            return grid

    def to_pygeos(self, dim):
        attrs = self.mesh_topology.attrs
        if dim == attrs.get("face_dimension", "face"):
            return conversion.faces_to_polygons(
                self.node_x,
                self.node_y,
                self.face_node_connectivity,
                self.fill_value,
            )
        elif dim == attrs.get("node_dimension", "node"):
            return conversion.nodes_to_points(
                self.node_x,
                self.node_y,
            )
        elif dim == attrs.get("edge_dimension", "edge"):
            return conversion.edges_to_linestrings(
                self.node_x,
                self.node_y,
                self.edge_node_connectivity,
            )
        else:
            raise ValueError(
                f"Dimension {dim} is not a face, node, or edge dimension of the"
                " Ugrid topology."
            )


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
    node_dimension: str, optional, default "node"
    edge_dimension: str, optional, default "edge"
    dataset: xr.Dataset, optional
    crs: Any, optional
        Coordinate Reference System of the geometry objects. Can be anything accepted by
        :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
        such as an authority string (eg "EPSG:4326") or a WKT string.
    """

    def __init__(
        self,
        node_x: FloatArray,
        node_y: FloatArray,
        fill_value: int,
        edge_node_connectivity: IntArray = None,
        node_dimension: str = "node",
        edge_dimension: str = "edge",
        dataset: xr.Dataset = None,
        crs: Any = None,
    ):
        self.node_x = node_x
        self.node_y = node_y
        self.fill_value = fill_value
        self.edge_node_connectivity = edge_node_connectivity
        self.node_dimension = node_dimension
        self.edge_dimension = edge_dimension
        self.dataset = dataset

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
        self._edge_dimension = None
        self._edge_x = None
        self._edge_y = None
        # Connectivity
        self._node_edge_connectivity = None
        # crs
        self.crs = pyproj.CRS.from_user_input(crs)

    @staticmethod
    def from_dataset(dataset: xr.Dataset):
        """
        Extract the 1D UGRID topology information from an xarray Dataset.

        Parameters
        ----------
        dataset: xr.Dataset
            Dataset containing topology information stored according to UGRID conventions.

        Returns
        -------
        grid: Ugrid1d
        """
        ds = dataset
        if not isinstance(ds, xr.Dataset):
            raise TypeError(
                "Ugrid should be initialized with xarray.Dataset. "
                f"Received instead: {type(ds)}"
            )
        variables = tuple(ds.variables.keys())
        mesh_1d_variables = ugrid_io.get_values_with_attribute(
            ds, "cf_role", "mesh_topology"
        )
        if len(mesh_1d_variables) > 1:
            raise ValueError("dataset may only contain a single mesh topology variable")

        mesh_topology = mesh_1d_variables[0]
        node_coord_array_names = ugrid_io.get_topology_array_with_role(
            mesh_topology, "node_coordinates", variables
        )
        edge_nodes_array_names = ugrid_io.get_topology_array_with_role(
            mesh_topology, "edge_node_connectivity", variables
        )

        node_coord_arrays = ugrid_io.get_coordinate_arrays_by_name(
            ds, node_coord_array_names
        )
        edge_nodes_array = ugrid_io.get_data_arrays_by_name(ds, edge_nodes_array_names)
        edge_nodes = edge_nodes_array[0]

        # Get dimension names
        edge_dimension = mesh_topology.attrs.get("edge_dimension", edge_nodes.dims[0])
        node_dimension = mesh_topology.attrs.get(
            "node_dimension", node_coord_arrays[0].dims[0]
        )

        # Collect values
        edge_node_connectivity = edge_nodes.values.astype(IntDType)
        fill_value = -1  # TODO: coerce fill value
        node_x = node_coord_arrays[0].values
        node_y = node_coord_arrays[1].values

        return Ugrid1d(
            node_x,
            node_y,
            fill_value,
            edge_node_connectivity,
            node_dimension,
            edge_dimension,
            dataset,
        )

    def topology_dataset(self):
        """
        Store the UGRID topology information in an xarray Dataset according to
        the UGRID conventions.
        """
        return ugrid_io.ugrid1d_dataset(
            self.node_x,
            self.node_y,
            self.fill_value,
            self.edge_node_connectivity,
            self.node_dimension,
            self.edge_dimension,
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

    # These are all optional UGRID attributes. They are not computed by
    # default, only when called upon.
    @property
    def topology_dimension(self):
        return 1

    @property
    def mesh(self):
        if self._mesh is None:
            self._mesh = mk.Mesh1d(
                node_x=self.node_x,
                node_y=self.node_y,
                edge_nodes=self.edge_nodes.ravel(),
            )
        return self._mesh

    @property
    def meshkernel(self):
        if self._meshkernel is None:
            self._meshkernel = mk.MeshKernel(is_geographic=False)
            self._meshkernel.mesh1d_set(self.mesh)
        return self._meshkernel

    @staticmethod
    def from_geodataframe(geodataframe: gpd.GeoDataFrame) -> "Ugrid1d":
        x, y, edge_node_connectivity, fill_value = conversion.linestrings_to_edges(
            geodataframe.geometry
        )
        return Ugrid1d(x, y, fill_value, edge_node_connectivity)


class Ugrid2d(AbstractUgrid):
    """
    This class stores the topological data of a 2-D unstructured grid.

    Parameters
    ----------
    node_x: ndarray of floats
    node_y: ndarray of floats
    fill_value: int
    face_node_connectivity: ndarray of integers
    edge_node_connectivity: ndarray of integers, optional
    node_dimension: str, optional, default "node"
    face_dimension: str, optional, default "face"
    edge_dimension: str, optional, default "edge"
    dataset: xr.Dataset, optional
    crs: Any, optional
        Coordinate Reference System of the geometry objects. Can be anything accepted by
        :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
        such as an authority string (eg "EPSG:4326") or a WKT string.
    """

    def __init__(
        self,
        node_x: FloatArray,
        node_y: FloatArray,
        fill_value: int,
        face_node_connectivity: IntArray,
        edge_node_connectivity: IntArray = None,
        node_dimension: str = "node",
        face_dimension: str = "face",
        edge_dimension: str = "edge",
        dataset: xr.Dataset = None,
        mesh_topology: str = None,
        crs: Any = None,
    ):
        self.node_x = node_x
        self.node_y = node_y
        self.face_node_connectivity = face_node_connectivity
        self.fill_value = fill_value
        self.node_dimension = node_dimension
        self.face_dimension = face_dimension
        self.edge_dimension = edge_dimension

        # Optional attributes, deferred initialization
        # Meshkernel
        self._mesh = None
        self._meshkernel = None
        # Celltree
        self._celltree = None
        # Centroids
        self._centroids = None
        # Bounds
        self._xmin = None
        self._xmax = None
        self._ymin = None
        self._ymax = None
        # Edges
        self._edge_dimension = None
        self._edge_x = None
        self._edge_y = None
        # Connectivity
        self._edge_node_connectivity = edge_node_connectivity
        self._edge_face_connectivity = None
        self._face_face_connectivity = None
        self._face_edge_connectivity = None
        self._node_edge_connectivity = None
        self._node_face_connectivity = None
        # Derived topology
        self._triangulation = None
        self._voronoi_topology = None
        self._centroid_triangulation = None
        # crs
        if crs is None:
            self.crs = None
        else:
            self.crs = pyproj.CRS.from_user_input(crs)
        # Gather things in a dataset
        if (dataset is None) ^ (mesh_topology is None):
            raise ValueError(
                "Either both dataset and mesh topology must be provided or neither."
            )
        elif dataset is None:
            self.dataset = self.topology_dataset()
            self.mesh_topology = self.dataset["mesh2d"]
        else:
            self.dataset = dataset
            self.mesh_topology = mesh_topology

    def _clear_geometry_properties(self):
        """Clear all properties that may have been invalidated"""
        # Meshkernel
        self._mesh = None
        self._meshkernel = None
        # Celltree
        self._celltree = None
        # Centroids
        self._centroids = None
        # Bounds
        self._xmin = None
        self._xmax = None
        self._ymin = None
        self._ymax = None
        # Edges
        self._edge_x = None
        self._edge_y = None
        # Derived topology
        self._triangulation = None
        self._voronoi_topology = None
        self._centroid_triangulation = None

    @staticmethod
    def from_dataset(dataset: xr.Dataset):
        """
        Extract the 2D UGRID topology information from an xarray Dataset.

        Parameters
        ----------
        dataset: xr.Dataset
            Dataset containing topology information stored according to UGRID conventions.

        Returns
        -------
        grid: Ugrid2d
        """
        ds = dataset
        if not isinstance(ds, xr.Dataset):
            raise TypeError(
                "Ugrid should be initialized with xarray.Dataset. "
                f"Received instead: {type(ds)}"
            )

        # Get topology information variable
        variables = tuple(ds.variables.keys())

        mesh_2d_variables = ugrid_io.get_variable_with_attribute(
            ds, "cf_role", "mesh_topology"
        )
        if len(mesh_2d_variables) > 1:
            raise ValueError("dataset may only contain a single mesh topology variable")
        mesh_topology = mesh_2d_variables[0]

        # Collect names
        node_coord_array_names = ugrid_io.get_topology_array_with_role(
            mesh_topology, "node_coordinates", variables
        )
        connectivity_array_names = ugrid_io.get_topology_array_with_role(
            mesh_topology, "face_node_connectivity", variables
        )
        edge_nodes_array_names = ugrid_io.get_topology_array_with_role(
            mesh_topology, "edge_node_connectivity", variables
        )

        # Collect values
        node_coord_arrays = ugrid_io.get_coordinate_arrays_by_name(
            ds, node_coord_array_names
        )
        node_x = node_coord_arrays[0].values
        node_y = node_coord_arrays[1].values
        face_nodes = ugrid_io.get_data_arrays_by_name(ds, connectivity_array_names)[0]
        face_node_connectivity = face_nodes.values.astype(IntDType)
        fill_value = -1  # TODO: coerce fill_value

        # Get dimension names
        face_dimension = mesh_topology.attrs.get("face_dimension", face_nodes.dims[0])
        node_dimension = mesh_topology.attrs.get(
            "node_dimension", node_coord_arrays[0].dims[0]
        )

        if len(edge_nodes_array_names) > 0:
            edge_nodes_array = ugrid_io.get_data_arrays_by_name(
                ds, edge_nodes_array_names
            )
            edge_nodes = edge_nodes_array[0]
            edge_node_connectivity = edge_nodes.values.astype(connectivity.IntDType)
            edge_dimension = mesh_topology.attrs.get(
                "edge_dimension", edge_nodes.dims[0]
            )
        else:
            edge_node_connectivity = None
            edge_dimension = f"{mesh_topology.name}_edge"

        return Ugrid2d(
            node_x,
            node_y,
            fill_value,
            face_node_connectivity,
            edge_node_connectivity,
            node_dimension,
            face_dimension,
            edge_dimension,
            ds,
            mesh_topology,
        )

    def remove_topology(self, obj: Union[xr.Dataset, xr.DataArray]):
        """
        removes the grid topology data from a dataset. Use after creating an
        Ugrid object from the dataset
        """
        return obj
        obj = obj.drop_vars(
            self.mesh_topology.attrs["face_node_connectivity"], errors="ignore"
        )
        x_name, y_name = self.mesh_topology.attrs["node_coordinates"].split()
        obj = obj.drop_vars(x_name, errors="ignore")
        obj = obj.drop_vars(y_name, errors="ignore")
        obj = obj.drop_vars(self.mesh_topology.name, errors="ignore")
        return obj

    def topology_dataset(self):
        return ugrid_io.ugrid2d_dataset(
            self.node_x,
            self.node_y,
            self.fill_value,
            self.face_node_connectivity,
            self._edge_node_connectivity,
            self.node_dimension,
            self.face_dimension,
            self.edge_dimension,
        )

    # These are all optional/derived UGRID attributes. They are not computed by
    # default, only when called upon.

    @property
    def topology_dimension(self):
        return 2

    def _edge_connectivity(self):
        (
            self._edge_node_connectivity,
            self._face_edge_connectivity,
        ) = connectivity.edge_connectivity(
            self.face_node_connectivity,
            self.fill_value,
        )

    @property
    def edge_node_connectivity(self) -> FloatArray:
        if self._edge_node_connectivity is None:
            self._edge_connectivity()
        return self._edge_node_connectivity

    @property
    def face_edge_connectivity(self) -> FloatArray:
        if self._face_edge_connectivity is None:
            self._edge_connectivity()
        return self._face_edge_connectivity

    @property
    def centroids(self) -> FloatArray:
        if self._centroids is None:
            self._centroids = connectivity.centroids(
                self.face_node_connectivity,
                self.fill_value,
                self.node_x,
                self.node_y,
            )
        return self._centroids

    @property
    def face_x(self):
        return self.centroids[:, 0]

    @property
    def face_y(self):
        return self.centroids[:, 1]

    @property
    def face_coordinates(self) -> FloatArray:
        return self.centroids

    @property
    def edge_face_connectivity(self) -> IntArray:
        if self._edge_face_connectivity is None:
            self._edge_face_connectivity = connectivity.invert_dense(
                self.face_edge_connectivity, self.fill_value
            )
        return self._edge_face_connectivity

    @property
    def face_face_connectivity(self) -> csr_matrix:
        if self._face_face_connectivity is None:
            self._face_face_connectivity = connectivity.face_face_connectivity(
                self.edge_face_connectivity, self.fill_value
            )
        return self._face_face_connectivity

    @property
    def node_face_connectivity(self):
        if self._node_face_connectivity is None:
            self._node_face_connectivity = connectivity.invert_dense_to_sparse(
                self.face_node_connectivity, self.fill_value
            )
        return self._node_face_connectivity

    @property
    def mesh(self):
        if self._mesh is None:
            self._mesh = mk.Mesh1d(
                node_x=self.node_x,
                node_y=self.node_y,
                edge_nodes=self.edge_nodes.ravel(),
            )
        return self._mesh

    @property
    def meshkernel(self) -> mk.MeshKernel:
        if self._meshkernel is None:
            self._meshkernel = mk.MeshKernel(is_geographic=False)
            self._meshkernel.mesh2d_set(self.mesh)
        return self._meshkernel

    @property
    def voronoi_topology(self):
        if self._voronoi_topology is None:
            vertices, faces, face_index = voronoi_topology(
                self.node_face_connectivity,
                self.node_coordinates,
                self.centroids,
            )
            self._voronoi_topology = vertices, faces, face_index
        return self._voronoi_topology

    @property
    def centroid_triangulation(self):
        if self._centroid_triangulation is None:
            nodes, faces, face_index = self.voronoi_topology
            triangles, _ = connectivity.triangulate(faces, self.fill_value)
            triangulation = (nodes[:, 0].copy(), nodes[:, 1].copy(), triangles)
            self._centroid_triangulation = (triangulation, face_index)
        return self._centroid_triangulation

    @property
    def triangulation(self):
        if self._triangulation is None:
            triangles, triangle_face_connectivity = connectivity.triangulate(
                self.face_node_connectivity, self.fill_value
            )
            triangulation = (self.node_x, self.node_y, triangles)
            self._triangulation = (triangulation, triangle_face_connectivity)
        return self._triangulation

    @property
    def exterior_edges(self) -> IntArray:
        """
        Get all exterior edges, i.e. edges with no other face.
        """
        return np.argwhere(self.edge_face_connectivity[:, 1] == self.fill_value)

    @property
    def exterior_faces(self) -> IntArray:
        """
        Get all exterior faces, i.e. faces with an unshared edge.
        """
        exterior_edges = self.exterior_edges
        exterior_faces = self.edge_face_connectivity[exterior_edges].ravel()
        return np.unique(exterior_faces[exterior_faces != self.fill_value])

    @property
    def celltree(self):
        """
        initializes the celltree, a search structure for spatial lookups in 2d grids
        """
        if self._celltree is None:
            self._celltree = CellTree2d(
                self.node_coordinates, self.face_node_connectivity, self.fill_value
            )
        return self._celltree

    def locate_faces(self, points):
        """
        returns the face indices in which the points lie. The points should be
        provided as an (N,2) array. The result is an (N) array
        """
        return self.celltree.locate_points(points)

    def locate_faces_bounding_box(self, xmin, xmax, ymin, ymax):
        """
        given a rectangular bounding box, this function returns the face
        indices which lie in it.
        """
        x = self.node_x
        y = self.node_y
        nodemask = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
        node_face_connectivity = connectivity.invert_dense_to_sparse(
            self.face_node_connectivity, self.fill_value
        )
        selected_faces = np.unique(node_face_connectivity[nodemask].indices)
        return selected_faces

    def locate_faces_polygon(self, polygon: sg.Polygon):
        geometry_list = mku.to_geometry_list(polygon)
        node_indices = self._meshkernel.mesh2d_get_nodes_in_polygons(
            geometry_list, inside=True
        )
        node_face_connectivity = connectivity.invert_dense_to_sparse(
            self.face_node_connectivity, self.fill_value
        )
        selected_faces = np.unique(node_face_connectivity[node_indices].indices)
        return selected_faces

    def rasterize(self, resolution: float) -> Tuple[FloatArray, FloatArray, IntArray]:
        xmin, ymin, xmax, ymax = self.bounds
        d = abs(resolution)
        xmin = np.floor(xmin / d) * d
        xmax = np.ceil(xmax / d) * d
        ymin = np.floor(ymin / d) * d
        ymax = np.ceil(ymax / d) * d
        x = np.arange(xmin + 0.5 * d, xmax, d)
        y = np.arange(ymax - 0.5 * d, ymin, -d)
        yy, xx = np.meshgrid(y, x, indexing="ij")
        nodes = np.column_stack([xx.ravel(), yy.ravel()])
        index = self.celltree.locate_points(nodes).reshape((y.size, x.size))
        return x, y, index

    def as_vector_geometry(self, data_on: str):
        if data_on == "node":
            return conversion.nodes_to_points(self.node_x, self.node_y)
        elif data_on == "edge":
            return conversion.edges_to_linestrings(
                self.node_x, self.node_y, self.node_edge_connectivity
            )
        elif data_on == "face":
            return conversion.faces_to_polygons(
                self.node_x, self.node_y, self.face_edge_connectivity
            )
        else:
            raise ValueError(
                "data_on for Ugrid2d should be one of {node, edge, face}. "
                f"Received instead {data_on}"
            )

    def topology_subset(self, face_indices: IntArray):
        """
        Create a valid ugrid dataset describing the topology of a subset of
        this unstructured mesh topology.
        """
        # If faces are repeated: not a valid mesh
        # _, count = np.unique(face_indices, return_counts=True)
        # assert count.max() <= 1?
        # If no faces are repeated, and size is the same, it's the same mesh
        if face_indices.size == len(self.face_node_connectivity):
            return self
        # Subset of faces, create new topology data
        else:
            face_selection = self.face_node_connectivity[face_indices]
            node_indices = np.unique(face_selection.ravel())
            face_node_connectivity = connectivity.renumber(face_selection)
            node_x = self.node_x[node_indices]
            node_y = self.node_y[node_indices]
            return Ugrid2d(node_x, node_y, self.fill_value, face_node_connectivity)

    def triangulate(self):
        triangles, _ = connectivity.triangulate(
            self.face_node_connectivity, self.fill_value
        )
        return Ugrid2d(self.node_x, self.node_y, self.fill_value, triangles)

    def tesselate_centroidal_voronoi(self, add_exterior=True, add_vertices=True):
        if add_exterior:
            edge_face_connectivity = self.edge_face_connectivity
            edge_node_connectivity = self.edge_node_connectivity
        else:
            edge_face_connectivity = None
            edge_node_connectivity = None

        vertices, faces, _ = voronoi_topology(
            self.node_face_connectivity,
            self.node_coordinates,
            self.centroids,
            edge_face_connectivity,
            edge_node_connectivity,
            self.fill_value,
            add_exterior,
            add_vertices,
        )
        faces = connectivity.to_dense(faces, self.fill_value)
        return Ugrid2d(vertices[:, 0], vertices[:, 1], self.fill_value, faces)

    def reverse_cuthill_mckee(self, dimension=None):
        # Todo: dispatch on dimension
        reordering = reverse_cuthill_mckee(
            graph=self.face_face_connectivity,
            symmetric_mode=True,
        )
        reordered_grid = Ugrid2d(
            self.node_x,
            self.node_y,
            self.fill_value,
            self.face_node_connectivity[reordering],
        )
        return reordered_grid, reordering

    def refine_polygon(
        self,
        polygon: sg.Polygon,
        min_face_size: float,
        refine_intersected: bool = True,
        use_mass_center_when_refining: bool = True,
        refinement_type: str = "refinement_levels",
        connect_hanging_nodes: bool = True,
        account_for_samples_outside_face: bool = True,
        max_refinement_iterations: int = 10,
    ):
        geometry_list = mku.to_geometry_list(polygon)
        refinement_type = mku.either_string_or_enum(refinement_type, mk.RefinementType)

        self._initialize_mesh_kernel()
        mesh_refinement_params = mk.MeshRefinementParameters(
            refine_intersected,
            use_mass_center_when_refining,
            min_face_size,
            refinement_type,
            connect_hanging_nodes,
            account_for_samples_outside_face,
            max_refinement_iterations,
        )
        self._meshkernel.mesh2d_refine_based_on_polygon(
            geometry_list,
            mesh_refinement_params,
        )

    def delete_polygon(
        self,
        polygon: sg.Polygon,
        delete_option: str = "all_face_circumenters",
        invert_deletion: bool = False,
    ):
        geometry_list = mku.to_geometry_list(polygon)
        delete_option = mku.either_string_or_enum(delete_option, mk.DeleteMeshOption)
        self._initialize_mesh_kernel()
        self._meshkernel.mesh2d_delete(geometry_list, delete_option, invert_deletion)

    @staticmethod
    def from_polygon(
        polygon: sg.Polygon,
    ):
        geometry_list = mku.to_geometry_list(polygon)
        _mesh_kernel = mk.MeshKernel()
        _mesh_kernel.mesh2d_make_mesh_from_polygon(geometry_list)
        mesh = _mesh_kernel.mesh2d_get()
        n_max_node = mesh.nodes_per_face.max()
        ds = Ugrid2d.topology_dataset(
            mesh.node_x,
            mesh.node_y,
            mesh.face_nodes.reshape((-1, n_max_node)),
        )
        ugrid = Ugrid2d(ds)
        ugrid.mesh = mesh
        ugrid._meshkernel = _mesh_kernel
        return ugrid

    @staticmethod
    def from_geodataframe(geodataframe: gpd.GeoDataFrame):
        x, y, face_node_connectivity, fill_value = conversion.polygons_to_faces(
            geodataframe.geometry.values
        )
        return Ugrid2d(x, y, fill_value, face_node_connectivity)

    @staticmethod
    def from_structured(data: Union[xr.DataArray, xr.Dataset]) -> "Ugrid2d":
        """
        Derive the 2D-UGRID quadrilateral mesh topology from a structured DataArray
        or Dataset, with (2D-dimensions) "y" and "x".

        Parameters
        ----------
        data: Union[xr.DataArray, xr.Dataset]
            Structured data from which the "x" and "y" coordinate will be used to
            define the UGRID-2D topology.

        Returns
        -------
        ugrid_topology: Ugrid2d
        """

        def _coord(da, dim):
            """
            Transform N xarray midpoints into N + 1 vertex edges
            """
            dxs = np.diff(da[dim].values)
            dx = dxs[0]
            atolx = abs(1.0e-4 * dx)
            if not np.allclose(dxs, dx, atolx):
                raise ValueError(
                    f"DataArray has to be equidistant along {dim}, or cellsizes"
                    " must be provided as a coordinate."
                )
            dxs = np.full(da[dim].size, dx)
            dxs = np.abs(dxs)
            x = da[dim].values
            if not da.indexes[dim].is_monotonic_increasing:
                x = x[::-1]
                dxs = dxs[::-1]
            # This assumes the coordinate to be monotonic increasing
            x0 = x[0] - 0.5 * dxs[0]
            x = np.full(dxs.size + 1, x0)
            x[1:] += np.cumsum(dxs)
            return x

        # Transform midpoints into vertices
        # These are always returned monotonically increasing
        xcoord = _coord(data, "x")
        ycoord = _coord(data, "y")
        # Compute all vertices, these are the ugrid nodes
        node_y, node_x = (a.ravel() for a in np.meshgrid(ycoord, xcoord, indexing="ij"))
        linear_index = np.arange(node_x.size, dtype=np.int).reshape(
            ycoord.size, xcoord.size
        )
        # Allocate face_node_connectivity
        nfaces = (ycoord.size - 1) * (xcoord.size - 1)
        face_nodes = np.empty((nfaces, 4))
        # Set connectivity in counterclockwise manner
        face_nodes[:, 0] = linear_index[:-1, 1:].ravel()  # upper right
        face_nodes[:, 1] = linear_index[:-1, :-1].ravel()  # upper left
        face_nodes[:, 2] = linear_index[1:, :-1].ravel()  # lower left
        face_nodes[:, 3] = linear_index[1:, 1:].ravel()  # lower right
        return Ugrid2d(node_x, node_y, -1, face_nodes)


def grid_from_geodataframe(geodataframe):
    gdf = geodataframe
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError(f"Cannot convert a {type(gdf)}, expected a GeoDataFrame")

    geom_types = gdf.geom_type.unique()
    if len(geom_types) == 0:
        raise ValueError("geodataframe contains no geometry")
    elif len(geom_types) > 1:
        message = ", ".join(geom_types)
        raise ValueError(f"Multiple geometry types detected: {message}")

    geom_type = geom_types[0]
    if geom_type == "Linestring":
        grid = Ugrid1d.from_geodataframe(gdf)
    elif geom_type == "Polygon":
        grid = Ugrid2d.from_geodataframe(gdf)
    else:
        raise ValueError(
            f"Invalid geometry type: {geom_type}. Expected Linestring or Polygon."
        )
    return grid
