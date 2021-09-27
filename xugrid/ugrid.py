from typing import Tuple, Union

import geopandas as gpd
import matplotlib.tri as mtri
import meshkernel as mk
import numpy as np
import shapely.geometry as sg
import xarray as xr
from numba_celltree import CellTree2d
from meshkernel.meshkernel import MeshKernel
from meshkernel.py_structures import Mesh2d
from scipy.sparse.csr import csr_matrix

from . import connectivity, conversion
from . import meshkernel_utils as mku
from . import ugrid_io
from .typing import FloatArray, IntArray, IntDType
from .voronoi import voronoi_topology


class Ugrid1d:
    """
    Stores the topological data of a "1D unstructured mesh": a collection of
    connected line elements, such as a river network.
    """

    _topology_keys = [
        "edge_coordinates ",
        "edge_dimension",
        "node_coordinates",
        "node_dimension",
        "edge_node_connectivity",
    ]

    def __init__(self, dataset: xr.Dataset):
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
        assert len(mesh_1d_variables) == 1
        self.mesh_topology = mesh_1d_variables[0]

        node_coord_array_names = ugrid_io.get_topology_array_with_role(
            self.mesh_topology, "node_coordinates", variables
        )
        edge_nodes_array_names = ugrid_io.get_topology_array_with_role(
            self.mesh_topology, "edge_node_connectivity", variables
        )
        self.fill_value = -1

        node_coord_arrays = ugrid_io.get_coordinate_arrays_by_name(
            ds, node_coord_array_names
        )
        edge_nodes_array = ugrid_io.get_data_arrays_by_name(ds, edge_nodes_array_names)
        self.edge_nodes = edge_nodes_array[0].astype(connectivity.IntDType)
        self.node_x = node_coord_arrays[0]
        self.node_y = node_coord_arrays[1]
        self.dataset = ds
        self._mesh = None
        self._meshkernel = None

    # These are all optional UGRID attributes. They are not computed by
    # default, only when called upon.
    @property
    def edge_x(self):
        if self._edge_x is None:
            self._edge_x = xr.DatArray(
                data=self.node_x[self.edge_node_connectivity],
                dims=[self.edge_dim],
            )

    @property
    def edge_y(self):
        if self._edge_y is None:
            self._edge_y = xr.DatArray(
                data=self.node_y[self.edge_node_connectivity],
                dims=[self.edge_dim],
            )

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
    def topology_dataset(
        node_x: FloatArray, node_y: FloatArray, edge_node_connectivity: IntArray
    ) -> xr.Dataset:
        return ugrid_io.ugrid1d_dataset(node_x, node_y, edge_node_connectivity)

    def as_vector_geometry(self, data_on: str):
        if data_on == "node":
            return conversion.nodes_to_points(self.node_x, self.node_y)
        elif data_on == "edge":
            return conversion.edges_to_linestrings(
                self.node_x, self.node_y, self.node_edge_connectivity
            )
        else:
            raise ValueError(
                "data_on for Ugrid2d should be one of {node, edge}. "
                f"Received instead {data_on}"
            )

    @staticmethod
    def from_geodataframe(geodataframe: gpd.GeoDataFrame) -> "Ugrid1d":
        coords, edge_node_connectivity = conversion.linestrings_to_edges(
            geodataframe.geometry
        )
        return Ugrid1d(Ugrid1d.topology_dataset(coords, edge_node_connectivity))


class Ugrid2d:
    """
    this class stores the topological data of a 2d unstructured grid.  it
    contains a search tree for search operations, such as finding the face
    index in which a give point lies.  to avoid data duplication, it contains a
    method to remove topological data from a dataset
    """

    _topology_keys = [
        "node_dimension",
        "node_coordinates",
        "max_face_nodes_dimension",
        "edge_node_connectivity",
        "edge_dimension",
        "edge_coordinates ",
        "face_node_connectivity",
        "face_dimension",
        "edge_face_connectivity",
        "face_coordinates",
    ]

    def __init__(self, dataset: xr.Dataset):
        """
        Initialize Ugrid object from the topology data in an xarray Dataset
        """
        ds = dataset
        if not isinstance(ds, xr.Dataset):
            raise TypeError(
                "Ugrid should be initialized with xarray.Dataset. "
                f"Received instead: {type(ds)}"
            )

        # Get topology information variable
        variables = tuple(ds.variables.keys())
        mesh_2d_variables = ugrid_io.get_values_with_attribute(
            ds, "cf_role", "mesh_topology"
        )
        assert len(mesh_2d_variables) == 1
        self.mesh_topology = mesh_2d_variables[0]

        # Collect topology variable names
        node_coord_array_names = ugrid_io.get_topology_array_with_role(
            self.mesh_topology, "node_coordinates", variables
        )
        connectivity_array_names = ugrid_io.get_topology_array_with_role(
            self.mesh_topology, "face_node_connectivity", variables
        )
        edge_nodes_array_names = ugrid_io.get_topology_array_with_role(
            self.mesh_topology, "edge_node_connectivity", variables
        )

        # Extract the values of the arrays, store in object.
        node_coord_arrays = ugrid_io.get_coordinate_arrays_by_name(
            ds, node_coord_array_names
        )
        self.node_x = node_coord_arrays[0].values
        self.node_y = node_coord_arrays[1].values

        face_nodes = ugrid_io.get_data_arrays_by_name(ds, connectivity_array_names)[0]
        self.face_node_connectivity = face_nodes.values.astype(connectivity.IntDType)
        self.fill_value = -1

        # Get dimension names
        self.face_dimension = self.mesh_topology.attrs.get(
            "face_dimension", face_nodes.dims[0]
        )
        self.node_dimension = self.mesh_topology.attrs.get(
            "node_dimension", node_coord_arrays[0].dims[0]
        )

        if len(edge_nodes_array_names) > 0:
            edge_nodes_array = ugrid_io.get_data_arrays_by_name(
                ds, edge_nodes_array_names
            )
            edge_nodes = edge_nodes_array[0]
            self._edge_node_connectivity = edge_nodes.values.astype(
                connectivity.IntDType
            )
            self.edge_dimension = self.mesh_topology.attrs.get(
                "edge_dimension", edge_nodes.dims[0]
            )
        else:
            self._edge_node_connectivity = None
            self.edge_dimension = f"{self.mesh_topology.name}_edge"

        # Misc.
        self.build_celltree()
        self.dataset = ds
        # Optional attributes
        self._init_optional_attrs()

    def remove_topology(self, obj: Union[xr.Dataset, xr.DataArray]):
        """
        removes the grid topology data from a dataset. Use after creating an
        Ugrid object from the dataset
        """
        obj = obj.drop_vars(
            self.mesh_topology.attrs["face_node_connectivity"], errors="ignore"
        )
        x_name, y_name = self.mesh_topology.attrs["node_coordinates"].split()
        obj = obj.drop_vars(x_name, errors="ignore")
        obj = obj.drop_vars(y_name, errors="ignore")
        obj = obj.drop_vars(self.mesh_topology.name, errors="ignore")
        return obj

    def _init_optional_attrs(self):
        # Optional attributes, deferred initialization
        # Meshkernel
        self._mesh = None
        self._meshkernel = None
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
        self._edge_face_connectivity = None
        self._face_face_connectivity = None
        self._face_edge_connectivity = None
        self._node_edge_connectivity = None
        self._node_face_connectivity = None
        # Derived topology
        self._triangulation = None
        self._voronoi_topology = None
        self._centroid_triangulation = None

    @staticmethod
    def from_topology(vertices: FloatArray, face_node_connectivity: IntArray):
        # TODO: switch around with __init__?
        # and call this from_dataset?
        raise NotImplementedError

    def _edge_connectivity(self):
        (
            self._edge_node_connectivity,
            self._face_edge_connectivity,
        ) = connectivity.edge_connectivity(
            self.face_node_connectivity,
            self.fill_value,
        )

    # These are all optional/derived UGRID attributes. They are not computed by
    # default, only when called upon.
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
    def edge_x(self) -> FloatArray:
        if self._edge_x is None:
            self._edge_x = self.node_x[self.edge_node_connectivity]
        return self._edge_x

    @property
    def edge_y(self):
        if self._edge_y is None:
            self._edge_y = self.node_y[self.edge_node_connectivity]
        return self._edge_y

    @property
    def node_edge_connectivity(self) -> csr_matrix:
        if self._node_edge_connectivity is None:
            self._node_edge_connectivity = connectivity.invert_dense_to_sparse(
                self.edge_node_connectivity, self.fill_value
            )
        return self._node_edge_connectivity

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
    def nodes(self) -> FloatArray:
        return np.column_stack([self.node_x, self.node_y])

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
    def voronoi_topology(self):
        if self._voronoi_topology is None:
            vertices, faces, face_index = voronoi_topology(
                self.node_face_connectivity,
                self.nodes,
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
    def bounds(self) -> Tuple[float, float, float, float]:
        if any([
            self._xmin is None,
            self._ymin is None,
            self._xmax is None,
            self._ymax is None,
        ]):
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

    def build_celltree(self):
        """
        initializes the celltree, a search structure for spatial lookups in 2d grids
        """
        nodes = np.column_stack([self.node_x, self.node_y])
        self._cell_tree = CellTree2d(nodes, self.face_node_connectivity, self.fill_value)

    def locate_faces(self, points):
        """
        returns the face indices in which the points lie. The points should be
        provided as an (N,2) array. The result is an (N) array
        """
        if self._cell_tree is None:
            self.build_celltree()
        indices = self._cell_tree.locate(points)
        return indices

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
        index = self._cell_tree.locate_points(nodes).reshape((y.size, x.size))
        return x, y, index

    def locate_cells(points):
        raise NotImplementedError()

    def locate_edges(points):
        raise NotImplementedError()

    @staticmethod
    def topology_dataset(
        node_x: FloatArray, node_y: FloatArray, face_node_connectivity: IntArray
    ) -> xr.Dataset:
        return ugrid_io.ugrid2d_dataset(node_x, node_y, face_node_connectivity)

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
            ds = Ugrid2d.topology_dataset(node_x, node_y, face_node_connectivity)
            return Ugrid2d(ds)

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
        coords, face_node_connectivity = conversion.polygons_to_faces(
            geodataframe.geometry.values
        )
        return Ugrid2d(Ugrid2d.topology_dataset(coords, face_node_connectivity))
