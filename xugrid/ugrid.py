from typing import Union

import matplotlib.tri as mtri
import meshkernel as mk
import numpy as np
import shapely.geometry as sg
import xarray as xr
from cell_tree2d import CellTree
from meshkernel.meshkernel import MeshKernel
from meshkernel.py_structures import Mesh2d
from scipy.sparse.csr import csr_matrix

from . import connectivity
from . import conversion
from . import meshkernel_utils as mku
from . import ugrid_io
from .typing import IntArray, FloatArray, IntDType


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
        mesh_1d_variables = self._get_values_with_attribute(
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
                data=self.node_x.values[self.edge_node_connectivity.values],
                dims=[self.edge_dim],
            )

    @property
    def edge_y(self):
        if self._edge_y is None:
            self._edge_y = xr.DatArray(
                data=self.node_y.values[self.edge_node_connectivity.values],
                dims=[self.edge_dim],
            )

    @property
    def mesh(self):
        if self._mesh is None:
            self._mesh = mk.Mesh1d(
                node_x=self.node_x.values,
                node_y=self.node_y.values,
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
            return conversion.edges_to_linestrings(self.node_x, self.node_y, self.node_edge_connectivity)
        else:
            raise ValueError(
                "data_on for Ugrid2d should be one of {node, edge}. "
                f"Received instead {data_on}"
            )

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
        variables = tuple(ds.variables.keys())
        mesh_2d_variables = self._get_values_with_attribute(
            ds, "cf_role", "mesh_topology"
        )
        assert len(mesh_2d_variables) == 1
        self.mesh_topology = mesh_2d_variables[0]

        node_coord_array_names = ugrid_io.get_topology_array_with_role(
            self.mesh_topology, "node_coordinates", variables
        )
        connectivity_array_names = ugrid_io.get_topology_array_with_role(
            self.mesh_topology, "face_node_connectivity", variables
        )
        edge_nodes_array_names = ugrid_io.get_topology_array_with_role(
            self.mesh_topology, "edge_node_connectivity", variables
        )
        self.fill_value = -1

        connectivity_array = ugrid_io.get_data_arrays_by_name(
            ds, connectivity_array_names
        )
        if len(edge_nodes_array_names) > 0:
            edge_nodes_array = ugrid_io.get_data_arrays_by_name(
                ds, edge_nodes_array_names
            )
        else:
            edge_nodes_array = []
        node_coord_arrays = ugrid_io.get_coordinate_arrays_by_name(
            ds, node_coord_array_names
        )

        self.connectivity_array = connectivity_array[0].astype(connectivity.IntDType)
        self.edge_nodes = edge_nodes_array[0].astype(connectivity.IntDType)
        self.node_x = node_coord_arrays[0]
        self.node_y = node_coord_arrays[1]
        self.build_celltree()
        self.dataset = ds
        self._mesh = None
        self._meshkernel = None

    # These are all optional/derived UGRID attributes. They are not computed by
    # default, only when called upon.
    @property
    def edge_dimension(self) -> str:
        if self._edge_dimension is None:
            self._edge_dimension = f"{self.mesh_topology.name()}_edge"
        return self._edge_dimension

    @property
    def edge_node_connectivity(self) -> xr.DataArray:
        if self._edge_node_connectivity is None:
            self._edge_node_connectivity = xr.DataArray(
                data=connectivity.edge_node_connectivity(
                    self.face_node_connectivity.values,
                    self.fill_value,
                ),
                dims=[self.edge_dimension, 2],
            )
        return self._edge_node_connectivity

    @property
    def edge_x(self) -> xr.DataArray:
        if self._edge_x is None:
            self._edge_x = xr.DatArray(
                data=self.node_x.values[self.edge_node_connectivity.values],
                dims=[self.edge_dim],
            )

    @property
    def edge_y(self):
        if self._edge_y is None:
            self._edge_y = xr.DatArray(
                data=self.node_y.values[self.edge_node_connectivity.values],
                dims=[self.edge_dim],
            )

    @property
    def node_edge_connectivity(self) -> csr_matrix:
        if self._node_edge_connectivity is None:
            self._node_edge_connectivity = connectivity.invert_dense_to_sparse(
                self.edge_face_connectivity.values, self.fill_value
            )
        return self._node_edge_connectivity

    @property
    def face_edge_connectivity(self) -> csr_matrix:
        if self.face_edge_connectivity is None:
            self._face_edge_connectivity = connectivity.face_edge_connectivity(
                self.face_node_connectivity.values,
                self.node_edge_connectivity,
                self.fill_value,
            )
        return self._face_edge_connectivity

    @property
    def mesh(self):
        if self._mesh is None:
            self._mesh = mk.Mesh1d(
                node_x=self.node_x.values,
                node_y=self.node_y.values,
                edge_nodes=self.edge_nodes.ravel(),
            )
        return self._mesh

    @property
    def mesh(self) -> mk.Mesh2d:
        if self._mesh is None:
            self._mesh = mk.Mesh2d(
                node_x=self.node_x.values,
                node_y=self.node_y.values,
                face_nodes=self.connectivity_array.values.ravel(),
                edge_nodes=self.edge_nodes.ravel(),
            )
        return self._mesh

    @property
    def meshkernel(self) -> mk.MeshKernel:
        if self._meshkernel is None:
            self._meshkernel = mk.MeshKernel(is_geographic=False)
            self._meshkernel.mesh2d_set(self.mesh)
        return self._meshkernel

    def remove_topology(self, obj: Union[xr.Dataset, xr.DataArray]):
        """
        removes the grid topology data from a dataset. Use after creating an
        Ugrid object from the dataset
        """
        obj = obj.drop_vars(self.connectivity_array.name, errors="ignore")
        obj = obj.drop_vars(self.node_x.name, errors="ignore")
        obj = obj.drop_vars(self.node_y.name, errors="ignore")
        obj = obj.drop_vars(self.mesh_2d.name, errors="ignore")
        return obj

    def build_celltree(self):
        """
        initializes the celltree, a search structure for spatial lookups in 2d grids
        """
        nodes = np.column_stack([self.node_x.values, self.node_y.values])
        self._cell_tree = CellTree(nodes, self.connectivity_array.values)

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
        x = self.node_x.values
        y = self.node_y.values
        nodemask = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
        node_face_connectivity = connectivity.invert_dense_to_sparse(
            self.connectivity_array.values, self.fill_value
        )
        selected_faces = np.unique(node_face_connectivity[nodemask].indices)
        return selected_faces

    def locate_faces_polygon(self, polygon: sg.Polygon):
        geometry_list = mku.to_geometry_list(polygon)
        node_indices = self._meshkernel.mesh2d_get_nodes_in_polygons(
            geometry_list, inside=True
        )
        node_face_connectivity = connectivity.invert_dense_to_sparse(
            self.connectivity_array.values, self.fill_value
        )
        selected_faces = np.unique(node_face_connectivity[node_indices].indices)
        return selected_faces

    def locate_cells(points):
        raise NotImplementedError()

    def locate_edges(points):
        raise NotImplementedError()

    def triangulate(self):
        raise NotImplementedError()

    def matplotlib_triangulation(self):
        return mtri.Triangulation(
            x=self.node_x,
            y=self.node_y,
            triangles=self.connectivity_array,
        )

    @staticmethod
    def topology_dataset(
        node_x: FloatArray, node_y: FloatArray, face_node_connectivity: IntArray
    ) -> xr.Dataset:
        return ugrid_io.ugrid2d_dataset(node_x, node_y, face_node_connectivity)
    
    def as_vector_geometry(self, data_on: str):
        if data_on == "node": 
            return conversion.nodes_to_points(self.node_x, self.node_y)
        elif data_on == "edge":
            return conversion.edges_to_linestrings(self.node_x, self.node_y, self.node_edge_connectivity)
        elif data_on == "face":
            return conversion.faces_to_polygons(self.node_x, self.node_y, self.face_edge_connectivity)
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
        if face_indices.size == len(self.connectivity_array):
            return self
        # Subset of faces, create new topology data
        else:
            face_selection = self.connectivity_array.values[face_indices]
            node_indices = np.unique(face_selection.ravel())
            face_node_connectivity = connectivity.renumber(face_selection)
            node_x = self.node_x.values[node_indices]
            node_y = self.node_y.values[node_indices]
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
