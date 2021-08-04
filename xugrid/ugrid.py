from typing import Union

import matplotlib.tri as mtri
import numpy as np
import xarray as xr
from cell_tree2d import CellTree

from . import connectivity

IntArray = np.ndarray
FloatArray = np.ndarray


class Ugrid:
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

    def __init__(self, ds: xr.Dataset):
        """
        Initialize Ugrid object from the topology data in an xarray Dataset
        """
        assert isinstance(ds, xr.Dataset)
        mesh_2d_variables = self._get_values_with_attribute(
            ds, "cf_role", "mesh_topology"
        )
        assert len(mesh_2d_variables) == 1
        self.mesh_2d = mesh_2d_variables[0]

        node_coord_array_names = self._get_topology_array_with_role(
            self.mesh_2d, "node_coordinates"
        )
        connectivity_array_names = self._get_topology_array_with_role(
            self.mesh_2d, "face_node_connectivity"
        )
        self.fill_value = -1

        connectivity_array = self._get_data_arrays_by_name(ds, connectivity_array_names)
        node_coord_arrays = self._get_coordinate_arrays_by_name(
            ds, node_coord_array_names
        )

        self.connectivity_array = connectivity_array[0].astype(connectivity.INT_DTYPE)
        self.nodes_x = node_coord_arrays[0]
        self.nodes_y = node_coord_arrays[1]
        self.build_celltree()

    def to_dataset(self):
        ds = xr.Dataset()
        ds["mesh2d"] = self.mesh_2d
        ds["face_node_connectivity"] = self.connectivity_array
        ds["node_x"] = self.nodes_x
        ds["node_x"] = self.nodes_y
        return ds

    def remove_topology(self, obj: Union[xr.Dataset, xr.DataArray]):
        """
        removes the grid topology data from a dataset. Use after creating an
        Ugrid object from the dataset
        """
        obj = obj.drop_vars(self.connectivity_array.name, errors="ignore")
        obj = obj.drop_vars(self.nodes_x.name, errors="ignore")
        obj = obj.drop_vars(self.nodes_y.name, errors="ignore")
        obj = obj.drop_vars(self.mesh_2d.name, errors="ignore")
        return obj

    # returns the names of the arrays that have the specified role on the Mesh2D variable
    def _get_topology_array_with_role(
        self, mesh_2d_variable: xr.core.dataarray.DataArray, role: str
    ):
        topology_array_names = []
        for variable_name in mesh_2d_variable.attrs:
            if variable_name == role:
                split_names = str(mesh_2d_variable.attrs[variable_name]).split()
                topology_array_names.extend(split_names)
        return topology_array_names

    # returns values of dataset which have a given attribute value
    @staticmethod
    def _get_values_with_attribute(dataset, attribute_name, attribute_value):
        result = []
        for da in dataset.values():
            if da.attrs.get(attribute_name) == attribute_value:
                result.append(da)
        return result

    # return those data arrays whose name appears in nameList
    @staticmethod
    def _get_data_arrays_by_name(dataset, name_list):
        result = []
        for da in dataset.values():
            if da.name in name_list:
                result.append(da)
        return result

    # return those coordinate arrays whose name appears in nameList
    @staticmethod
    def _get_coordinate_arrays_by_name(dataset, name_list):
        result = []
        for da in dataset.coords:
            if da in name_list:
                result.append(dataset.coords[da])
        return result

    def build_celltree(self):
        """
        initializes the celltree, a search structure for spatial lookups in 2d grids
        """
        nodes = np.column_stack([self.nodes_x.values, self.nodes_y.values])

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
        x = self.nodes_x.values
        y = self.nodes_y.values
        nodemask = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
        node_face_connectivity = connectivity.invert_dense_to_sparse(
            self.connectivity_array.values, self.fill_value
        )
        selected_faces = np.unique(node_face_connectivity[nodemask].indices)
        return selected_faces

    def locate_cells(points):
        raise NotImplementedError()

    def locate_edges(points):
        raise NotImplementedError()

    def triangulate(self):
        raise NotImplementedError()

    def matplotlib_triangulation(self):
        return mtri.Triangulation(
            x=self.nodes_x,
            y=self.nodes_y,
            triangles=self.connectivity_array,
        )

    @staticmethod
    def topology_dataset(
        node_x: FloatArray, node_y: FloatArray, face_node_connectivity: IntArray
    ) -> xr.Dataset:
        # TODO: parametrize dataset variable names (node, node_x, node_y, node, etc.)
        # mesh2d variable could just be deep-copied in case of subset
        ds = xr.Dataset()
        ds["mesh2d"] = xr.DataArray(
            data=0,
            attrs={
                "cf_role": "mesh_topology",
                "long_name": "Topology data of 2D mesh",
                "topology_dimension": 2,
                "node_coordinates": "node_x node_y",
                "face_node_connectivity": "face_nodes",
                "edge_node_connectivity": "edge_nodes",
            },
        )
        ds = ds.assign_coords(
            node_x=xr.DataArray(
                data=node_x,
                dims=["node"],
            )
        )
        ds = ds.assign_coords(
            node_y=xr.DataArray(
                data=node_y,
                dims=["node"],
            )
        )
        ds["face_nodes"] = xr.DataArray(
            data=face_node_connectivity,
            dims=["face", "nmax_face"],
            attrs={
                "cf_role": "face_node_connectivity",
                "long_name": "Vertex nodes of mesh faces (counterclockwise)",
                "start_index": 0,
                "_FillValue": -1,
            },
        )
        ds.attrs = {"Conventions": "CF-1.8 UGRID-1.0"}
        return ds

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
            node_x = self.nodes_x.values[node_indices]
            node_y = self.nodes_y.values[node_indices]
            ds = Ugrid.topology_dataset(node_x, node_y, face_node_connectivity)
            return Ugrid(ds)
