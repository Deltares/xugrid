from typing import Union

import matplotlib.tri as mtri
import numpy as np
import xarray as xr
from cell_tree2d import CellTree


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

        connectivity_array = self._get_data_arrays_by_name(ds, connectivity_array_names)
        node_coord_arrays = self._get_coordinate_arrays_by_name(
            ds, node_coord_array_names
        )

        self.connectivity_array = connectivity_array[0]
        self.nodes_x = node_coord_arrays[0]
        self.nodes_y = node_coord_arrays[1]

        self.build_celltree()

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
        maskx = (self.nodes_x >= xmin) & (self.nodes_x <= xmax)
        masky = (self.nodes_y >= ymin) & (self.nodes_y <= ymax)
        nodemask = maskx & masky
        result = []

        def cell_in_box(node_indices: xr.DataArray, nodeMask: xr.DataArray, missing):
            """returns true if all the node indices are included in the mask (or missing)"""
            result = True
            for i in range(len(node_indices)):
                index = int(node_indices.values[i])
                if not index == missing:
                    result = result and nodeMask[index]
            return result

        for i in range(len(self.connectivity_array)):
            if cell_in_box(self.connectivity_array[i], nodemask, -999):
                result.append(i)
        return result

    def locate_cells(points):
        raise NotImplementedError()

    def locate_edges(points):
        raise NotImplementedError()
