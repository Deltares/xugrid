
import xarray as xr
import matplotlib.tri as mtri
import numpy as np
from cell_tree2d import *
from typing import  Union


class Ugrid:

    _topology_keys = ["node_dimension", "node_coordinates", "max_face_nodes_dimension","edge_node_connectivity", "edge_dimension",
    "edge_coordinates ", "face_node_connectivity", "face_dimension" ,"edge_face_connectivity","face_coordinates"]

    def __init__(self,  ds: xr.Dataset ):
        assert(isinstance(ds, xr.Dataset))
        mesh2dVariables = self._get_values_with_attribute(ds, "cf_role", "mesh_topology")
        assert(len(mesh2dVariables) == 1)
        self.mesh2d = mesh2dVariables[0]

        node_coord_array_names = self._get_topology_array_with_role( self.mesh2d, "node_coordinates")
        connectivity_array_names = self._get_topology_array_with_role( self.mesh2d, "face_node_connectivity")

        connectivity_array = self._get_data_arrays_by_name(ds, connectivity_array_names)
        node_coord_arrays = self._get_coordinate_arrays_by_name(ds, node_coord_array_names)

        self.connectivity_array = connectivity_array[0]
        self.nodes_x = node_coord_arrays[0]
        self.nodes_y = node_coord_arrays[1]
       
        self.build_celltree()

    def remove_topology(self, obj: Union[xr.Dataset, xr.DataArray]):
       obj = obj.drop_vars(self.connectivity_array.name, errors='ignore')
       obj = obj.drop_vars(self.nodes_x.name, errors='ignore')
       obj = obj.drop_vars(self.nodes_y.name, errors='ignore')
       obj = obj.drop_vars(self.mesh2d.name, errors='ignore')
       return obj

    # returns the names of the arrays that have the specified role on the Mesh2D variable
    def _get_topology_array_with_role(self, mesh2dVariable: xr.core.dataarray.DataArray, role: str):
        topologyArrayNames = []
        for variableName in mesh2dVariable.attrs:
            if variableName == role:
                splitNames = str(mesh2dVariable.attrs[variableName]).split()
                topologyArrayNames.extend(splitNames)
        return topologyArrayNames

    #returns values of dataset which have a given attribute value
    @staticmethod
    def _get_values_with_attribute( dataset, attributeName, attributeValue):
        result = []
        for da in dataset.values():
            if da.attrs.get(attributeName) == attributeValue:
                result.append(da)
        return result

    #return those data arrays whose name appears in nameList
    @staticmethod
    def _get_data_arrays_by_name(dataset, nameList):
        result = []
        for da in dataset.values():
            if  da.name in nameList:
                result.append(da)
        return result 

    #return those coordinate arrays whose name appears in nameList
    @staticmethod
    def _get_coordinate_arrays_by_name(dataset, nameList):   
        result = []
        for da in dataset.coords:
            if  da in nameList:
                result.append(dataset.coords[da])
        return result

    def build_celltree(self):
        nodes = []
        nrNodes = len(self.nodes_x)
        for i in range(nrNodes):
            element = [self.nodes_x.values[i],self.nodes_y.values[i]]
            nodes.append(element)

        self._cell_tree = CellTree(nodes, self.connectivity_array.values)

    def locate_faces(self, points) :
        if self._cell_tree is None:
            self.build_celltree()
        indices = self._cell_tree.locate(points)
        return indices


    def locate_faces_bounding_box(self, xmin, xmax, ymin, ymax):
        maskx = (self.nodes_x >= xmin) & (self.nodes_x <= xmax)
        masky = (self.nodes_y >= ymin) & (self.nodes_y <= ymax )
        nodemask = maskx & masky
        result = []
        def cellInBox(cellNodeNumbers: xr.DataArray, nodeMask: xr.DataArray, missing):
            result = True
            for i in range(len(cellNodeNumbers)):
                index=  int(cellNodeNumbers.values[i])
                if not index == missing:
                    result = result & nodeMask[index]
            return result
        for i in range(len(self.connectivity_array)):
            if  cellInBox(self.connectivity_array[i], nodemask, -999):
                result.append(i)
        return result

    def locate_cells(points):
        raise Exception("not implemented")

    def locate_edges(points):
        raise Exception("not implemented")
