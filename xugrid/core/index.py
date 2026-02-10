import abc

from xarray import Index

import xugrid


class UgridIndex(Index, abc.ABC):
    pass

    @classmethod
    def from_variables(cls, variables, options):
        return cls(*variables.values())

    def should_add_coord_to_array(self, name, var, dims):
        return True


class UgridIndex1d(UgridIndex):
    def __init__(
        self,
        node_x,
        node_y,
        edge_node_connectivity,
    ):
        edge_nodes = edge_node_connectivity.to_numpy().astype(int)
        self._ugrid = xugrid.Ugrid1d(node_x.values, node_y.values, -1, edge_nodes)


class UgridIndex2d(UgridIndex):
    def __init__(
        self,
        node_x,
        node_y,
        face_node_connectivity,
    ):
        face_nodes = face_node_connectivity.fillna(-1).to_numpy().astype(int)
        self._ugrid = xugrid.Ugrid2d(node_x.values, node_y.values, -1, face_nodes)
