import abc

import xarray as xr

import xugrid
from xugrid.ugrid.ugrid1d import Ugrid1d
from xugrid.ugrid.ugridbase import AbstractUgrid


class UgridIndex(xr.Index, abc.ABC):
    @classmethod
    def from_variables(cls, variables, options):
        return cls(*variables.values(), **options)

    def should_add_coord_to_array(self, name, var, dims):
        return True

    def equals(self, other, **kwargs) -> bool:
        if type(self) is not type(other):
            return False
        return self._ugrid.equals(other._ugrid)

    def join(self, other, how="inner"):
        # UgridIndex coordinates are not label-aligned; just return self.
        # Two different grids are kept independent; alignment is a no-op.
        if isinstance(other, type(self)):
            return self
        return NotImplemented

    def reindex_like(self, other):
        # UgridIndex has no label-based indexing; reindexing is a no-op.
        return {}

    @classmethod
    def from_ugrid(cls, grid: AbstractUgrid):
        index = cls.__new__(cls)
        index._ugrid = grid
        return index

    @staticmethod
    @abc.abstractmethod
    def _variables_from_dataset(dataset, topology):
        pass


class UgridIndex1d(UgridIndex):
    def __init__(
        self,
        node_x,
        node_y,
        edge_node_connectivity,
    ):
        edge_nodes = edge_node_connectivity.to_numpy().astype(int)
        self._ugrid = xugrid.Ugrid1d(node_x.values, node_y.values, -1, edge_nodes)

    @classmethod
    def from_ugrid(cls, grid: Ugrid1d):
        index = cls.__new__(cls)
        index._ugrid = grid
        return index

    def isel(self, indexers):
        numpy_indexers = {
            k: v.values if hasattr(v, "values") else v for k, v in indexers.items()
        }
        try:
            new_grid = self._ugrid.isel(numpy_indexers)
            return type(self).from_ugrid(new_grid)
        except Exception:
            return None

    def create_variables(self, variables=None):
        ugrid = self._ugrid
        return {
            f"{ugrid.name}_node_x": xr.Variable(ugrid.node_dimension, ugrid.node_x),
            f"{ugrid.name}_node_y": xr.Variable(ugrid.node_dimension, ugrid.node_y),
            f"{ugrid.name}_edge_x": xr.Variable(ugrid.edge_dimension, ugrid.edge_x),
            f"{ugrid.name}_edge_y": xr.Variable(ugrid.edge_dimension, ugrid.edge_y),
        }

    @staticmethod
    def _variables_from_dataset(dataset, topology):
        ugrid_vars = dataset.ugrid_roles[topology]
        node_xs, node_ys = ugrid_vars["node_coordinates"]
        variables = (node_xs[0], node_ys[0])
        edge_nodes = dataset[ugrid_vars["edge_node_connectivity"]]
        return variables, {"edge_node_connectivity": edge_nodes}


class UgridIndex2d(UgridIndex):
    def __init__(
        self,
        node_x,
        node_y,
        face_node_connectivity,
    ):
        face_nodes = face_node_connectivity.fillna(-1).to_numpy().astype(int)
        self._ugrid = xugrid.Ugrid2d(node_x.values, node_y.values, -1, face_nodes)

    def isel(self, indexers):
        numpy_indexers = {
            k: v.values if hasattr(v, "values") else v for k, v in indexers.items()
        }
        try:
            new_grid = self._ugrid.isel(numpy_indexers)
            return type(self).from_ugrid(new_grid)
        except Exception:
            return None

    def create_variables(self, variables=None):
        ugrid = self._ugrid
        return {
            f"{ugrid.name}_face_x": xr.Variable(ugrid.face_dimension, ugrid.face_x),
            f"{ugrid.name}_face_y": xr.Variable(ugrid.face_dimension, ugrid.face_y),
            f"{ugrid.name}_node_x": xr.Variable(ugrid.node_dimension, ugrid.node_x),
            f"{ugrid.name}_node_y": xr.Variable(ugrid.node_dimension, ugrid.node_y),
            f"{ugrid.name}_edge_x": xr.Variable(ugrid.edge_dimension, ugrid.edge_x),
            f"{ugrid.name}_edge_y": xr.Variable(ugrid.edge_dimension, ugrid.edge_y),
        }

    @staticmethod
    def _variables_from_dataset(dataset, topology):
        ugrid_vars = dataset.ugrid_roles[topology]
        node_xs, node_ys = ugrid_vars["node_coordinates"]
        variables = (node_xs[0], node_ys[0])
        face_nodes = dataset[ugrid_vars["face_node_connectivity"]]
        return variables, {"face_node_connectivity": face_nodes}


UGRID_INDEXES = {
    1: UgridIndex1d,
    2: UgridIndex2d,
}


def drop_ugrid_index(obj):
    """Strip all UgridIndex-backed coordinates from an xarray object."""
    existing = [k for k, v in obj.xindexes.items() if isinstance(v, UgridIndex)]
    if existing:
        return obj.drop_indexes(existing).drop_vars(existing)
    return obj
