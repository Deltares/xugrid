import abc
import copy
from typing import Tuple, Union

import numpy as np
import pyproj
import xarray as xr
from scipy.sparse import csr_matrix

from .. import connectivity, conversion
from ..typing import FloatArray, IntArray
from . import ugrid_io


class AbstractUgrid(abc.ABC):
    @abc.abstractproperty
    def topology_dimension(self):
        return

    @abc.abstractmethod
    def _get_dimension(self):
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
    def remove_topology(self):
        return

    @abc.abstractmethod
    def topology_coords(self):
        return

    @abc.abstractmethod
    def _clear_geometry_properties(self):
        return

    def copy(self):
        return copy.deepcopy(self)

    @property
    def node_dimension(self):
        return self._get_dimension("node")

    @property
    def edge_dimension(self):
        return self._get_dimension("edge")

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

    def _topology_subset(self, indices: IntArray, connectivity: IntArray):
        # If faces are repeated: not a valid mesh
        # _, count = np.unique(face_indices, return_counts=True)
        # assert count.max() <= 1?
        # If no faces are repeated, and size is the same, it's the same mesh
        if indices.size == len(connectivity):
            return self
        # Subset of faces, create new topology data
        else:
            subset = connectivity[indices]
            node_indices = np.unique(subset.ravel())
            new_connectivity = connectivity.renumber(subset)
            node_x = self.node_x[node_indices]
            node_y = self.node_y[node_indices]
            return self.__class__(node_x, node_y, self.fill_value, new_connectivity)

    def _remove_topology(
        self,
        obj: Union[xr.Dataset, xr.DataArray],
        topology_variables: ugrid_io.UgridTopologyAttributes,
    ) -> Union[xr.Dataset, xr.DataArray]:
        """
        removes the grid topology data from a dataset. Use after creating an
        Ugrid object from the dataset
        """
        attrs = self.topology_attrs
        names = []
        for topology_attr in topology_variables.coordinates:
            varname = attrs.get(topology_attr)
            if varname and varname in obj:
                names.extend(varname.split())
        for topology_attr in topology_variables.connectivity:
            varname = attrs.get(topology_attr)
            if varname and varname in obj:
                names.append(varname)
        return obj.drop_vars(names)

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
        try:
            import pyproj
        except ImportError:
            raise ImportError("pyproj must be installed to use to_crs")

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
        if dim == self.face_dimension:
            return conversion.faces_to_polygons(
                self.node_x,
                self.node_y,
                self.face_node_connectivity,
                self.fill_value,
            )
        elif dim == self.node_dimension:
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
                f"Dimension {dim} is not a face, node, or edge dimension of the"
                " Ugrid topology."
            )
