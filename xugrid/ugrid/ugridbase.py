import abc
import copy
from typing import Tuple, Union

import numpy as np
import xarray as xr
from scipy.sparse import csr_matrix

from .. import connectivity
from ..typing import BoolArray, FloatArray, IntArray
from . import ugrid_io


class AbstractUgrid(abc.ABC):
    @abc.abstractproperty
    def topology_dimension(self):
        """ """

    @abc.abstractmethod
    def _get_dimension(self):
        """ """

    @abc.abstractstaticmethod
    def from_dataset():
        """ """

    @abc.abstractmethod
    def sel(self):
        """ """

    @abc.abstractmethod
    def topology_dataset(self):
        """ """

    @abc.abstractmethod
    def topology_subset(self):
        """ """

    @abc.abstractmethod
    def remove_topology(self):
        """ """

    @abc.abstractmethod
    def topology_coords(self):
        """ """

    @abc.abstractmethod
    def _clear_geometry_properties(self):
        """ """

    def copy(self):
        """Creates deepcopy"""
        return copy.deepcopy(self)

    @property
    def node_dimension(self):
        """Name of node dimension"""
        return self._get_dimension("node")

    @property
    def edge_dimension(self):
        """Name of edge dimension"""
        return self._get_dimension("edge")

    @property
    def node_coordinates(self) -> FloatArray:
        """Coordinates (x, y) of the nodes (vertices)"""
        return np.column_stack([self.node_x, self.node_y])

    @property
    def n_node(self) -> int:
        """Number of nodes (vertices) in the UGRID topology"""
        return self.node_x.size

    @property
    def n_edge(self) -> int:
        """Number of edges in the UGRID topology"""
        return self.edge_node_connectivity.shape[0]

    @property
    def edge_x(self):
        """x-coordinate of every edge in the UGRID topology"""
        if self._edge_x is None:
            self._edge_x = self.node_x[self.edge_node_connectivity].mean(axis=1)
        return self._edge_x

    @property
    def edge_y(self):
        """y-coordinate of every edge in the UGRID topology"""
        if self._edge_y is None:
            self._edge_y = self.node_y[self.edge_node_connectivity].mean(axis=1)
        return self._edge_y

    @property
    def edge_coordinates(self) -> FloatArray:
        """Centroid (x,y) coordinates of every edge in the UGRID topology"""
        return np.column_stack([self.edge_x, self.edge_y])

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Returns a tuple with the node bounds: xmin, ymin, xmax, ymax"""
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

    def _topology_subset(
        self, indices: Union[BoolArray, IntArray], node_connectivity: IntArray
    ):
        is_same = False
        if np.issubdtype(indices.dtype, np.bool_):
            is_same = indices.all()
        else:
            # TODO: check for unique indices if integer?
            is_same = np.array_equal(indices, np.arange(node_connectivity.shape[0]))

        if is_same:
            return self
        # Subset of faces, create new topology data
        else:
            subset = node_connectivity[indices]
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
        Removes the grid topology data from a dataset. Use after creating an
        Ugrid object from the dataset.
        """
        attrs = self.topology_attrs
        names = []
        if self.name in obj:
            names.append(self.name)
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
        """
        Node to edge connectivity.

        Returns
        -------
        connectivity: csr_matrix
        """
        if self._node_edge_connectivity is None:
            self._node_edge_connectivity = connectivity.invert_dense_to_sparse(
                self.edge_node_connectivity, self.fill_value
            )
        return self._node_edge_connectivity

    def set_crs(
        self,
        crs: Union["pyproj.CRS", str] = None,  # type: ignore # noqa
        epsg: int = None,
        allow_override: bool = False,
    ):
        """
        Set the Coordinate Reference System (CRS) of a UGRID topology.

        NOTE: The underlying geometries are not transformed to this CRS. To
        transform the geometries to a new CRS, use the ``to_crs`` method.

        Parameters
        ----------
        crs : pyproj.CRS, optional if `epsg` is specified
            The value can be anything accepted
            by :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
            such as an authority string (eg "EPSG:4326") or a WKT string.
        epsg : int, optional if `crs` is specified
            EPSG code specifying the projection.
        inplace : bool, default False
            If True, the CRS of the UGRID topology will be changed in place
            (while still returning the result) instead of making a copy of the
            GeoSeries.
        allow_override : bool, default False
            If the the UGRID topology already has a CRS, allow to replace the
            existing CRS, even when both are not equal.
        """
        import pyproj

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
        self.crs = crs

    def to_crs(
        self,
        crs: Union["pyproj.CRS", str] = None,  # type: ignore # noqa
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
        import pyproj

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

        if self.crs.is_exact_same(crs):
            if inplace:
                return
            else:
                return grid

        transformer = pyproj.Transformer.from_crs(
            crs_from=self.crs, crs_to=crs, always_xy=True
        )
        node_x, node_y = transformer.transform(xx=grid.node_x, yy=grid.node_y)
        grid.node_x = node_x
        grid.node_y = node_y
        grid._clear_geometry_properties()
        grid.crs = crs

        if not inplace:
            return grid
