from typing import Any, Tuple, Union

import geopandas as gpd
import meshkernel as mk
import numpy as np
import xarray as xr

from .. import conversion
from ..typing import BoolArray, FloatArray, FloatDType, IntArray, IntDType
from . import ugrid_io
from .ugridbase import AbstractUgrid


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
    dataset: xr.Dataset, optional
    name: string, optional
        Network name. Defaults to "network1d" in dataset.
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
        dataset: xr.Dataset = None,
        name: str = None,
        crs: Any = None,
    ):
        self.node_x = np.ascontiguousarray(node_x)
        self.node_y = np.ascontiguousarray(node_y)
        self.fill_value = fill_value
        self.edge_node_connectivity = edge_node_connectivity
        if name is None:
            name = ugrid_io.UGRID1D_DEFAULT_NAME
        self.name = name
        self.topology_attrs = None

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
        self._edge_x = None
        self._edge_y = None
        # Connectivity
        self._node_edge_connectivity = None
        # crs
        if crs is None:
            self.crs = None
        else:
            import pyproj

            self.crs = pyproj.CRS.from_user_input(crs)
        # Store dataset
        if dataset is None:
            dataset = self.topology_dataset()
        self.dataset = dataset
        self.topology_attrs = dataset[self.name].attrs

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

        # Collect names
        mesh_topology = ugrid_io.get_topology_variable(ds)
        ugrid_roles = ugrid_io.get_ugrid2d_variables(dataset, mesh_topology)
        # Rename the arrays to their standard UGRID roles
        topology_ds = ds[set(ugrid_roles.keys())]
        ugrid_ds = topology_ds.rename(ugrid_roles)
        # Coerce type and fill value
        ugrid_ds["node_x"] = ugrid_ds["node_x"].astype(FloatDType)
        ugrid_ds["node_y"] = ugrid_ds["node_y"].astype(FloatDType)
        fill_value = -1
        ugrid_ds["edge_node_connectivity"] = ugrid_ds["edge_node_connectivity"].astype(
            IntDType
        )

        return Ugrid1d(
            ugrid_ds["node_x"].values,
            ugrid_ds["node_y"].values,
            fill_value,
            ugrid_ds["edge_node_connectivity"].values,
            topology_ds,
            mesh_topology.name,
        )

    def remove_topology(
        self, obj: Union[xr.DataArray, xr.Dataset]
    ) -> Union[xr.DataArray, xr.Dataset]:
        """
        Remove UGRID specific variables from the object.

        Parameters
        ----------
        obj: Union[xr.DataArray, xr.Dataset]

        Returns
        -------
        sanitized: Union[xr.DataArray, xr.Dataset]
        """
        return self._remove_topology(obj, ugrid_io.UGRID1D_TOPOLOGY_VARIABLES)

    def topology_coords(self, obj: Union[xr.DataArray, xr.Dataset]) -> dict:
        """
        Return a dictionary with the coordinates required for the dimension of
        the object.

        Parameters
        ----------
        obj: Union[xr.DataArray, xr.Dataset]

        Returns
        -------
        coords: dict
        """
        coords = {}
        dims = obj.dims
        attrs = self.topology_attrs
        edgedim = self.edge_dimension
        nodedim = self.node_dimension
        if edgedim in dims:
            # It's not a required attribute.
            if "edge_coordinates" in attrs:
                name_x, name_y = attrs["edge_coordinates"].split()
            else:
                name_x = f"{self.name}_edge_x"
                name_y = f"{self.name}_edge_y"
            coords[name_x] = (edgedim, self.edge_x)
            coords[name_y] = (edgedim, self.edge_y)
            attrs["edge_coordinates"] = f"{name_x} {name_y}"
        if nodedim in dims:
            name_x, name_y = attrs["node_coordinates"].split()
            coords[name_x] = (nodedim, self.node_x)
            coords[name_y] = (nodedim, self.node_y)
        return coords

    def topology_dataset(self):
        """
        Store the UGRID topology information in an xarray Dataset according to
        the UGRID conventions.

        Returns
        -------
        ugrid_topology: xr.Dataset
        """
        return ugrid_io.ugrid1d_dataset(
            self.node_x,
            self.node_y,
            self.fill_value,
            self.edge_node_connectivity,
            self.name,
            self.topology_attrs,
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
        """Highest dimensionality of the geometric elements: 1"""
        return 1

    def _get_dimension(self, dim):
        key = f"{dim}_dimension"
        if key not in self.topology_attrs:
            self.topology_attrs[key] = ugrid_io.UGrid.mesh1d_get_attributes(self.name)[
                key
            ]
        return self.topology_attrs[key]

    @property
    def mesh(self):
        """
        Create if needed, and return meshkernel Mesh1d object.

        Returns
        -------
        mesh: meshkernel.Mesh1d
        """
        if self._mesh is None:
            self._mesh = mk.Mesh1d(
                node_x=self.node_x,
                node_y=self.node_y,
                edge_nodes=self.edge_node_connectivity.ravel(),
            )
        return self._mesh

    @property
    def meshkernel(self):
        """
        Create if needed, and return meshkernel MeshKernel instance.

        Returns
        -------
        meshkernel: meshkernel.MeshKernel
        """
        if self._meshkernel is None:
            self._meshkernel = mk.MeshKernel(is_geographic=False)
            self._meshkernel.mesh1d_set(self.mesh)
        return self._meshkernel

    @staticmethod
    def from_geodataframe(geodataframe: gpd.GeoDataFrame) -> "Ugrid1d":
        """
        Convert geodataframe of linestrings into a UGRID1D topology.

        Parameters
        ----------
        geodataframe: gpd.GeoDataFrame

        Returns
        -------
        topology: Ugrid1d
        """
        x, y, edge_node_connectivity = conversion.linestrings_to_edges(
            geodataframe.geometry
        )
        fill_value = -1
        return Ugrid1d(x, y, fill_value, edge_node_connectivity)

    def to_pygeos(self, dim):
        """
        Convert UGRID topology to pygeos objects.

        * nodes: points
        * edges: linestrings

        Parameters
        ----------
        dim: str
            Node or edge dimension.

        Returns
        -------
        geometry: ndarray of pygeos.Geometry
        """
        if dim == self.node_dimension:
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
                f"Dimension {dim} is not a node or edge dimension of the"
                " Ugrid1d topology."
            )

    def _validate_indexer(self, indexer) -> Tuple[float, float]:
        if isinstance(indexer, slice):
            if indexer.step is not None:
                raise ValueError("Ugrid1d does not support steps in slices")
            if indexer.start >= indexer.stop:
                raise ValueError("slice start should be smaller than slice stop")
        else:
            raise ValueError("Ugrid1d only supports slice indexing")
        return indexer.start, indexer.stop

    def sel(self, x, y) -> Tuple[str, bool, IntArray, dict]:
        """
        Select a selection of edges, based on edge centroids.

        Parameters
        ----------
        x: slice
        y: slice

        Returns
        -------
        dimension: str
        as_ugrid: bool
        index: 1d array of integers
        coords: dict
        """
        xmin, xmax = self._validate_indexer(x)
        ymin, ymax = self._validate_indexer(y)
        index = np.nonzero(
            (self.edge_x >= xmin)
            & (self.edge_x < xmax)
            & (self.edge_y >= ymin)
            & (self.edge_y < ymax)
        )[0]
        return self.edge_dimension, True, index, {}

    def topology_subset(self, edge_index: Union[BoolArray, IntArray]):
        """
        Create a new UGRID1D topology for a subset of this topology.

        Parameters
        ----------
        edge_index: 1d array of integers or bool
            Edges of the subset.

        Returns
        -------
        subset: Ugrid1d
        """
        edge_index = np.atleast_1d(edge_index)
        return self._topology_subset(edge_index, self.edge_node_connectivity)
