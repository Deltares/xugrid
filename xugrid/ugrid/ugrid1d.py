from typing import Any, Union

import geopandas as gpd
import meshkernel as mk
import pyproj
import xarray as xr

from .. import conversion
from ..typing import FloatArray, FloatDType, IntArray, IntDType
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
        self.node_x = node_x
        self.node_y = node_y
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
        ugrid_ds = ds[set(ugrid_roles.keys())].rename(ugrid_roles)
        # Coerce type and fill value
        ugrid_ds["node_x"] = ugrid_ds["node_x"].astype(FloatDType)
        ugrid_ds["node_y"] = ugrid_ds["node_y"].astype(FloatDType)
        fill_value = -1
        ugrid_ds["edge_node_connectivity"] = ugrid_ds["edge_node_connectivity"].astype(
            IntDType
        )

        # Set back to their original names
        topology_ds = ugrid_ds.rename({v: k for k, v in ugrid_roles.items()})

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
        """
        return self._remove_topology(obj, ugrid_io.UGRID1D_TOPOLOGY_VARIABLES)

    def topology_coords(self, obj: Union[xr.DataArray, xr.Dataset]) -> dict:
        """
        Return a dictionary with the coordinates required for the dimension of
        the object.
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
        if self._mesh is None:
            self._mesh = mk.Mesh1d(
                node_x=self.node_x,
                node_y=self.node_y,
                edge_nodes=self.edge_node_connectivity.ravel(),
            )
        return self._mesh

    @property
    def meshkernel(self):
        if self._meshkernel is None:
            self._meshkernel = mk.MeshKernel(is_geographic=False)
            self._meshkernel.mesh1d_set(self.mesh)
        return self._meshkernel

    @staticmethod
    def from_geodataframe(geodataframe: gpd.GeoDataFrame) -> "Ugrid1d":
        x, y, edge_node_connectivity = conversion.linestrings_to_edges(
            geodataframe.geometry
        )
        fill_value = -1
        return Ugrid1d(x, y, fill_value, edge_node_connectivity)

    def to_pygeos(self, dim):
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

    def topology_subset(self, edge_indices: IntArray):
        return self._topology_subset(edge_indices, self.edge_node_connectivity)
