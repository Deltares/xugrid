from itertools import chain
from typing import Any, Tuple, Union

import numpy as np
import xarray as xr

from .. import conversion
from ..typing import BoolArray, FloatArray, FloatDType, IntArray, IntDType
from . import conventions
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
        name: str = "network1d",
        dataset: xr.Dataset = None,
        indexes: dict[str, str] = None,
        projected: bool = True,
        crs: Any = None,
    ):
        if dataset is not None and indexes is None:
            raise ValueError("indexes must be provided for a dataset")

        self.node_x = np.ascontiguousarray(node_x)
        self.node_y = np.ascontiguousarray(node_y)
        self.fill_value = fill_value
        self.edge_node_connectivity = edge_node_connectivity
        self.name = name
        self.projected = projected

        # Store topology attributes
        defaults = conventions.default_topology_attrs(name, self.topology_dimension)
        if dataset is None:
            self._attrs = defaults
            x, y = defaults["node_coordinates"].split(" ")
            self._indexes = {"node_x": x, "node_y": y}
        else:
            derived_dims = dataset.ugrid_roles.dimensions[name]
            self._attrs = {**defaults, **derived_dims, **dataset[name].attrs}
            self._indexes = indexes

        self._dataset = dataset

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

    @classmethod
    def from_dataset(cls, dataset: xr.Dataset, topology: str = None):
        """
        Extract the 1D UGRID topology information from an xarray Dataset.

        Parameters
        ----------
        dataset: xr.Dataset
            Dataset containing topology information stored according to UGRID conventions.

        Returns
        -------
        grid: Ugrid1dAdapter
        """
        ds = dataset
        if not isinstance(ds, xr.Dataset):
            raise TypeError(
                "Ugrid should be initialized with xarray.Dataset. "
                f"Received instead: {type(ds)}"
            )
        if topology is None:
            topology = cls._single_topology(ds)

        indexes = {}

        # Collect names
        connectivity = ds.ugrid_roles.connectivity[topology]
        coordinates = ds.ugrid_roles.coordinates[topology]
        ugrid_vars = (
            [topology]
            + list(connectivity.values())
            + list(chain.from_iterable(chain.from_iterable(coordinates.values())))
        )
        fill_value = -1

        # Take the first coordinates by default.
        # They can be reset with .set_node_coords()
        x_index = coordinates["node_coordinates"][0][0]
        y_index = coordinates["node_coordinates"][1][0]
        node_x_coordinates = ds[x_index].astype(FloatDType).values
        node_y_coordinates = ds[y_index].astype(FloatDType).values

        edge_nodes = connectivity["edge_node_connectivity"]
        edge_node_connectivity = cls._prepare_connectivity(
            ds[edge_nodes], fill_value, dtype=IntDType
        ).values

        indexes["node_x"] = x_index
        indexes["node_y"] = y_index
        projected = False  # TODO

        return cls(
            node_x_coordinates,
            node_y_coordinates,
            fill_value,
            edge_node_connectivity,
            name=topology,
            dataset=dataset[ugrid_vars],
            indexes=indexes,
            projected=projected,
            crs=None,
        )

    def to_dataset(self, other: xr.Dataset = None) -> xr.Dataset:
        node_x = self._indexes["node_x"]
        node_y = self._indexes["node_y"]
        edge_nodes = self._attrs["edge_node_connectivity"]
        edge_nodes_attrs = conventions.DEFAULT_ATTRS["edge_node_connectivity"]

        data_vars = {
            self.name: 0,
            edge_nodes: xr.DataArray(
                data=self.edge_node_connectivity,
                attrs=edge_nodes_attrs,
                dims=(self.edge_dimension, "two"),
            ),
        }
        dataset = xr.Dataset(data_vars, attrs={"Conventions": "CF-1.8 UGRID-1.0"})

        if self._dataset:
            dataset.update(self._dataset)
        if other is not None:
            dataset = dataset.merge(other)
        if node_x not in dataset or node_y not in dataset:
            dataset = self.assign_node_coords(dataset)

        dataset[self.name].attrs = self._filtered_attrs(dataset)
        return dataset

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

    @property
    def topology_dimension(self):
        """Highest dimensionality of the geometric elements: 1"""
        return 1

    @property
    def dimensions(self):
        return self.node_dimension, self.edge_dimension

    # These are all optional UGRID attributes. They are not computed by
    # default, only when called upon.

    @property
    def mesh(self) -> "mk.Mesh1d":  # type: ignore # noqa
        """
        Create if needed, and return meshkernel Mesh1d object.

        Returns
        -------
        mesh: meshkernel.Mesh1d
        """
        import meshkernel as mk

        edge_nodes = self.edge_node_connectivity.ravel().astype(np.int32)

        if self._mesh is None:
            self._mesh = mk.Mesh1d(
                node_x=self.node_x,
                node_y=self.node_y,
                edge_nodes=edge_nodes,
            )
        return self._mesh

    @property
    def meshkernel(self) -> "mk.MeshKernel":  # type: ignore # noqa
        """
        Create if needed, and return meshkernel MeshKernel instance.

        Returns
        -------
        meshkernel: meshkernel.MeshKernel
        """
        import meshkernel as mk

        if self._meshkernel is None:
            self._meshkernel = mk.MeshKernel(is_geographic=False)
            self._meshkernel.mesh1d_set(self.mesh)
        return self._meshkernel

    @staticmethod
    def from_geodataframe(geodataframe: "geopandas.GeoDataFrame") -> "Ugrid1d":  # type: ignore # noqa
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

    def isel(self, dim, indexer):
        if dim != self.edge_dimension:
            raise NotImplementedError("Can only index edge_dimension for Ugrid1D")
        return self.topology_subset(indexer)

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
