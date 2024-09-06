from __future__ import annotations

import warnings
from itertools import chain
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
from numba_celltree import CellTree2d
from numpy.typing import ArrayLike
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

import xugrid
from xugrid import conversion
from xugrid import meshkernel_utils as mku
from xugrid.constants import (
    FILL_VALUE,
    BoolArray,
    FloatArray,
    FloatDType,
    IntArray,
    IntDType,
    PolygonArray,
    SparseMatrix,
)
from xugrid.core.utils import either_dict_or_kwargs
from xugrid.ugrid import connectivity, conventions
from xugrid.ugrid.ugridbase import AbstractUgrid, as_pandas_index
from xugrid.ugrid.voronoi import voronoi_topology


def section_coordinates(
    edges: FloatArray, xy: FloatArray, dim: str, index: IntArray, name: str
) -> Tuple[IntArray, dict]:
    # TODO: add boundaries xy[:, 0] and xy[:, 1]
    xy_mid = 0.5 * (xy[:, 0, :] + xy[:, 1, :])
    s = np.linalg.norm(xy_mid - edges[0, 0], axis=1)
    order = np.argsort(s)
    coords = {
        f"{name}_x": (dim, xy_mid[order, 0]),
        f"{name}_y": (dim, xy_mid[order, 1]),
        f"{name}_s": (dim, s[order]),
    }
    return coords, index[order]


def numeric_bound(v: Union[float, None], other: float):
    if v is None:
        return other
    else:
        return v


class Ugrid2d(AbstractUgrid):
    """
    This class stores the topological data of a 2-D unstructured grid.

    Parameters
    ----------
    node_x: ndarray of floats
    node_y: ndarray of floats
    fill_value: int
    face_node_connectivity: ndarray of integers
    name: string, optional
        Mesh name. Defaults to "mesh2d".
    edge_node_connectivity: ndarray of integers, optional
    dataset: xr.Dataset, optional
    indexes: Dict[str, str], optional
        When a dataset is provided, a mapping from the UGRID role to the dataset
        variable name. E.g. {"face_x": "mesh2d_face_lon"}.
    projected: bool, optional
        Whether node_x and node_y are longitude and latitude or projected x and
        y coordinates. Used to write the appropriate standard_name in the
        coordinate attributes.
    crs: Any, optional
        Coordinate Reference System of the geometry objects. Can be anything accepted by
        :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
        such as an authority string (eg "EPSG:4326") or a WKT string.
    attrs: Dict[str, str], optional
        UGRID topology attributes. Should not be provided together with
        dataset: if other names are required, update the dataset instead.
        A name entry is ignored, as name is given explicitly.
    start_index: int, 0 or 1, default is 0.
        Start index of the connectivity arrays. Must match the start index
        of the provided face_node_connectivity and edge_node_connectivity.
    """

    def __init__(
        self,
        node_x: FloatArray,
        node_y: FloatArray,
        fill_value: int,
        face_node_connectivity: Union[IntArray, SparseMatrix],
        name: str = "mesh2d",
        edge_node_connectivity: IntArray = None,
        dataset: xr.Dataset = None,
        indexes: Dict[str, str] = None,
        projected: bool = True,
        crs: Any = None,
        attrs: Dict[str, str] = None,
        start_index: int = 0,
    ):
        self.node_x = np.ascontiguousarray(node_x)
        self.node_y = np.ascontiguousarray(node_y)
        self.fill_value = fill_value
        self.start_index = start_index
        self.name = name
        self.projected = projected

        if isinstance(face_node_connectivity, np.ndarray):
            self.face_node_connectivity = face_node_connectivity.copy()
        elif isinstance(face_node_connectivity, (coo_matrix, csr_matrix)):
            self.face_node_connectivity = connectivity.to_dense(face_node_connectivity)
        else:
            raise TypeError(
                "face_node_connectivity should be an array of integers or a sparse matrix"
            )

        # Ensure the fill value is FILL_VALUE (-1) and the array is 0-based.
        if self.fill_value != -1 or self.start_index != 0:
            is_fill = self.face_node_connectivity == self.fill_value
            if self.start_index != 0:
                self.face_node_connectivity[~is_fill] -= self.start_index
            if self.fill_value != FILL_VALUE:
                self.face_node_connectivity[is_fill] = FILL_VALUE

        # TODO: do this in validation instead. While UGRID conventions demand it,
        # where does it go wrong?
        # self.face_node_connectivity = connectivity.counterclockwise(
        #    face_node_connectivity, self.fill_value, self.node_coordinates
        # )

        self._initialize_indexes_attrs(name, dataset, indexes, attrs)
        self._dataset = dataset

        # Optional attributes, deferred initialization
        # Meshkernel
        self._mesh = None
        self._meshkernel = None
        # Celltree
        self._celltree = None
        # Perimeter
        self._perimeter = None
        # Area
        self._area = None
        # Centroids
        self._centroids = None
        self._circumcenters = None
        # Bounds
        self._xmin = None
        self._xmax = None
        self._ymin = None
        self._ymax = None
        # Edges
        self._edge_x = None
        self._edge_y = None
        # Connectivity
        self._edge_node_connectivity = edge_node_connectivity
        if self._edge_node_connectivity is not None:
            self._edge_node_connectivity -= self.start_index
        self._edge_face_connectivity = None
        self._node_node_connectivity = None
        self._node_edge_connectivity = None
        self._node_face_connectivity = None
        self._face_edge_connectivity = None
        self._face_face_connectivity = None
        self._boundary_node_connectivity = None
        # Derived topology
        self._triangulation = None
        self._voronoi_topology = None
        self._centroid_triangulation = None
        # crs
        if crs is None:
            self.crs = None
        else:
            import pyproj

            self.crs = pyproj.CRS.from_user_input(crs)

    def _clear_geometry_properties(self):
        """Clear all properties that may have been invalidated"""
        # Meshkernel
        self._mesh = None
        self._meshkernel = None
        # Celltree
        self._celltree = None
        # Perimeter
        self._perimeter = None
        # Area
        self._area = None
        # Centroids
        self._centroids = None
        self._circumcenters = None
        # Bounds
        self._xmin = None
        self._xmax = None
        self._ymin = None
        self._ymax = None
        # Edges
        self._edge_x = None
        self._edge_y = None
        # Derived topology
        self._triangulation = None
        self._voronoi_topology = None
        self._centroid_triangulation = None

    @classmethod
    def from_meshkernel(
        cls,
        mesh,
        name: str = "mesh2d",
        projected: bool = True,
        crs: Any = None,
    ):
        """
        Create a 2D UGRID topology from a MeshKernel Mesh2d object.

        Parameters
        ----------
        mesh: MeshKernel.Mesh2d
        name: str
            Mesh name. Defaults to "mesh2d".
        projected: bool
            Whether node_x and node_y are longitude and latitude or projected x and
            y coordinates. Used to write the appropriate standard_name in the
            coordinate attributes.
        crs: Any, optional
            Coordinate Reference System of the geometry objects. Can be anything accepted by
            :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
            such as an authority string (eg "EPSG:4326") or a WKT string.

        Returns
        -------
        grid: Ugrid2d
        """
        n_face = len(mesh.nodes_per_face)
        n_max_node = mesh.nodes_per_face.max()
        face_node_connectivity = np.full((n_face, n_max_node), FILL_VALUE)
        isnode = connectivity.ragged_index(n_face, n_max_node, mesh.nodes_per_face)
        face_node_connectivity[isnode] = mesh.face_nodes
        edge_node_connectivity = np.reshape(mesh.edge_nodes, (-1, 2))
        return cls(
            node_x=mesh.node_x,
            node_y=mesh.node_y,
            fill_value=FILL_VALUE,
            face_node_connectivity=face_node_connectivity,
            edge_node_connectivity=edge_node_connectivity,
            name=name,
            projected=projected,
            crs=crs,
        )

    @classmethod
    def from_dataset(cls, dataset: xr.Dataset, topology: str = None):
        """
        Extract the 2D UGRID topology information from an xarray Dataset.

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

        x_index = coordinates["node_coordinates"][0][0]
        y_index = coordinates["node_coordinates"][1][0]
        node_x_coordinates = ds[x_index].astype(FloatDType).to_numpy()
        node_y_coordinates = ds[y_index].astype(FloatDType).to_numpy()

        face_nodes = connectivity["face_node_connectivity"]
        fill_value = ds[face_nodes].encoding.get("_FillValue", -1)
        start_index = ds[face_nodes].attrs.get("start_index", 0)
        face_node_connectivity = cls._prepare_connectivity(
            ds[face_nodes], fill_value, dtype=IntDType
        ).to_numpy()

        edge_nodes = connectivity.get("edge_node_connectivity")
        if edge_nodes:
            edge_node_connectivity = cls._prepare_connectivity(
                ds[edge_nodes], fill_value, dtype=IntDType
            ).to_numpy()
            # Make sure the single passed start index is valid for both
            # connectivity arrays.
            edge_start_index = ds[edge_nodes].attrs.get("start_index", 0)
            if edge_start_index != start_index:
                # start_index = 1, edge_start_index = 0, then add one
                # start_index = 0, edge_start_index = 1, then subtract one
                edge_node_connectivity += start_index - edge_start_index
        else:
            edge_node_connectivity = None

        indexes["node_x"] = x_index
        indexes["node_y"] = y_index
        projected = False  # TODO

        return cls(
            node_x_coordinates,
            node_y_coordinates,
            fill_value,
            face_node_connectivity,
            name=topology,
            edge_node_connectivity=edge_node_connectivity,
            dataset=ds[ugrid_vars],
            indexes=indexes,
            projected=projected,
            crs=None,
            start_index=start_index,
        )

    def _get_name_and_attrs(self, name: str):
        key = f"{name}_connectivity"
        attrs = conventions.DEFAULT_ATTRS[key]
        if "start_index" in attrs:
            attrs["start_index"] = self.start_index
        if "_FillValue" in attrs:
            attrs["_FillValue"] = self.fill_value
        return self._attrs[key], attrs

    def to_dataset(
        self, other: xr.Dataset = None, optional_attributes: bool = False
    ) -> xr.Dataset:
        node_x = self._indexes["node_x"]
        node_y = self._indexes["node_y"]
        face_nodes, face_nodes_attrs = self._get_name_and_attrs("face_node")
        nmax_node_dim = self._attrs["max_face_nodes_dimension"]
        edge_nodes, edge_nodes_attrs = self._get_name_and_attrs("edge_node")

        data_vars = {
            self.name: 0,
            face_nodes: xr.DataArray(
                data=self._adjust_connectivity(self.face_node_connectivity),
                attrs=face_nodes_attrs,
                dims=(self.face_dimension, nmax_node_dim),
            ),
        }
        if self.edge_node_connectivity is not None or optional_attributes:
            data_vars[edge_nodes] = xr.DataArray(
                data=self._adjust_connectivity(self.edge_node_connectivity),
                attrs=edge_nodes_attrs,
                dims=(self.edge_dimension, "two"),
            )
        if optional_attributes:
            face_edges, face_edges_attrs = self._get_name_and_attrs("face_edge")
            face_faces, face_faces_attrs = self._get_name_and_attrs("face_face")
            edge_faces, edge_faces_attrs = self._get_name_and_attrs("edge_face")
            bound_nodes, bound_nodes_attrs = self._get_name_and_attrs("boundary_node")
            boundary_edge_dim = self._attrs["boundary_edge_dimension"]

            data_vars[face_edges] = xr.DataArray(
                data=self._adjust_connectivity(self.face_edge_connectivity),
                attrs=face_edges_attrs,
                dims=(self.face_dimension, nmax_node_dim),
            )
            data_vars[face_faces] = xr.DataArray(
                data=self._adjust_connectivity(
                    connectivity.to_dense(
                        self.face_face_connectivity, self.n_max_node_per_face
                    )
                ),
                attrs=face_faces_attrs,
                dims=(self.face_dimension, nmax_node_dim),
            )
            data_vars[edge_faces] = xr.DataArray(
                data=self._adjust_connectivity(self.edge_face_connectivity),
                attrs=edge_faces_attrs,
                dims=(self.edge_dimension, "two"),
            )
            data_vars[bound_nodes] = xr.DataArray(
                data=self._adjust_connectivity(self.boundary_node_connectivity),
                attrs=bound_nodes_attrs,
                dims=(boundary_edge_dim, "two"),
            )

        attrs = {"Conventions": "CF-1.9 UGRID-1.0"}
        if other is not None:
            attrs.update(other.attrs)

        dataset = xr.Dataset(data_vars, attrs=attrs)
        if self._dataset:
            dataset = dataset.merge(self._dataset, compat="override")
        if other is not None:
            dataset = dataset.merge(other)
        if node_x not in dataset or node_y not in dataset:
            dataset = self.assign_node_coords(dataset)
        if optional_attributes:
            dataset = self.assign_face_coords(dataset)
            dataset = self.assign_edge_coords(dataset)

        dataset[self.name].attrs = self._filtered_attrs(dataset)
        return dataset

    # These are all optional/derived UGRID attributes. They are not computed by
    # default, only when called upon.
    @property
    def n_face(self) -> int:
        """Return the number of faces in the UGRID2D topology."""
        return self.face_node_connectivity.shape[0]

    @property
    def n_max_node_per_face(self) -> int:
        """
        Return the maximum number of nodes that a face can contain in the
        UGRID2D topology.
        """
        return self.face_node_connectivity.shape[1]

    @property
    def n_node_per_face(self) -> IntArray:
        return (self.face_node_connectivity != FILL_VALUE).sum(axis=1)

    @property
    def core_dimension(self):
        return self.face_dimension

    @property
    def dims(self):
        """Set of UGRID dimension names: node dimension, edge dimension, face_dimension."""
        return {
            self.node_dimension,
            self.edge_dimension,
            self.face_dimension,
        }

    @property
    def sizes(self):
        return {
            self.node_dimension: self.n_node,
            self.edge_dimension: self.n_edge,
            self.face_dimension: self.n_face,
        }

    @property
    def max_face_node_dimension(self) -> str:
        return self._attrs["max_face_nodes_dimension"]

    @property
    def max_connectivity_sizes(self) -> dict[str, int]:
        return {
            self.max_face_node_dimension: self.n_max_node_per_face,
        }

    @property
    def max_connectivity_dimensions(self) -> tuple[str]:
        return (self.max_face_node_dimension,)

    @property
    def topology_dimension(self):
        """Highest dimensionality of the geometric elements: 2"""
        return 2

    @property
    def face_dimension(self):
        """Return the name of the face dimension."""
        return self._attrs["face_dimension"]

    def _edge_connectivity(self):
        (
            self._edge_node_connectivity,
            self._face_edge_connectivity,
        ) = connectivity.edge_connectivity(
            self.face_node_connectivity,
            self._edge_node_connectivity,
        )

    @property
    def edge_node_connectivity(self) -> IntArray:
        """
        Edge to node connectivity. Every edge consists of a connection between
        two nodes.

        Returns
        -------
        connectivity: ndarray of integers with shape ``(n_edge, 2)``.
        """
        if self._edge_node_connectivity is None:
            self._edge_connectivity()
        return self._edge_node_connectivity

    @edge_node_connectivity.setter
    def edge_node_connectivity(self, value):
        self._edge_node_connectivity = value

    @property
    def face_edge_connectivity(self) -> csr_matrix:
        """
        Face to edge connectivity.

        Returns
        -------
        connectivity: csr_matrix
        """
        if self._face_edge_connectivity is None:
            self._edge_connectivity()
        return self._face_edge_connectivity

    @property
    def boundary_node_connectivity(self) -> IntArray:
        """
        Boundary node connectivity

        Returns
        -------
        connectivity: ndarray of integers with shape ``(n_boundary_edge, 2)``
        """
        if self._boundary_node_connectivity is None:
            self._boundary_node_connectivity = connectivity.boundary_node_connectivity(
                self.edge_face_connectivity,
                self.edge_node_connectivity,
            )
        return self._boundary_node_connectivity

    @property
    def centroids(self) -> FloatArray:
        """
        Centroid (x, y) of every face.

        Returns
        -------
        centroids: ndarray of floats with shape ``(n_face, 2)``
        """
        if self._centroids is None:
            self._centroids = connectivity.centroids(
                self.face_node_connectivity,
                self.node_x,
                self.node_y,
            )
        return self._centroids

    @property
    def circumcenters(self):
        """
        Circumenter (x, y) of every face; only works for fully triangular
        grids.
        """
        if self._circumcenters is None:
            self._circumcenters = connectivity.circumcenters(
                self.face_node_connectivity,
                self.node_x,
                self.node_y,
            )
        return self._circumcenters

    @property
    def area(self) -> FloatArray:
        """Area of every face."""
        if self._area is None:
            self._area = connectivity.area(
                self.face_node_connectivity,
                self.node_x,
                self.node_y,
            )
        return self._area

    @property
    def perimeter(self) -> FloatArray:
        """Perimeter length of every face."""
        if self._perimeter is None:
            self._perimeter = connectivity.perimeter(
                self.face_node_connectivity,
                self.node_x,
                self.node_y,
            )
        return self._perimeter

    @property
    def face_bounds(self):
        """
        Returns a numpy array with columns ``minx, miny, maxx, maxy``,
        describing the bounds of every face in the grid.

        Returns
        -------
        face_bounds: np.ndarray of shape (n_face, 4)
        """
        x = self.node_x[self.face_node_connectivity]
        y = self.node_y[self.face_node_connectivity]
        isfill = self.face_node_connectivity == FILL_VALUE
        x[isfill] = np.nan
        y[isfill] = np.nan
        return np.column_stack(
            [
                np.nanmin(x, axis=1),
                np.nanmin(y, axis=1),
                np.nanmax(x, axis=1),
                np.nanmax(y, axis=1),
            ]
        )

    @property
    def face_x(self):
        """x-coordinate of centroid of every face"""
        return self.centroids[:, 0]

    @property
    def face_y(self):
        """y-coordinate of centroid of every face"""
        return self.centroids[:, 1]

    @property
    def face_coordinates(self) -> FloatArray:
        """
        Centroid (x, y) of every face.

        Returns
        -------
        centroids: ndarray of floats with shape ``(n_face, 2)``
        """
        return self.centroids

    @property
    def face_node_coordinates(self) -> FloatArray:
        """
        Node coordinates of every face.

        "Fill node" coordinates are set as NaN.

        Returns
        -------
        face_node_coordinates: ndarray of floats with shape ``(n_face, n_max_node_per_face, 2)``
        """
        coords = np.full(
            (self.n_face, self.n_max_node_per_face, 2), np.nan, dtype=FloatDType
        )
        is_node = self.face_node_connectivity != FILL_VALUE
        index = self.face_node_connectivity[is_node]
        coords[is_node, :] = self.node_coordinates[index]
        return coords

    @property
    def edge_face_connectivity(self) -> IntArray:
        """
        Edge to face connectivity. An edge may belong to a single face
        (exterior edge), or it may be shared by two faces (interior edge).

        An exterior edge will contain a FILL_VALUE of -1 for the second column.

        Returns
        -------
        connectivity: ndarray of integers with shape ``(n_edge, 2)``.
        """
        if self._edge_face_connectivity is None:
            self._edge_face_connectivity = connectivity.invert_dense(
                self.face_edge_connectivity
            )
        return self._edge_face_connectivity

    @property
    def face_face_connectivity(self) -> csr_matrix:
        """
        Face to face connectivity. Derived from shared edges.

        The connectivity is represented as an adjacency matrix in CSR format,
        with the row and column indices as a (0-based) face index. The data of
        the matrix contains the edge index as every connection is formed by a
        shared edge.

        Returns
        -------
        connectivity: csr_matrix
        """
        if self._face_face_connectivity is None:
            self._face_face_connectivity = connectivity.face_face_connectivity(
                self.edge_face_connectivity
            )
        return self._face_face_connectivity

    @property
    def node_face_connectivity(self):
        """
        Node to face connectivity. Inverted from face node connectivity.

        Returns
        -------
        connectivity: csr_matrix
        """
        if self._node_face_connectivity is None:
            self._node_face_connectivity = connectivity.invert_dense_to_sparse(
                self.face_node_connectivity
            )
        return self._node_face_connectivity

    @property
    def coords(self):
        """Dictionary for grid coordinates."""
        return {
            self.node_dimension: self.node_coordinates,
            self.edge_dimension: self.edge_coordinates,
            self.face_dimension: self.face_coordinates,
        }

    def get_coordinates(self, dim: str) -> FloatArray:
        """Return the coordinates for the specified UGRID dimension."""
        if dim == self.node_dimension:
            return self.node_coordinates
        elif dim == self.edge_dimension:
            return self.edge_coordinates
        elif dim == self.face_dimension:
            return self.face_coordinates
        else:
            raise ValueError(
                f"Expected {self.node_dimension}, {self.edge_dimension}, or "
                f"{self.face_dimension}; got: {dim}",
            )

    def get_connectivity_matrix(self, dim: str, xy_weights: bool):
        """Return the connectivity matrix for the specified UGRID dimension."""
        if dim == self.node_dimension:
            connectivity = self.node_node_connectivity.copy()
            coordinates = self.node_coordinates
        elif dim == self.face_dimension:
            connectivity = self.face_face_connectivity.copy()
            coordinates = self.centroids
        else:
            raise ValueError(
                f"Expected {self.node_dimension} or {self.face_dimension}; got: {dim}"
            )

        if xy_weights:
            connectivity.data = self._connectivity_weights(connectivity, coordinates)

        return connectivity

    @property
    def mesh(self) -> "mk.Mesh2d":  # type: ignore # noqa
        """
        Create if needed, and return meshkernel Mesh2d object.

        Returns
        -------
        mesh: meshkernel.Mesh2d
        """
        import meshkernel as mk

        edge_nodes = self.edge_node_connectivity.ravel().astype(np.int32)
        is_node = self.face_node_connectivity != FILL_VALUE
        nodes_per_face = is_node.sum(axis=1).astype(np.int32)
        face_nodes = self.face_node_connectivity[is_node].ravel().astype(np.int32)

        if self._mesh is None:
            self._mesh = mk.Mesh2d(
                node_x=self.node_x,
                node_y=self.node_y,
                edge_nodes=edge_nodes,
                face_nodes=face_nodes,
                nodes_per_face=nodes_per_face,
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
            if self.is_geographic:
                mk_projection = mk.ProjectionType.SPHERICAL
            else:
                mk_projection = mk.ProjectionType.CARTESIAN
            self._meshkernel = mk.MeshKernel(mk_projection)
            self._meshkernel.mesh2d_set(self.mesh)
        return self._meshkernel

    @property
    def voronoi_topology(self):
        """
        Centroidal Voronoi tesselation of this UGRID2D topology.

        Returns
        -------
        vertices: ndarray of floats with shape ``(n_centroids, 2)``
        face_node_connectivity: csr_matrix
            Describes face node connectivity of voronoi topology.
        face_index: 1d array of integers
        """
        if self._voronoi_topology is None:
            vertices, faces, face_index, _ = voronoi_topology(
                self.node_face_connectivity,
                self.node_coordinates,
                self.centroids,
                self.edge_face_connectivity,
                self.edge_node_connectivity,
                add_exterior=True,
                add_vertices=False,
            )
            self._voronoi_topology = vertices, faces, face_index
        return self._voronoi_topology

    @property
    def centroid_triangulation(self):
        """
        Triangulation of centroidal voronoi tesselation.

        Required for e.g. contouring face data, which takes triangles and
        associated values at the triangle vertices.

        Returns
        -------
        vertices: ndarray of floats with shape ``(n_centroids, 2)``
        face_node_connectivity: ndarray of integers with shape ``(n_triangle, 3)``
            Describes face node connectivity of triangle topology.
        face_index: 1d array of integers
        """
        if self._centroid_triangulation is None:
            nodes, faces, face_index = self.voronoi_topology
            triangles, _ = connectivity.triangulate(faces)
            triangulation = (nodes[:, 0].copy(), nodes[:, 1].copy(), triangles)
            self._centroid_triangulation = (triangulation, face_index)
        return self._centroid_triangulation

    @property
    def triangulation(self):
        """
        Triangulation of the UGRID2D topology.

        Returns
        -------
        triangulation: tuple
            Contains node_x, node_y, triangle face_node_connectivity.
        triangle_face_connectivity: 1d array of integers
            Identifies the original face for every triangle.
        """
        if self._triangulation is None:
            triangles, triangle_face_connectivity = connectivity.triangulate(
                self.face_node_connectivity
            )
            triangulation = (self.node_x, self.node_y, triangles)
            self._triangulation = (triangulation, triangle_face_connectivity)
        return self._triangulation

    @property
    def exterior_edges(self) -> IntArray:
        """
        Get all exterior edges, i.e. edges with no other face.

        Returns
        -------
        edge_index: 1d array of integers
        """
        # Numpy argwhere doesn't return a 1D array
        return np.nonzero(self.edge_face_connectivity[:, 1] == FILL_VALUE)[0]

    @property
    def exterior_faces(self) -> IntArray:
        """
        Get all exterior faces, i.e. faces with an unshared edge.

        Returns
        -------
        face_index: 1d array of integers
        """
        exterior_edges = self.exterior_edges
        exterior_faces = self.edge_face_connectivity[exterior_edges].ravel()
        return np.unique(exterior_faces[exterior_faces != FILL_VALUE])

    @property
    def celltree(self):
        """
        Initializes the celltree if needed, and returns celltree.

        A celltree is a search structure for spatial lookups in unstructured grids.
        """
        if self._celltree is None:
            self._celltree = CellTree2d(
                self.node_coordinates, self.face_node_connectivity, FILL_VALUE
            )
        return self._celltree

    def validate_edge_node_connectivity(self):
        """
        Mark valid edges, by comparing face_node_connectivity and
        edge_node_connectivity. Edges that are not part of a face, as well as
        duplicate edges are marked ``False``.

        An error is raised if the face_node_connectivity defines more unique
        edges than the edge_node_connectivity.

        Returns
        -------
        valid: np.ndarray of bool
            Marks for every edge whether it is valid.

        Examples
        --------
        To purge invalid edges and associated data from a dataset that contains
        un-associated or duplicate edges:

        >>> uds = xugrid.open_dataset("example.nc")
        >>> valid = uds.ugrid.grid.validate_edge_node_connectivity()
        >>> purged = uds.isel({grid.edge_dimension: valid})
        """
        return connectivity.validate_edge_node_connectivity(
            self.face_node_connectivity,
            self.edge_node_connectivity,
        )

    def assign_face_coords(
        self,
        obj: Union[xr.DataArray, xr.Dataset],
    ) -> Union[xr.DataArray, xr.Dataset]:
        """
        Assign face coordinates from the grid to the object.

        Returns a new object with all the original data in addition to the new
        node coordinates of the grid.

        Parameters
        ----------
        obj: xr.DataArray or xr.Dataset

        Returns
        -------
        assigned (same type as obj)
        """
        xname = self._indexes.get("face_x", f"{self.name}_face_x")
        yname = self._indexes.get("face_y", f"{self.name}_face_y")
        x_attrs = conventions.DEFAULT_ATTRS["face_x"][self.projected]
        y_attrs = conventions.DEFAULT_ATTRS["face_y"][self.projected]
        coords = {
            xname: xr.DataArray(
                data=self.face_x,
                dims=(self.face_dimension,),
                attrs=x_attrs,
            ),
            yname: xr.DataArray(
                data=self.face_y,
                dims=(self.face_dimension,),
                attrs=y_attrs,
            ),
        }
        return obj.assign_coords(coords)

    def locate_points(self, points: FloatArray):
        """
        Find in which face points are located.

        Parameters
        ----------
        points: ndarray of floats with shape ``(n_point, 2)``

        Returns
        -------
        face_index: ndarray of integers with shape ``(n_points,)``
        """
        return self.celltree.locate_points(points)

    def intersect_edges(self, edges: FloatArray):
        """
        Find in which face edges are located and compute the intersection with
        the face edges.

        Parameters
        ----------
        edges: ndarray of floats with shape ``(n_edge, 2, 2)``
            The first dimensions represents the different edges.
            The second dimensions represents the start and end of every edge.
            The third dimensions reresent the x and y coordinate of every vertex.

        Returns
        -------
        edge_index: ndarray of integers with shape ``(n_intersection,)``
        face_index: ndarray of integers with shape ``(n_intersection,)``
        intersections: ndarray of float with shape ``(n_intersection, 2, 2)``
        """
        return self.celltree.intersect_edges(edges)

    def locate_bounding_box(
        self, xmin: float, ymin: float, xmax: float, ymax: float
    ) -> IntArray:
        """
        Find which faces are located in the bounding box. The centroids of the
        faces are used.

        Parameters
        ----------
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float

        Returns
        -------
        face_index: ndarray of bools with shape ``(n_face,)``
        """
        return np.nonzero(
            (self.face_x >= xmin)
            & (self.face_x < xmax)
            & (self.face_y >= ymin)
            & (self.face_y < ymax)
        )[0]

    def compute_barycentric_weights(
        self, points: FloatArray
    ) -> Tuple[IntArray, FloatArray]:
        """
        Find in which face the points are located, and compute the barycentric
        weight for every vertex of the face.

        Parameters
        ----------
        points: ndarray of floats with shape ``(n_point, 2)``

        Returns
        -------
        face_index: ndarray of integers with shape ``(n_points,)``
        weights: ndarray of floats with shape ```(n_points, n_max_node)``
        """
        return self.celltree.compute_barycentric_weights(points)

    def rasterize_like(
        self, x: FloatArray, y: FloatArray
    ) -> Tuple[FloatArray, FloatArray, IntArray]:
        """
        Rasterize unstructured grid by sampling on the x and y coordinates.

        Parameters
        ----------
        x: 1d array of floats with shape ``(ncol,)``
        y: 1d array of floats with shape ``(nrow,)``

        Returns
        -------
        x: 1d array of floats with shape ``(ncol,)``
        y: 1d array of floats with shape ``(nrow,)``
        face_index: 1d array of integers with shape ``(nrow * ncol,)``
        """
        yy, xx = np.meshgrid(y, x, indexing="ij")
        nodes = np.column_stack([xx.ravel(), yy.ravel()])
        index = self.celltree.locate_points(nodes).reshape((y.size, x.size))
        return x, y, index

    def rasterize(
        self,
        resolution: float,
        bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> Tuple[FloatArray, FloatArray, IntArray]:
        """
        Rasterize unstructured grid by sampling.

        x and y coordinates are generated from the bounds of the UGRID2D
        topology and the provided resolution.

        Parameters
        ----------
        resolution: float
            Spacing in x and y.
        bounds: tuple of four floats, optional
            xmin, ymin, xmax, ymax

        Returns
        -------
        x: 1d array of floats with shape ``(ncol,)``
        y: 1d array of floats with shape ``(nrow,)``
        face_index: 1d array of integers with shape ``(nrow * ncol,)``
        """
        if bounds is None:
            bounds = self.bounds
        xmin, ymin, xmax, ymax = bounds
        d = abs(resolution)
        xmin = np.floor(xmin / d) * d
        xmax = np.ceil(xmax / d) * d
        ymin = np.floor(ymin / d) * d
        ymax = np.ceil(ymax / d) * d
        x = np.arange(xmin + 0.5 * d, xmax, d)
        y = np.arange(ymax - 0.5 * d, ymin, -d)
        return self.rasterize_like(x, y)

    def topology_subset(
        self, face_index: Union[BoolArray, IntArray], return_index: bool = False
    ):
        """
        Create a new UGRID1D topology for a subset of this topology.

        Parameters
        ----------
        face_index: 1d array of integers or bool
            Edges of the subset.
        return_index: bool, optional
            Whether to return node_index, edge_index, face_index.

        Returns
        -------
        subset: Ugrid2d
        indexes: dict
            Dictionary with keys node dimension, edge dimension, face dimension
            and values their respective index. Only returned if return_index is
            True.
        """
        if not isinstance(face_index, pd.Index):
            face_index = as_pandas_index(face_index, self.n_face)

        # The pandas index may only contain uniques. So if size matches, it may
        # be the identity.
        range_index = pd.RangeIndex(0, self.n_face)
        if face_index.size == self.n_face and face_index.equals(range_index):
            # TODO: return self.copy instead?
            if return_index:
                indexes = {
                    self.node_dimension: pd.RangeIndex(0, self.n_node),
                    self.edge_dimension: pd.RangeIndex(0, self.n_edge),
                    self.face_dimension: range_index,
                }
                return self, indexes
            else:
                return self

        index = face_index.to_numpy()
        face_subset = self.face_node_connectivity[index]
        node_index = np.unique(face_subset.ravel())
        node_index = node_index[node_index != FILL_VALUE]
        new_faces = connectivity.renumber(face_subset)
        node_x = self.node_x[node_index]
        node_y = self.node_y[node_index]

        edge_index = None
        new_edges = None
        if self.edge_node_connectivity is not None:
            edge_index = np.unique(self.face_edge_connectivity[index].ravel())
            edge_index = edge_index[edge_index != FILL_VALUE]
            edge_subset = self.edge_node_connectivity[edge_index]
            new_edges = connectivity.renumber(edge_subset)

        grid = Ugrid2d(
            node_x,
            node_y,
            FILL_VALUE,
            new_faces,
            name=self.name,
            edge_node_connectivity=new_edges,
            indexes=self._indexes,
            projected=self.projected,
            crs=self.crs,
            attrs=self._attrs,
        )
        self._propagate_properties(grid)
        if return_index:
            indexes = {
                self.node_dimension: pd.Index(node_index),
                self.face_dimension: face_index,
            }
            if edge_index is not None:
                indexes[self.edge_dimension] = pd.Index(edge_index)
            return grid, indexes
        else:
            return grid

    def clip_box(
        self,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
    ):
        xmin, ymin, xmax, ymax = self.bounds
        bounds = [xmin, ymin, xmax, ymax]
        face_index = self.locate_bounding_box(*bounds)
        return self.topology_subset(face_index)

    def isel(self, indexers=None, return_index=False, **indexers_kwargs):
        """
        Select based on node, edge, or face.

        Face selection always results in a valid UGRID topology.
        Node or edge selection may result in invalid topologies (incomplete
        faces), and will error in such a case.

        Parameters
        ----------
        indexers: dict of str to np.ndarray of integers or bools
        return_index: bool, optional
            Whether to return node_index, edge_index, face_index.

        Returns
        -------
        obj: xr.Dataset or xr.DataArray
        grid: Ugrid2d
        indexes: dict
            Dictionary with keys node dimension, edge dimension, face dimension
            and values their respective index. Only returned if return_index is
            True.
        """
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "isel")
        alldims = set(self.dims)
        invalid = indexers.keys() - alldims
        if invalid:
            raise ValueError(
                f"Dimensions {invalid} do not exist. Expected one of {alldims}"
            )

        indexers = {k: as_pandas_index(v, self.sizes[k]) for k, v in indexers.items()}
        nodedim = self.node_dimension
        edgedim = self.edge_dimension
        facedim = self.face_dimension

        face_index = {}
        if nodedim in indexers:
            node_index = indexers[nodedim]
            face_index[nodedim] = np.unique(
                self.node_face_connectivity[node_index].data
            )
        if edgedim in indexers:
            edge_index = indexers[edgedim]
            index = np.unique(self.edge_face_connectivity[edge_index])
            face_index[edgedim] = index[index != FILL_VALUE]
        if facedim in indexers:
            face_index[facedim] = indexers[facedim]

        # Convert all to pandas index.
        face_index = {k: as_pandas_index(v, self.n_face) for k, v in face_index.items()}

        # Check the indexes against each other.
        index = self._precheck(face_index)
        grid, finalized_indexers = self.topology_subset(index, return_index=True)
        self._postcheck(indexers, finalized_indexers)

        if return_index:
            return grid, finalized_indexers
        else:
            return grid

    def _validate_indexer(self, indexer) -> Union[slice, np.ndarray]:
        if isinstance(indexer, slice):
            s = indexer
            if s.start is not None and s.stop is not None:
                if s.start >= s.stop:
                    raise ValueError(
                        "slice stop should be larger than slice start, received: "
                        f"start: {s.start}, stop: {s.stop}"
                    )
                if s.step is not None:
                    indexer = np.arange(s.start, s.stop, s.step)
            elif s.start is None or s.stop is None:
                if s.step is not None:
                    raise ValueError(
                        "step should be None if slice start or stop is None"
                    )

        else:  # Convert it into a 1d numpy array
            if isinstance(indexer, xr.DataArray):
                indexer = indexer.to_numpy()
            if isinstance(indexer, (list, np.ndarray, int, float)):
                indexer = np.atleast_1d(indexer)
            else:
                raise TypeError(
                    f"Invalid indexer type: {type(indexer).__name__}, "
                    "allowed types: integer, float, list, numpy array, xarray DataArray"
                )
            if indexer.ndim > 1:
                raise ValueError("index should be 0d or 1d")

        return indexer

    def _sel_box(
        self,
        obj,
        x: slice,
        y: slice,
    ):
        xmin, ymin, xmax, ymax = self.bounds
        bounds = [
            numeric_bound(x.start, xmin),
            numeric_bound(y.start, ymin),
            numeric_bound(x.stop, xmax),
            numeric_bound(y.stop, ymax),
        ]
        face_index = self.locate_bounding_box(*bounds)
        grid, indexes = self.topology_subset(face_index, return_index=True)
        indexes = {k: v for k, v in indexes.items() if k in obj.dims}
        new_obj = obj.isel(indexes)
        return new_obj, grid

    def _sel_line(
        self,
        obj,
        start,
        end,
    ):
        edges = np.array([[start, end]])
        _, index, xy = self.intersect_edges(edges)
        coords, index = section_coordinates(
            edges, xy, self.face_dimension, index, self.name
        )
        return obj.isel({self.face_dimension: index}).assign_coords(coords)

    def _sel_yline(
        self,
        obj,
        x: float,
        y: slice,
    ):
        xmin, _, xmax, _ = self.bounds
        if y.size != 1:
            raise ValueError(
                "If x is a slice without steps, y should be a single value"
            )
        y = y[0]
        xstart = numeric_bound(x.start, xmin)
        xstop = numeric_bound(x.stop, xmax)
        return self._sel_line(obj, start=(xstart, y), end=(xstop, y))

    def _sel_xline(
        self,
        obj,
        x: float,
        y: slice,
    ):
        _, ymin, _, ymax = self.bounds
        if x.size != 1:
            raise ValueError(
                "If y is a slice without steps, x should be a single value"
            )
        x = x[0]
        ystart = numeric_bound(y.start, ymin)
        ystop = numeric_bound(y.stop, ymax)
        return self._sel_line(obj, start=(x, ystart), end=(x, ystop))

    def sel_points(
        self, obj, x: FloatArray, y: FloatArray, out_of_bounds="warn", fill_value=np.nan
    ):
        """
        Select points in the unstructured grid.


        Parameters
        ----------
        x: 1d array of floats with shape ``(n_points,)``
        y: 1d array of floats with shape ``(n_points,)``
        obj: xr.DataArray or xr.Dataset
        out_of_bounds: str, default ``"warn"``
            What to do when points are located outside of any feature:

            * raise: raise a ValueError.
            * ignore: return ``fill_value`` for the out of bounds points.
            * warn: give a warning and return NaN for the out of bounds points.
            * drop: drop the out of bounds points. They may be identified
              via the ``index`` coordinate of the returned selection.
        fill_value: scalar, DataArray, Dataset, or callable, optional, default: np.nan
            Value to assign to out-of-bounds points if out_of_bounds is warn
            or ignore. Forwarded to xarray's ``.where()`` method.

        Returns
        -------
        selection: xr.DataArray or xr.Dataset
            The name of the topology is prefixed in the x, y coordinates.
        """
        options = ("warn", "raise", "ignore", "drop")
        if out_of_bounds not in options:
            str_options = ", ".join(options)
            raise ValueError(
                f"out_of_bounds must be one of {str_options}, "
                f"received: {out_of_bounds}"
            )

        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        if x.shape != y.shape:
            raise ValueError("shape of x does not match shape of y")
        if x.ndim != 1:
            raise ValueError("x and y must be 1d")
        dim = self.face_dimension
        xy = np.column_stack([x, y])
        index = self.locate_points(xy)

        keep = slice(None, None)  # keep all by default
        condition = None
        valid = index != -1
        if not valid.all():
            msg = "Not all points are located inside of the grid."
            if out_of_bounds == "raise":
                raise ValueError(msg)
            elif out_of_bounds in ("warn", "ignore"):
                if out_of_bounds == "warn":
                    warnings.warn(msg)
                condition = xr.DataArray(valid, dims=(dim,))
            elif out_of_bounds == "drop":
                index = index[valid]
                keep = valid

        # Create the selection DataArray or Dataset
        coords = {
            f"{self.name}_index": (dim, np.arange(len(xy))[keep]),
            f"{self.name}_x": (dim, xy[keep, 0]),
            f"{self.name}_y": (dim, xy[keep, 1]),
        }
        selection = obj.isel({dim: index}).assign_coords(coords)

        # Set values to fill_value for out-of-bounds
        if condition is not None:
            selection = selection.where(condition, other=fill_value)
        return selection

    def intersect_line(self, obj, start: Sequence[float], end: Sequence[float]):
        """
        Intersect a line with this grid, and fetch the values of the
        intersected faces.

        Parameters
        ----------
        obj: xr.DataArray or xr.Dataset
        start: sequence of two floats
            coordinate pair (x, y), designating the start point of the line.
        end: sequence of two floats
            coordinate pair (x, y), designating the end point of the line.

        Returns
        -------
        selection: xr.DataArray or xr.Dataset
            The name of the topology is prefixed in the x, y and s
            (spatium=distance) coordinates.
        """
        if (len(start) != 2) or (len(end) != 2):
            raise ValueError("Start and end coordinate pairs must have length two")
        return self._sel_line(obj, start, end)

    def intersect_linestring(
        self,
        obj: Union[xr.DataArray, xr.Dataset],
        linestring: "shapely.geometry.LineString",  # type: ignore # noqa
    ) -> Union[xr.DataArray, xr.Dataset]:
        """
        Intersect linestrings with this grid, and fetch the values of the
        intersected faces.

        Parameters
        ----------
        obj: xr.DataArray or xr.Dataset
        linestring: shapely.geometry.lineString

        Returns
        -------
        selection: xr.DataArray or xr.Dataset
            The name of the topology is prefixed in the x, y and s
            (spatium=distance) coordinates.
        """
        import shapely

        xy = shapely.get_coordinates([linestring])
        edges = np.stack((xy[:-1], xy[1:]), axis=1)
        edge_index, face_index, intersections = self.intersect_edges(edges)

        # Compute the cumulative length along the edges
        edge_length = np.linalg.norm(edges[:, 1] - edges[:, 0], axis=1)
        cumulative_length = np.empty_like(edge_length)
        cumulative_length[0] = 0
        np.cumsum(edge_length[:-1], out=cumulative_length[1:])

        # Compute the distance for every intersection to the start of the linestring.
        intersection_centroid = intersections.mean(axis=1)
        distance_node_to_intersection = np.linalg.norm(
            intersection_centroid - edges[edge_index, 0], axis=1
        )
        s = distance_node_to_intersection + cumulative_length[edge_index]

        # Now sort everything according to s.
        sorter = np.argsort(s)
        face_index = face_index[sorter]
        intersection_centroid = intersection_centroid[sorter]
        intersections = intersections[sorter]

        facedim = self.face_dimension
        coords = {
            f"{self.name}_s": (facedim, s[sorter]),
            f"{self.name}_x": (facedim, intersection_centroid[:, 0]),
            f"{self.name}_y": (facedim, intersection_centroid[:, 1]),
        }
        return obj.isel({facedim: face_index}).assign_coords(coords)

    def sel(self, obj, x=None, y=None):
        """
        Find selection in the UGRID x and y coordinates.

        The indexing for x and y always occurs orthogonally, i.e.:
        ``.sel(x=[0.0, 5.0], y=[10.0, 15.0])`` results in a four points. For
        vectorized indexing (equal to ``zip``ing through x and y), see
        ``.sel_points``.

        Parameters
        ----------
        obj: xr.DataArray or xr.Dataset
        x: float, 1d array, slice
        y: float, 1d array, slice

        Returns
        -------
        dimension: str
        as_ugrid: bool
        index: 1d array of integers
        coords: dict
        """

        if x is None:
            x = slice(None, None)
        if y is None:
            y = slice(None, None)

        x = self._validate_indexer(x)
        y = self._validate_indexer(y)
        if isinstance(x, slice) and isinstance(y, slice):
            f = self._sel_box
        elif isinstance(x, slice) and isinstance(y, np.ndarray):
            f = self._sel_yline
        elif isinstance(x, np.ndarray) and isinstance(y, slice):
            f = self._sel_xline
        elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            # Orthogonal points
            y, x = [a.ravel() for a in np.meshgrid(y, x, indexing="ij")]
            f = self.sel_points
        else:
            raise TypeError(
                f"Invalid indexer types: {type(x).__name__}, and {type(y).__name__}"
            )
        return f(obj, x, y)

    def label_partitions(self, n_part: int) -> "xugrid.UgridDataArray":
        """
        Generate partition labesl for this grid topology using METIS:
        https://github.com/KarypisLab/METIS

        This method utilizes the pymetis Python bindings:
        https://github.com/inducer/pymetis

        Parameters
        ----------
        n_part: integer
            The number of parts to partition the mesh.

        Returns
        -------
        partition_labels: UgridDataArray of integers
        """
        import pymetis

        adjacency_matrix = self.face_face_connectivity
        _, partition_index = pymetis.part_graph(
            nparts=n_part,
            xadj=adjacency_matrix.indptr,
            adjncy=adjacency_matrix.indices,
        )
        return xugrid.UgridDataArray(
            obj=xr.DataArray(
                data=np.array(partition_index),
                dims=(self.core_dimension,),
                name="labels",
            ),
            grid=self,
        )

    def partition(self, n_part: int):
        """
        Partition this grid topology using METIS:
        https://github.com/KarypisLab/METIS

        This method utilizes the pymetis Python bindings:
        https://github.com/inducer/pymetis

        Parameters
        ----------
        n_part: integer
            The number of parts to partition the mesh.

        Returns
        -------
        partitions
        """
        from xugrid.ugrid.partitioning import labels_to_indices

        labels = self.label_partitions(n_part)
        indices = labels_to_indices(labels.values)
        return [self.topology_subset(index) for index in indices]

    @staticmethod
    def merge_partitions(
        grids: Sequence["Ugrid2d"],
    ) -> tuple["Ugrid2d", dict[str, np.array]]:
        """
        Merge grid partitions into a single whole.

        Duplicate faces are included only once, and removed from subsequent
        partitions before merging.

        Parameters
        ----------
        grids: sequence of Ugrid2d

        Returns
        -------
        merged: Ugrid2d
        """

        from xugrid.ugrid import partitioning

        # Grab a sample grid
        grid = next(iter(grids))
        node_coordinates, node_indexes, node_inverse = partitioning.merge_nodes(grids)
        new_faces, face_indexes = partitioning.merge_faces(grids, node_inverse)
        indexes = {
            grid.node_dimension: node_indexes,
            grid.face_dimension: face_indexes,
        }

        if grid._edge_node_connectivity is not None:
            new_edges, edge_indexes = partitioning.merge_edges(grids, node_inverse)
            indexes[grid.edge_dimension] = edge_indexes
        else:
            new_edges = None

        merged_grid = Ugrid2d(
            *node_coordinates.T,
            FILL_VALUE,
            new_faces,
            name=grid.name,
            edge_node_connectivity=new_edges,
            indexes=grid._indexes,
            projected=grid.projected,
            crs=grid.crs,
            attrs=grid._attrs,
        )
        # Maintain fill_value, start_index
        grid._propagate_properties(merged_grid)
        return merged_grid, indexes

    def to_periodic(self, obj=None):
        """
        Convert this grid to a periodic grid, where the rightmost nodes are
        equal to the leftmost nodes. Note: for this to work, the y-coordinates
        on the left boundary must match those on the right boundary exactly.

        Returns
        -------
        periodic_grid: Ugrid2d
        aligned: xr.DataArray or xr.Dataset
        """
        xmin, _, xmax, _ = self.bounds
        coordinates = self.node_coordinates
        is_right = np.isclose(coordinates[:, 0], xmax)
        is_left = np.isclose(coordinates[:, 0], xmin)

        node_y = coordinates[:, 1]
        if not np.allclose(np.sort(node_y[is_left]), np.sort(node_y[is_right])):
            raise ValueError(
                "y-coordinates of the left and right boundaries do not match"
            )

        # Discard the rightmost nodes. Preserve the order in the faces, and the
        # order of the nodes.
        coordinates[is_right, 0] = xmin
        _, node_index, inverse = np.unique(
            coordinates, return_index=True, return_inverse=True, axis=0
        )
        inverse = inverse.ravel()
        # Create a mapping of the inverse index to the new node index.
        new_index = connectivity.renumber(node_index)
        new_faces = new_index[inverse[self.face_node_connectivity]]
        # Get the selection of nodes, and keep the order.
        node_index.sort()
        new_xy = self.node_coordinates[node_index]

        # Preserve the order of the edge_node_connectivity if it is present.
        new_edges = None
        edge_index = None
        if self._edge_node_connectivity is not None:
            new_edges = inverse[self.edge_node_connectivity]
            new_edges.sort(axis=1)
            _, edge_index = np.unique(new_edges, axis=0, return_index=True)
            edge_index.sort()
            new_edges = new_index[new_edges][edge_index]

        new = Ugrid2d(
            node_x=new_xy[:, 0],
            node_y=new_xy[:, 1],
            face_node_connectivity=new_faces,
            fill_value=FILL_VALUE,
            name=self.name,
            edge_node_connectivity=new_edges,
            indexes=self._indexes,
            projected=self.projected,
            crs=self.crs,
            attrs=self.attrs,
        )
        self._propagate_properties(new)

        if obj is not None:
            indexes = {
                self.face_dimension: pd.RangeIndex(0, self.n_face),
                self.node_dimension: pd.Index(node_index),
            }
            if edge_index is not None:
                indexes[self.edge_dimension] = pd.Index(edge_index)
            indexes = {k: v for k, v in indexes.items() if k in obj.dims}
            return new, obj.isel(**indexes)
        else:
            return new

    def to_nonperiodic(self, xmax: float, obj=None):
        """
        Convert this grid from a periodic grid (where the rightmost boundary shares its
        nodes with the leftmost boundary) to an aperiodic grid, where the leftmost nodes
        are separate from the rightmost nodes.

        Parameters
        ----------
        xmax: float
            The x-value of the newly created rightmost boundary nodes.
        obj: xr.DataArray or xr.Dataset

        Returns
        -------
        nonperiodic_grid: Ugrid2d
        aligned: xr.DataArray or xr.Dataset
        """
        xleft, _, xright, _ = self.bounds
        half_domain = 0.5 * (xright - xleft)

        # Extract all x coordinates for every face. Then identify the nodes
        # which have a value of e.g. -180, while the max x value for the face
        # is 180.0. These nodes should be duplicated.
        x = self.face_node_coordinates[..., 0]
        is_periodic = (np.nanmax(x, axis=1)[:, np.newaxis] - x) > half_domain
        periodic_nodes = self.face_node_connectivity[is_periodic]

        uniques, new_nodes = np.unique(periodic_nodes, return_inverse=True)
        new_x = np.full(uniques.size, xmax)
        new_y = self.node_y[uniques]
        new_faces = self.face_node_connectivity.copy()
        new_faces[is_periodic] = new_nodes + self.n_node

        # edge_node_connectivity must be rederived, since we've added a number
        # of new edges and new nodes.
        new = Ugrid2d(
            node_x=np.concatenate((self.node_x, new_x)),
            node_y=np.concatenate((self.node_y, new_y)),
            face_node_connectivity=new_faces,
            fill_value=FILL_VALUE,
            name=self.name,
            edge_node_connectivity=None,
            indexes=self._indexes,
            projected=self.projected,
            crs=self.crs,
            attrs=self.attrs,
        )
        self._propagate_properties(new)

        edge_index = None
        if self._edge_node_connectivity is not None:
            # If there is edge associated data, we need to duplicate the data
            # of the edges. It is impossible(?) to do this on the edges
            # directly, due to the possible presence of "symmetric" edges:
            #     2
            #    /|\
            #   / | \
            #  0__1__0
            #
            # (0, 1) and (1, 0) are topologically distinct, but only in the
            # face definition. In the new grid, the 0 on the right will have
            # become node 3, creating distinct edges.
            #
            # Note that any data with the edge is only stored once, which is
            # incorrect(!), but a given for these grids and would be a problem
            # for the simulation code producing these results.
            #
            # We use a casting trick to collapse two integers into one so we
            # can use searchsorted easily.
            edges = (
                np.sort(self.edge_node_connectivity, axis=1)
                .astype(np.int32)
                .view(np.int64)
                .ravel()
            )
            # Create a mapping of the new nodes created above, to the original nodes.
            # Then, find the new edges in the old using searchsorted.
            mapping = np.concatenate((np.arange(self.n_node), uniques))
            new_edges = (
                np.sort(mapping[new.edge_node_connectivity], axis=1)
                .astype(np.int32)
                .view(np.int64)
                .ravel()
            )
            edge_index = np.searchsorted(edges, new_edges, sorter=np.argsort(edges))
            # Reshuffle to keep the original order as intact as possible; how
            # much benefit does this actually give?
            sorter = np.argsort(edge_index)
            new._edge_node_connectivity = new._edge_node_connectivity[sorter]
            edge_index = edge_index[sorter]

        if obj is not None:
            indexes = {
                self.face_dimension: pd.RangeIndex(0, self.n_face),
                self.node_dimension: pd.Index(
                    np.concatenate((np.arange(self.n_node), uniques))
                ),
            }
            if edge_index is not None:
                indexes[self.edge_dimension] = pd.Index(edge_index)
            indexes = {k: v for k, v in indexes.items() if k in obj.dims}
            return new, obj.isel(**indexes)
        else:
            return new

    def reindex_like(
        self,
        other: "Ugrid2d",
        obj: Union[xr.DataArray, xr.Dataset],
        tolerance: float = 0.0,
    ):
        """
        Conform a DataArray or Dataset to match the topology of another Ugrid2D
        topology. The topologies must be exactly equivalent: only the order of
        the nodes, edges, and faces may differ.

        Parameters
        ----------
        other: Ugrid2d
        obj: DataArray or Dataset
        tolerance: float, default value 0.0.
            Maximum distance between inexact coordinate matches.

        Returns
        -------
        reindexed: DataArray or Dataset
        """
        if not isinstance(other, Ugrid2d):
            raise TypeError(f"Expected Ugrid2d, received: {type(other).__name__}")

        indexers = {
            self.node_dimension: connectivity.index_like(
                xy_a=self.node_coordinates,
                xy_b=other.node_coordinates,
                tolerance=tolerance,
            ),
            self.face_dimension: connectivity.index_like(
                xy_a=self.centroids,
                xy_b=other.centroids,
                tolerance=tolerance,
            ),
        }
        if other._edge_node_connectivity is not None:
            indexers[self.edge_dimension] = connectivity.index_like(
                xy_a=self.edge_coordinates,
                xy_b=other.edge_coordinates,
                tolerance=tolerance,
            )
        return obj.isel(indexers, missing_dims="ignore")

    def triangulate(self):
        """
        Triangulate this UGRID2D topology, breaks more complex polygons down
        into triangles.

        Returns
        -------
        triangles: Ugrid2d
        """
        triangles, _ = connectivity.triangulate(self.face_node_connectivity)
        grid = Ugrid2d(self.node_x, self.node_y, FILL_VALUE, triangles)
        self._propagate_properties(grid)
        return grid

    def _tesselate_voronoi(self, centroids, add_exterior, add_vertices, skip_concave):
        if add_exterior:
            edge_face_connectivity = self.edge_face_connectivity
            edge_node_connectivity = self.edge_node_connectivity
        else:
            edge_face_connectivity = None
            edge_node_connectivity = None

        vertices, faces, _, _ = voronoi_topology(
            self.node_face_connectivity,
            self.node_coordinates,
            centroids,
            edge_face_connectivity,
            edge_node_connectivity,
            add_exterior,
            add_vertices,
            skip_concave,
        )
        grid = Ugrid2d(vertices[:, 0], vertices[:, 1], FILL_VALUE, faces)
        self._propagate_properties(grid)
        return grid

    def tesselate_centroidal_voronoi(
        self, add_exterior=True, add_vertices=True, skip_concave=False
    ):
        """
        Create a centroidal Voronoi tesselation of this UGRID2D topology.

        Such a tesselation is not guaranteed to produce convex cells. To ensure
        convexity, set ``add_vertices=False`` -- this will result in a
        different exterior, however.

        Parameters
        ----------
        add_exterior: bool, default: True
        add_vertices: bool, default: True
        skip_concave: bool, default: False

        Returns
        -------
        tesselation: Ugrid2d
        """
        return self._tesselate_voronoi(
            self.centroids, add_exterior, add_vertices, skip_concave
        )

    def tesselate_circumcenter_voronoi(
        self, add_exterior=True, add_vertices=True, skip_concave=False
    ):
        """
        Create a circumcenter Voronoi tesselation of this UGRID2D topology.

        Such a tesselation is not guaranteed to produce convex cells. To ensure
        convexity, set ``add_vertices=False`` -- this will result in a
        different exterior, however.

        Parameters
        ----------
        add_exterior: bool, default: True
        add_vertices: bool, default: True
        skip_concave: bool, default: False

        Returns
        -------
        tesselation: Ugrid2d
        """
        return self._tesselate_voronoi(
            self.circumcenters, add_exterior, add_vertices, skip_concave
        )

    def reverse_cuthill_mckee(self, dimension=None):
        """
        Reduces bandwith of the connectivity matrix.

        Wraps :py:func:`scipy.sparse.csgraph.reverse_cuthill_mckee`.

        Returns
        -------
        reordered: Ugrid2d
        """
        # TODO: dispatch on dimension?
        reordering = reverse_cuthill_mckee(
            graph=self.face_face_connectivity,
            symmetric_mode=True,
        )
        reordered_grid = Ugrid2d(
            self.node_x,
            self.node_y,
            FILL_VALUE,
            self.face_node_connectivity[reordering],
        )
        self._propagate_properties(reordered_grid)
        return reordered_grid, reordering

    def refine_polygon(
        self,
        polygon: "shapely.geometry.Polygon",  # type: ignore # noqa
        min_face_size: float,
        refine_intersected: bool = True,
        use_mass_center_when_refining: bool = True,
        refinement_type: str = "refinement_levels",
        connect_hanging_nodes: bool = True,
        account_for_samples_outside_face: bool = True,
        max_refinement_iterations: int = 10,
    ):
        import meshkernel as mk

        geometry_list = mku.to_geometry_list(polygon)
        refinement_type = mku.either_string_or_enum(refinement_type, mk.RefinementType)

        self._initialize_mesh_kernel()
        mesh_refinement_params = mk.MeshRefinementParameters(
            refine_intersected,
            use_mass_center_when_refining,
            min_face_size,
            refinement_type,
            connect_hanging_nodes,
            account_for_samples_outside_face,
            max_refinement_iterations,
        )
        self._meshkernel.mesh2d_refine_based_on_polygon(
            geometry_list,
            mesh_refinement_params,
        )

    def delete_polygon(
        self,
        polygon: "shapely.geometry.Polygon",  # type: ignore # noqa
        delete_option: str = "all_face_circumenters",
        invert_deletion: bool = False,
    ):
        import meshkernel as mk

        geometry_list = mku.to_geometry_list(polygon)
        delete_option = mku.either_string_or_enum(delete_option, mk.DeleteMeshOption)
        self._initialize_mesh_kernel()
        self._meshkernel.mesh2d_delete(geometry_list, delete_option, invert_deletion)

    @staticmethod
    def from_polygon(
        polygon: "shapely.geometry.Polygon",  # type: ignore # noqa
    ):
        import meshkernel as mk

        geometry_list = mku.to_geometry_list(polygon)
        _mesh_kernel = mk.MeshKernel()
        _mesh_kernel.mesh2d_make_mesh_from_polygon(geometry_list)
        mesh = _mesh_kernel.mesh2d_get()
        n_max_node = mesh.nodes_per_face.max()
        ds = Ugrid2d.topology_dataset(
            mesh.node_x,
            mesh.node_y,
            mesh.face_nodes.reshape((-1, n_max_node)),
        )
        ugrid = Ugrid2d(ds)
        ugrid.mesh = mesh
        ugrid._meshkernel = _mesh_kernel
        return ugrid

    @staticmethod
    def earcut_triangulate_polygons(polygons, return_index: bool = False):
        """
        Break down polygons using mapbox_earcut, and create a mesh from the
        resulting triangles.

        Parameters
        ----------
        polygons: ndarray of shapely polygons
        return_index: bool, default is False.

        Returns
        -------
        grid: xugrid.Ugrid2d
        index: ndarray of integer, optional
            The polygon index for each triangle. Only provided if ``return_index``
            is True.
        """
        return xugrid.ugrid.burn.grid_from_earcut_polygons(
            polygons, return_index=return_index
        )

    @classmethod
    def from_geodataframe(cls, geodataframe: "geopandas.GeoDataFrame") -> "Ugrid2d":  # type: ignore # noqa
        """
        Convert a geodataframe of polygons to UGRID2D topology.

        Parameters
        ----------
        geodataframe: geopandas GeoDataFrame

        Returns
        -------
        topology: Ugrid2d
        """
        import geopandas as gpd

        if not isinstance(geodataframe, gpd.GeoDataFrame):
            raise TypeError(
                f"Expected GeoDataFrame, received: {type(geodataframe).__name__}"
            )
        return cls.from_shapely(geodataframe.geometry.to_numpy(), crs=geodataframe.crs)

    @staticmethod
    def from_shapely(geometry: PolygonArray, crs=None) -> "Ugrid2d":
        """
        Convert an array of shapely polygons to UGRID2D topology.

        Parameters
        ----------
        geometry: np.ndarray of shapely polygons
        crs: Any, optional
            Coordinate Reference System of the geometry objects. Can be anything accepted by
            :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
            such as an authority string (eg "EPSG:4326") or a WKT string.

        Returns
        -------
        topology: Ugrid2d
        """
        import shapely

        if not (shapely.get_type_id(geometry) == shapely.GeometryType.POLYGON).all():
            raise TypeError(
                "Can only create Ugrid2d from shapely Polygon geometries, "
                "geometry contains other types of geometries."
            )

        x, y, face_node_connectivity = conversion.polygons_to_faces(geometry)
        return Ugrid2d(x, y, FILL_VALUE, face_node_connectivity, crs=crs)

    @staticmethod
    def _from_intervals_helper(
        node_x: np.ndarray,
        node_y: np.ndarray,
        nx: int,
        ny: int,
    ) -> "Ugrid2d":
        linear_index = np.arange(node_x.size, dtype=IntDType).reshape((ny + 1, nx + 1))
        # Allocate face_node_connectivity
        face_nodes = np.empty((ny * nx, 4), dtype=IntDType)
        # Set connectivity in counterclockwise manner
        left, right = slice(None, -1), slice(1, None)
        lower, upper = slice(None, -1), slice(1, None)
        if node_x[1] < node_x[0]:  # x_decreasing
            left, right = right, left
        if node_y[ny + 1] < node_y[0]:  # y_decreasing
            lower, upper = upper, lower
        face_nodes[:, 0] = linear_index[lower, left].ravel()
        face_nodes[:, 1] = linear_index[lower, right].ravel()
        face_nodes[:, 2] = linear_index[upper, right].ravel()
        face_nodes[:, 3] = linear_index[upper, left].ravel()
        return Ugrid2d(node_x, node_y, -1, face_nodes)

    @staticmethod
    def from_structured_intervals1d(
        x_intervals: np.ndarray,
        y_intervals: np.ndarray,
    ) -> "Ugrid2d":
        """
        Create a Ugrid2d topology from a structured topology based on 1D intervals.

        Parameters
        ----------
        x_intervals: np.ndarray of shape (M + 1,)
            x-coordinate interval values for N row and M columns.
        y_intervals: np.ndarray of shape (N + 1,)
            y-coordinate interval values for N row and M columns.
        """
        x_intervals = np.asarray(x_intervals)
        y_intervals = np.asarray(y_intervals)
        nx = x_intervals.shape[0] - 1
        ny = y_intervals.shape[0] - 1
        node_y, node_x = (
            a.ravel() for a in np.meshgrid(y_intervals, x_intervals, indexing="ij")
        )
        return Ugrid2d._from_intervals_helper(node_x, node_y, nx, ny)

    @staticmethod
    def from_structured_intervals2d(
        x_intervals: np.ndarray,
        y_intervals: np.ndarray,
    ) -> "Ugrid2d":
        """
        Create a Ugrid2d topology from a structured topology based on 2D intervals.

        Parameters
        ----------
        x_intervals: np.ndarray of shape shape (N + 1, M + 1)
            x-coordinate interval values for N row and M columns.
        y_intervals: np.ndarray of shape shape (N + 1, M + 1)
            y-coordinate interval values for N row and M columns.
        """
        x_intervals = np.asarray(x_intervals)
        y_intervals = np.asarray(y_intervals)
        shape = x_intervals.shape
        if (x_intervals.ndim != 2) or (y_intervals.ndim != 2):
            raise ValueError("Dimensions of intervals must be 2D.")
        if shape != y_intervals.shape:
            raise ValueError(
                "Interval shapes must match. Found: "
                f"x_intervals: {shape}, versus y_intervals: {y_intervals.shape}"
            )
        nx = shape[1] - 1
        ny = shape[0] - 1
        node_x = x_intervals.ravel()
        node_y = y_intervals.ravel()
        return Ugrid2d._from_intervals_helper(node_x, node_y, nx, ny)

    @staticmethod
    def from_structured_bounds(
        x_bounds: np.ndarray,
        y_bounds: np.ndarray,
    ) -> "Ugrid2d":
        """
        Create a Ugrid2d topology from a structured topology based on 1D bounds.

        The bounds contain the lower and upper cell boundary for each cell.

        Parameters
        ----------
        x_bounds: np.ndarray of shape (M, 2)
            x-coordinate bounds for N row and M columns.
        y_bounds: np.ndarray of shape (N, 2)
            y-coordinate bounds for N row and M columns.

        Returns
        -------
        grid: Ugrid2d
        """
        nx, _ = x_bounds.shape
        ny, _ = y_bounds.shape
        x = conversion.bounds_to_vertices(x_bounds)
        y = conversion.bounds_to_vertices(y_bounds)
        node_y, node_x = (a.ravel() for a in np.meshgrid(y, x, indexing="ij"))
        return Ugrid2d._from_intervals_helper(node_x, node_y, nx, ny)

    @staticmethod
    def from_structured(
        data: Union[xr.DataArray, xr.Dataset],
        x: str | None = None,
        y: str | None = None,
    ) -> "Ugrid2d":
        """
        Create a Ugrid2d topology from an axis-aligned rectilinear structured topology.

        This method assumes the coordinates are 1D.

        Use ``from_structured_multicoord`` for 2D x and y coordinates, e.g. for
        (approximated) curvilinear and rotated structured topologies.

        Parameters
        ----------
        data: xr.DataArray or xr.Dataset
        x: str, optional
            Name of the 1D coordinate to use as the UGRID x-coordinate.
        y: str, optional
            Name of the 1D coordinate to use as the UGRID y-coordinate.

        Returns
        -------
        grid: Ugrid2d
        """
        if x is None or y is None:
            x, y = conversion.infer_xy_coords(data)
            if x is None or y is None:
                raise ValueError(
                    "Could not infer bounds. Please provide x and y explicitly."
                )

        x_intervals = conversion.infer_interval_breaks1d(data, x)
        y_intervals = conversion.infer_interval_breaks1d(data, y)
        return Ugrid2d.from_structured_intervals1d(x_intervals, y_intervals)

    @staticmethod
    def from_structured_multicoord(
        data: Union[xr.DataArray, xr.Dataset],
        x: str,
        y: str,
    ) -> "Ugrid2d":
        """
        Create a Ugrid2d topology from a structured topology, including rotated
        and (approximated) curvilinear topologies.

        This method assumes the coordinates are 2D.

        Use ``from_structured`` for 1D x and y coordinates, which is generally
        the case for axis-aligned rectilinear topologies (most rasters).

        Parameters
        ----------
        data: xr.DataArray or xr.Dataset
        x: str
            Name of the 2D coordinate to use as the UGRID x-coordinate.
        y: str
            Name of the 2D coordinate to use as the UGRID y-coordinate.

        Returns
        -------
        grid: Ugrid2d
        """
        xv = conversion.infer_interval_breaks(data[x], axis=1, check_monotonic=True)
        xv = conversion.infer_interval_breaks(xv, axis=0)
        yv = conversion.infer_interval_breaks(data[y], axis=1)
        yv = conversion.infer_interval_breaks(yv, axis=0, check_monotonic=True)
        return Ugrid2d.from_structured_intervals2d(xv, yv)

    def to_shapely(self, dim):
        """
        Convert UGRID topology to shapely objects.

        * nodes: points
        * edges: linestrings
        * faces: polygons

        Parameters
        ----------
        dim: str
            Node, edge, or face dimension.

        Returns
        -------
        geometry: ndarray of shapely.Geometry
        """
        if dim == self.face_dimension:
            return conversion.faces_to_polygons(
                self.node_x,
                self.node_y,
                self.face_node_connectivity,
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
                " Ugrid2d topology."
            )

    def bounding_polygon(self) -> "shapely.Polygon":  # type: ignore # noqa
        """
        Construct the bounding polygon of the grid. This polygon may include
        holes if the grid also contains holes.
        """
        import shapely

        def _bbox_area(bounds):
            return (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])

        edges = self.node_coordinates[self.boundary_node_connectivity]
        collection = shapely.polygonize(shapely.linestrings(edges))
        polygon = max(collection.geoms, key=lambda x: _bbox_area(x.bounds))
        return polygon

    def create_data_array(self, data: ArrayLike, facet: str) -> "xugrid.UgridDataArray":
        """
        Create a UgridDataArray from this grid and a 1D array of values.

        Parameters
        ----------
        data: array like
            Values for this array. Must be a ``numpy.ndarray`` or castable to
            it.
        grid: Ugrid1d, Ugrid2d
        facet: str
            With which facet to associate the data. Options for Ugrid1d are,
            ``"node"`` or ``"edge"``. Options for Ugrid2d are ``"node"``,
            ``"edge"``, or ``"face"``.

        Returns
        -------
        uda: UgridDataArray
        """
        if facet == "node":
            dimension = self.node_dimension
        elif facet == "edge":
            dimension = self.edge_dimension
        elif facet == "face":
            dimension = self.face_dimension
        else:
            raise ValueError(f"Invalid facet: {facet}. Must be one of: node, edge.")
        return self._create_data_array(data, dimension)
