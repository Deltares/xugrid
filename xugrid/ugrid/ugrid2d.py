from typing import Any, Tuple, Union

import geopandas as gpd
import numpy as np
import pyproj
import shapely.geometry as sg
import xarray as xr
from numba_celltree import CellTree2d
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

from .. import connectivity, conversion
from .. import meshkernel_utils as mku
from ..typing import BoolArray, FloatArray, FloatDType, IntArray, IntDType, SparseMatrix
from ..voronoi import voronoi_topology
from . import ugrid_io
from .ugridbase import AbstractUgrid


def section_coordinates(
    edges: FloatArray, xy: FloatArray, dim: str, index: IntArray
) -> Tuple[IntArray, dict]:
    # TODO: add boundaries xy[:, 0] and xy[:, 1]
    xy_mid = 0.5 * (xy[:, 0, :] + xy[:, 1, :])
    s = np.linalg.norm(xy_mid - edges[0, 0], axis=1)
    order = np.argsort(s)
    coords = {
        "x": (dim, xy_mid[order, 0]),
        "y": (dim, xy_mid[order, 1]),
        "s": (dim, s[order]),
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
    edge_node_connectivity: ndarray of integers, optional
    dataset: xr.Dataset, optional
    name: string, optional
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
        face_node_connectivity: Union[IntArray, SparseMatrix],
        edge_node_connectivity: IntArray = None,
        dataset: xr.Dataset = None,
        name: str = None,
        crs: Any = None,
    ):
        self.node_x = np.ascontiguousarray(node_x)
        self.node_y = np.ascontiguousarray(node_y)

        if isinstance(face_node_connectivity, np.ndarray):
            face_node_connectivity = face_node_connectivity
        elif isinstance(face_node_connectivity, (coo_matrix, csr_matrix)):
            face_node_connectivity = connectivity.to_dense(
                face_node_connectivity, fill_value
            )
        else:
            raise TypeError(
                "face_node_connectivity should be an array of integers or a sparse matrix"
            )

        self.fill_value = fill_value
        self.face_node_connectivity = connectivity.counterclockwise(
            face_node_connectivity, self.fill_value, self.node_coordinates
        )
        if name is None:
            name = ugrid_io.UGRID2D_DEFAULT_NAME
        self.name = name
        self.topology_attrs = None

        # Optional attributes, deferred initialization
        # Meshkernel
        self._mesh = None
        self._meshkernel = None
        # Celltree
        self._celltree = None
        # Centroids
        self._centroids = None
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
        self._edge_face_connectivity = None
        self._node_edge_connectivity = None
        self._node_face_connectivity = None
        self._face_edge_connectivity = None
        self._face_face_connectivity = None
        # Derived topology
        self._triangulation = None
        self._voronoi_topology = None
        self._centroid_triangulation = None
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

    def _clear_geometry_properties(self):
        """Clear all properties that may have been invalidated"""
        # Meshkernel
        self._mesh = None
        self._meshkernel = None
        # Celltree
        self._celltree = None
        # Centroids
        self._centroids = None
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

    @staticmethod
    def from_dataset(dataset: xr.Dataset):
        """
        Extract the 2D UGRID topology information from an xarray Dataset.

        Parameters
        ----------
        dataset: xr.Dataset
            Dataset containing topology information stored according to UGRID conventions.

        Returns
        -------
        grid: Ugrid2d
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
        ugrid_ds["face_node_connectivity"] = ugrid_io.cast(
            ugrid_ds["face_node_connectivity"], fill_value, IntDType
        )
        if "edge_node_connectivity" in ugrid_ds:
            edge_node_connectivity = (
                ugrid_ds["edge_node_connectivity"].astype(IntDType).values
            )
        else:
            edge_node_connectivity = None

        return Ugrid2d(
            ugrid_ds["node_x"].values,
            ugrid_ds["node_y"].values,
            fill_value,
            ugrid_ds["face_node_connectivity"].values,
            edge_node_connectivity,
            topology_ds,
            mesh_topology.name,
        )

    def topology_dataset(self):
        """
        Store the UGRID topology information in an xarray Dataset according to
        the UGRID conventions.

        Returns
        -------
        ugrid_topology: xr.Dataset
        """
        return ugrid_io.ugrid2d_dataset(
            self.node_x,
            self.node_y,
            self.fill_value,
            self.face_node_connectivity,
            self._edge_node_connectivity,
            self.name,
            self.topology_attrs,
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
        return self._remove_topology(obj, ugrid_io.UGRID2D_TOPOLOGY_VARIABLES)

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
        facedim = self.face_dimension
        edgedim = self.edge_dimension
        nodedim = self.node_dimension
        attrs = self.topology_attrs

        if facedim in dims:
            # It's not a required attribute.
            if "face_coordinates" in attrs:
                name_x, name_y = attrs["face_coordinates"].split()
            else:
                name_x = f"{self.name}_face_x"
                name_y = f"{self.name}_face_y"
            coords[name_x] = (facedim, self.face_x)
            coords[name_y] = (facedim, self.face_y)
            attrs["face-coordinates"] = f"{name_x} {name_y}"

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

    # These are all optional/derived UGRID attributes. They are not computed by
    # default, only when called upon.
    @property
    def n_face(self):
        """
        Return the number of faces in the UGRID2D topology.
        """
        return self.face_node_connectivity.shape[0]

    @property
    def topology_dimension(self):
        """Highest dimensionality of the geometric elements: 2"""
        return 2

    def _get_dimension(self, dim):
        key = f"{dim}_dimension"
        if key not in self.topology_attrs:
            self.topology_attrs[key] = ugrid_io.UGrid.mesh2d_get_attributes(self.name)[
                key
            ]
        return self.topology_attrs[key]

    @property
    def face_dimension(self):
        """
        Return the name of the face dimension.
        """
        return self._get_dimension("face")

    def _edge_connectivity(self):
        (
            self._edge_node_connectivity,
            self._face_edge_connectivity,
        ) = connectivity.edge_connectivity(
            self.face_node_connectivity,
            self.fill_value,
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
                self.fill_value,
                self.node_x,
                self.node_y,
            )
        return self._centroids

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
    def edge_face_connectivity(self) -> IntArray:
        """
        Edge to face connectivity. An edge may belong to a single face
        (exterior edge), or it may be shared by two faces (interior edge).

        An exterior edge will contain a ``fill_value`` for the second column.

        Returns
        -------
        connectivity: ndarray of integers with shape ``(n_edge, 2)``.
        """
        if self._edge_face_connectivity is None:
            self._edge_face_connectivity = connectivity.invert_dense(
                self.face_edge_connectivity, self.fill_value
            )
        return self._edge_face_connectivity

    @property
    def face_face_connectivity(self) -> csr_matrix:
        """
        Face to face connectivity. Derived from shared edges.

        Returns
        -------
        connectivity: csr_matrix
        """
        if self._face_face_connectivity is None:
            self._face_face_connectivity = connectivity.face_face_connectivity(
                self.edge_face_connectivity, self.fill_value
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
                self.face_node_connectivity, self.fill_value
            )
        return self._node_face_connectivity

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
        is_node = self.face_node_connectivity != self.fill_value
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
            self._meshkernel = mk.MeshKernel(is_geographic=False)
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
            vertices, faces, face_index = voronoi_topology(
                self.node_face_connectivity,
                self.node_coordinates,
                self.centroids,
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
            triangles, _ = connectivity.triangulate(faces, self.fill_value)
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
                self.face_node_connectivity, self.fill_value
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
        return np.nonzero(self.edge_face_connectivity[:, 1] == self.fill_value)[0]

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
        return np.unique(exterior_faces[exterior_faces != self.fill_value])

    @property
    def celltree(self):
        """
        Initializes the celltree if needed, and returns celltree.

        A celltree is a search structure for spatial lookups in unstructured grids.
        """
        if self._celltree is None:
            self._celltree = CellTree2d(
                self.node_coordinates, self.face_node_connectivity, self.fill_value
            )
        return self._celltree

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

    def rasterize(self, resolution: float) -> Tuple[FloatArray, FloatArray, IntArray]:
        """
        Rasterize unstructured grid by sampling.

        x and y coordinates are generated from the bounds of the UGRID2D
        topology and the provided resolution.

        Parameters
        ----------
        resolution: float
            Spacing in x and y.

        Returns
        -------
        x: 1d array of floats with shape ``(ncol,)``
        y: 1d array of floats with shape ``(nrow,)``
        face_index: 1d array of integers with shape ``(nrow * ncol,)``
        """
        xmin, ymin, xmax, ymax = self.bounds
        d = abs(resolution)
        xmin = np.floor(xmin / d) * d
        xmax = np.ceil(xmax / d) * d
        ymin = np.floor(ymin / d) * d
        ymax = np.ceil(ymax / d) * d
        x = np.arange(xmin + 0.5 * d, xmax, d)
        y = np.arange(ymax - 0.5 * d, ymin, -d)
        return self.rasterize_like(x, y)

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
                indexer = indexer.values
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

    def sel_points(self, x: FloatArray, y: FloatArray):
        """
        Select points in the unstructured grid.

        Parameters
        ----------
        x: 1d array of floats with shape ``(n_points,)``
        y: 1d array of floats with shape ``(n_points,)``

        Returns
        -------
        dimension: str
        as_ugrid: bool
        index: 1d array of integers
        coords: dict
        """
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        if x.shape != y.shape:
            raise ValueError("shape of x does not match shape of y")
        if x.ndim != 1:
            raise ValueError("x and y must be 1d")
        dim = self.face_dimension
        xy = np.column_stack([x, y])
        index = self.locate_points(xy)
        valid = index != -1
        index = index[valid]
        coords = {
            "x": (dim, xy[valid, 0]),
            "y": (dim, xy[valid, 1]),
        }
        return dim, False, index, coords

    def sel(self, x=None, y=None):
        """
        Find selection in the UGRID x and y coordinates.

        The indexing for x and y always occurs orthogonally, i.e.:
        ``.sel(x=[0.0, 5.0], y=[10.0, 15.0])`` results in a four points. For
        vectorized indexing (equal to ``zip``ing through x and y), see
        ``.sel_points``.

        Parameters
        ----------
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
        dim = self.face_dimension
        as_ugrid = False
        xmin, ymin, xmax, ymax = self.bounds
        if isinstance(x, slice) and isinstance(y, slice):
            # Bounding box selection
            # Fill None values by infinite
            bounds = [
                numeric_bound(x.start, xmin),
                numeric_bound(y.start, ymin),
                numeric_bound(x.stop, xmax),
                numeric_bound(y.stop, ymax),
            ]
            as_ugrid = True
            coords = {}
            index = self.locate_bounding_box(*bounds)
        elif isinstance(x, slice) and isinstance(y, np.ndarray):
            if y.size != 1:
                raise ValueError(
                    "If x is a slice without steps, y should be a single value"
                )
            y = y[0]
            xstart = numeric_bound(x.start, xmin)
            xstop = numeric_bound(x.stop, xmax)
            edges = np.array([[[xstart, y], [xstop, y]]])
            _, index, xy = self.intersect_edges(edges)
            coords, index = section_coordinates(edges, xy, dim, index)

        elif isinstance(x, np.ndarray) and isinstance(y, slice):
            if x.size != 1:
                raise ValueError(
                    "If y is a slice without steps, x should be a single value"
                )
            x = x[0]
            ystart = numeric_bound(y.start, ymin)
            ystop = numeric_bound(y.stop, ymax)
            edges = np.array([[[x, ystart], [x, ystop]]])
            _, index, xy = self.intersect_edges(edges)
            coords, index = section_coordinates(edges, xy, dim, index)

        elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            # Orthogonal points
            yy, xx = np.meshgrid(y, x, indexing="ij")
            return self.sel_points(xx.ravel(), yy.ravel())

        else:
            raise TypeError(
                f"Invalid indexer types: {type(x).__name__}, and {type(y).__name__}"
            )
        return dim, as_ugrid, index, coords

    def topology_subset(self, index: Union[BoolArray, IntArray]):
        """
        Create a new UGRID1D topology for a subset of this topology.

        Parameters
        ----------
        face_index: 1d array of integers or bool
            Edges of the subset.

        Returns
        -------
        subset: Ugrid2d
        """
        index = np.atleast_1d(index)
        return self._topology_subset(index, self.face_node_connectivity)

    def triangulate(self):
        """
        Triangulate this UGRID2D topology, breaks more complex polygons down
        into triangles.

        Returns
        -------
        triangles: Ugrid2d
        """
        triangles, _ = connectivity.triangulate(
            self.face_node_connectivity, self.fill_value
        )
        return Ugrid2d(self.node_x, self.node_y, self.fill_value, triangles)

    def tesselate_centroidal_voronoi(self, add_exterior=True, add_vertices=True):
        """
        Create a centroidal Voronoi tesselation of this UGRID2D topology.

        Such a tesselation is not guaranteed to produce convex cells. To ensure
        convexity, set ``add_vertices=False`` -- this will result in a
        different exterior, however.

        Parameters
        ----------
        add_exterior: bool, default: True
        add_vertices: bool, default: True

        Returns
        -------
        tesselation: Ugrid2d
        """
        if add_exterior:
            edge_face_connectivity = self.edge_face_connectivity
            edge_node_connectivity = self.edge_node_connectivity
        else:
            edge_face_connectivity = None
            edge_node_connectivity = None

        vertices, faces, _ = voronoi_topology(
            self.node_face_connectivity,
            self.node_coordinates,
            self.centroids,
            edge_face_connectivity,
            edge_node_connectivity,
            self.fill_value,
            add_exterior,
            add_vertices,
        )
        faces = connectivity.to_dense(faces, self.fill_value)
        return Ugrid2d(vertices[:, 0], vertices[:, 1], self.fill_value, faces)

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
            self.fill_value,
            self.face_node_connectivity[reordering],
        )
        return reordered_grid, reordering

    def refine_polygon(
        self,
        polygon: sg.Polygon,
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
        polygon: sg.Polygon,
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
        polygon: sg.Polygon,
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
    def from_geodataframe(geodataframe: gpd.GeoDataFrame):
        """
        Convert a geodataframe of polygons to UGRID2D topology.

        Returns
        -------
        topology: Ugrid2d
        """
        x, y, face_node_connectivity, fill_value = conversion.polygons_to_faces(
            geodataframe.geometry.values
        )
        return Ugrid2d(x, y, fill_value, face_node_connectivity)

    @staticmethod
    def from_structured(data: Union[xr.DataArray, xr.Dataset]) -> "Ugrid2d":
        """
        Derive the 2D-UGRID quadrilateral mesh topology from a structured DataArray
        or Dataset, with (2D-dimensions) "y" and "x".

        Parameters
        ----------
        data: Union[xr.DataArray, xr.Dataset]
            Structured data from which the "x" and "y" coordinate will be used to
            define the UGRID-2D topology.

        Returns
        -------
        ugrid_topology: Ugrid2d
        """

        def _coord(da, dim):
            """
            Transform N xarray midpoints into N + 1 vertex edges

            This assumes cell sizes can be provided by a "dx" or "dy"
            coordinate. This is messy and should almost certainly be replaced
            by an actual standard such as CF bounds -- however, xarray
            initially did decode bounds as coordinates, see:

            https://github.com/pydata/xarray/pull/2844

            See also:
            https://cf-xarray.readthedocs.io/en/latest/generated/cf_xarray.bounds_to_vertices.html
            """
            delta_dim = "d" + dim  # e.g. dx, dy, dz, etc.

            # If empty array, return empty
            if da[dim].size == 0:
                raise ValueError(f"{dim} size must be >= 1")

            if delta_dim in da.coords:  # equidistant or non-equidistant
                dx = da[delta_dim].values
                if dx.shape == () or dx.shape == (1,):  # scalar -> equidistant
                    dxs = np.full(da[dim].size, dx)
                else:  # array -> non-equidistant
                    dxs = dx

                if not ((dxs > 0.0).all() ^ (dxs < 0.0).all()):
                    raise ValueError(f"{dim} is not only increasing or only decreasing")

            else:  # undefined -> equidistant
                if da[dim].size == 1:
                    raise ValueError(
                        f"DataArray has size 1 along {dim}, so cellsize must be provided"
                        " as a coordinate."
                    )
                dxs = np.diff(da[dim].values)
                dx = dxs[0]
                atolx = abs(1.0e-4 * dx)
                if not np.allclose(dxs, dx, atolx):
                    raise ValueError(
                        f"DataArray has to be equidistant along {dim}, or cellsizes"
                        " must be provided as a coordinate."
                    )
                dxs = np.full(da[dim].size, dx)

            dxs = np.abs(dxs)
            x = da[dim].values
            if not da.indexes[dim].is_monotonic_increasing:
                x = x[::-1]
                dxs = dxs[::-1]

            # This assumes the coordinate to be monotonic increasing
            x0 = x[0] - 0.5 * dxs[0]
            x = np.full(dxs.size + 1, x0)
            x[1:] += np.cumsum(dxs)
            return x

        # Transform midpoints into vertices
        # These are always returned monotonically increasing
        xcoord = _coord(data, "x")
        ycoord = _coord(data, "y")
        # Compute all vertices, these are the ugrid nodes
        node_y, node_x = (a.ravel() for a in np.meshgrid(ycoord, xcoord, indexing="ij"))
        linear_index = np.arange(node_x.size, dtype=np.intp).reshape(
            ycoord.size, xcoord.size
        )
        # Allocate face_node_connectivity
        nfaces = (ycoord.size - 1) * (xcoord.size - 1)
        face_nodes = np.empty((nfaces, 4), dtype=IntDType)
        # Set connectivity in counterclockwise manner
        face_nodes[:, 0] = linear_index[:-1, 1:].ravel()  # upper right
        face_nodes[:, 1] = linear_index[:-1, :-1].ravel()  # upper left
        face_nodes[:, 2] = linear_index[1:, :-1].ravel()  # lower left
        face_nodes[:, 3] = linear_index[1:, 1:].ravel()  # lower right
        return Ugrid2d(node_x, node_y, -1, face_nodes)

    def to_pygeos(self, dim):
        """
        Convert UGRID topology to pygeos objects.

        * nodes: points
        * edges: linestrings
        * faces: polygons

        Parameters
        ----------
        dim: str
            Node, edge, or face dimension.

        Returns
        -------
        geometry: ndarray of pygeos.Geometry
        """
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
                " Ugrid2d topology."
            )
