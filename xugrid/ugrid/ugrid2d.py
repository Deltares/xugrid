from typing import Any, Dict, Tuple, Union

import geopandas as gpd
import meshkernel as mk
import numpy as np
import pyproj
import shapely.geometry as sg
import xarray as xr
from meshkernel.meshkernel import MeshKernel
from meshkernel.py_structures import Mesh2d
from numba_celltree import CellTree2d
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

from .. import connectivity, conversion
from .. import meshkernel_utils as mku
from ..typing import FloatArray, FloatDType, IntArray, IntDType, SparseMatrix
from ..voronoi import voronoi_topology
from . import ugrid_io
from .ugridbase import AbstractUgrid


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
        self._face_face_connectivity = None
        self._face_edge_connectivity = None
        self._node_edge_connectivity = None
        self._node_face_connectivity = None
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
        return self._remove_topology(obj, ugrid_io.UGRID2D_TOPOLOGY_VARIABLES)

    def topology_coords(self, obj: Union[xr.DataArray, xr.Dataset]) -> dict:
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
        return self.face_node_connectivity.shape[0]

    @property
    def topology_dimension(self):
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
    def edge_node_connectivity(self) -> FloatArray:
        if self._edge_node_connectivity is None:
            self._edge_connectivity()
        return self._edge_node_connectivity

    @property
    def face_edge_connectivity(self) -> FloatArray:
        if self._face_edge_connectivity is None:
            self._edge_connectivity()
        return self._face_edge_connectivity

    @property
    def centroids(self) -> FloatArray:
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
        return self.centroids[:, 0]

    @property
    def face_y(self):
        return self.centroids[:, 1]

    @property
    def face_coordinates(self) -> FloatArray:
        return self.centroids

    @property
    def edge_face_connectivity(self) -> IntArray:
        if self._edge_face_connectivity is None:
            self._edge_face_connectivity = connectivity.invert_dense(
                self.face_edge_connectivity, self.fill_value
            )
        return self._edge_face_connectivity

    @property
    def face_face_connectivity(self) -> csr_matrix:
        if self._face_face_connectivity is None:
            self._face_face_connectivity = connectivity.face_face_connectivity(
                self.edge_face_connectivity, self.fill_value
            )
        return self._face_face_connectivity

    @property
    def node_face_connectivity(self):
        if self._node_face_connectivity is None:
            self._node_face_connectivity = connectivity.invert_dense_to_sparse(
                self.face_node_connectivity, self.fill_value
            )
        return self._node_face_connectivity

    @property
    def mesh(self):
        if self._mesh is None:
            self._mesh = mk.Mesh2d(
                node_x=self.node_x,
                node_y=self.node_y,
                edge_nodes=self.edge_node_connectivity.flatten().astype(np.int32),
            )
        return self._mesh

    @property
    def meshkernel(self) -> mk.MeshKernel:
        if self._meshkernel is None:
            self._meshkernel = mk.MeshKernel(is_geographic=False)
            self._meshkernel.mesh2d_set(self.mesh)
        return self._meshkernel

    @property
    def voronoi_topology(self):
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
        if self._centroid_triangulation is None:
            nodes, faces, face_index = self.voronoi_topology
            triangles, _ = connectivity.triangulate(faces, self.fill_value)
            triangulation = (nodes[:, 0].copy(), nodes[:, 1].copy(), triangles)
            self._centroid_triangulation = (triangulation, face_index)
        return self._centroid_triangulation

    @property
    def triangulation(self):
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
        """
        # Numpy argwhere doesn't return a 1D array
        return np.nonzero(self.edge_face_connectivity[:, 1] == self.fill_value)[0]

    @property
    def exterior_faces(self) -> IntArray:
        """
        Get all exterior faces, i.e. faces with an unshared edge.
        """
        exterior_edges = self.exterior_edges
        exterior_faces = self.edge_face_connectivity[exterior_edges].ravel()
        return np.unique(exterior_faces[exterior_faces != self.fill_value])

    @property
    def celltree(self):
        """
        initializes the celltree, a search structure for spatial lookups in 2d grids
        """
        if self._celltree is None:
            self._celltree = CellTree2d(
                self.node_coordinates, self.face_node_connectivity, self.fill_value
            )
        return self._celltree

    def locate_faces(self, points):
        """
        returns the face indices in which the points lie. The points should be
        provided as an (N,2) array. The result is an (N) array
        """
        return self.celltree.locate_points(points)

    def locate_faces_bounding_box(self, xmin, xmax, ymin, ymax):
        """
        given a rectangular bounding box, this function returns the face
        indices which lie in it.
        """
        x = self.node_x
        y = self.node_y
        nodemask = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
        node_face_connectivity = connectivity.invert_dense_to_sparse(
            self.face_node_connectivity, self.fill_value
        )
        selected_faces = np.unique(node_face_connectivity[nodemask].indices)
        return selected_faces

    def rasterize(self, resolution: float) -> Tuple[FloatArray, FloatArray, IntArray]:
        xmin, ymin, xmax, ymax = self.bounds
        d = abs(resolution)
        xmin = np.floor(xmin / d) * d
        xmax = np.ceil(xmax / d) * d
        ymin = np.floor(ymin / d) * d
        ymax = np.ceil(ymax / d) * d
        x = np.arange(xmin + 0.5 * d, xmax, d)
        y = np.arange(ymax - 0.5 * d, ymin, -d)
        yy, xx = np.meshgrid(y, x, indexing="ij")
        nodes = np.column_stack([xx.ravel(), yy.ravel()])
        index = self.celltree.locate_points(nodes).reshape((y.size, x.size))
        return x, y, index

    def topology_subset(self, face_indices: IntArray):
        return self._topology_subset(face_indices, self.face_node_connectivity)

    def tesselate_centroidal_voronoi(self, add_exterior=True, add_vertices=True):
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
        geometry_list = mku.to_geometry_list(polygon)
        delete_option = mku.either_string_or_enum(delete_option, mk.DeleteMeshOption)
        self._initialize_mesh_kernel()
        self._meshkernel.mesh2d_delete(geometry_list, delete_option, invert_deletion)

    @staticmethod
    def from_polygon(
        polygon: sg.Polygon,
    ):
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
            """
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
