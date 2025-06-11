from typing import Optional

import numba as nb
import numpy as np
import xarray as xr

import xugrid as xu
from xugrid.constants import FloatDType
from xugrid.regrid.grid.basegrid import Grid
from xugrid.ugrid import voronoi
from xugrid.ugrid.ugrid2d import Ugrid2d


@nb.njit(cache=True)
def replace_interpolated_weights(
    vertices,
    faces,
    face_index,
    weights,
    node_to_node_map,
    node_index_threshold,
):
    n, m = weights.shape
    for i in range(n):
        face = faces[face_index[i]]
        weights_row = weights[i]
        for j in range(m):
            p = face[j]
            w = weights_row[j]
            if (p < node_index_threshold) or (w <= 0):
                continue
            # Find the two surrounding nodes (q and r)
            index = p - node_index_threshold
            q, r = node_to_node_map[index]
            px, py = vertices[p]
            qx, qy = vertices[q]
            rx, ry = vertices[r]
            # Compute the euclidian distance to both
            p_q = np.sqrt((qx - px) ** 2 + (qy - py) ** 2)
            p_r = np.sqrt((rx - px) ** 2 + (ry - py) ** 2)
            total = p_q + p_r
            # Redistribute weight according to inverse distance.
            weight_q = (p_r / total) * w
            weight_r = (p_q / total) * w
            # Set weights to zero for p, and add to r and q weights.
            weights[i, j] = 0.0
            # Search for p and q
            for jj in range(m):
                node = face[jj]
                if node == q:
                    weights[i, jj] += weight_q
                if node == r:
                    weights[i, jj] += weight_r
    return


class UnstructuredGrid2d(Grid):
    """
    Stores only the grid topology.

    e.g. face -> face

    Parameters
    ----------
    grid: Ugrid2d
    """

    def __init__(self, obj, dim: str):
        # TODO: do not omit type check on grid!
        if isinstance(obj, (xu.UgridDataArray, xu.UgridDataset)):
            self.ugrid_topology = obj.grid
        elif isinstance(obj, Ugrid2d):
            self.ugrid_topology = obj
        else:
            options = {"Ugrid2d", "UgridDataArray", "UgridDataset"}
            raise TypeError(
                f"Expected one of {options}, received: {type(obj).__name__}"
            )

        grid = self.ugrid_topology
        if dim == grid.node_dimension:
            self._coordinates = grid.node_coordinates
            self._dims = (grid.node_dimension,)
            self._shape = (grid.n_node,)
            self._size = grid.n_node
        elif dim == grid.face_dimension:
            self._coordinates = grid.face_coordinates
            self._dims = (grid.face_dimension,)
            self._shape = (grid.n_face,)
            self._size = grid.n_face
        else:
            raise ValueError(
                f"Expected {grid.node_dimension} or {grid.face_dimension}, "
                f"received: {dim}"
            )
        self.dimension = dim

    @property
    def ndim(self):
        return 1

    @property
    def coordinates(self) -> np.ndarray:
        return self._coordinates

    @property
    def dims(self):
        return self._dims

    @property
    def shape(self):
        return self._shape

    @property
    def size(self):
        return self._size

    @property
    def area(self):
        return self.ugrid_topology.area

    @property
    def kdtree(self):
        # This instantiates the kdtree, which is expensive.
        # Only do so when required!
        grid = self.ugrid_topology
        if self.dimension == grid.node_dimension:
            return grid.node_kdtree
        elif self.dimension == grid.face_dimension:
            return grid.face_kdtree

    def convert_to(self, matched_type):
        if isinstance(self, matched_type):
            return self
        else:
            TypeError(f"Cannot convert UnstructuredGrid2d to {matched_type.__name__}")

    def overlap(self, other, relative: bool):
        """
        Parameters
        ----------
        other: UnstructuredGrid2d
        relative: bool
            Whether to divide by the original area. Used for e.g.
            first-order-conservative methods.

        Returns
        -------
        source_index: 1d np.ndarray of int
        target_index: 1d np.ndarray of int
        weights: 1d np.ndarray of float
        """
        (
            target_index,
            source_index,
            weights,
        ) = self.ugrid_topology.celltree.intersect_faces(
            vertices=other.ugrid_topology.node_coordinates,
            faces=other.ugrid_topology.face_node_connectivity,
            fill_value=other.ugrid_topology.fill_value,
        )
        if relative:
            weights /= self.area[source_index]
        return source_index, target_index, weights

    def locate_nearest(self, other, tolerance: Optional[float] = None):
        _, source_index = self.kdtree.query(other.coordinates, workers=-1)
        celltree = self.ugrid_topology.celltree
        inside = celltree.locate_points(other.coordinates, tolerance=tolerance) != -1
        source_index = source_index[inside]
        target_index = np.arange(other.size, dtype=source_index.dtype)[inside]
        weight_values = np.ones_like(source_index, dtype=FloatDType)
        return source_index, target_index, weight_values

    def locate_inside(self, other, tolerance: Optional[float] = None):
        celltree = self.ugrid_topology.celltree
        source_index = celltree.locate_points(other.coordinates, tolerance)
        inside = source_index != -1
        source_index = source_index[inside]
        target_index = np.arange(other.size, dtype=source_index.dtype)[inside]
        weight_values = np.ones_like(source_index, dtype=FloatDType)
        return source_index, target_index, weight_values

    def _nodes_barycentric(self, other, tolerance: Optional[float] = None):
        points = other.coordinates
        grid = self.ugrid_topology
        outside = grid.locate_points(points) == -1
        face_index, weights = grid.compute_barycentric_weights(points, tolerance)
        weights[outside] = 0
        source_index = grid.face_node_connectivity[face_index].ravel()
        return source_index, weights

    def _faces_barycentric(self, other, tolerance: Optional[float] = None):
        points = other.coordinates
        grid = self.ugrid_topology
        outside = grid.locate_points(points) == -1
        (
            vertices,
            faces,
            node_to_face_index,
            node_to_node_map,
        ) = voronoi.voronoi_topology(
            grid.node_face_connectivity,
            grid.node_coordinates,
            grid.centroids,
            edge_face_connectivity=grid.edge_face_connectivity,
            edge_node_connectivity=grid.edge_node_connectivity,
            add_exterior=True,
            add_vertices=True,
            skip_concave=True,
        )
        voronoi_grid = Ugrid2d(
            vertices[:, 0],
            vertices[:, 1],
            -1,
            faces,
        )
        face_index, weights = voronoi_grid.compute_barycentric_weights(
            points, tolerance
        )

        # Find which nodes are interpolated. Redistribute their weights
        # according to distance to projection vertex.
        replace_interpolated_weights(
            vertices=vertices,
            faces=faces,
            face_index=face_index,
            weights=weights,
            node_to_node_map=node_to_node_map,
            node_index_threshold=len(vertices) - len(node_to_node_map),
        )
        weights[outside] = 0
        source_index = node_to_face_index[
            voronoi_grid.face_node_connectivity[face_index]
        ].ravel()
        return source_index, weights

    def barycentric(self, other, tolerance: Optional[float] = None):
        grid = self.ugrid_topology
        if self.dimension == grid.node_dimension:
            source_index, weights = self._nodes_barycentric(other, tolerance)
        elif self.dimension == grid.face_dimension:
            source_index, weights = self._faces_barycentric(other, tolerance)

        n_points, n_max_node = weights.shape
        keep = weights.ravel() > 0
        source_index = source_index[keep]
        target_index = np.repeat(np.arange(n_points), n_max_node)[keep]
        weights = weights.ravel()[keep]
        return self.sorted_output(source_index, target_index, weights)

    def intersection_length(self, other: "xu.regrid.network.Network1d", relative: bool):
        (
            target_index,
            source_index,
            intersections,
        ) = self.ugrid_topology.celltree.intersect_edges(
            other.ugrid_topology.edge_node_coordinates
        )
        length = np.linalg.norm(np.diff(intersections, axis=1)[:, 0, :], axis=-1)
        if relative:
            length /= other.length[source_index]
        return self.sorted_output(source_index, target_index, length)

    def to_dataset(self, name: str):
        ds = self.ugrid_topology.rename(name).to_dataset()
        ds[name + "_type"] = xr.DataArray(-1, attrs={"type": "UnstructuredGrid2d"})
        return ds
