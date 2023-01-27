import numpy as np

from xugrid.ugrid import voronoi
from xugrid.ugrid.ugrid2d import Ugrid2d


class UnstructuredGrid2d:
    """
    e.g. face -> face

    Parameters
    ----------
    grid: Ugrid2d
    """

    def __init__(self, grid):
        self.grid = grid

    @property
    def dims(self):
        return (self.grid.face_dimension,)

    @property
    def shape(self):
        return (self.grid.n_face,)

    @property
    def size(self):
        return self.grid.n_face

    @property
    def area(self):
        return self.grid.area

    def overlap(self, other, relative: bool):
        """
        Parameters
        ----------
        other: UnstructuredGrid2d
        """
        target_index, source_index, weights = self.grid.celltree.intersect_faces(
            vertices=other.grid.node_coordinates,
            faces=other.grid.face_node_connectivity,
            fill_value=other.grid.fill_value,
        )
        if relative:
            weights /= self.area[source_index]
        return source_index, target_index, weights

    def locate(self, points):
        grid = self.grid
        face_index = grid.locate_points(points)
        inside = face_index != grid.fill_value
        source_index = face_index[inside]
        target_index = np.arange(len(points))[inside]
        weights = np.full(source_index.size, 1.0, dtype=float)
        return source_index, target_index, weights

    def barycentric(self, other):
        points = other.grid.centroids
        grid = self.grid

        # Create a voronoi grid to get surrounding nodes as vertices
        vertices, faces, node_to_face_index = voronoi.voronoi_topology(
            grid.node_face_connectivity,
            grid.node_coordinates,
            grid.centroids,
            #    edge_face_connectivity=grid.edge_face_connectivity,
            #    edge_node_connectivity=grid.edge_node_connectivity,
            #    fill_value=grid.fill_value,
            #    add_exterior=True,
            #    add_vertices=False,
        )

        voronoi_grid = Ugrid2d(
            vertices[:, 0],
            vertices[:, 1],
            -1,
            faces,
        )

        face_index, weights = voronoi_grid.compute_barycentric_weights(points)
        n_points, n_max_node = weights.shape
        keep = weights.ravel() > 0
        source_index = node_to_face_index[
            voronoi_grid.face_node_connectivity[face_index]
        ].ravel()[keep]
        target_index = np.repeat(np.arange(n_points), n_max_node)[keep]
        weights = weights.ravel()[keep]

        # Look for points falling outside.
        outside = face_index == -1
        other_points = points[outside]
        sampled_index = grid.locate_points(other_points)
        sampled_inside = sampled_index != grid.fill_value
        other_target = np.arange(n_points)[outside][sampled_inside]
        other_source = sampled_index[sampled_inside]

        # Combine first and second
        source_index = np.concatenate((source_index, other_source))
        target_index = np.concatenate((target_index, other_target))
        weights = np.concatenate((weights, np.ones(other_target.size, dtype=float)))

        order = np.argsort(target_index)
        return source_index[order], target_index[order], weights[order]
