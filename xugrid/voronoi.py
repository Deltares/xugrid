from typing import Tuple

import numpy as np
from scipy import sparse

from .typing import FloatArray
from .connectivity import renumber


def voronoi_topology(
    node_face_connectivity: sparse.csr_matrix,
    vertices: FloatArray,
    centroids: FloatArray,
) -> Tuple[FloatArray, sparse.csr_matrix]:
    """
    Parameters
    ----------
    node_face_connectivity : csr matrix
    centroids: ndarray of floats with shape ``(n_centroid, 2)``

    Returns
    -------
    nodes: ndarray of floats with shape ``(n_vertex, 2)``
    face_node_connectivity: scipy.sparse.csr_matrix
    """
    # Select only the nodes with a finite voronoi polygon: at least trilateral.
    # TODO: maybe extend 1- or 2-connected nodes with a boundary edge.
    ncol_per_row = node_face_connectivity.getnnz(axis=1)
    valid = np.repeat(ncol_per_row >= 3, ncol_per_row)
    # Grab the centroids for all these faces.
    coo = node_face_connectivity.tocoo()
    node_i = coo.row[valid]
    face_i = coo.col[valid]
    voronoi_centroids = vertices[node_i]
    voronoi_vertices = centroids[face_i]
    # Compute the angle between voronoi centroids and vertices.
    x = voronoi_vertices[:, 0] - voronoi_centroids[:, 0]
    y = voronoi_vertices[:, 1] - voronoi_centroids[:, 1]
    angle = np.arctan2(y, x)
    # Now sort to create a counter clockwise oriented polygon.
    # Use a lexsort to make sure nodes are grouped together.
    order = np.lexsort((angle, node_i))
    i = renumber(node_i)
    j = renumber(face_i[order])
    coo_content = (j, (i, j))
    face_node_connectivity = sparse.coo_matrix(coo_content)
    # Get rid of excess vertices
    valid_nodes = np.unique(face_i)
    vertices = centroids[valid_nodes]
    return vertices, face_node_connectivity
