from typing import Tuple

import numpy as np
from scipy import sparse

from .typing import BoolArray, FloatDType, IntArray, FloatArray
from .connectivity import face_face_connectivity, renumber, triangulate


def exterior_topology(
    edge_face_connectivity: IntArray,
    edge_node_connectivity: IntArray,
    node_edge_connectivity: sparse.csr_matrix,
    fill_value: int,
    vertices: FloatArray,
    centroids: FloatArray,
    concave: bool,
):
    is_exterior = edge_face_connectivity[:, 1] == fill_value
    exterior_nodes = edge_node_connectivity[is_exterior]

    # Find associated edges
    coo = node_edge_connectivity[exterior_nodes.ravel()].tocoo()
    n_edge_per_node = coo.getnnz(axis=1)
    nodes = np.repeat(exterior_nodes, repeats=n_edge_per_node)
    assoc_edges = coo.col
    # Find all centroids that are associated with this node
    # We're only dealing with the interior edges and their faces here.
    is_interior = ~is_exterior[assoc_edges]
    # These contains duplicate pointers to face centroids, which we'll filter
    # away.
    centroid_nodes = np.repeat(nodes[is_interior], 2)
    face_i = edge_face_connectivity[assoc_edges[is_interior]].ravel()
    ij = np.column_stack([centroid_nodes, face_i])
    centroid_nodes, face_i = np.unique(ij, axis=0).T
    
    # For every exterior node, project the centroids to exterior edges
    edge_vertices = vertices[exterior_nodes]
    selected_faces = edge_face_connectivity[is_exterior, 0]
    centroid_vertices = centroids[selected_faces]
    a = edge_vertices[:, 0, :]
    b = edge_vertices[:, 1, :]
    U = b - a
    V = b - centroid_vertices
    norm = np.linalg.norm(U, axis=1)
    # np.dot(U, V) doesn't do the desired thing here
    projected_vertices = a + ((U[:, 0] * V[:, 0] + U[:, 1] * V[:, 1]) / norm ** 2 * U.T).T
    # Every one of these newly created vertices is used twice in forming
    # polygons, for exterior_nodes[:, 0] and exterior_nodes[:, 1].
    # Their values are unique since the edges are unique.
    # The associated face for these projected vertices are those of face_i.
    
    # Add the new vertices to the centroids, which will become the vertices of
    # the voronoi polygons.
    n_vertex = len(centroids)
    n_new = len(projected_vertices)
    # Create the numbering pointing to these new vertices
    new_numbers = np.repeat(np.arange(n_vertex, n_vertex + n_new), 2)
    new_vertices = np.concatenate([centroids, projected_vertices])
    face_index = np.concatenate([np.arange(len(centroids)), selected_faces])

    # Summing duplicates gets rid of duplicate entries.
    i = np.concatenate([centroid_nodes, exterior_nodes.ravel()])
    j = np.concatenate([face_i, new_numbers])

    # Now form valid counter-clockwise polygons 
    voronoi_vertices = new_vertices[j]
    voronoi_centroids = vertices[i]
    x = voronoi_vertices[:, 0] - voronoi_centroids[:, 0]
    y = voronoi_vertices[:, 1] - voronoi_centroids[:, 1]
    angle = np.arctan2(y, x)
    # Sort to create a counter clockwise oriented polygon.
    # Use a lexsort to make sure nodes are grouped together.
    order = np.lexsort((angle, i))
    i = i[order]
    j = j[order]
    return new_vertices, i, j, face_index


def voronoi_topology(
    node_face_connectivity: sparse.csr_matrix,
    vertices: FloatArray,
    centroids: FloatArray,
    edge_face_connectivity: IntArray=None,
    edge_node_connectivity: IntArray=None,
    node_edge_connectivity: sparse.csr_matrix=None,
    fill_value: int=None,
    exterior: bool=True,
    concave: bool=False,
) -> Tuple[FloatArray, sparse.csr_matrix]:
    """
    Parameters
    ----------
    node_face_connectivity : csr matrix
    vertices: ndarray of floats with shape ``(n_vertex, 2)``
    centroids: ndarray of floats with shape ``(n_centroid, 2)``

    Returns
    -------
    nodes: ndarray of floats with shape ``(n_vertex, 2)``
    face_node_connectivity: scipy.sparse.csr_matrix
    face_index: ndarray of ints with shape ``(n_vertex,)``
        Connects the nodes of the voronoi topology to the faces of the original
        grid.
    """
    #if exterior:
    #    if any(arg is None for arg in [edge_face_connectivity, edge_node_connectivity, node_edge_connectivity, fill_value]):
    #        raise ValueError()
    
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
    j = face_i[order]
    if exterior:
        vertices, exterior_i, exterior_j, face_i = exterior_topology(
            edge_face_connectivity,
            edge_node_connectivity,
            node_edge_connectivity,
            fill_value,
            vertices,
            centroids,
            concave,
        )     
        i = np.concatenate([node_i, exterior_i + node_i.max() + 1])
        j = np.concatenate([j, exterior_j])
    else:
        face_i = np.unique(face_i)
        vertices = centroids[face_i]
        i = node_i
        j = renumber(j)
    
    i = renumber(i)
    coo_content = (j, (i, j))
    face_node_connectivity = sparse.coo_matrix(coo_content)
    return vertices, face_node_connectivity, face_i
