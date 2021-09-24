from typing import Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from .typing import IntArray, FloatArray
from .connectivity import renumber


def exterior_centroids(node_face_connectivity: sparse.csr_matrix):
    n, _ = node_face_connectivity.shape
    # Find exterior nodes NOT associated with any interior edge
    is_exterior_only = node_face_connectivity.getnnz(axis=1) == 1
    j = node_face_connectivity[is_exterior_only].indices
    i = np.arange(n)[is_exterior_only]
    return i, j


def interior_centroids(
    node_face_connectivity: sparse.csr_matrix,
    edge_face_connectivity: IntArray,
    edge_node_connectivity: IntArray,
    fill_value: int,
):
    # Find exterior nodes associated with interior edges
    is_exterior = edge_face_connectivity[:, 1] == fill_value
    exterior_nodes = edge_node_connectivity[is_exterior].ravel()
    m_per_node = node_face_connectivity.getnnz(axis=1)
    is_interior_only = m_per_node > 1

    selected_nodes = exterior_nodes[is_interior_only[exterior_nodes]]
    selection = node_face_connectivity[selected_nodes]
    m_per_selected_node = selection.getnnz(axis=1)

    j = selection.indices
    i = np.repeat(selected_nodes, repeats=m_per_selected_node)
    return i, j


def project_vertices(
    edge_face_connectivity: IntArray,
    edge_node_connectivity: IntArray,
    fill_value: int,
    vertices: FloatArray,
    centroids: FloatArray,
    add_vertices: bool,
):
    is_exterior = edge_face_connectivity[:, 1] == fill_value
    exterior_nodes = edge_node_connectivity[is_exterior]
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
    projected_vertices = (
        a + ((U[:, 0] * V[:, 0] + U[:, 1] * V[:, 1]) / norm ** 2 * U.T).T
    )

    # Create the numbering pointing to these new vertices
    n_new = len(projected_vertices)
    n_centroid = len(centroids)
    i = exterior_nodes.ravel()
    n = n_centroid + n_new
    j = np.repeat(np.arange(n_centroid, n), 2)
    n_interpolated = 0

    # We add a substitution value for the actual vertex
    if add_vertices:
        order = np.argsort(i)
        jj = np.repeat(np.arange(n_new), 2)[order]
        to_interpolate = projected_vertices[jj]
        interpolated = 0.5 * (to_interpolate[::2] + to_interpolate[1::2])
        n_interpolated = len(interpolated)
        i = np.concatenate([i, i[order][::2]])
        j = np.concatenate([j, np.arange(n, n + n_interpolated)])
        projected_vertices = np.concatenate([projected_vertices, interpolated])

    return i, j, projected_vertices, n_interpolated


def exterior_topology(
    edge_face_connectivity: IntArray,
    edge_node_connectivity: IntArray,
    node_face_connectivity: sparse.csr_matrix,
    fill_value: int,
    vertices: FloatArray,
    centroids: FloatArray,
    add_vertices: bool,
):
    i0, j0 = interior_centroids(
        node_face_connectivity,
        edge_face_connectivity,
        edge_node_connectivity,
        fill_value,
    )
    i1, j1 = exterior_centroids(node_face_connectivity)
    i2, j2, projected_vertices, n_interpolated = project_vertices(
        edge_face_connectivity,
        edge_node_connectivity,
        fill_value,
        vertices,
        centroids,
        add_vertices,
    )

    i = np.concatenate([i0, i1, i2])
    j = np.concatenate([j0, j1, j2])
    vor_vertices = np.concatenate([centroids, projected_vertices])
    orig_vertices = vertices[i][-n_interpolated:]

    # Create face index: the face of the original mesh associated with every
    # voronoi vertex is a centroid. The exterior vertices are an exception to
    # this: these are associated with two faces. So we set a value of -1 here.
    face_index = j.copy()
    face_index[-n_interpolated:] = -1

    # Now form valid counter-clockwise polygons
    voronoi_vertices = vor_vertices[j]
    voronoi_centroids = (
        pd.DataFrame({"i": i, "x": voronoi_vertices[:, 0], "y": voronoi_vertices[:, 1]})
        .groupby("i")
        .mean()
    )
    renumbered_i = renumber(i)
    x = voronoi_vertices[:, 0] - voronoi_centroids["x"].values[renumbered_i]
    y = voronoi_vertices[:, 1] - voronoi_centroids["y"].values[renumbered_i]
    angle = np.arctan2(y, x)
    # Sort to create a counter clockwise oriented polygon.
    # Use a lexsort to make sure nodes are grouped together.
    order = np.lexsort((angle, i))
    i = i[order]
    j = j[order]
    face_index = face_index[order]

    # If add_vertices is True, we have substituted interpolated points before
    # to generate the proper ordering. We overwrite those substituted points
    # here by their possibly concave true vertices.
    if add_vertices:
        vor_vertices[-n_interpolated:] = orig_vertices

    return vor_vertices, i, j, face_index


def voronoi_topology(
    node_face_connectivity: sparse.csr_matrix,
    vertices: FloatArray,
    centroids: FloatArray,
    edge_face_connectivity: IntArray = None,
    edge_node_connectivity: IntArray = None,
    fill_value: int = None,
    add_exterior: bool = False,
    add_vertices: bool = False,
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
    # if exterior:
    #    if any(arg is None for arg in [edge_face_connectivity, edge_node_connectivity, node_edge_connectivity, fill_value]):
    #        raise ValueError()

    # Avoid overlapping polygons: if exterior is included, the exterior
    # algorithm will construct those polygons. If exterior is not included, we
    # take any valid internal polygon we can construct: at least a triangle.
    ncol_per_row = node_face_connectivity.getnnz(axis=1)
    if add_exterior:
        is_exterior = edge_face_connectivity[:, 1] == fill_value
        exterior_nodes = edge_node_connectivity[is_exterior]
        valid = np.full(len(vertices), True)
        valid[exterior_nodes.ravel()] = False
        valid = np.repeat(valid, ncol_per_row)
    else:
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

    if add_exterior:
        vor_vertices, exterior_i, exterior_j, face_i = exterior_topology(
            edge_face_connectivity,
            edge_node_connectivity,
            node_face_connectivity,
            fill_value,
            vertices,
            centroids,
            add_vertices,
        )
        offset = node_i.max() + 1 if len(node_i > 0) else 0
        i = np.concatenate([node_i, exterior_i + offset])
        j = np.concatenate([j, exterior_j])
    else:
        face_i = np.arange(face_i.max())
        vor_vertices = centroids.copy()
        i = node_i
        j = renumber(j)

    i = renumber(i)
    coo_content = (j, (i, j))
    face_node_connectivity = sparse.coo_matrix(coo_content)
    return vor_vertices, face_node_connectivity, face_i
