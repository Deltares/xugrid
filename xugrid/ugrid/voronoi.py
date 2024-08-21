"""
Module to compute centroidal voronoi tesselation mesh from an existing mesh of
convex cells.

In principle, this is easy to compute:

* Compute the centroids of the cells.
* Invert the face_node_connectivity index array.
* For every node, find the connected faces.
* Use the connected faces to find the centroids.
* Order the centroids around the vertex in a counter-clockwise manner.

Dealing with the mesh exterior (beyond which no centroids are located) is the
tricky part. Refer to the docstrings and especially the visual examples in the
developer documentation.
"""
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from xugrid.constants import X_EPSILON, FloatArray, IntArray
from xugrid.ugrid.connectivity import (
    area_from_coordinates,
    close_polygons,
    ragged_index,
    renumber,
)


def dot_product2d(U: FloatArray, V: FloatArray):
    return U[:, 1] * V[:, 1] + U[:, 0] * V[:, 0]


def _centroid_pandas(i: IntArray, x: FloatArray, y: FloatArray):
    grouped = pd.DataFrame({"i": i, "x": x, "y": y}).groupby("i").mean()
    x_centroid = grouped["x"].to_numpy()
    y_centroid = grouped["y"].to_numpy()
    return x_centroid, y_centroid


def _centroid_scipy(i: IntArray, x: FloatArray, y: FloatArray):
    j = np.arange(len(i))
    coo_content = (x, (i, j))
    mat = sparse.coo_matrix(coo_content)
    n = mat.getnnz(axis=1)
    x_centroid = mat.sum(axis=1).flat / n
    mat.data = y
    y_centroid = mat.sum(axis=1).flat / n
    return x_centroid, y_centroid


def compute_centroid(i: IntArray, x: FloatArray, y: FloatArray):
    return _centroid_pandas(i, x, y)


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
    exterior_nodes = np.unique(edge_node_connectivity[is_exterior].ravel())
    m_per_node = node_face_connectivity.getnnz(axis=1)
    is_interior_only = m_per_node > 1

    selected_nodes = exterior_nodes[is_interior_only[exterior_nodes]]
    selection = node_face_connectivity[selected_nodes]
    m_per_selected_node = selection.getnnz(axis=1)

    j = selection.indices
    i = np.repeat(selected_nodes, repeats=m_per_selected_node)
    return i, j


def exterior_vertices(
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
    face_i = edge_face_connectivity[is_exterior, 0]
    centroid_vertices = centroids[face_i]
    a = edge_vertices[:, 0, :]
    b = edge_vertices[:, 1, :]
    V = b - a
    U = centroid_vertices - a
    # np.dot(U, V) doesn't do the desired thing here
    projected_vertices = a + ((dot_product2d(U, V) / dot_product2d(V, V)) * V.T).T

    # Create the numbering pointing to these new vertices.
    # Discard vertices that overlap with e.g. circumcenters.
    keep = np.linalg.norm(projected_vertices - centroid_vertices, axis=1) > (
        X_EPSILON * X_EPSILON
    )
    face_i = face_i[keep]
    vertices_keep = projected_vertices[keep]
    n_centroid = len(centroids)
    i_keep = exterior_nodes[keep].ravel()
    n = n_centroid + len(vertices_keep)
    j_keep = np.repeat(np.arange(n_centroid, n), 2)
    n_interpolated = 0
    interpolation_map = None

    # We add a substitution value for the actual vertex
    if add_vertices:
        n_new = len(projected_vertices)
        i = exterior_nodes.ravel()
        order = np.argsort(i)
        jj = np.repeat(np.arange(n_new), 2)[order]
        to_interpolate = projected_vertices[jj]
        interpolated = 0.5 * (to_interpolate[::2] + to_interpolate[1::2])
        n_interpolated = len(interpolated)
        i_keep = np.concatenate([i_keep, i[order][::2]])
        j_keep = np.concatenate([j_keep, np.arange(n, n + n_interpolated)])
        vertices_keep = np.concatenate([vertices_keep, interpolated])
        # Create face index: the face of the original mesh associated with every
        # voronoi vertex is a centroid. The exterior vertices are an exception to
        # this: these are associated with two faces. So we set a value of -1 here.
        face_i = np.concatenate([face_i, np.full(n_interpolated, -1)])

        # For the interpolated vertices, it depends on the two other nodes (which
        # depend on centroids). Store for each interpolation on which two nodes it relies.
        interpolation_map = jj.reshape((-1, 2))

    return i_keep, j_keep, vertices_keep, face_i, n_interpolated, interpolation_map


def choose_convex(
    i: IntArray,
    j: IntArray,
    nodes: FloatArray,
    original_vertices: FloatArray,
    n_interpolated: int,
) -> None:
    # Determine whether the original vertex or the interpolated vertex
    # generates the largest area (since the concave face will be smaller than
    # the convex face.)
    # Create a face_node_connectivity array.
    n_vertex = np.bincount(i)
    n_vertex = n_vertex[n_vertex > 0]
    n = len(n_vertex)
    m = n_vertex.max()
    index = ragged_index(n, m, n_vertex)
    faces = np.full((n, m), -1)
    faces[index] = j
    # Close the polygons so we can easily compute areas.
    closed, _ = close_polygons(faces, -1)
    # Make a copy and insert the original vertices.
    modified_nodes = nodes.copy()
    modified_nodes[-n_interpolated:] = original_vertices

    # Compare areas of faces
    convex_area = area_from_coordinates(nodes[closed])
    modified_area = area_from_coordinates(modified_nodes[closed])
    original_is_convex = (modified_area >= convex_area)[:, np.newaxis]
    # All the newly created vertices are found at the end.
    is_interpolated = faces >= len(nodes) - n_interpolated
    # No need for unique: every exterior vertex is featured exactly once.
    use_original = faces[original_is_convex & is_interpolated]
    nodes[use_original] = modified_nodes[use_original]
    return


def exterior_topology(
    edge_face_connectivity: IntArray,
    edge_node_connectivity: IntArray,
    node_face_connectivity: sparse.csr_matrix,
    fill_value: int,
    vertices: FloatArray,
    centroids: FloatArray,
    add_vertices: bool,
    skip_concave: bool,
):
    """
    Create the exterior topology of the voronoi tesselation.

    The exterior topology of this voronoi tesselation consists of three kinds
    of vertices:

    * Centroids of faces that have a vertex located on the exterior, but have
      no exterior edges. These centroids are used twice in the resulting
      voronoi mesh topology.
    * Centroids of faces that have an exterior edge. These centroids are used
      once.
    * Vertices of the intersection of (infinite) rays with the mesh exterior.
      These vertices are used twice; these intersections are the orthogonal
      projection of the centroid of the face on its exterior edges.
    * Original vertices of the mesh exterior. These vertices are used once, and
      are always located in between the projected vertices.

    The last ones are trickiest, as they may create concave angles; with a
    concave angle in a polygon, we cannot easily compute the counter clockwise
    order of the vertices to form a polygon. However, since these vertices are
    always located in between the projected vertices, we can introduce a
    subsitute: a linear interpolation of the two projected vertices. We
    can then utilize this interpolated vertex to compute the counter clockwise
    order, and then replace it by the original vertex -- which then may
    introduce a concave angle.

    A note on the index arrays:

    * i refers to the node number of the original mesh.
    * In principle, j refers to the face number (and associated centroids) of
    the original mesh; exterior vertices require new numbers.

    Finally, in the resulting new (voronoi) face_node_connectivity, i becomes
    the face number, and j becomes the node number.
    """
    i0, j0 = interior_centroids(
        node_face_connectivity,
        edge_face_connectivity,
        edge_node_connectivity,
        fill_value,
    )
    i1, j1 = exterior_centroids(node_face_connectivity)
    (
        i2,
        j2,
        projected_vertices,
        face_i,
        n_interpolated,
        interpolation_map,
    ) = exterior_vertices(
        edge_face_connectivity,
        edge_node_connectivity,
        fill_value,
        vertices,
        centroids,
        add_vertices,
    )

    i = np.concatenate([i0, i1, i2])
    j = np.concatenate([j0, j1, j2])
    _, n_face = node_face_connectivity.shape
    vor_vertices = np.concatenate([centroids, projected_vertices])
    face_i = np.concatenate([np.arange(n_face), face_i])
    orig_vertices = vertices[i][-n_interpolated:]

    # Now form valid counter-clockwise polygons
    voronoi_vertices = vor_vertices[j]
    x = voronoi_vertices[:, 0]
    y = voronoi_vertices[:, 1]
    centroid_x, centroid_y = compute_centroid(i, x, y)
    renumbered_i = renumber(i)
    dx = x - centroid_x[renumbered_i]
    dy = y - centroid_y[renumbered_i]
    angle = np.arctan2(dy, dx)
    # Sort to create a counter clockwise oriented polygon.
    # Use a lexsort to make sure nodes are grouped together.
    order = np.lexsort((angle, i))
    i = i[order]
    j = j[order]

    # If add_vertices is True, we have substituted interpolated points before
    # to generate the proper ordering. We overwrite those substituted points
    # here by their possibly concave true vertices.
    if add_vertices:
        if skip_concave:
            choose_convex(i, j, vor_vertices, orig_vertices, n_interpolated)
        else:
            vor_vertices[-n_interpolated:] = orig_vertices

    return vor_vertices, i, j, face_i, interpolation_map


def voronoi_topology(
    node_face_connectivity: sparse.csr_matrix,
    vertices: FloatArray,
    centroids: FloatArray,
    edge_face_connectivity: IntArray = None,
    edge_node_connectivity: IntArray = None,
    fill_value: int = None,
    add_exterior: bool = False,
    add_vertices: bool = False,
    skip_concave: bool = False,
) -> Tuple[FloatArray, sparse.csr_matrix, IntArray, IntArray]:
    """
    Compute the centroidal voronoi tesslation (CVT) of an existing mesh of
    (convex!) cells using connectivity index arrays.

    If the exterior boundary of the mesh forms a concave polygon, this will
    result in some concave voronoi cells as well. Since concave cells are often
    undesirable, there are three options for dealing with the exterior
    boundary:

    * if ``add_exterior=True`` and ``add_vertices=True``, infinite voronoi rays
      (projections) are intersected at exterior edges and all exterior vertices
      are included. If the mesh exterior forms a concave polygon, the resulting
      tesselation will contain concave cells as well.
    * if ``add_exterior=True`` and ``add_vertices=False``, the infinite voronoi
      rays are intersected, but no exterior vertices are included. This will
      always result in convex voronoi cells if the original cells are convex.
    * if ``add_exterior=False`` and ``add_vertices=False``, no vertices on the
      exterior edges are considered. Only centroids of the original mesh are
      considered. This will always result in convex voronoi cells if the
      original cells are convex.

    A direct correspondence exists between the voronoi vertices and the
    original faces: the vertices of the voronoi polygons are the centroids of
    the original faces; and the intersected rays "belong" to a single face too.
    This is not the case when all exterior vertices are included: these may
    correspond to multiple faces of the original mesh. These vertices are
    marked by a face_index of -1.

    Parameters
    ----------
    node_face_connectivity : csr matrix
    vertices: ndarray of floats with shape ``(n_vertex, 2)``
    centroids: ndarray of floats with shape ``(n_centroid, 2)``
    edge_face_connectivity: ndarray of integers with shape ``(n_edge, 2)``, optional
    edge_node_connectivity: ndarray of integers with shape ``(n_edge, 2)``, optional
    fill_value: int, optional
        Fill value for edge_face_connectivity.
    add_exterior: bool, optional
        Whether to consider exterior edges of the original mesh, or to consider
        exclusively centroids.
    add_vertices: bool, optional
        Whether to use existing exterior vertices.
    skip_concave: bool, optional
        Whether to skip existing exterior vertices if they generate concave
        faces.

    Returns
    -------
    nodes: ndarray of floats with shape ``(n_vertex, 2)``
    face_node_connectivity: scipy.sparse.csr_matrix
    face_index: ndarray of ints with shape ``(n_vertex,)``
        Connects the nodes of the voronoi topology to the faces of the original
        grid. Exterior vertices (when ``add_vertices=True``) are given an index
        of -1.
    interpolation_map: ndarray of ints with shape ``(n_interpolated, 2)``
        Marks for each interpolated point from which nodes it has been
        interpolated.
    """
    if add_exterior:
        if any(
            arg is None
            for arg in [edge_face_connectivity, edge_node_connectivity, fill_value]
        ):
            raise ValueError(
                "edge_face_connectivity, edge_node_connectivity, fill_value "
                "must be provided if add_exterior is True."
            )

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
        (
            vor_vertices,
            exterior_i,
            exterior_j,
            face_i,
            interpolation_map,
        ) = exterior_topology(
            edge_face_connectivity,
            edge_node_connectivity,
            node_face_connectivity,
            fill_value,
            vertices,
            centroids,
            add_vertices,
            skip_concave,
        )
        offset = node_i.max() + 1 if len(node_i > 0) else 0
        i = np.concatenate([node_i, exterior_i + offset])
        j = np.concatenate([j, exterior_j])
    else:
        interpolation_map = None
        vor_vertices = centroids[np.unique(face_i)]
        face_i = np.arange(face_i.max() + 1)
        i = node_i
        j = renumber(j)

    i = renumber(i)
    coo_content = (j, (i, j))
    face_node_connectivity = sparse.coo_matrix(coo_content)

    return vor_vertices, face_node_connectivity, face_i, interpolation_map
