from typing import NamedTuple, Tuple

import numba as nb
import numpy as np
from scipy import sparse

from xugrid.constants import BoolArray, FloatArray, IntArray, IntDType, SparseMatrix


class AdjacencyMatrix(NamedTuple):
    indices: IntArray
    indptr: IntArray
    nnz: int
    n: int
    m: int


def _csr_to_adjacency(A: sparse.csr_matrix) -> AdjacencyMatrix:
    if not isinstance(A, sparse.csr_matrix):
        raise TypeError(
            f"Expected scipy.sparse.csr_matrix, received: {type(A).__name__}"
        )

    n, m = A.shape
    adj = AdjacencyMatrix(A.indices, A.indptr, A.nnz, n, m)
    return adj


@nb.njit(inline="always")
def neighbors(A: AdjacencyMatrix, cell: int) -> IntArray:
    start = A.indptr[cell]
    end = A.indptr[cell + 1]
    return A.indices[start:end]


@nb.njit(inline="always")
def pop(array, size):
    return array[size - 1], size - 1


@nb.njit(inline="always")
def push(array, value, size):
    array[size] = value
    return size + 1


@nb.njit
def _topological_sort_by_dfs(A: AdjacencyMatrix):
    # This code is almost a direct port of the BSD-2 licensed code in Graphs.jl:
    #
    # https://github.com/JuliaGraphs/Graphs.jl/blob/master/LICENSE.md
    #
    # Copyright (c) 2015: Seth Bromberger and other contributors. Copyright (c)
    # 2012: John Myles White and other contributors.
    #
    # Redistribution and use in source and binary forms, with or without
    # modification, are permitted provided that the following conditions are met:
    #
    # Redistributions of source code must retain the above copyright notice, this
    # list of conditions and the following disclaimer. Redistributions in binary
    # form must reproduce the above copyright notice, this list of conditions and
    # the following disclaimer in the documentation and/or other materials provided
    # with the distribution. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
    # CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
    # NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    # PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
    # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
    # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
    # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
    # OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
    # WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
    # OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
    # ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    vcolor = np.zeros(A.m, dtype=np.uint8)
    verts = np.empty(A.m, dtype=np.int64)
    verts_size = 0
    S = np.empty(A.m, dtype=np.int64)
    S_size = 0

    for v in range(A.m):
        if vcolor[v] != 0:
            continue
        S_size = 0
        S_size = push(S, v, S_size)
        vcolor[v] = 1
        while S_size > 0:
            u = S[S_size - 1]
            w = 0
            for n in neighbors(A, u):
                if vcolor[n] == 1:
                    raise ValueError("The graph contains at least one cycle")
                elif vcolor[n] == 0:
                    w = n
                    break

            if w != 0:
                vcolor[w] = 1
                S_size = push(S, w, S_size)
            else:
                vcolor[u] = 2
                verts_size = push(verts, u, verts_size)
                S_size -= 1

    return verts[::-1]


def topological_sort_by_dfs(A: sparse.csr_matrix) -> IntArray:
    """
    Returns an array of vertices in topological order.

    Parameters
    ----------
    A: sparse.csr_matrix
        CSR adjacency matrix of the directed acyclical graph.

    Returns
    -------
    sorted_vertices: np.ndarray of integer
    """
    return _topological_sort_by_dfs(_csr_to_adjacency(A))


@nb.njit
def _contract_vertices(A: AdjacencyMatrix, indices: IntArray) -> IntArray:
    vcolor = np.zeros(A.m, dtype=np.uint8)
    vcolor[indices] = 2

    edge_node_connectivity = np.empty((A.m, 2), dtype=np.int64)
    n_edge = 0

    S = np.empty(A.m, dtype=np.int64)
    size = 0

    for v in indices:
        size = 0
        size = push(S, v, size)
        while size > 0:
            u, size = pop(S, size)

            # Do not paint over the marked indices
            if vcolor[u] == 0:
                vcolor[u] == 1

            for n in neighbors(A, u):
                if (n == v) or (vcolor[n] == 1):
                    raise ValueError("The graph contains at least one cycle")
                elif vcolor[n] == 2:
                    edge_node_connectivity[n_edge, 0] = v
                    edge_node_connectivity[n_edge, 1] = n
                    n_edge += 1
                    size -= 1
                else:
                    size = push(S, n, size)

    return edge_node_connectivity[:n_edge]


def contract_vertices(A: sparse.csr_matrix, indices: IntArray) -> IntArray:
    """
    Contract vertices to the set defined by indices.

    Parameters
    ----------
    A: AdjacencyMatrix
        CSR adjacency matrix of the directed acyclical graph.
    indices: np.ndarray of integer
        Which vertices to preserve.

    Returns
    -------
    edge_node_connectivity: np.ndarray of shape (n_edge, 2)
        New edge_node_connectivity containing edges of the vertices contained
        in indices.
    """
    return _contract_vertices(_csr_to_adjacency(A), np.array(indices))


# Conversion between dense and sparse
# -----------------------------------
def _to_ij(conn: IntArray, fill_value: int, invert: bool) -> Tuple[IntArray, IntArray]:
    n, m = conn.shape
    j = conn.ravel()
    valid = j != fill_value
    i = np.repeat(np.arange(n), m)[valid]
    j = j[valid]
    if invert:
        return j, i
    else:
        return i, j


def _to_sparse(
    conn: IntArray, fill_value: int, invert: bool, sort_indices: bool
) -> sparse.csr_matrix:
    i, j = _to_ij(conn, fill_value, invert)
    coo_content = (j, (i, j))
    coo_matrix = sparse.coo_matrix(coo_content)
    csr_matrix = coo_matrix.tocsr()
    # Conversion to csr format results in a sorting of indices. We require
    # only sorting of i, not j, as this would e.g. mess up the order for
    # counter clockwise vertex orientation, e.g.
    if not sort_indices:
        order = np.argsort(i)
        csr_matrix.indices = j[order]
        csr_matrix.has_sorted_indices = False
    return csr_matrix


def ragged_index(n: int, m: int, m_per_row: IntArray) -> BoolArray:
    """
    Given an array of n rows by m columns, starting from left mark the values
    True such that the number of True values equals m_per_row.

    For example:

    n = 3
    m = 4
    m_per_row = np.array([1, 2, 3])

    Then the result of _ragged_index(n, m, m_per_row) is:

    np.array([
        [True, False, False, False],
        [True, True, False, False],
        [True, True, True, False],
    ])

    This can be used as boolean index to set a variable number of values per
    row.
    """
    column_number = np.tile(np.arange(m), n).reshape((n, m))
    return (column_number.T < m_per_row).T


def to_sparse(
    conn: IntArray, fill_value: int, sort_indices: bool = True
) -> sparse.csr_matrix:
    return _to_sparse(conn, fill_value, invert=False, sort_indices=sort_indices)


def to_dense(conn: SparseMatrix, fill_value: int, n_columns: int = None) -> IntArray:
    n, _ = conn.shape
    m_per_row = conn.getnnz(axis=1)
    m = m_per_row.max()
    if n_columns is not None:
        if n_columns < m:
            raise ValueError(
                f"n_columns {n_columns} is too small for the data, requires {m}"
            )
        m = n_columns

    # Allocate 2D array and create a flat view of the dense connectivity
    dense_conn = np.empty((n, m), dtype=IntDType)
    flat_conn = dense_conn.ravel()
    if (n * m) == conn.nnz:
        # Shortcut if fill_value is not present, when all of same geom. type
        # e.g. all triangles or all quadrangles
        valid = slice(None)  # a[:] equals a[slice(None)]
    else:
        valid = ragged_index(n, m, m_per_row).ravel()
        flat_conn[~valid] = fill_value

    if isinstance(conn, sparse.csr_matrix):
        flat_conn[valid] = conn.indices
    elif isinstance(conn, sparse.coo_matrix):
        flat_conn[valid] = conn.col
    else:
        raise TypeError("Can only invert coo or csr matrix")
    return dense_conn


# Inverting connectivities
# ------------------------
def invert_dense_to_sparse(
    conn: IntArray, fill_value: int, sort_indices: bool = True
) -> sparse.csr_matrix:
    return _to_sparse(conn, fill_value, invert=True, sort_indices=sort_indices)


def invert_dense(
    conn: IntArray, fill_value: int, sort_indices: bool = True
) -> IntArray:
    sparse_inverted = _to_sparse(
        conn, fill_value, invert=True, sort_indices=sort_indices
    )
    return to_dense(sparse_inverted, fill_value)


def invert_sparse(conn: sparse.csr_matrix) -> sparse.csr_matrix:
    coo = conn.tocoo()
    j = coo.row
    i = coo.col
    coo_content = (j, (i, j))
    inverted = sparse.coo_matrix(coo_content)
    return inverted.tocsr()


def invert_sparse_to_dense(conn: sparse.csr_matrix, fill_value: int) -> IntArray:
    inverted = invert_sparse(conn)
    return to_dense(inverted, fill_value)


# Renumbering
# -----------
def _renumber(a: IntArray) -> IntArray:
    # Taken from https://github.com/scipy/scipy/blob/v1.7.1/scipy/stats/stats.py#L8631-L8737
    # (scipy is BSD-3-Clause License)
    arr = np.ravel(np.asarray(a))
    sorter = np.argsort(arr, kind="quicksort")
    inv = np.empty(sorter.size, dtype=IntDType)
    inv[sorter] = np.arange(sorter.size, dtype=IntDType)
    arr = arr[sorter]
    obs = np.r_[True, arr[1:] != arr[:-1]]
    dense = obs.cumsum()[inv] - 1
    return dense.reshape(a.shape)


def renumber(a: IntArray, fill_value: int = None):
    if fill_value is None:
        return _renumber(a)

    valid = a != fill_value
    renumbered = np.full_like(a, fill_value)
    renumbered[valid] = _renumber(a[valid])
    return renumbered


def close_polygons(face_node_connectivity: IntArray, fill_value: int) -> IntArray:
    # Wrap around and create closed polygon: put the first node at the end of the row
    # In case of fill values, replace all fill values
    n, m = face_node_connectivity.shape
    closed = np.full((n, m + 1), fill_value, dtype=IntDType)
    closed[:, :-1] = face_node_connectivity
    first_node = face_node_connectivity[:, 0]
    # Identify fill value, and replace by first node also
    isfill = closed == fill_value
    closed.ravel()[isfill.ravel()] = np.repeat(first_node, isfill.sum(axis=1))
    return closed, isfill


def reverse_orientation(face_node_connectivity: IntArray, fill_value: int):
    # We cannot simply reverse the rows with [:, ::-1], since there may be fill
    # values present.
    reversed_orientation = face_node_connectivity.copy()
    in_reverse = face_node_connectivity[:, ::-1]
    in_reverse = in_reverse[in_reverse != fill_value]
    replace = face_node_connectivity != fill_value
    reversed_orientation[replace] = in_reverse
    return reversed_orientation


def counterclockwise(
    face_node_connectivity: IntArray, fill_value: int, nodes: FloatArray
) -> IntArray:
    closed, _ = close_polygons(face_node_connectivity, fill_value)
    p = nodes[closed]
    dxy = np.diff(p, axis=1)
    reverse = (np.cross(dxy[:, :-1], dxy[:, 1:])).sum(axis=1) < 0
    ccw = face_node_connectivity.copy()
    if reverse.any():
        ccw[reverse] = reverse_orientation(face_node_connectivity[reverse], fill_value)
    return ccw


# Derived connectivities
# ----------------------
def boundary_node_connectivity(
    edge_face_connectivity: IntArray,
    fill_value: int,
    edge_node_connectivity: IntArray,
) -> IntArray:
    """Is a subset of the edge_node_connectivity"""
    is_boundary = (edge_face_connectivity == fill_value).any(axis=1)
    return edge_node_connectivity[is_boundary]


def edge_connectivity(
    face_node_connectivity: IntArray,
    fill_value: int,
    edge_node_connectivity=None,
) -> Tuple[IntArray, IntArray]:
    """Derive new edge_node_connectivity and face_edge_connectivity."""
    prior = edge_node_connectivity
    n, m = face_node_connectivity.shape
    # Close the polygons: [0 1 2 3] -> [0 1 2 3 0]
    closed, isfill = close_polygons(face_node_connectivity, fill_value)
    # Allocate array for edge_node_connectivity: includes duplicate edges
    edge_node_connectivity = np.empty((n * m, 2), dtype=IntDType)
    edge_node_connectivity[:, 0] = closed[:, :-1].ravel()
    edge_node_connectivity[:, 1] = closed[:, 1:].ravel()
    # Cleanup: delete invalid edges (same node to same node)
    edge_node_connectivity = edge_node_connectivity[
        edge_node_connectivity[:, 0] != edge_node_connectivity[:, 1]
    ]
    # Now find the unique rows == unique edges
    edge_node_connectivity.sort(axis=1)
    edge_node_connectivity, inverse_indices = np.unique(
        ar=edge_node_connectivity, return_inverse=True, axis=0
    )

    if prior is not None:  # prior edge_node_connectivity exists
        unique, index = np.unique(np.sort(prior, axis=1), axis=0, return_index=True)
        # Check whether everything looks okay:
        if not np.array_equal(unique, edge_node_connectivity):
            raise ValueError("Invalid edge_node_connectivity")
        inverse_indices = index[inverse_indices]
        edge_node_connectivity = prior

    # Create face_edge_connectivity
    face_edge_connectivity = np.full((n, m), fill_value, dtype=np.int64)
    isnode = ~isfill[:, :-1]
    face_edge_connectivity[isnode] = inverse_indices
    return edge_node_connectivity, face_edge_connectivity


def face_face_connectivity(
    edge_face_connectivity: IntArray,
    fill_value: int,
) -> sparse.csr_matrix:
    """
    An edge can be shared by two faces at most. If this is the case, they are
    neighbors.
    """
    i = edge_face_connectivity[:, 0]
    j = edge_face_connectivity[:, 1]
    is_connection = j != fill_value
    i = i[is_connection]
    j = j[is_connection]
    ij = np.concatenate([i, j])
    ji = np.concatenate([j, i])
    coo_content = (ji, (ij, ji))
    coo_matrix = sparse.coo_matrix(coo_content)
    return coo_matrix.tocsr()


def directed_node_node_connectivity(
    edge_node_connectivity: IntArray,
) -> sparse.csr_matrix:
    i = edge_node_connectivity[:, 0]
    j = edge_node_connectivity[:, 1]
    coo_content = (j, (i, j))
    coo_matrix = sparse.coo_matrix(coo_content)
    return coo_matrix.tocsr()


def node_node_connectivity(edge_node_connectivity: IntArray) -> sparse.csr_matrix:
    i = edge_node_connectivity[:, 0]
    j = edge_node_connectivity[:, 1]
    ij = np.concatenate([i, j])
    ji = np.concatenate([j, i])
    coo_content = (ji, (ij, ji))
    coo_matrix = sparse.coo_matrix(coo_content)
    return coo_matrix.tocsr()


def structured_connectivity(active: IntArray) -> AdjacencyMatrix:
    nrow, ncol = active.shape
    nodes = np.arange(nrow * ncol).reshape(nrow, ncol)
    nodes[~active] = -1
    left = nodes[:, :-1].ravel()
    right = nodes[:, 1:].ravel()
    front = nodes[:-1].ravel()
    back = nodes[1:].ravel()
    # valid x connection
    valid = (left != -1) & (right != -1)
    left = left[valid]
    right = right[valid]
    # valid y connection
    valid = (front != -1) & (back != -1)
    front = front[valid]
    back = back[valid]
    i = renumber(np.concatenate([left, right, front, back]))
    j = renumber(np.concatenate([right, left, back, front]))
    coo_content = (j, (i, j))
    A = sparse.coo_matrix(coo_content).tocsr()
    n, m = A.shape
    return AdjacencyMatrix(A.indices, A.indptr, A.nnz, n, m)


def area(
    face_node_connectivity: IntArray,
    fill_value: int,
    node_x: FloatArray,
    node_y: FloatArray,
):
    nodes = np.column_stack([node_x, node_y])
    closed, _ = close_polygons(face_node_connectivity, fill_value)
    coordinates = nodes[closed]
    # Shift coordinates to avoid precision loss
    coordinates[..., 0] -= node_x.mean()
    coordinates[..., 1] -= node_y.mean()
    a = coordinates[:, :-1]
    b = coordinates[:, 1:]
    determinant = np.cross(a, b)
    return 0.5 * abs(determinant.sum(axis=1))


def centroids(
    face_node_connectivity: IntArray,
    fill_value: int,
    node_x: FloatArray,
    node_y: FloatArray,
) -> FloatArray:
    n_face, n_max_node = face_node_connectivity.shape
    nodes = np.column_stack([node_x, node_y])
    # Check if it's fully triangular
    if n_max_node == 3:
        # Since it's triangular, computing the average of the vertices suffices
        coordinates = nodes[face_node_connectivity]
        return np.nanmean(coordinates, axis=1)
    else:
        # This is mathematically equivalent to triangulating, computing triangle centroids
        # and computing the area weighted average of those centroids
        centroid_coordinates = np.empty((n_face, 2), dtype=np.float64)
        closed, _ = close_polygons(face_node_connectivity, fill_value)
        coordinates = nodes[closed]
        # Shift coordinates to avoid precision loss
        a = coordinates[:, :-1]
        b = coordinates[:, 1:]
        c = a + b
        determinant = np.cross(a, b)
        area_weight = 1.0 / (3.0 * determinant.sum(axis=1))
        centroid_coordinates[:, 0] = area_weight * (c[..., 0] * determinant).sum(axis=1)
        centroid_coordinates[:, 1] = area_weight * (c[..., 1] * determinant).sum(axis=1)
        return centroid_coordinates


@nb.njit(cache=True, parallel=True)
def _circumcenters_triangle(xxx: FloatArray, yyy: FloatArray):
    """Numba should nicely fuse these operations."""
    a_x, b_x, c_x = xxx
    a_y, b_y, c_y = yyy
    D_inv = 0.5 / (
        (a_y * c_x + b_y * a_x - b_y * c_x - a_y * b_x - c_y * a_x + c_y * b_x)
    )
    x = ((a_x - c_x) * (a_x + c_x) + (a_y - c_y) * (a_y + c_y)) * (b_y - c_y) - (
        (b_x - c_x) * (b_x + c_x) + (b_y - c_y) * (b_y + c_y)
    ) * (a_y - c_y)
    y = ((b_x - c_x) * (b_x + c_x) + (b_y - c_y) * (b_y + c_y)) * (a_x - c_x) - (
        (a_x - c_x) * (a_x + c_x) + (a_y - c_y) * (a_y + c_y)
    ) * (b_x - c_x)
    return D_inv * x, D_inv * y


def circumcenters(
    face_node_connectivity: IntArray,
    fill_value: int,
    node_x: FloatArray,
    node_y: FloatArray,
) -> FloatArray:
    # TODO: Skyum or Welzl implementation for polygons -- although it's
    # practical use is dubious?
    n_max_node = face_node_connectivity.shape[1]
    # Check if it's fully triangular
    if n_max_node == 3:
        xxx = node_x[face_node_connectivity.T]
        yyy = node_y[face_node_connectivity.T]
        x, y = _circumcenters_triangle(xxx, yyy)
    else:
        raise NotImplementedError(
            "Circumcenters are only supported for triangular grids"
        )
    return np.column_stack((x, y))


def _triangulate(i: IntArray, j: IntArray, n_triangle_per_row: IntArray) -> IntArray:
    n_triangle = n_triangle_per_row.sum()
    n_face = len(i)
    index_first = np.argwhere(np.diff(i, prepend=-1) != 0)
    index_second = index_first + 1
    index_last = np.argwhere(np.diff(i, append=-1) != 0)

    first = np.full(n_face, False)
    first[index_first] = True
    second = np.full(n_face, True) & ~first
    second[index_last] = False
    third = np.full(n_face, True) & ~first
    third[index_second] = False

    triangles = np.empty((n_triangle, 3), IntDType)
    triangles[:, 0] = np.repeat(j[first], n_triangle_per_row)
    triangles[:, 1] = j[second]
    triangles[:, 2] = j[third]
    return triangles


def triangulate_dense(face_node_connectivity: IntArray, fill_value: int) -> None:
    n_face, n_max = face_node_connectivity.shape

    if n_max == 3:
        triangles = face_node_connectivity.copy()
        return triangles, np.arange(n_face)

    valid = face_node_connectivity != fill_value
    n_per_row = valid.sum(axis=1)
    n_triangle_per_row = n_per_row - 2
    i = np.repeat(np.arange(n_face), n_per_row)
    j = face_node_connectivity.ravel()[valid.ravel()]
    triangles = _triangulate(i, j, n_triangle_per_row)

    triangle_face_connectivity = np.repeat(
        np.arange(n_face), repeats=n_triangle_per_row
    )
    return triangles, triangle_face_connectivity


def triangulate_coo(face_node_connectivity: sparse.coo_matrix) -> IntArray:
    ncol_per_row = face_node_connectivity.getnnz(axis=1)

    if ncol_per_row.max() == 3:
        triangles = face_node_connectivity.row.copy().reshape((-1, 3))
        return triangles, np.arange(len(triangles))

    n_triangle_per_row = ncol_per_row - 2
    i = face_node_connectivity.row
    j = face_node_connectivity.col
    triangles = _triangulate(i, j, n_triangle_per_row)

    n_face = face_node_connectivity.shape[0]
    triangle_face_connectivity = np.repeat(
        np.arange(n_face), repeats=n_triangle_per_row
    )
    return triangles, triangle_face_connectivity


def triangulate(face_node_connectivity, fill_value: int = None) -> IntArray:
    """
    Convert polygons into its constituent triangles.

    Triangulation runs from the first node of every face:

    * first, second, third,
    * first, third, fourth,
    * and so forth ...

    If the grid is already triangular, a copy is returned.

    Returns
    -------
    triangles: ndarray of integers with shape ``(n_triangle, 3)``
    triangle_face_connectivity: ndarray of integers with shape ``(n_triangle,)``
    """
    if isinstance(face_node_connectivity, IntArray):
        if fill_value is None:
            raise ValueError("fill_value is required for dense connectivity")
        return triangulate_dense(face_node_connectivity, fill_value)
    elif isinstance(face_node_connectivity, sparse.coo_matrix):
        return triangulate_coo(face_node_connectivity)
    else:
        raise TypeError("connectivity must be ndarray or sparse matrix")


def _mutate(output: BoolArray, i: IntArray, j: IntArray, value: bool, mask: BoolArray):
    a = output[i]
    b = output[j]
    mutate = a != b
    output[i[mutate]] = value
    output[j[mutate]] = value
    if mask is not None:
        output[mask] = not value
    return None


def _binary_iterate(
    connectivity: sparse.csr_matrix,
    input: BoolArray,
    value: bool,
    iterations: int,
    mask: BoolArray,
    exterior: IntArray,
    border_value: bool,
) -> BoolArray:
    if input.dtype != np.bool_:
        raise TypeError("input dtype should be bool")

    coo = connectivity.tocoo()
    i = coo.row
    j = coo.col
    output = input.copy()

    # First iteration, mutate borders.
    _mutate(output, i, j, value, mask)
    if exterior is not None and value == border_value:
        output[exterior] = value

    # Subsequent iterations, disregard border.
    for _ in range(iterations - 1):
        _mutate(output, i, j, value, mask)

    return output


def binary_erosion(
    connectivity: sparse.csr_matrix,
    input: BoolArray,
    iterations: int = 1,
    mask: BoolArray = None,
    exterior: IntArray = None,
    border_value: bool = False,
) -> BoolArray:
    """
    By default, erodes inwards from the exterior.
    """
    return _binary_iterate(
        connectivity=connectivity,
        input=input,
        value=False,
        iterations=iterations,
        mask=mask,
        exterior=exterior,
        border_value=border_value,
    )


def binary_dilation(
    connectivity: sparse.csr_matrix,
    input: BoolArray,
    iterations: int = 1,
    mask: BoolArray = None,
    exterior: IntArray = None,
    border_value: bool = False,
) -> BoolArray:
    """
    By default, does not dilate inward from the exterior.
    """
    return _binary_iterate(
        connectivity=connectivity,
        input=input,
        value=True,
        iterations=iterations,
        mask=mask,
        exterior=exterior,
        border_value=border_value,
    )
