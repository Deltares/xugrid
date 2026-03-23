from typing import Literal

import numpy as np

from xugrid.constants import FILL_VALUE


def build_regular_prisms(n, face_node_connectivity, n_layer):
    """
    Build wedges (3), hexahedrons (4), pentagonal (5) and hexahedral prisms (6).

    Each cell owns 2*n consecutive nodes in xyz_flat
    Cell (layer, face) starts at: (layer * n_faces + face) * 2 * n
    """

    n_faces = face_node_connectivity.shape[0]
    cell_idx = (
        np.arange(n_layer)[:, np.newaxis] * n_faces + np.arange(n_faces)[np.newaxis, :]
    )  # (n_layer, n_faces)
    base = cell_idx * (2 * n)  # (n_layer, n_faces)

    bottom_ring = base[:, :, np.newaxis] + np.arange(n)
    top_ring = base[:, :, np.newaxis] + np.arange(n) + n

    out = np.empty((n_layer, n_faces, 2 * n + 1), dtype=int)
    out[:, :, 0] = 2 * n
    out[:, :, 1 : n + 1] = bottom_ring
    out[:, :, n + 1 :] = top_ring
    is_index = np.ones_like(out, dtype=bool)
    is_index[:, :, 0] = False
    return out.ravel(), is_index.ravel()


def build_polyhedric_prisms(n, face_node_connectivity, n_layer):
    """
    General prismatic polyhedra for n > 6. VTK POLYHEDRON (42).
    VTK POLYHEDRON face stream per cell:
    [n_faces, n0, v0..vN-1, n1, v0..vN-1, ...]
    Each prismatic cell has n+2 faces:
    - 1 bottom cap  (n nodes, reversed for outward normal)
    - 1 top cap     (n nodes, CCW)
    - n lateral quads
    """
    n_faces = face_node_connectivity.shape[0]
    cell_idx = (
        np.arange(n_layer)[:, np.newaxis] * n_faces + np.arange(n_faces)[np.newaxis, :]
    )  # (n_layer, n_faces)
    base = cell_idx * (2 * n)  # (n_layer, n_faces)

    bot = base[:, :, np.newaxis] + np.arange(n)  # (n_layer, n_faces, n)
    top = base[:, :, np.newaxis] + np.arange(n) + n  # (n_layer, n_faces, n)
    bot_next = np.roll(bot, -1, axis=2)
    top_next = np.roll(top, -1, axis=2)

    # (n_layer, n_faces, n, 4) -> (n_layer, n_faces, n*5)
    quads = np.stack([bot, bot_next, top_next, top], axis=3)
    lateral = np.empty((n_layer, n_faces, n, 5), dtype=int)
    lateral[:, :, :, 0] = 4
    lateral[:, :, :, 1:] = quads
    lateral = lateral.reshape(n_layer, n_faces, -1)

    # cell width: 1 (n_items) + 1 (n_cell_faces) + (1+n) + (1+n) + n*5 = 7n+4
    out = np.empty((n_layer, n_faces, 7 * n + 4), dtype=int)
    is_index = np.zeros((n_layer, n_faces, 7 * n + 4), dtype=bool)

    assert 4 + 2 * n + (n - 1) * 5 + 4 <= 7 * n + 3, (
        "quad node positions exceed cell stream bounds"
    )

    face_stream_len = 7 * n + 3  # everything after n_items
    out[:, :, 0] = face_stream_len  # n_items
    out[:, :, 1] = n + 2  # n_cell_faces
    out[:, :, 2] = n  # bot cap count
    out[:, :, 3 : 3 + n] = bot[:, :, ::-1]  # bot cap nodes
    out[:, :, 3 + n] = n  # top cap count
    out[:, :, 4 + n : 4 + 2 * n] = top  # top cap nodes
    out[:, :, 4 + 2 * n :] = lateral  # lateral quads

    is_index[:, :, 3 : 3 + n] = True
    is_index[:, :, 4 + n : 4 + 2 * n] = True
    quad_node_positions = 4 + 2 * n + np.arange(n) * 5
    is_index[:, :, quad_node_positions[:, np.newaxis] + np.arange(1, 5)] = True
    return out.ravel(), is_index.ravel()


def extrude_faces(face_node_coordinates, z_bottom, z_top):
    """
    Parameters
    ----------
    face_node_coordinates: (n_face, n_vert, 2) float
    z_bottom: (n_layer, n_face) float
    z_top: (n_layer, n_face) float

    Returns
    -------
    xyz: (n_layer * n_face * 2 * n_vert, 3) float
    """
    n_face, n_vert, _ = face_node_coordinates.shape
    n_layer = z_top.shape[0]
    xy = np.broadcast_to(face_node_coordinates, (n_layer, n_face, n_vert, 2)).copy()
    xy = np.tile(xy, (1, 1, 2, 1))  # (n_layer, n_face, 2*n_vert, 2)
    z_bot_repeated = np.repeat(z_bottom[:, :, np.newaxis], n_vert, axis=2)
    z_top_repeated = np.repeat(z_top[:, :, np.newaxis], n_vert, axis=2)
    z = np.concatenate(
        [z_bot_repeated, z_top_repeated], axis=2
    )  # (n_layer, n_face, 2*n_vert)
    return np.concatenate([xy, z[..., np.newaxis]], axis=-1).reshape(-1, 3)


def extrude_nodes(face_node_coordinates, z_bottom, z_top):
    """
    Parameters
    ----------
    face_node_coordinates: (n_face, n_vert, 2) float
    z_bottom: (n_layer, n_face, n_vert) float
    z_top: (n_layer, n_face, n_vert) float

    Returns
    -------
    xyz: (n_layer * n_face * 2 * n_vert, 3) float
    """
    n_face, n_vert, _ = face_node_coordinates.shape
    n_layer = z_top.shape[0]
    xy = np.broadcast_to(face_node_coordinates, (n_layer, n_face, n_vert, 2)).copy()
    xy = np.tile(xy, (1, 1, 2, 1))  # (n_layer, n_face, 2*n_vert, 2)
    z = np.concatenate([z_bottom, z_top], axis=2)  # (n_layer, n_face, 2*n_vert)
    return np.concatenate([xy, z[..., np.newaxis]], axis=-1).reshape(-1, 3)


def ugrid2d_to_pyvista(
    face_node_connectivity,
    face_node_coordinates,
    z_bottom,
    z_top,
    z_location: Literal["face", "node"],
):
    import pyvista as pv
    import vtk

    if z_location not in ("face", "node"):
        raise ValueError(
            "z_location should be face or node, received instead: {z_location}"
        )

    PRISM_CELL_TYPES = {
        3: vtk.VTK_WEDGE,  # 3
        4: vtk.VTK_HEXAHEDRON,  # 4
        5: vtk.VTK_PENTAGONAL_PRISM,  # 5
        6: vtk.VTK_HEXAGONAL_PRISM,  # 6
        # vtk.VTK_POLYHEDRON otherwise
    }

    n_node = (face_node_connectivity != FILL_VALUE).sum(axis=1)
    unique_n = np.unique(n_node)
    n_layer = z_top.shape[0]

    all_points = []
    all_cells = []
    all_celltypes = []
    all_is_index = []

    point_offset = 0
    for n in unique_n:
        indices = n_node == n
        n_group_faces = indices.sum()
        group_connectivity = face_node_connectivity[indices]

        if n > 6:
            group_cells, group_is_index = build_polyhedric_prisms(
                n, group_connectivity, n_layer
            )
        else:
            group_cells, group_is_index = build_regular_prisms(
                n, group_connectivity, n_layer
            )

        group_celltypes = np.full(
            n_group_faces * n_layer,
            PRISM_CELL_TYPES.get(n, vtk.VTK_POLYHEDRON),
            dtype=int,
        )

        if z_location == "face":
            group_points = extrude_faces(
                face_node_coordinates[indices, :n],
                z_bottom[:, indices],
                z_top[:, indices],
            )
        else:
            node_indices = group_connectivity[:, :n]
            group_points = extrude_nodes(
                face_node_coordinates[indices, :n],
                z_bottom[:, node_indices],
                z_top[:, node_indices],
            )

        group_cells[group_is_index] += point_offset

        all_cells.append(group_cells)
        all_celltypes.append(group_celltypes)
        all_is_index.append(group_is_index)
        all_points.append(group_points)
        point_offset += n_group_faces * n_layer * 2 * n

    points = np.concatenate(all_points)
    cells = np.concatenate(all_cells)
    celltypes = np.concatenate(all_celltypes)
    is_index = np.concatenate(all_is_index)
    # Deduplicate vertices shared across groups, then remap cell point indices.
    unique_points, inverse = np.unique(points, return_inverse=True, axis=0)
    cells[is_index] = inverse[cells[is_index]]
    return pv.UnstructuredGrid(cells, celltypes, unique_points)
