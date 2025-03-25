import numpy as np
from typing import Tuple

from xugrid.constants import FloatArray, IntArray

def section_coordinates_1d(
    edges: FloatArray, xy: FloatArray, dim: str, index: IntArray, name: str
) -> Tuple[IntArray, dict]:
    s = np.linalg.norm(xy - edges[0, 0], axis=1)
    order = np.argsort(s)
    coords = {
        f"{name}_x": (dim, xy[order, 0]),
        f"{name}_y": (dim, xy[order, 1]),
        f"{name}_s": (dim, s[order]),
    }
    return coords, index[order]

def section_coordinates_2d(
    edges: FloatArray, xy: FloatArray, dim: str, index: IntArray, name: str
) -> Tuple[IntArray, dict]:
    # TODO: add boundaries xy[:, 0] and xy[:, 1]
    xy_mid = 0.5 * (xy[:, 0, :] + xy[:, 1, :])
    return section_coordinates_1d(edges, xy_mid, dim, index, name)
