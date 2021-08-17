"""
Snapes nodes at an arbitrary distance together.
"""
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import KDTree

FloatArray = np.ndarray
IntArray = np.ndarray


def snap(
    x: FloatArray, y: FloatArray, distance: float
) -> Tuple[FloatArray, FloatArray, IntArray]:
    # First, find all the points that lie within distance of each other
    coords = np.column_stack((x, y))
    n = len(coords)
    tree = KDTree(coords)
    pairs = np.array([*tree.query_pairs(r=distance)])
    # If no pairs are found, there is nothing to snap
    if len(pairs) > 0:
        # Next, group the points together.
        # Point A might lie close to point B, point B might lie close to
        # Point C, and so on. These points are grouped, and their centroid
        # is computed.
        i = pairs[:, 0]
        j = pairs[:, 1]
        coo_content = (np.ones(i.size), (i, j))
        coo_matrix = sparse.coo_matrix(coo_content, shape=(n, n))
        _, labels = connected_components(coo_matrix, directed=False)
        new = (
            pd.DataFrame({"label": labels, "x": x, "y": y})
            .groupby("label")
            .agg(
                {
                    "x": "mean",
                    "y": "mean",
                }
            )
        )
        # In case of a connectivity array, labels can be used to index into,
        # yielding the updated numbers. E.g.:
        # updated_face_nodes = labels[face_nodes]
        return labels, new["x"].values, new["y"].values
    else:
        return None, x, y
