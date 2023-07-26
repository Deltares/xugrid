"""
The functions in this module serve to polygonize
"""
from typing import Tuple

import numpy as np
from scipy import sparse

from xugrid.constants import IntArray


def _bbox_area(bounds):
    return (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])


def _classify(
    fill_value: int, i: IntArray, j: IntArray, face_values: np.ndarray
) -> Tuple[int, IntArray]:
    """
    Find out how many discrete polygons are created. Identify the connectivity,
    such that we can select a single polygon afterwards.

    Parameters
    ----------
    fill_value: int
        Fill value in j: marks exterior edges.
    i: np.ndarray of int
        First face of the edge.
    j: np.ndarray of int
        Second face of the edge.
    face_values: np.ndarray

    Returns
    -------
    n_polygon: int
    polygon_id: np.ndarray of int
    """
    # Every face connects up to two faces. vi holds the values of the first face,
    # vj holds the value of the second face.
    vi = face_values[i]
    vj = face_values[j]
    n = face_values.size
    # For labelling, only those parts of the mesh that have the same value
    # should be connected with each other.
    # Since we dropped NaN values before, we needn't worry about those.
    is_connection = (i != fill_value) & (j != fill_value) & (vi == vj)
    i = i[is_connection]
    j = j[is_connection]
    ij = np.concatenate([i, j])
    ji = np.concatenate([j, i])
    coo_content = (ji, (ij, ji))
    # Make sure to explicitly set the matrix shape: otherwise, isolated
    # elements witout any connection might disappear, and connected_components
    # will not return a value for every face.
    coo_matrix = sparse.coo_matrix(coo_content, shape=(n, n))
    # We can classify the grid faces using this (reduced) connectivity
    return sparse.csgraph.connected_components(coo_matrix)


def polygonize(uda: "UgridDataArray") -> "gpd.GeoDataFrame":  # type: ignore # noqa
    """
    This function creates vector polygons for all connected regions of cells
    (faces) in the Ugrid2d topology sharing a common value.

    The produced polygon edges will follow exactly the cell boundaries. When
    the data consists of many unique values (e.g. unbinned elevation data), the
    result will essentially be one polygon per face. In such cases, it is much
    more efficient to use ``xugrid.UgridDataArray.to_geodataframe``, which
    directly converts every cell to a polygon. This function is meant for data
    with relatively few unique values such as classification results.

    Parameters
    ----------
    uda: UgridDataArray
        The DataArray should only contain the face dimension. Additional
        dimensions, such as time, are not allowed.

    Returns
    -------
    polygonized: GeoDataFrame
    """

    import geopandas as gpd
    import shapely

    facedim = uda.ugrid.grid.face_dimension
    if uda.dims != (facedim,):
        raise ValueError(
            "Cannot polygonize non-xy spatial dimensions. Expected only"
            f"({facedim},), but received {uda.dims}."
        )

    # First remove the NaN values. These will not be polygonized anyway.
    dropped = uda.dropna(dim=uda.ugrid.grid.face_dimension)
    face_values = dropped.to_numpy()
    grid = dropped.ugrid.grid
    i, j = grid.edge_face_connectivity.T
    fill_value = grid.fill_value
    n_polygon, polygon_id = _classify(fill_value, i, j, face_values)

    # Now we identify for each label the subset of edges. These are the
    # "exterior" edges: either the exterior edge of the mesh identified by a
    # fill value, or by being connected to a cell with a different value.
    coordinates = grid.node_coordinates
    data_i = face_values[i]
    vi = polygon_id[i]
    vj = polygon_id[j]
    # Ensure that no result thas has been created by indexing with the
    # fill_value remains. Since polygon_id starts counting a 0, we may use -1.
    vi[i == fill_value] = -1
    vj[j == fill_value] = -1
    boundary = vi != vj

    polygons = []
    values = []
    for label in range(n_polygon):
        keep = ((vi == label) | (vj == label)) & boundary
        # The result of shapely polygonize is always a GeometryCollection.
        # Holes are included twice: once as holes in the largest body, and once
        # more as polygons on their own. We are interested in the largest
        # polygon, which we identify through its bounding box.
        edges = grid.edge_node_connectivity[keep]
        collection = shapely.polygonize(shapely.linestrings(coordinates[edges]))
        polygon = max(collection.geoms, key=lambda x: _bbox_area(x.bounds))
        # Find the first True value in keep, use that to fetch the polygon
        # value.
        value = data_i[keep.argmax()]
        polygons.append(polygon)
        values.append(value)

    return gpd.GeoDataFrame({"values": values}, geometry=polygons)
