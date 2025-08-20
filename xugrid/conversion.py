"""
Conversion from and to other data structures:

* GIS vector data (e.g. geopackage, shapefile)
* Structured data (e.g. rasters)

"""

import warnings
from typing import Tuple, Union

import numpy as np
import xarray as xr

import xugrid
from xugrid.constants import (
    FILL_VALUE,
    BoolArray,
    FloatArray,
    IntArray,
    IntDType,
    LineArray,
    MissingOptionalModule,
    PointArray,
    PolygonArray,
)
from xugrid.ugrid.connectivity import cross2d, ragged_index

try:
    import shapely
except ImportError:
    shapely = MissingOptionalModule("shapely")


def contiguous_xy(xy: FloatArray) -> Tuple[FloatArray, FloatArray]:
    x, y = [np.ascontiguousarray(a) for a in xy.T]
    return x, y


def nodes_to_points(x: FloatArray, y: FloatArray) -> PointArray:
    return shapely.points(x, y)


def edges_to_linestrings(
    x: FloatArray, y: FloatArray, edge_node_connectivity: IntArray
) -> LineArray:
    c = edge_node_connectivity.ravel()
    xy = np.column_stack((x[c], y[c]))
    i = np.repeat(np.arange(len(edge_node_connectivity)), 2)
    return shapely.linestrings(xy, indices=i)


def faces_to_polygons(
    x: FloatArray, y: FloatArray, face_node_connectivity: IntArray
) -> PolygonArray:
    is_data = face_node_connectivity != FILL_VALUE
    m_per_row = is_data.sum(axis=1)
    i = np.repeat(np.arange(len(face_node_connectivity)), m_per_row)
    c = face_node_connectivity.ravel()[is_data.ravel()]
    xy = np.column_stack((x[c], y[c]))
    rings = shapely.linearrings(xy, indices=i)
    return shapely.polygons(rings)


def points_to_nodes(points: PointArray) -> Tuple[FloatArray, FloatArray]:
    return contiguous_xy(shapely.get_coordinates(points))


def linestrings_to_edges(edges: LineArray) -> Tuple[FloatArray, FloatArray, IntArray]:
    xy, index = shapely.get_coordinates(edges, return_index=True)
    linear_index = np.arange(index.size)
    segments = np.column_stack([linear_index[:-1], linear_index[1:]])
    segments = segments[np.diff(index) == 0]
    unique, inverse = np.unique(xy, return_inverse=True, axis=0)
    inverse = inverse.ravel()
    segments = inverse[segments]
    x, y = contiguous_xy(unique)
    return x, y, segments


def _remove_last_vertex(xy: FloatArray, indices: IntArray):
    """
    GEOS polygons are always closed: the first and last vertex are the same.
    UGRID faces are by definition closed. So we'll have to remove the last,
    repeated, vertex of every polygon here.
    """
    no_repeats = np.diff(indices, append=-1) == 0
    xy = xy[no_repeats]
    indices = indices[no_repeats]
    return xy, indices


def polygons_to_faces(
    polygons: PolygonArray,
) -> Tuple[FloatArray, FloatArray, IntArray, int]:
    xy, indices = _remove_last_vertex(
        *shapely.get_coordinates(polygons, return_index=True)
    )
    unique, inverse = np.unique(xy, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    n = len(polygons)
    m_per_row = np.bincount(indices)
    m = m_per_row.max()
    # Allocate 2D array and create a flat view of the dense connectivity
    conn = np.empty((n, m), dtype=IntDType)
    flat_conn = conn.ravel()
    if (n * m) == indices.size:
        # Shortcut if fill_value is not present, when all of same geom. type
        # e.g. all triangles or all quadrangles
        valid = slice(None)  # a[:] equals a[slice(None)]
    else:
        valid = ragged_index(n, m, m_per_row).ravel()
        flat_conn[~valid] = FILL_VALUE
    flat_conn[valid] = inverse
    x, y = contiguous_xy(unique)
    return x, y, conn


def _scalar_spacing(coords, spacing):
    dim = coords.dims[0]
    diff = coords.diff(dim)
    spacing_value = abs(spacing.item())
    if not np.allclose(
        abs(diff.to_numpy()), spacing_value, atol=abs(1.0e-4 * spacing.item())
    ):
        raise ValueError(
            f"spacing of {coords.name} does not match value of {spacing.name}"
        )
    halfdiff = xr.full_like(coords, 0.5 * spacing_value)
    return halfdiff


def _array_spacing(coords, spacing):
    if coords.size != spacing.size:
        raise ValueError(f"size of {coords.name} does not match size of {spacing.name}")
    halfdiff = 0.5 * abs(spacing)
    return halfdiff


def _is_monotonic_and_increasing(coord, axis=0) -> bool:
    """
    Test if monotonic and retun whether increasing along axis.
    Raises error if not monotonic.

    Copied and slightly adapted from xarray.utils.

    >>> _is_monotonic(np.array([0, 1, 2]))
    True
    >>> _is_monotonic(np.array([2, 1, 0]))
    True
    >>> _is_monotonic(np.array([0, 2, 1]))
    False
    """
    coord = np.asarray(coord)
    n = coord.shape[axis]
    delta_pos = coord.take(np.arange(1, n), axis=axis) >= coord.take(
        np.arange(0, n - 1), axis=axis
    )
    delta_neg = coord.take(np.arange(1, n), axis=axis) <= coord.take(
        np.arange(0, n - 1), axis=axis
    )
    if np.all(delta_pos):
        return True
    elif np.all(delta_neg):
        return False
    else:
        raise ValueError("The input coordinate is not monotonic.")


def infer_interval_breaks(coord, axis: int = 0, check_monotonic: bool = False):
    """
    Infer intervals from cell center coordinates.

    Copied and adapted from xarray.utils.

    >>> _infer_interval_breaks(np.arange(5))
    array([-0.5,  0.5,  1.5,  2.5,  3.5,  4.5])
    >>> _infer_interval_breaks([[0, 1], [3, 4]], axis=1)
    array([[-0.5,  0.5,  1.5],
           [ 2.5,  3.5,  4.5]])
    """
    coord = np.asarray(coord)
    if check_monotonic:
        _is_monotonic_and_increasing(coord, axis=axis)

    deltas = 0.5 * np.diff(coord, axis=axis)
    if deltas.size == 0:
        deltas = np.array(0.0)

    first = np.take(coord, [0], axis=axis) - np.take(deltas, [0], axis=axis)
    last = np.take(coord, [-1], axis=axis) + np.take(deltas, [-1], axis=axis)
    trim_last = tuple(
        slice(None, -1) if n == axis else slice(None) for n in range(coord.ndim)
    )
    interval_breaks = np.concatenate(
        [first, coord[trim_last] + deltas, last], axis=axis
    )
    return interval_breaks


def infer_interval_breaks1d(
    obj: Union[xr.DataArray, xr.Dataset],
    var: str,
) -> np.ndarray:
    """
    Infer the breaks for 1D coordinates.

        * For non-equidistant grids, taking half of each cell size result in
          wrong answers.
        * Cell size may be provided by a dx or dy attribute instead.
        * We also want to take 1-row or 1 column topologies into account.
    """
    coord = obj[var]
    spacing_name = f"d{var}"

    # Spacing name is provided:
    if spacing_name in obj.coords:
        spacing = obj[spacing_name]
        if spacing.ndim > 1:
            raise NotImplementedError(
                f"More than one dimension in spacing variable: {spacing_name}"
            )
        if spacing.shape in ((), (1,)):
            halfdiff = _scalar_spacing(coord, spacing)
        else:
            halfdiff = _array_spacing(coord, spacing)

        # Now check if monotonic and take orientation into account.
        if _is_monotonic_and_increasing(coord):
            intervals = np.insert(coord + halfdiff, 0, coord[0] - halfdiff[0])
        else:
            intervals = np.insert(coord - halfdiff, 0, coord[0] + halfdiff[0])

    # Implicit spacing, infer from coordinates instead:
    else:
        if coord.size == 1:
            raise ValueError(
                f"Cannot derive spacing of 1-sized coordinate: {var} \n"
                f"Assign a d{var} variable with spacing instead."
            )
        intervals = infer_interval_breaks(coord.to_numpy(), check_monotonic=True)

    return intervals


def infer_xy_coords(obj):
    # First check names, then check whether CF roles are specified.
    x = None
    y = None
    if "x" in obj.dims and "y" in obj.dims:
        x, y = "x", "y"
    elif "longitude" in obj.dims and "latitude" in obj.dims:
        x, y = "longitude", "latitude"
    else:
        for name, da in obj.coords.items():
            # Only 1D dimensions are allowed.
            if da.ndim != 1:
                continue

            attrs = da.attrs
            axis = attrs.get("axis", "").lower()
            stdname = attrs.get("standard_name", "").lower()
            if axis == "x" or stdname in ("longitude", "projection_x_coordinate"):
                x = name
            elif axis == "y" or stdname in ("latitude", "projection_y_coordinate"):
                y = name

    return x, y


def bounds1d_to_vertices(bounds: np.ndarray):
    diff = np.diff(bounds, axis=0)
    ascending = (diff >= 0.0).all()
    descending = (diff <= 0.0).all()
    if ascending:
        vertices = np.concatenate((bounds[:, 0], bounds[-1:, 1]))
    elif descending:
        vertices = np.concatenate((bounds[:, 1], bounds[-1:, 0]))
    else:
        raise ValueError("Bounds are not monotonic ascending or monotonic descending")
    return vertices


def quad_area(coordinates: FloatArray):
    # Subtly different from connectivity.area_from_coordinates
    # Due to lexsort (for repeated values), coordinates are not in CCW order.
    # We might get conflicting sign on the determinant; we always get two triangles,
    # so call abs() before summing.
    xy0 = coordinates[:, 0]
    a = coordinates[:, :-1] - xy0[:, np.newaxis]
    b = coordinates[:, 1:] - xy0[:, np.newaxis]
    determinant = cross2d(a, b)
    return 0.5 * abs(determinant).sum(axis=1)


def bounds2d_to_topology2d(
    x_bounds: np.ndarray, y_bounds: np.ndarray
) -> Tuple[FloatArray, BoolArray]:
    x = x_bounds.reshape(-1, 4)
    y = y_bounds.reshape(-1, 4)
    # Make sure repeated nodes are consecutive so we can find them later.
    # lexsort along axis 1.
    sorter = np.lexsort((y, x))
    face_node_coordinates = np.stack(
        (
            np.take_along_axis(x, sorter, axis=1),
            np.take_along_axis(y, sorter, axis=1),
        ),
        axis=-1,
    )

    # Check whether all coordinates form valid UGRID topologies.
    # We can only maintain triangles and quadrangles.
    # We also have to discard collinear triangles or quadrangles.
    n_unique = (
        (face_node_coordinates != np.roll(face_node_coordinates, 1, axis=1))
        .any(axis=-1)
        .sum(axis=1)
    )
    valid = (n_unique >= 3) & (quad_area(face_node_coordinates) > 0)
    if not valid.all():
        warnings.warn(
            "A UGRID2D face requires at least three unique non-collinear vertices.\n"
            f"Your structured bounds contain {len(valid) - valid.sum()} invalid faces.\n"
            "These will be omitted from the Ugrid2d topology.",
        )
    # Also check for NaNs.
    index = np.isfinite(face_node_coordinates.reshape(-1, 8)).all(axis=-1) & valid
    face_node_coordinates = face_node_coordinates[index]

    # Guarantee counterclockwise orientation.
    face_centroids = np.mean(face_node_coordinates, axis=1)
    dx = face_node_coordinates[..., 0] - face_centroids[:, np.newaxis, 0]
    dy = face_node_coordinates[..., 1] - face_centroids[:, np.newaxis, 1]
    angle = np.arctan2(dy, dx)
    # When nodes are repeated, make sure the repeated node ends up as last so we can
    # construct the face_node_connectivity directly from it. We do so by inserting
    # an angle of np.inf for the repeated nodes.
    angle[:, 1:][angle[:, 1:] == angle[:, :-1]] = np.inf
    counterclockwise = np.argsort(angle, axis=1)
    face_node_coordinates = np.take_along_axis(
        face_node_coordinates, counterclockwise[..., None], axis=1
    )
    # TODO: this assumes bounds align exactly. Do we need a tolerance?
    xy, inverse = np.unique(
        face_node_coordinates.reshape((-1, 2)), return_inverse=True, axis=0
    )
    face_node_connectivity = inverse.reshape((-1, 4))
    # For triangles, set the last node to the fill value of -1.
    face_node_connectivity[n_unique[index] == 3, -1] = -1
    return *xy.T, face_node_connectivity, index


def grid_from_geodataframe(geodataframe: "geopandas.GeoDataFrame"):  # type: ignore # noqa
    import geopandas as gpd

    gdf = geodataframe
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError(
            f"Cannot convert a {type(gdf).__name__}, expected a GeoDataFrame"
        )

    geom_types = gdf.geom_type.unique()
    if len(geom_types) == 0:
        raise ValueError("geodataframe contains no geometry")
    elif len(geom_types) > 1:
        message = ", ".join(geom_types)
        raise ValueError(f"Multiple geometry types detected: {message}")

    geom_type = geom_types[0]
    if geom_type == "LineString":
        grid = xugrid.Ugrid1d.from_geodataframe(gdf)
    elif geom_type == "Polygon":
        grid = xugrid.Ugrid2d.from_geodataframe(gdf)
    else:
        raise ValueError(
            f"Invalid geometry type: {geom_type}. Expected Linestring or Polygon."
        )
    return grid


def grid_from_dataset(dataset: xr.Dataset, topology: str):
    topodim = dataset[topology].attrs["topology_dimension"]
    if topodim == 1:
        return xugrid.Ugrid1d.from_dataset(dataset, topology)
    elif topodim == 2:
        return xugrid.Ugrid2d.from_dataset(dataset, topology)
    elif topodim == 3:
        raise NotImplementedError
    else:
        raise ValueError(f"Invalid topology dimension: {topodim}")
