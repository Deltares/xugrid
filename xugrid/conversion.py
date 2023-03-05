"""
Conversion from and to other data structures:

* GIS vector data (e.g. geopackage, shapefile)
* Structured data (e.g. rasters)

"""
from typing import Tuple, Union

import numpy as np
import xarray as xr

from xugrid.constants import (
    FloatArray,
    IntArray,
    IntDType,
    LineArray,
    MissingOptionalModule,
    PointArray,
    PolygonArray,
)
from xugrid.ugrid.connectivity import ragged_index
from xugrid.ugrid.ugrid1d import Ugrid1d
from xugrid.ugrid.ugrid2d import Ugrid2d

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
    x: FloatArray, y: FloatArray, face_node_connectivity: IntArray, fill_value: int
) -> PolygonArray:
    is_data = face_node_connectivity != fill_value
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
    n = len(polygons)
    m_per_row = np.bincount(indices)
    m = m_per_row.max()
    fill_value = -1
    # Allocate 2D array and create a flat view of the dense connectivity
    conn = np.empty((n, m), dtype=IntDType)
    flat_conn = conn.ravel()
    if (n * m) == indices.size:
        # Shortcut if fill_value is not present, when all of same geom. type
        # e.g. all triangles or all quadrangles
        valid = slice(None)  # a[:] equals a[slice(None)]
    else:
        valid = ragged_index(n, m, m_per_row).ravel()
        flat_conn[~valid] = fill_value
    flat_conn[valid] = inverse
    x, y = contiguous_xy(unique)
    return x, y, conn, fill_value


def _scalar_spacing(coords, spacing):
    dim = coords.dims[0]
    diff = coords.diff(dim)
    spacing_value = abs(spacing.item())
    if not np.allclose(
        abs(diff.values), spacing_value, atol=abs(1.0e-4 * spacing.item())
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


def _implicit_spacing(coords):
    dim = coords.dims[0]
    if coords.size == 1:
        raise ValueError(
            f"Cannot derive spacing of 1-sized coordinate: {coords.name} \n"
            f"Set bounds yourself or assign a d{coords.name} variable with spacing"
        )
    halfdiff = 0.5 * abs(coords.diff(dim)).values
    return np.insert(halfdiff, 0, halfdiff[0])


def infer_bounds(
    obj: Union[xr.DataArray, xr.Dataset],
    var: str,
):
    coords = obj[var]
    index = obj.indexes[var]
    if not (index.is_monotonic_increasing or index.is_monotonic_decreasing):
        raise ValueError(f"{var} is not monotonic")

    # e.g. rioxarray will set dx, dy as (scalar) values.
    spacing_name = f"d{var}"
    if spacing_name in obj.coords:
        spacing = obj[spacing_name]
        spacing_shape = spacing.shape
        if len(spacing_shape) > 1:
            raise NotImplementedError(
                f"More than one dimension in spacing variable: {spacing_name}"
            )

        if spacing_shape in ((), (1,)):
            halfdiff = _scalar_spacing(coords, spacing)
        else:
            halfdiff = _array_spacing(coords, spacing)
    # Implicit spacing
    else:
        halfdiff = _implicit_spacing(coords)

    lower = coords - halfdiff
    upper = coords + halfdiff
    bounds = xr.concat([lower, upper], dim="bounds").transpose()
    return bounds


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


def bounds_to_vertices(bounds):
    diff = np.diff(bounds.values, axis=0)
    ascending = (diff >= 0.0).all()
    descending = (diff <= 0.0).all()
    if ascending:
        vertices = np.concatenate((bounds[:, 0], bounds[-1:, 1]))
    elif descending:
        vertices = np.concatenate((bounds[:, 1], bounds[-1:, 0]))
    else:
        raise ValueError("Bounds are not monotonic ascending or monotonic descending")
    return vertices


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
        grid = Ugrid1d.from_geodataframe(gdf)
    elif geom_type == "Polygon":
        grid = Ugrid2d.from_geodataframe(gdf)
    else:
        raise ValueError(
            f"Invalid geometry type: {geom_type}. Expected Linestring or Polygon."
        )
    return grid


def grid_from_dataset(dataset: xr.Dataset, topology: str):
    topodim = dataset[topology].attrs["topology_dimension"]
    if topodim == 1:
        return Ugrid1d.from_dataset(dataset, topology)
    elif topodim == 2:
        return Ugrid2d.from_dataset(dataset, topology)
    elif topodim == 3:
        raise NotImplementedError
    else:
        raise ValueError(f"Invalid topology dimension: {topodim}")
