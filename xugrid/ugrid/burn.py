from __future__ import annotations

from typing import List, Union

import numba as nb
import numpy as np
import xarray as xr
from numba_celltree.constants import TOLERANCE_ON_EDGE, Point, Triangle
from numba_celltree.geometry_utils import (
    as_point,
    as_triangle,
    cross_product,
    to_vector,
)

import xugrid
from xugrid.constants import FloatArray, IntArray, MissingOptionalModule

try:
    import shapely

except ImportError:
    shapely = MissingOptionalModule("shapely")

try:
    import mapbox_earcut

except ImportError:
    mapbox_earcut = MissingOptionalModule("mapbox_earcut")


@nb.njit(inline="always")
def in_bounds(p: Point, a: Point, b: Point) -> bool:
    """
    Check whether point p falls within the bounding box created by a and b
    (after we've checked the size of the cross product).

    However, we must take into account that a line may be either vertical
    (dx=0) or horizontal (dy=0) and only evaluate the non-zero value.

    If the area created by p, a, b is tiny AND p is within the bounds of a and
    b, the point lies very close to the edge.
    """
    # Already in numba_celltree, unreleased.
    dx = b.x - a.x
    dy = b.y - a.y
    if abs(dx) >= abs(dy):
        if dx > 0:
            return a.x <= p.x and p.x <= b.x
        return b.x <= p.x and p.x <= a.x
    else:
        if dy > 0:
            return a.y <= p.y and p.y <= b.y
        return b.y <= p.y and p.y <= a.y


@nb.njit(inline="always")
def point_in_triangle(p: Point, t: Triangle) -> bool:
    # TODO: move this into numba_celltree instead?
    ap = to_vector(t.a, p)
    bp = to_vector(t.b, p)
    cp = to_vector(t.c, p)
    ab = to_vector(t.a, t.b)
    bc = to_vector(t.b, t.c)
    ca = to_vector(t.c, t.a)
    # Do a half plane check.
    A = cross_product(ab, ap)
    B = cross_product(bc, bp)
    C = cross_product(ca, cp)
    signA = A > 0
    signB = B > 0
    signC = C > 0
    if (signA == signB) and (signB == signC):
        return True

    # Test whether p is located on/very close to edges.
    if (
        (abs(A) < TOLERANCE_ON_EDGE)
        and in_bounds(p, t.a, t.b)
        or (abs(B) < TOLERANCE_ON_EDGE)
        and in_bounds(p, t.b, t.c)
        or (abs(C) < TOLERANCE_ON_EDGE)
        and in_bounds(p, t.c, t.a)
    ):
        return True

    return False


@nb.njit(inline="always", parallel=True, cache=True)
def points_in_triangles(
    points: FloatArray,
    face_indices: IntArray,
    faces: IntArray,
    vertices: FloatArray,
):
    # TODO: move this into numba_celltree instead?
    n_points = len(points)
    inside = np.empty(n_points, dtype=np.bool_)
    for i in nb.prange(n_points):
        face_index = face_indices[i]
        face = faces[face_index]
        triangle = as_triangle(vertices, face)
        point = as_point(points[i])
        inside[i] = point_in_triangle(point, triangle)
    return inside


def _locate_polygon(
    grid: "xu.Ugrid2d",  # type: ignore # noqa
    exterior: FloatArray,
    interiors: List[FloatArray],
    all_touched: bool,
) -> IntArray:
    """
    Locate a single polygon.

    This algorithm burns a polygon vector geometry in a 2d topology by:

    * Extracting the exterior and interiors (holes) coordinates from the
      polygon.
    * Breaking every polygon down into a triangles using an "earcut" algorithm.
    * Searching the grid for these triangles.

    Due to the use of the separating axes theorem, _locate_faces finds all
    intersecting triangles, including those who only touch the edge.
    To enable all_touched=False, we have to search the centroids of candidates
    in the intersecting triangles.

    Parameters
    ----------
    grid: Ugrid2d
    exterior: FloatArray
        Exterior of the polygon.
    interiors: List[FloatArray]
        Interior holes of the polygon.
    all_touched: bool
        Whether to include include cells whose centroid falls inside, of every
        cell that is touched.
    """

    import mapbox_earcut

    rings = np.cumsum([len(exterior)] + [len(interior) for interior in interiors])
    vertices = np.vstack([exterior] + interiors).astype(np.float64)
    triangles = mapbox_earcut.triangulate_float64(vertices, rings).reshape((-1, 3))
    triangle_indices, grid_indices = grid.celltree._locate_faces(vertices, triangles)
    if all_touched:
        return grid_indices
    else:
        centroids = grid.centroids[grid_indices]
        inside = points_in_triangles(
            points=centroids,
            face_indices=triangle_indices,
            faces=triangles,
            vertices=vertices,
        )
        return grid_indices[inside]


def _burn_polygons(
    polygons: "geopandas.GeoSeries",  # type: ignore # noqa
    like: "xugrid.Ugrid2d",
    values: np.ndarray,
    all_touched: bool,
    output: FloatArray,
) -> None:
    exterior_coordinates = [
        shapely.get_coordinates(exterior) for exterior in polygons.exterior
    ]
    interior_coordinates = [
        [shapely.get_coordinates(p_interior) for p_interior in p_interiors]
        for p_interiors in polygons.interiors
    ]
    to_burn = np.empty(like.n_face, dtype=bool)

    for exterior, interiors, value in zip(
        exterior_coordinates, interior_coordinates, values
    ):
        to_burn = _locate_polygon(like, exterior, interiors, all_touched)
        output[to_burn] = value

    return


def _burn_points(
    points: "geopandas.GeoSeries",  # type: ignore # noqa
    like: "xugrid.Ugrid2d",
    values: np.ndarray,
    output: FloatArray,
) -> None:
    """Simply searches the points in the ``like`` 2D topology."""
    xy = shapely.get_coordinates(points)
    to_burn = like.locate_points(xy)
    output[to_burn] = values
    return


def _burn_lines(
    lines: "geopandas.GeoSeries",  # type: ignore # noqa
    like: "xugrid.Ugrid2d",
    values: np.ndarray,
    output: FloatArray,
) -> None:
    """
    Burn the line values into the underlying faces.

    This algorithm breaks any linestring down into edges (two x, y points). We
    search and intersect every edge in the ``like`` grid, the intersections are
    discarded.
    """
    xy, index = shapely.get_coordinates(lines, return_index=True)
    # From the coordinates and the index, create the (n_edge, 2, 2) shape array
    # containing the edge_coordinates.
    linear_index = np.arange(index.size)
    segments = np.column_stack([linear_index[:-1], linear_index[1:]])
    # Only connections with vertices with the same index are valid.
    valid = np.diff(index) == 0
    segments = segments[valid]
    edges = xy[segments]
    # Now query the grid for these edges.
    edge_index, face_index, _ = like.intersect_edges(edges)
    # Find the associated values.
    line_index = index[1:][valid]
    value_index = line_index[edge_index]
    output[face_index] = values[value_index]
    return


def burn_vector_geometry(
    gdf: "geopandas.GeoDataframe",  # type: ignore # noqa
    like: Union["xugrid.Ugrid2d", "xugrid.UgridDataArray", "xugrid.UgridDataset"],
    column: str | None = None,
    fill: Union[int, float] = np.nan,
    all_touched: bool = False,
) -> xugrid.UgridDataArray:
    """
    Burn vector geometries (points, lines, polygons) into a Ugrid2d mesh.

    If no ``column`` argument is provided, a value of 1.0 will be burned in to
    the mesh.

    Parameters
    ----------
    gdf: geopandas.GeoDataFrame
        Polygons, points, and/or lines to be burned into the grid.
    like: UgridDataArray, UgridDataset, or Ugrid2d
        Grid to burn the vector data into.
    column: str, optional
        Name of the geodataframe column of which to the values to burn into
        grid.
    fill: int, float, optional, default value ``np.nan``.
        Fill value for nodata areas.
    all_touched: bool, optional, default value ``False``.
        All mesh faces (cells) touched by polygons will be updated, not just
        those whose center point is within the polygon.

    Returns
    -------
    burned: UgridDataArray
    """
    import geopandas as gpd

    POINT = shapely.GeometryType.POINT
    LINESTRING = shapely.GeometryType.LINESTRING
    LINEARRING = shapely.GeometryType.LINEARRING
    POLYGON = shapely.GeometryType.POLYGON
    GEOM_NAMES = {v: k for k, v in shapely.GeometryType.__members__.items()}

    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError(f"gdf must be GeoDataFrame, received: {type(gdf).__name__}")
    if isinstance(like, (xugrid.UgridDataArray, xugrid.UgridDataset)):
        like = like.ugrid.grid
    if not isinstance(like, xugrid.Ugrid2d):
        raise TypeError(
            "Like must be Ugrid2d, UgridDataArray, or UgridDataset;"
            f"received: {type(like).__name__}"
        )
    geometry_id = shapely.get_type_id(gdf.geometry)
    allowed_types = (POINT, LINESTRING, LINEARRING, POLYGON)
    if not np.isin(geometry_id, allowed_types).all():
        received = ", ".join(
            [GEOM_NAMES[geom_id] for geom_id in np.unique(geometry_id)]
        )
        raise TypeError(
            "GeoDataFrame contains unsupported geometry types. Can only burn "
            "Point, LineString, LinearRing, and Polygon geometries. Received: "
            f"{received}"
        )

    points = gdf.loc[geometry_id == POINT]
    lines = gdf.loc[(geometry_id == LINESTRING) | (geometry_id == LINEARRING)]
    polygons = gdf.loc[geometry_id == POLYGON]

    if column is None:
        point_values = np.ones(len(points), dtype=float)
        line_values = np.ones(len(lines), dtype=float)
        poly_values = np.ones(len(polygons), dtype=float)
    else:
        point_values = points[column].to_numpy()
        line_values = lines[column].to_numpy()
        poly_values = polygons[column].to_numpy()

    output = np.full(like.n_face, fill)
    if len(polygons) > 0:
        _burn_polygons(polygons.geometry, like, poly_values, all_touched, output)
    if len(lines) > 0:
        _burn_lines(lines.geometry, like, line_values, output)
    if len(points) > 0:
        _burn_points(points.geometry, like, point_values, output)

    return xugrid.UgridDataArray(
        obj=xr.DataArray(output, dims=[like.face_dimension], name=column),
        grid=like,
    )


def grid_from_earcut_polygons(
    polygons: "geopandas.GeoDataFrame",  # type: ignore # noqa
    return_index: bool = False,
):
    import geopandas as gpd

    if not isinstance(polygons, gpd.GeoDataFrame):
        raise TypeError(f"Expected GeoDataFrame, received: {type(polygons).__name__}")

    geometry = polygons.geometry
    POLYGON = shapely.GeometryType.POLYGON
    GEOM_NAMES = {v: k for k, v in shapely.GeometryType.__members__.items()}
    geometry_id = shapely.get_type_id(geometry)

    if not (geometry_id == POLYGON).all():
        GEOM_NAMES = {v: k for k, v in shapely.GeometryType.__members__.items()}
        received = ", ".join(
            [GEOM_NAMES[geom_id] for geom_id in np.unique(geometry_id)]
        )
        raise TypeError(
            "geometry contains unsupported geometry types. Can only triangulate "
            f"Polygon geometries. Received: {received}"
        )

    # Shapely does not provide a vectorized manner to get the interior rings
    # easily, an index argument is always required, which is a poor fit for
    # heterogeneous polygons (where some have no holes, and some may have
    # hundreds).
    # map_box_earcut is also not vectorized for polygons, so we simply loop
    # over the geometries here.
    exterior_coordinates = [
        shapely.get_coordinates(exterior) for exterior in geometry.exterior
    ]
    interior_coordinates = [
        [shapely.get_coordinates(p_interior) for p_interior in p_interiors]
        for p_interiors in geometry.interiors
    ]

    all_triangles = []
    offset = 0
    for exterior, interiors in zip(exterior_coordinates, interior_coordinates):
        rings = np.cumsum([len(exterior)] + [len(interior) for interior in interiors])
        vertices = np.vstack([exterior] + interiors).astype(np.float64)
        triangles = mapbox_earcut.triangulate_float64(vertices, rings).reshape((-1, 3))
        all_triangles.append(triangles + offset)
        offset += len(vertices)

    face_nodes = np.concatenate(all_triangles).reshape((-1, 3))
    all_vertices = shapely.get_coordinates(geometry)
    node_x = all_vertices[:, 0]
    node_y = all_vertices[:, 1]
    grid = xugrid.Ugrid2d(node_x, node_y, -1, face_nodes)

    if return_index:
        n_triangles = [len(triangles) for triangles in all_triangles]
        index = np.repeat(np.arange(len(geometry)), n_triangles)
        return grid, index
    else:
        return grid


def earcut_triangulate_polygons(
    polygons: "geopandas.GeoDataframe",  # type: ignore # noqa
    column: str | None = None,
) -> xugrid.UgridDataArray:
    """
    Break down polygons using mapbox_earcut, and create a mesh from the
    resulting triangles.

    If no ``column`` argument is provided, the polygon index will be assigned
    to the grid faces.

    Parameters
    ----------
    polygons: geopandas.GeoDataFrame
        Polygons to convert to triangles.
    column: str, optional
        Name of the geodataframe column of which to the values to assign
        to the grid faces.

    Returns
    -------
    triangulated: UgridDataArray
    """
    grid, index = grid_from_earcut_polygons(polygons, return_index=True)

    if column is not None:
        da = (
            polygons[column]
            .reset_index(drop=True)
            .to_xarray()
            .isel(index=index)
            .rename({"index": grid.face_dimension})
        )
    else:
        da = xr.DataArray(data=index, dims=(grid.face_dimension,))

    return xugrid.UgridDataArray(da, grid)
