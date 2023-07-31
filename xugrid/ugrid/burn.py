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
    This algorithm burns polygon vector geometries in a 2d topology by:

    * Extracting the exterior and interiors (holes) coordinates from the
      polygons.
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
    """
    Simply searches the points in the ``like`` 2D topology.
    """
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
    like: "xugrid.Ugrid2d",
    column: str = None,
    fill: Union[int, float] = np.nan,
    all_touched: bool = False,
) -> None:
    """
    Burn vector geometries (points, lines, polygons) into a Ugrid2d mesh.

    If no ``column`` argument is provided, a value of 1.0 will be burned in to
    the mesh.

    Parameters
    ----------
    gdf: geopandas.GeoDataFrame
    like: UgridDataArray, UgridDataset, or Ugrid2d
    column: str
        Column name of geodataframe to burn into mesh
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

    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError(f"gdf must be GeoDataFrame, received: {type(like).__name__}")
    if isinstance(like, (xugrid.UgridDataArray, xugrid.UgridDataset)):
        like = like.ugrid.grid
    if not isinstance(like, xugrid.Ugrid2d):
        raise TypeError(
            "Like must be Ugrid2d, UgridDataArray, or UgridDataset;"
            f"received: {type(like).__name__}"
        )
    geometry_id = shapely.get_type_id(gdf.geometry)
    allowed_types = (
        shapely.GeometryType.POINT,
        shapely.GeometryType.LINESTRING,
        shapely.GeometryType.POLYGON,
    )
    if not np.isin(geometry_id, allowed_types).all():
        raise TypeError(
            "GeoDataFrame contains unsupported geometry types. Can only burn "
            "Point, LineString, and Polygon geometries."
        )

    points = gdf.loc[geometry_id == shapely.GeometryType.POINT]
    lines = gdf.loc[geometry_id == shapely.GeometryType.LINESTRING]
    polygons = gdf.loc[geometry_id == shapely.GeometryType.POLYGON]

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
