from typing import List, Union

import numpy as np
import xarray as xr

import xugrid
from xugrid.constants import BoolArray, FloatArray, MissingOptionalModule

try:
    import shapely
except ImportError:
    shapely = MissingOptionalModule("shapely")

ShapelyArray = np.ndarray


def _locate_polygon(
    xy: FloatArray,
    exterior: FloatArray,
    interiors: List[FloatArray],
) -> BoolArray:
    import mapbox_earcut

    rings = np.cumsum([len(exterior)] + [len(interior) for interior in interiors])
    vertices = np.vstack([exterior] + interiors).astype(np.float64)
    triangles = mapbox_earcut.triangulate_float64(vertices, rings).reshape((-1, 3))
    grid = xugrid.Ugrid2d(*vertices.T, -1, triangles)
    return grid.locate_points(xy) != -1


def _burn_polygons(
    polygons: ShapelyArray,
    like: "xugrid.Ugrid2d",
    values: np.ndarray,
    all_touched: bool,
    output: FloatArray,
) -> None:
    """
    This algorithm burns polygon vector geometries in a 2d topology by:

    * Extracting the exterior and interiors (holes) coordinates from the
      polygons.
    * Breaking every polygon down into a triangles using an "earcut" algorithm.
    * Building a Ugrid2d from the triangles.

    When all_touched=False:

    * The grid created from the polygon is searched for every centroid of the
      ``like`` grid.

    When all_touched=True:

    * The grid created from the polygon is searched for every node of the
      ``like`` grid.
    * If any of the associated nodes of a face is found in the polygon, the
      value is burned into the face.

    """
    exterior_coordinates = [
        shapely.get_coordinates(exterior) for exterior in polygons.geometry.exterior
    ]
    interior_coordinates = [
        [shapely.get_coordinates(p_interior) for p_interior in p_interiors]
        for p_interiors in polygons.geometry.interiors
    ]

    if all_touched:
        # Pre-allocate work arrays so we don't have re-allocate for every
        # polygon.
        to_burn2d = np.empty_like(like.face_node_connectivity, dtype=bool)
        to_burn = np.empty(like.n_face, dtype=bool)
        # These data are static:
        mask = like.face_node_connectivity == like.fill_value
        xy = like.node_coordinates
        for exterior, interiors, value in zip(
            exterior_coordinates, interior_coordinates, values
        ):
            location = _locate_polygon(xy, exterior, interiors)
            # Equal to: to_burn2d = location[like.face_node_connectivity]
            location.take(like.face_node_connectivity, out=to_burn2d)
            to_burn2d[mask] = False
            # Equal to: to_burn = to_burn2d.any(axis=1)
            np.bitwise_or.reduce(to_burn2d, axis=1, out=to_burn)
            output[to_burn] = value

    else:
        xy = like.centroids
        for exterior, interiors, value in zip(
            exterior_coordinates, interior_coordinates, values
        ):
            to_burn = _locate_polygon(xy, exterior, interiors)
            output[to_burn] = value

    return


def _burn_points(
    points: ShapelyArray, like: "xugrid.Ugrid2d", values: np.ndarray, output: FloatArray
) -> None:
    """
    Simply searches the points in the ``like`` 2D topology.
    """
    xy = shapely.get_coordinates(points)
    to_burn = like.locate_points(xy)
    output[to_burn] = values
    return


def _burn_lines(
    lines: ShapelyArray, like: "xugrid.Ugrid2d", values: np.ndarray, output: FloatArray
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

    if column is not None:
        values = gdf["column"].to_numpy()
    else:
        values = np.ones(len(gdf), dtype=float)

    output = np.full(like.n_face, fill)
    if len(points) > 0:
        _burn_points(points.geometry.to_numpy(), like, values, output)
    if len(lines) > 0:
        _burn_lines(lines.geometry.to_numpy(), like, values, output)
    if len(polygons) > 0:
        _burn_polygons(polygons.geometry.to_numpy(), like, values, all_touched, output)

    return xugrid.UgridDataArray(
        obj=xr.DataArray(output, dims=[like.face_dimension], name=column),
        grid=like,
    )
