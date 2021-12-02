import geopandas as gpd

from .ugrid1d import Ugrid1d
from .ugrid2d import Ugrid2d


def grid_from_geodataframe(geodataframe):
    gdf = geodataframe
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError(f"Cannot convert a {type(gdf)}, expected a GeoDataFrame")

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
