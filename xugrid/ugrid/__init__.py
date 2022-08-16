import xarray as xr

from .conventions import UgridRolesAccessor
from .ugrid1d import Ugrid1d
from .ugrid2d import Ugrid2d
from .ugridbase import AbstractUgrid


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
