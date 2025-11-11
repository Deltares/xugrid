import importlib

import numpy as np
import pooch
import xarray as xr

import xugrid

REGISTRY = pooch.create(
    path=pooch.os_cache("xugrid"),
    base_url="https://github.com/deltares/xugrid/raw/main/data/",
    version=None,
    version_dev="main",
    env="XUGRID_DATA_DIR",
)
REGISTRY.load_registry(importlib.resources.files("xugrid.data") / "registry.txt")


def xoxo():
    """Fetch a simple two part synthetic unstructured grid topology."""
    fname_vertices = REGISTRY.fetch("xoxo_vertices.txt")
    fname_triangles = REGISTRY.fetch("xoxo_triangles.txt")
    vertices = np.loadtxt(fname_vertices, dtype=float)
    triangles = np.loadtxt(fname_triangles, dtype=int)
    grid = xugrid.Ugrid2d(
        node_x=vertices[:, 0],
        node_y=vertices[:, 1],
        fill_value=-1,
        face_node_connectivity=triangles,
    )
    return grid


def adh_san_diego(xarray=False):
    """Fetch time varying output of a hydraulic simulation."""
    fname = REGISTRY.fetch("ADH_SanDiego.nc")
    ds = xr.open_dataset(fname)
    ds["node_x"].attrs["standard_name"] = "projection_x_coordinate"
    ds["node_y"].attrs["standard_name"] = "projection_y_coordinate"
    if xarray:
        return ds
    else:
        grid = xugrid.Ugrid2d.from_dataset(ds)
        return xugrid.UgridDataset(ds, grid)


def elevation_nl(xarray=False):
    """Fetch surface elevation dataset for the Netherlands."""
    fname = REGISTRY.fetch("elevation_nl.nc")
    ds = xr.open_dataset(fname)
    ds["mesh2d_node_x"].attrs["standard_name"] = "projection_x_coordinate"
    ds["mesh2d_node_y"].attrs["standard_name"] = "projection_y_coordinate"
    ds["mesh2d_face_x"].attrs["standard_name"] = "projection_x_coordinate"
    ds["mesh2d_face_y"].attrs["standard_name"] = "projection_y_coordinate"
    if xarray:
        return ds
    else:
        grid = xugrid.Ugrid2d.from_dataset(ds)
        return xugrid.UgridDataArray(ds["elevation"], grid)


def provinces_nl():
    """Fetch provinces polygons for the Netherlands."""
    import geopandas as gpd

    fname = REGISTRY.fetch("provinces-nl.geojson")
    gdf = gpd.read_file(fname)
    return gdf


def hydamo_network():
    """Fetch some surface water data for the Netherlands."""
    import geopandas as gpd
    import pandas as pd

    def read_wkt_csv(path):
        df = pd.read_csv(path)
        return gpd.GeoDataFrame(
            data=df,
            geometry=gpd.GeoSeries.from_wkt(df["geometry"]),
        )

    objects_fname = REGISTRY.fetch("hydamo_objects.csv")
    points_fname = REGISTRY.fetch("hydamo_points.csv")
    profiles_fname = REGISTRY.fetch("hydamo_profiles.csv")

    return (
        read_wkt_csv(objects_fname),
        read_wkt_csv(points_fname),
        read_wkt_csv(profiles_fname),
    )
