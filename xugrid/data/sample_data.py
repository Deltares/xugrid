"""
Functions to load sample data.
"""
import pkg_resources

import numpy as np
import pooch

import xugrid

REGISTRY = pooch.create(
    path=pooch.os_cache("xugrid"),
    base_url="https://github.com/deltares/xugrid/raw/{version}/data",
    version=None,
    version_dev=None,
    env="XUGRID_DATA_DIR",
)
with pkg_resources.resource_stream("xugrid.data", "registry.txt") as registry_file:
    REGISTRY.load_registry(registry_file)


def xoxo():
    """
    Fetch a simple two part synthetic unstructured grid topology.
    """
    fname_vertices = REGISTRY.fetch("xoxo_vertices.txt")
    fname_triangles = REGISTRY.fetch("xoxo_triangles.txt")
    vertices = np.loadtxt(fname_vertices, dtype=float)
    triangles = np.loadtxt(fname_triangles, dtype=int)
    return vertices, triangles
