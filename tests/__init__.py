# __init__.py for pytest-cov
import importlib

import pytest


def _importorskip(modname):
    try:
        mod = importlib.import_module(modname)
        has = True
        # Special case: meshkernel requires a runtime check beyond just importing
        if modname == "meshkernel":
            try:
                mod.MeshKernel()
            except OSError:
                has = False
    except ImportError:
        has = False
    func = pytest.mark.skipif(not has, reason=f"requires {modname}")
    return has, func


has_meshkernel, requires_meshkernel = _importorskip("meshkernel")
has_geopandas, requires_geopandas = _importorskip("geopandas")
has_shapely, requires_shapely = _importorskip("shapely")
has_pyproj, requires_pyproj = _importorskip("pyproj")
has_matplotlib, requires_matplotlib = _importorskip("matplotlib")
has_dask, requires_dask = _importorskip("dask")
has_zarr, requires_zarr = _importorskip("zarr")
has_numba_celltree, requires_numba_celltree = _importorskip("numba_celltree")
has_netCDF4, requires_netCDF4 = _importorskip("netCDF4")
has_pymetis, requires_pymetis = _importorskip("pymetis")
