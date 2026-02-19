from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pyproj


class CrsPlaceholder:
    """Stands in for pyproj.CRS when pyproj is not installed."""

    def __init__(self, attrs: dict):
        self._attrs = dict(attrs)

    def __eq__(self, other):
        if isinstance(other, CrsPlaceholder):
            return self._attrs == other._attrs
        return False

    def __repr__(self):
        return f"CrsPlaceholder({self._attrs})"


def crs_from_attrs(ds_attrs: dict) -> pyproj.CRS | CrsPlaceholder:
    """
    Attempt to construct a pyproj.CRS from grid mapping attributes.

    Tries in order:
    1. CF grid mapping attributes
    2. WKT (crs_wkt or spatial_ref)
    3. EPSG code

    Returns CrsPlaceholder if no valid CRS can be determined.
    Note that it looks like the from_cf method prefers the WKT
    form even if the name is given.
    """
    try:
        import pyproj
    except ImportError:
        return CrsPlaceholder(ds_attrs)

    # 0. Lower keys just in case.
    attrs = {k.lower(): v for k, v in ds_attrs.items()}

    # 1. CF grid mapping attributes
    name = attrs.get("grid_mapping_name")
    if name is not None:
        try:
            return pyproj.CRS.from_cf(attrs)
        except pyproj.exceptions.CRSError:
            pass

    # 2. WKT
    wkt = attrs.get("crs_wkt") or attrs.get("spatial_ref")
    if wkt is not None:
        try:
            return pyproj.CRS.from_wkt(wkt)
        except pyproj.exceptions.CRSError:
            pass

    # 3. EPSG fallback
    epsg_entry = attrs.get("epsg") or attrs.get("epsg_code")
    if epsg_entry is not None:
        try:
            return pyproj.CRS.from_user_input(epsg_entry)
        except (ValueError, pyproj.exceptions.CRSError):
            pass

    # 4. pyproj couldn't make sense of it. Return a placeholder instead.
    return CrsPlaceholder(ds_attrs)


def crs_to_attrs(crs: pyproj.CRS) -> dict:
    if isinstance(crs, CrsPlaceholder):
        return crs._attrs

    attrs = crs.to_cf()  # already includes crs_wkt and CF attrs when possible.
    # GDAL compat: GDAL uses spatial_ref also.
    attrs["spatial_ref"] = attrs["crs_wkt"]
    attrs["name"] = crs.name
    epsg = crs.to_epsg()
    if epsg is not None:
        attrs["epsg"] = epsg
    return attrs
