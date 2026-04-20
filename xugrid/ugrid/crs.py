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

    CRS resolution from grid mapping attributes prefers the CRS that
    round-trips to a valid EPSG code, and raises ``ValueError`` if multiple
    attributes (e.g. ESPG identifier vs. WKT) resolve to conflicting EPSG codes.

    Returns CrsPlaceholder if no valid CRS can be determined.
    Note that it looks like the from_cf method prefers the WKT
    form even if the name is given.
    """
    try:
        import pyproj
    except ImportError:
        return CrsPlaceholder(ds_attrs)

    candidates: dict[str, pyproj.CRS] = {}

    # 0. Lower keys just in case.
    attrs = {k.lower(): v for k, v in ds_attrs.items()}

    # 1. CF grid mapping attributes
    name = attrs.get("grid_mapping_name")
    if name is not None:
        # Note that from_cf will also check whether crs_wkt or spatial_ref is defined.
        try:
            candidates["grid_mapping"] = pyproj.CRS.from_cf(attrs)
        except pyproj.exceptions.CRSError:
            pass
    else:
        # 2. WKT
        wkt = attrs.get("crs_wkt") or attrs.get("spatial_ref")
        if wkt is not None:
            try:
                candidates["wkt"] = pyproj.CRS.from_wkt(wkt)
            except pyproj.exceptions.CRSError:
                pass

    # 3. EPSG fallback
    epsg_entry = attrs.get("epsg") or attrs.get("epsg_code")
    if epsg_entry is not None:
        try:
            candidates["epsg"] = pyproj.CRS.from_user_input(epsg_entry)
        except (ValueError, pyproj.exceptions.CRSError):
            pass

    if not candidates:
        return CrsPlaceholder(ds_attrs)

    # All candidates agree. Return the first.
    crses = list(candidates.values())
    first = crses[0]
    if all(first.equals(crs) for crs in crses[1:]):
        return first

    # Differing CRS found: try to resolve by EPSG identifier.
    epsg_ids = {
        label: epsg
        for label, crs in candidates.items()
        if (epsg := crs.to_epsg()) is not None
    }
    unique_epsg = set(epsg_ids.values())
    if len(unique_epsg) > 1:
        msg = "\n".join(f"- {label}: EPSG={epsg}" for label, epsg in epsg_ids.items())
        raise ValueError(f"Contradictory CRS information in attributes:\n{msg}")

    # EPSG codes agree (or only one resolved): prefer the EPSG-backed one.
    for label, crs in candidates.items():
        if label in epsg_ids:
            return crs

    return first


def crs_to_attrs(crs: pyproj.CRS | CrsPlaceholder) -> dict:
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
