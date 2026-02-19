import pyproj

from xugrid.ugrid.crs import CrsPlaceholder, crs_from_attrs, crs_to_attrs


class TestCrsPlaceholder:
    def test_stores_attrs(self):
        attrs = {"epsg": 28992, "grid_mapping_name": "Unknown projected"}
        placeholder = CrsPlaceholder(attrs)
        assert placeholder._attrs == attrs
        # Check that it doesn't use the reference.
        attrs["epsg"] = 4326
        assert placeholder._attrs["epsg"] == 28992

    def test_eq_same_attrs(self):
        assert CrsPlaceholder({"epsg": 28992}) == CrsPlaceholder({"epsg": 28992})
        assert CrsPlaceholder({"epsg": 28992}) != CrsPlaceholder({"epsg": 4326})

    def test_repr(self):
        placeholder = CrsPlaceholder({"epsg": 28992})
        assert repr(placeholder) == "CrsPlaceholder({'epsg': 28992})"


class TestCrsFromAttrs:
    """Test the priority chain: CF attrs -> WKT -> EPSG -> placeholder."""

    # CF grid mapping attributes (priority 1)

    def test_name_only(self):
        # Name only: should still work according to CF.
        crs = crs_from_attrs({"grid_mapping_name": "latitude_longitude"})
        assert isinstance(crs, pyproj.CRS)
        assert crs.name == "undefined"

        attrs = {
            "geographic_crs_name": "WGS 84",
            "grid_mapping_name": "latitude_longitude",
        }
        crs = crs_from_attrs(attrs)
        assert isinstance(crs, pyproj.CRS)
        assert crs.name == "WGS 84"

        # British National Grid has a CF grid_mapping_name
        attrs = pyproj.CRS.from_epsg(27700).to_cf()
        attrs.pop("crs_wkt")
        crs = crs_from_attrs(attrs)
        assert isinstance(crs, pyproj.CRS)
        assert crs.to_epsg() == 27700

        # Nonsense name
        attrs = {"grid_mapping_name": "totally_invalid_projection"}
        result = crs_from_attrs(attrs)
        assert isinstance(result, CrsPlaceholder)

    # WKT (priority 2)

    def test_from_crs_wkt(self):
        wkt = pyproj.CRS.from_epsg(28992).to_wkt()
        crs = crs_from_attrs({"crs_wkt": wkt})
        assert isinstance(crs, pyproj.CRS)
        assert crs.to_epsg() == 28992

        crs = crs_from_attrs({"spatial_ref": wkt})
        assert isinstance(crs, pyproj.CRS)
        assert crs.to_epsg() == 28992

        # crs_wkt preferred over spatial_ref(
        wkt_28992 = pyproj.CRS.from_epsg(28992).to_wkt()
        wkt_4326 = pyproj.CRS.from_epsg(4326).to_wkt()
        crs = crs_from_attrs({"crs_wkt": wkt_28992, "spatial_ref": wkt_4326})
        assert crs.to_epsg() == 28992

        # Nonsense wkt
        attrs = {"crs_wkt": "not valid wkt at all"}
        result = crs_from_attrs(attrs)
        assert isinstance(result, CrsPlaceholder)

    # EPSG fallback (priority 3)

    def test_from_epsg(self):
        crs = crs_from_attrs({"epsg": 28992})
        assert isinstance(crs, pyproj.CRS)
        assert crs.to_epsg() == 28992

        crs = crs_from_attrs({"epsg": "EPSG:28992"})
        assert isinstance(crs, pyproj.CRS)
        assert crs.to_epsg() == 28992

        crs = crs_from_attrs({"epsg_code": 4326})
        assert isinstance(crs, pyproj.CRS)
        assert crs.to_epsg() == 4326

        # Invalid
        attrs = {"epsg": -9999}
        result = crs_from_attrs(attrs)
        assert isinstance(result, CrsPlaceholder)

    # Misc. tests

    def test_case_sensitivity(self):
        wkt = pyproj.CRS.from_epsg(28992).to_wkt()
        crs = crs_from_attrs({"CRS_WKT": wkt})
        assert isinstance(crs, pyproj.CRS)
        assert crs.to_epsg() == 28992

        crs = crs_from_attrs({"EPSG": 28992})
        assert isinstance(crs, pyproj.CRS)
        assert crs.to_epsg() == 28992

    def test_bad_attrs(self):
        assert isinstance(crs_from_attrs({}), CrsPlaceholder)
        result = crs_from_attrs({"foo": "bar", "baz": 42})
        assert isinstance(result, CrsPlaceholder)

    def test_DFM_case(self):
        # DFM case, name is "unknown", no WKT, but ESPG is present.
        attrs = {
            "grid_mapping_name": "Unknown projected",
            "epsg": 28992,
            "EPSG_code": "EPSG:28992",
            "semi_major_axis": 6378137.0,
            "semi_minor_axis": 6356752.314245,
            "inverse_flattening": 298.257223563,
        }
        crs = crs_from_attrs(attrs)
        assert isinstance(crs, pyproj.CRS)
        assert crs.to_epsg() == 28992


class TestCrsToAttrs:
    def test_roundtrip(self):
        crs = pyproj.CRS.from_epsg(4326)
        attrs = crs_to_attrs(crs)
        assert "crs_wkt" in attrs
        assert "spatial_ref" in attrs
        assert attrs["name"] == "WGS 84"
        assert attrs["crs_wkt"] == attrs["spatial_ref"]
        assert attrs["epsg"] == 4326
        assert "grid_mapping_name" in attrs
        # And back to crs
        assert crs_from_attrs(attrs) == crs

        crs = pyproj.CRS.from_epsg(28992)
        attrs = crs_to_attrs(crs)
        assert "crs_wkt" in attrs
        assert "spatial_ref" in attrs
        assert attrs["name"] == "Amersfoort / RD New"
        assert attrs["epsg"] == 28992
        # Oblique stereographic has no CF grid_mapping_name
        assert "grid_mapping_name" not in attrs
        assert crs_from_attrs(attrs) == crs

    def test_no_epsg(self):
        crs = crs_from_attrs({"grid_mapping_name": "latitude_longitude"})
        attrs = crs_to_attrs(crs)
        assert "epsg" not in attrs
        assert "crs_wkt" in attrs

    def test_placeholder_roundtrip(self):
        # Nonsense name should be preserved in roundtrip.
        original = {"grid_mapping_name": "totally_invalid_projection"}
        placeholder = crs_from_attrs(original)
        back = crs_to_attrs(placeholder)
        assert back == original

        # Now with a valid attrs (e.g. when pyproj isn't available)
        crs = pyproj.CRS.from_epsg(28992)
        original = crs_to_attrs(crs)
        placeholder = CrsPlaceholder(original)
        back = crs_to_attrs(placeholder)
        assert back == original
