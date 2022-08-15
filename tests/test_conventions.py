from collections import ChainMap

import pytest
import xarray as xr

import xugrid
from xugrid.ugrid import conventions as cv


def test_infer_xy_coords():
    def assign_node_coords(ds: xr.Dataset, name: str, stdname: str = None):
        if stdname:
            attrs = {"standard_name": stdname}
        else:
            attrs = None

        da = xr.DataArray([0.0, 1.0], dims=["node"], attrs=attrs)
        return ds.assign_coords({name: da})

    # All have standard names: take all
    ds = xr.Dataset()
    ds = assign_node_coords(ds, "x", "projection_x_coordinate")
    ds = assign_node_coords(ds, "y", "projection_y_coordinate")
    ds = assign_node_coords(ds, "lon", "longitude")
    ds = assign_node_coords(ds, "lat", "latitude")
    candidates = ["x", "y", "lon", "lat"]
    x, y = cv._infer_xy_coords(ds, candidates)
    assert x == ["x", "lon"]
    assert y == ["y", "lat"]

    # Only lon and lat have standard names: take them.
    ds = xr.Dataset()
    ds = assign_node_coords(ds, "x")
    ds = assign_node_coords(ds, "y")
    ds = assign_node_coords(ds, "lon", "longitude")
    ds = assign_node_coords(ds, "lat", "latitude")
    x, y = cv._infer_xy_coords(ds, candidates)
    assert x == ["lon"]
    assert y == ["lat"]

    # Non have standard names: take the first two.
    ds = xr.Dataset()
    ds = assign_node_coords(ds, "x")
    ds = assign_node_coords(ds, "y")
    ds = assign_node_coords(ds, "lon")
    ds = assign_node_coords(ds, "lat")
    candidates = ["x", "y", "lon", "lat"]
    with pytest.warns(UserWarning):
        x, y = cv._infer_xy_coords(ds, candidates)
    assert x == ["x"]
    assert y == ["y"]

    # Only one: error
    ds = xr.Dataset()
    ds = assign_node_coords(ds, "x", "projection_x_coordinate")
    ds = assign_node_coords(ds, "y")
    candidates = ["x", "y"]
    with pytest.raises(cv.UgridCoordinateError):
        x, y = cv._infer_xy_coords(ds, candidates)

    # Only one: error
    ds = xr.Dataset()
    ds = assign_node_coords(ds, "x")
    ds = assign_node_coords(ds, "y", "projection_y_coordinate")
    candidates = ["x", "y"]
    with pytest.raises(cv.UgridCoordinateError):
        x, y = cv._infer_xy_coords(ds, candidates)


class TestConventionsElevation:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = xugrid.data.elevation_nl(xarray=True)
        self.coordinates = {
            "mesh2d": {
                "node_coordinates": (
                    ["mesh2d_node_x"],
                    ["mesh2d_node_y"],
                ),
                "face_coordinates": (
                    ["mesh2d_face_x"],
                    ["mesh2d_face_y"],
                ),
            }
        }
        self.connectivity = {
            "mesh2d": {
                "face_node_connectivity": "mesh2d_face_nodes",
            }
        }
        self.dimensions = {
            "mesh2d": {
                "edge_dimension": "mesh2d_nEdges",
                "face_dimension": "mesh2d_nFaces",
                "node_dimension": "mesh2d_nNodes",
            },
        }

    def test_get_topology(self):
        assert cv._get_topology(self.ds) == ["mesh2d"]

    def test_get_coordinates(self):
        ds = xugrid.data.elevation_nl(xarray=True)
        actual = cv._get_coordinates(ds, ["mesh2d"])
        assert actual == self.coordinates

    def test_get_connectivity(self):
        ds = xugrid.data.elevation_nl(xarray=True)
        actual = cv._get_connectivity(ds, ["mesh2d"])
        assert actual == self.connectivity

    def test_get_dimensions(self):
        ds = xugrid.data.elevation_nl(xarray=True)
        connectivity = cv._get_connectivity(ds, ["mesh2d"])
        coordinates = cv._get_coordinates(ds, ["mesh2d"])
        actual = cv._get_dimensions(ds, ["mesh2d"], connectivity, coordinates)
        assert actual == self.dimensions

    def test_topology(self):
        assert self.ds.ugrid_roles.topology == ["mesh2d"]

    def test_coordinates(self):
        assert self.ds.ugrid_roles.coordinates == self.coordinates

    def test_dimensions(self):
        assert self.ds.ugrid_roles.dimensions == self.dimensions

    def test_connectivity(self):
        assert self.ds.ugrid_roles.connectivity == self.connectivity

    def test_getitem(self):
        result = self.ds.ugrid_roles["mesh2d"]
        assert isinstance(result, ChainMap)

        with pytest.raises(KeyError):
            self.ds.ugrid_roles["mesh1d"]

        assert self.ds.ugrid_roles["mesh2d"]["node_coordinates"] == (
            ["mesh2d_node_x"],
            ["mesh2d_node_y"],
        )

    def test_repr(self):
        result = self.ds.ugrid_roles.__repr__()
        assert isinstance(result, str)
