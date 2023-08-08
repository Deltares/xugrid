from collections import ChainMap

import numpy as np
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

        ds = xugrid.data.elevation_nl(xarray=True)
        ds["mesh2d"].attrs["edge_coordinates"] = "mesh2d_edge_x mesh2d_edge_y"
        with pytest.warns(UserWarning):
            cv._get_coordinates(ds, ["mesh2d"])

        ds = xugrid.data.elevation_nl(xarray=True)
        ds["mesh2d"].attrs["edge_coordinates"] = "mesh2d_edge_x"
        ds["mesh2d_edge_x"] = 0  # Put a dummy value in the dataset
        with pytest.raises(cv.UgridCoordinateError):
            cv._get_coordinates(ds, ["mesh2d"])

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


class TestCompleteSpecification:
    """
    This test contains all attributes and variables.

    It is based on some D-Flow output dataset with data on nodes, edges, faces.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        ds = xr.Dataset()
        ds["mesh2d"] = xr.DataArray(
            0,
            attrs={
                "cf_role": "mesh_topology",
                "long_name": "Topology data of 2D mesh",
                "topology_dimension": 2,
                "node_coordinates": "mesh2d_node_x mesh2d_node_y",
                "node_dimension": "mesh2d_nNodes",
                "max_face_nodes_dimension": "mesh2d_nMax_face_nodes",
                "edge_node_connectivity": "mesh2d_edge_nodes",
                "edge_dimension": "mesh2d_nEdges",
                "edge_coordinates": "mesh2d_edge_x mesh2d_edge_y",
                "face_node_connectivity": "mesh2d_face_nodes",
                "face_dimension": "mesh2d_nFaces",
                "edge_face_connectivity": "mesh2d_edge_faces",
                "face_coordinates": "mesh2d_face_x mesh2d_face_y",
            },
        )
        xy = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ]
        )
        xy_edge = np.array(
            [
                [0.5, 0.0],
                [1.0, 0.5],
                [0.5, 1.0],
                [0.0, 0.5],
            ]
        )
        x_attrs = {"standard_name": "projection_x_coordinate"}
        y_attrs = {"standard_name": "projection_y_coordinate"}
        ds = ds.assign_coords(
            mesh2d_node_x=xr.DataArray(xy[:, 0], dims=["mesh2d_nNodes"], attrs=x_attrs)
        )
        ds = ds.assign_coords(
            mesh2d_node_y=xr.DataArray(xy[:, 1], dims=["mesh2d_nNodes"], attrs=y_attrs)
        )
        ds = ds.assign_coords(
            mesh2d_edge_x=xr.DataArray(
                xy_edge[:, 0], dims=["mesh2d_nEdges"], attrs=x_attrs
            )
        )
        ds = ds.assign_coords(
            mesh2d_edge_y=xr.DataArray(
                xy_edge[:, 1], dims=["mesh2d_nEdges"], attrs=y_attrs
            )
        )
        ds = ds.assign_coords(
            mesh2d_face_x=xr.DataArray([0.5], dims=["mesh2d_nFaces"], attrs=x_attrs)
        )
        ds = ds.assign_coords(
            mesh2d_face_y=xr.DataArray([0.5], dims=["mesh2d_nFaces"], attrs=y_attrs)
        )
        ds["mesh2d_face_nodes"] = xr.DataArray(
            data=[[0, 1, 2, 3]],
            dims=["mesh2d_nFaces", "mesh2d_nMax_face_nodes"],
            attrs={"_FillValue": -1, "start_index": 0},
        )
        ds["mesh2d_edge_nodes"] = xr.DataArray(
            data=[
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],
            ],
            dims=["mesh2d_nEdges", "Two"],
            attrs={"_FillValue": -1, "start_index": 0},
        )
        ds["mesh2d_edge_faces"] = xr.DataArray(
            data=[
                [0, -1],
                [0, -1],
                [0, -1],
                [0, -1],
            ],
            dims=["mesh2d_nEdges", "Two"],
            attrs={"_FillValue": -1, "start_index": 0},
        )
        self.ds = ds

        self.coordinates = {
            "mesh2d": {
                "node_coordinates": (
                    ["mesh2d_node_x"],
                    ["mesh2d_node_y"],
                ),
                "edge_coordinates": (
                    ["mesh2d_edge_x"],
                    ["mesh2d_edge_y"],
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
                "edge_node_connectivity": "mesh2d_edge_nodes",
                "edge_face_connectivity": "mesh2d_edge_faces",
            }
        }
        self.dimensions = {
            "mesh2d": {
                "edge_dimension": "mesh2d_nEdges",
                "face_dimension": "mesh2d_nFaces",
                "node_dimension": "mesh2d_nNodes",
            },
        }

    def test_topology(self):
        assert self.ds.ugrid_roles.topology == ["mesh2d"]

    def test_coordinates(self):
        assert self.ds.ugrid_roles.coordinates == self.coordinates

    def test_dimensions(self):
        assert self.ds.ugrid_roles.dimensions == self.dimensions

    def test_connectivity(self):
        assert self.ds.ugrid_roles.connectivity == self.connectivity

    def test_dimension_name_mismatch_error(self):
        ds = self.ds.copy()

        ds["mesh2d_edge_nodes"] = xr.DataArray(
            data=[
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],
            ],
            dims=["nEdges", "Two"],
            attrs={"_FillValue": -1, "start_index": 0},
        )

        with pytest.raises(
            cv.UgridDimensionError,
            match="edge_dimension: nEdges not in edge_face_connectivity",
        ):
            ds.ugrid_roles.dimensions

    def test_dimension_size_error(self):
        ds = self.ds.copy()

        ds["mesh2d_edge_nodes"] = xr.DataArray(
            data=[
                [0, 1, -1],
                [1, 2, -1],
                [2, 3, -1],
                [3, 0, -1],
            ],
            dims=["mesh2d_nEdges", "Three"],
            attrs={"_FillValue": -1, "start_index": 0},
        )

        with pytest.raises(cv.UgridDimensionError, match="Expected size 2"):
            ds.ugrid_roles.dimensions
