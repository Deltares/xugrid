import re

import numpy as np
import pytest
import xarray as xr

from xugrid.ugrid import ugrid_io


def test_get_attributes():
    d = ugrid_io.UGrid.network1d_get_attributes("network1d")
    assert isinstance(d, dict)

    d = ugrid_io.UGrid.mesh2d_get_attributes("network1d")
    assert isinstance(d, dict)


def test_get_topology_variable():
    ds = xr.Dataset()
    with pytest.raises(ValueError, match="dataset does not contain a mesh"):
        ugrid_io.get_topology_variable(ds)

    ds["mesh"] = xr.DataArray(0, attrs={"cf_role": "mesh_topology"})
    assert ugrid_io.get_topology_variable(ds) == ds["mesh"]

    ds["mesh2"] = xr.DataArray(0, attrs={"cf_role": "mesh_topology"})
    with pytest.raises(
        NotImplementedError,
        match=re.escape(
            "dataset should contain a single mesh topology variable, "
            "contains: ['mesh', 'mesh2']"
        ),
    ):
        ugrid_io.get_topology_variable(ds)


def test_check_dim():
    attrs = {"node_dimension": "node"}
    dimrole = "node_dimension"
    dimname = "node"
    varname = "mesh"

    assert ugrid_io.check_dim(attrs, dimrole, dimname, varname)
    assert ugrid_io.check_dim({}, dimrole, dimname, varname)

    with pytest.warns(UserWarning):
        ok = ugrid_io.check_dim(attrs, dimrole, "n_nodes", varname)
    assert not ok


def test_cast():
    da = xr.DataArray([1.0, 2.0, np.nan])
    actual = ugrid_io.cast(da, fill_value=-1, dtype=int)
    assert np.issubdtype(actual.dtype, np.integer)
    assert np.array_equal(actual.values, [1, 2, -1])
    # Check whether it's a copy, not a view
    assert da.values is not actual.values

    actual = ugrid_io.cast(da, fill_value=-1, dtype=float)
    assert np.issubdtype(actual.dtype, np.floating)
    assert np.array_equal(actual.values, [1.0, 2.0, -1.0])


def test_ugrid1d_dataset__default_attrs():
    for name in [None, "my-network"]:
        ds = ugrid_io.ugrid1d_dataset(
            node_x=np.array([0.0, 1.0]),
            node_y=np.array([0.0, 1.0]),
            fill_value=-1,
            edge_node_connectivity=np.array([[0, 1]]),
            name=name,
            attrs=None,
        )

        if name is None:
            name = ugrid_io.UGRID1D_DEFAULT_NAME

        assert isinstance(ds, xr.Dataset)
        assert f"{name}" in ds
        assert f"{name}_nNodes" in ds.dims
        assert f"{name}_nEdges" in ds.dims
        assert f"{name}_node_x" in ds.coords
        assert f"{name}_node_y" in ds.coords
        assert f"{name}_edge_nodes" in ds
        attrs = ds[f"{name}_edge_nodes"].attrs
        assert attrs["cf_role"] == "edge_node_connectivity"
        assert attrs["start_index"] == 0
        assert attrs["_FillValue"] == -1

        variables = ugrid_io.get_ugrid1d_variables(ds, ds[name])
        assert isinstance(variables, dict)
        assert variables[f"{name}_edge_nodes"] == "edge_node_connectivity"
        assert variables[f"{name}_node_x"] == "node_x"
        assert variables[f"{name}_node_y"] == "node_y"


def test_ugrid1d_dataset():
    # Test arbitrary names
    attrs = {
        "cf_role": "mesh_topology",
        "long_name": "Topology data of 1D network",
        "topology_dimension": 1,
        "node_dimension": "vertex",
        "node_coordinates": "vertex_x vertex_y",
        "edge_dimension": "branch",
        "edge_node_connectivity": "branch_vertices",
    }
    ds = ugrid_io.ugrid1d_dataset(
        node_x=np.array([0.0, 1.0]),
        node_y=np.array([0.0, 1.0]),
        fill_value=-1,
        edge_node_connectivity=np.array([[0, 1]]),
        name="rivernetwork",
        attrs=attrs,
    )
    assert isinstance(ds, xr.Dataset)
    assert "rivernetwork" in ds
    assert "vertex" in ds.dims
    assert "branch" in ds.dims
    assert "vertex_x" in ds.coords
    assert "vertex_y" in ds.coords
    assert "branch_vertices" in ds
    attrs = ds["branch_vertices"].attrs
    assert attrs["cf_role"] == "edge_node_connectivity"
    assert attrs["start_index"] == 0
    assert attrs["_FillValue"] == -1

    variables = ugrid_io.get_ugrid1d_variables(ds, ds["rivernetwork"])
    assert isinstance(variables, dict)
    assert variables["branch_vertices"] == "edge_node_connectivity"
    assert variables["vertex_x"] == "node_x"
    assert variables["vertex_y"] == "node_y"


def test_ugrid2d_dataset__default_attrs():
    for name in [None, "my-mesh"]:
        kwargs = dict(
            node_x=np.array([0.0, 1.0, 0.0]),
            node_y=np.array([0.0, 0.0, 1.0]),
            fill_value=-1,
            face_node_connectivity=np.array([[0, 1, 2]]),
            name=name,
            attrs=None,
        )
        ds = ugrid_io.ugrid2d_dataset(**kwargs)

        if name is None:
            name = ugrid_io.UGRID2D_DEFAULT_NAME

        assert isinstance(ds, xr.Dataset)
        assert f"{name}" in ds
        assert f"{name}_nNodes" in ds.dims
        assert f"{name}_nFaces" in ds.dims
        assert f"{name}_nMax_face_nodes" in ds.dims
        assert f"{name}_node_x" in ds.coords
        assert f"{name}_node_y" in ds.coords
        assert f"{name}_face_nodes" in ds
        attrs = ds[f"{name}_face_nodes"].attrs
        assert attrs["cf_role"] == "face_node_connectivity"
        assert attrs["start_index"] == 0
        assert attrs["_FillValue"] == -1

        variables = ugrid_io.get_ugrid2d_variables(ds, ds[name])
        assert isinstance(variables, dict)
        assert variables[f"{name}_face_nodes"] == "face_node_connectivity"
        assert variables[f"{name}_node_x"] == "node_x"
        assert variables[f"{name}_node_y"] == "node_y"

        kwargs["edge_node_connectivity"] = np.array([[0, 1], [1, 2], [2, 0]])
        ds = ugrid_io.ugrid2d_dataset(**kwargs)
        assert f"{name}_nEdges" in ds.dims
        assert f"{name}_edge_nodes" in ds
        attrs = ds[f"{name}_edge_nodes"].attrs
        assert attrs["cf_role"] == "edge_node_connectivity"
        assert attrs["start_index"] == 0
        assert attrs["_FillValue"] == -1


def test_ugrid2d_dataset():
    # Test arbitrary names
    attrs = {
        "cf_role": "mesh_topology",
        "long_name": "Topology data of 2D mesh",
        "topology_dimension": 2,
        "node_dimension": "vertex",
        "node_coordinates": "vertex_x vertex_y",
        "edge_dimension": "branch",
        "edge_node_connectivity": "branch_vertices",
        "face_dimension": "cell",
        "face_node_connectivity": "cell_vertices",
        "max_face_nodes_dimension": "node_nmax",
    }
    ds = ugrid_io.ugrid2d_dataset(
        node_x=np.array([0.0, 1.0, 0.0]),
        node_y=np.array([0.0, 0.0, 1.0]),
        fill_value=-1,
        face_node_connectivity=np.array([[0, 1, 2]]),
        edge_node_connectivity=np.array([[0, 1], [1, 2], [2, 0]]),
        name="unstructuredmesh",
        attrs=attrs,
    )
    assert isinstance(ds, xr.Dataset)
    assert "unstructuredmesh" in ds
    assert "vertex" in ds.dims
    assert "branch" in ds.dims
    assert "cell" in ds.dims
    assert "vertex_x" in ds.coords
    assert "vertex_y" in ds.coords
    assert "branch_vertices" in ds
    assert "cell_vertices" in ds
    attrs = ds["branch_vertices"].attrs
    assert attrs["cf_role"] == "edge_node_connectivity"
    assert attrs["start_index"] == 0
    assert attrs["_FillValue"] == -1
    attrs = ds["cell_vertices"].attrs
    assert attrs["cf_role"] == "face_node_connectivity"
    assert attrs["start_index"] == 0
    assert attrs["_FillValue"] == -1

    variables = ugrid_io.get_ugrid2d_variables(ds, ds["unstructuredmesh"])
    assert isinstance(variables, dict)
    assert variables["branch_vertices"] == "edge_node_connectivity"
    assert variables["cell_vertices"] == "face_node_connectivity"
    assert variables["vertex_x"] == "node_x"
    assert variables["vertex_y"] == "node_y"
