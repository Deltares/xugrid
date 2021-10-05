"""
Helper functions for parsing and composing UGRID files

Most of these functions will be replaced by a centralized UGRID library with a
C API.
"""
import warnings
from typing import List

import xarray as xr

from .typing import FloatArray, IntArray


def get_topology_array_with_role(
    mesh_variable: xr.DataArray, role: str, variables: List[str]
):
    """
    returns the names of the arrays that have the specified role on the
    Mesh topology variable
    """
    topology_array_names = []
    for variable_name in mesh_variable.attrs:
        if variable_name == role:
            split_names = str(mesh_variable.attrs[variable_name]).split()
            topology_array_names.extend(split_names)
    filtered = []
    for name in topology_array_names:
        if name in variables:
            filtered.append(name)
        else:
            warnings.warn(
                f"Topology variable with role {role} specified under name "
                f"{name} specified, but variable {name} not found in dataset.",
                UserWarning,
            )

    return filtered


# returns values of dataset which have a given attribute value
def get_variable_with_attribute(dataset, attribute_name, attribute_value):
    result = []
    for da in dataset.values():
        if da.attrs.get(attribute_name) == attribute_value:
            result.append(da)
    return result


# return those data arrays whose name appears in nameList
def get_data_arrays_by_name(dataset, name_list):
    result = []
    for da in dataset.values():
        if da.name in name_list:
            result.append(da)
    return result


# return those coordinate arrays whose name appears in nameList
def get_coordinate_arrays_by_name(dataset, name_list):
    result = []
    for da in dataset.coords:
        if da in name_list:
            result.append(dataset.coords[da])
    return result


def ugrid1d_dataset(
    node_x: FloatArray,
    node_y: FloatArray,
    fill_value: int,
    edge_node_connectivity: IntArray,
    node_dimension: str = "node",
    edge_dimension: str = "edge",
) -> xr.Dataset:
    ds = xr.Dataset()
    ds["mesh1d"] = xr.DataArray(
        data=0,
        attrs={
            "cf_role": "mesh_topology",
            "long_name": "Topology data of 1D mesh",
            "topology_dimension": 1,
            "node_coordinates": "node_x node_y",
            "edge_node_connectivity": "edge_node_connectivity",
            "node_dimension": node_dimension,
            "edge_dimension": edge_dimension,
        },
    )
    ds = ds.assign_coords(
        node_x=xr.DataArray(
            data=node_x,
            dims=[node_dimension],
        )
    )
    ds = ds.assign_coords(
        node_y=xr.DataArray(
            data=node_y,
            dims=[node_dimension],
        )
    )
    ds["edge_node_connectivity"] = xr.DataArray(
        data=edge_node_connectivity,
        dims=[edge_dimension, "two"],
        attrs={
            "cf_role": "edge_node_connectivity",
            "long_name": "Vertex nodes of edges",
            "start_index": 0,
            "_FillValue": fill_value,
        },
    )
    ds.attrs = {"Conventions": "CF-1.8 UGRID-1.0"}
    return ds


def ugrid2d_dataset(
    node_x: FloatArray,
    node_y: FloatArray,
    fill_value: int,
    face_node_connectivity: IntArray,
    edge_node_connectivity: IntArray = None,
    node_dimension: str = "node",
    face_dimension: str = "face",
    edge_dimension: str = "edge",
) -> xr.Dataset:
    # TODO: parametrize dataset variable names (node, node_x, node_y, node, etc.)
    # mesh2d variable could just be deep-copied in case of subset
    ds = xr.Dataset()
    ds["mesh2d"] = xr.DataArray(
        data=0,
        attrs={
            "cf_role": "mesh_topology",
            "long_name": "Topology data of 2D mesh",
            "topology_dimension": 2,
            "node_coordinates": "node_x node_y",
            "face_node_connectivity": "face_node_connectivity",
            "edge_node_connectivity": "edge_node_connectivity",
        },
    )
    ds = ds.assign_coords(
        node_x=xr.DataArray(
            data=node_x,
            dims=[node_dimension],
        )
    )
    ds = ds.assign_coords(
        node_y=xr.DataArray(
            data=node_y,
            dims=[node_dimension],
        )
    )
    ds["face_node_connectivity"] = xr.DataArray(
        data=face_node_connectivity,
        dims=[face_dimension, "nmax_face"],
        attrs={
            "cf_role": "face_node_connectivity",
            "long_name": "Vertex nodes of mesh faces (counterclockwise)",
            "start_index": 0,
            "_FillValue": fill_value,
        },
    )
    if edge_node_connectivity is not None:
        ds["edge_node_connectivity"] = xr.DataArray(
            data=face_node_connectivity,
            dims=[edge_dimension, "two"],
            attrs={
                "cf_role": "edge_node_connectivity",
                "long_name": "Vertex nodes of edges",
                "start_index": 0,
                "_FillValue": fill_value,
            },
        )

    ds.attrs = {"Conventions": "CF-1.8 UGRID-1.0"}
    return ds
