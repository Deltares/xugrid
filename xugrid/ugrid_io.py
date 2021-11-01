"""
Helper functions for parsing and composing UGRID files

Most of these functions will be replaced by a centralized UGRID library with a
C API.
"""
import warnings
from typing import Dict, NamedTuple, Tuple, Union

import xarray as xr

from .typing import FloatArray, IntArray

# Ugrid convention data:


class UgridTopologyAttributes(NamedTuple):
    coordinates: Tuple[str]
    metadata: Tuple[str]
    connectivity: Tuple[str]


UGRID1D_DEFAULT_NAME = "network1d"
UGRID1D_TOPOLOGY_VARIABLES = UgridTopologyAttributes(
    coordinates=("node_coordinates", "edge_coordinates"),
    metadata=("node_dimension", "edge_dimension"),
    connectivity=("edge_node_connectivity",),
)

UGRID2D_DEFAULT_NAME = "mesh2d"
UGRID2D_TOPOLOGY_VARIABLES = UgridTopologyAttributes(
    coordinates=("node_coordinates", "face_coordinates", "edge_coordinates"),
    metadata=("node_dimension", "face_dimension", "edge_dimension"),
    connectivity=(
        "face_node_connectivity",
        "edge_node_connectivity",
        "face_edge_connectivity",
        "face_face_connectivity",
        "edge_face_connectivity",
        "boundary_node_connectivity",
    ),
)


class UGrid:
    # TODO: to be replaced by https://github.com/Deltares/UGridPy UGrid class
    @staticmethod
    def network1d_get_attributes(name: str):
        """Get a dictionary of network1d default attribute names and the corresponding default values.
        Args:
            name (str): The network1d name.
        Returns:
            dict: A dictionary containing the attribute names and the corresponding default values.
        """

        return {
            "cf_role": "mesh_topology",
            "long_name": "Topology data of 1D network",
            "topology_dimension": 1,
            "node_dimension": f"{name}_nNodes",
            "node_coordinates": f"{name}_node_x {name}_node_y",
            "node_id": f"{name}_node_id",
            "node_long_name": f"{name}_node_long_name",
            "edge_dimension": f"{name}_nEdges",
            "edge_node_connectivity": f"{name}_edge_nodes",
            "edge_length": f"{name}_edge_length",
            "edge_id": f"{name}_edge_id",
            "edge_long_name": f"{name}_edge_long_name",
            "edge_geometry": f"{name}_edge_geometry",
        }

    @staticmethod
    def mesh2d_get_attributes(name: str):
        """Get a dictionary of mesh2d default attribute names.
        Args:
            name (str): The mesh2d name.
        Returns:
            dict: A dictionary containing the attribute names and the corresponding default values.
        """
        return {
            "cf_role": "mesh_topology",
            "long_name": "Topology data of 2D mesh",
            "topology_dimension": 2,
            "node_dimension": f"{name}_nNodes",
            "node_coordinates": f"{name}_node_x {name}_node_y",
            "edge_dimension": f"{name}_nNodes",
            "edge_node_connectivity": f"{name}_edge_nodes",
            "face_dimension": f"{name}_nFaces",
            "face_node_connectivity": f"{name}_face_nodes",
            "max_face_nodes_dimension": f"{name}_nMax_face_nodes",
            "face_coordinates": f"{name}_face_x {name}_face_y",
        }


def default_ugrid1d_attrs(prefix: str) -> Dict[str, str]:
    return {
        "cf_role": "mesh_topology",
        "long_name": "Topology data of 1D network",
        "topology_dimension": 1,
        "node_coordinates": f"{prefix}node_x {prefix}node_y",
        "edge_node_connectivity": f"{prefix}edge_node_connectivity",
        "node_dimension": f"{prefix}node",
        "edge_dimension": f"{prefix}edge",
    }


def default_ugrid2d_attrs(prefix: str) -> Dict[str, str]:
    return {
        "cf_role": "mesh_topology",
        "long_name": "Topology data of 2D mesh",
        "topology_dimension": 2,
        "node_coordinates": f"{prefix}node_x {prefix}node_y",
        "edge_node_connectivity": f"{prefix}edge_node_connectivity",
        "node_dimension": f"{prefix}node",
        "face_dimension": f"{prefix}face",
        "edge_dimension": f"{prefix}edge",
        "face_node_connectivity": f"{prefix}face_node_connectivity",
        "edge_node_connectivity": f"{prefix}edge_node_connectivity",
    }


# Reading / parsing:


def get_topology_variable(dataset):
    variables = [
        da for da in dataset.values() if da.attrs.get("cf_role") == "mesh_topology"
    ]
    if len(variables) > 1:
        raise ValueError("dataset may only contain a single mesh topology variable")
    return variables[0]


def set_name(variables, dataset, name, role):
    if name in dataset:
        variables[name] = role
    else:
        warnings.warn(
            f"Topology variable with role {role} specified under name "
            f"{name} specified, but variable {name} not found in dataset.",
            UserWarning,
        )


def _extract_topology_variables(
    dataset, mesh_topology, ugrid_attrs: UgridTopologyAttributes
):
    attrs = mesh_topology.attrs
    variables = {mesh_topology.name: "mesh_topology"}
    for role in ugrid_attrs.connectivity:
        name = attrs.get(role)
        set_name(variables, dataset, name, role)
    for role in ugrid_attrs.coordinates:
        name = attrs.get(role)
        if name:
            name_x, name_y = name.split()
            set_name(variables, dataset, name_x, role)
            set_name(variables, dataset, name_y, role)
    return variables


def get_ugrid1d_variables(dataset, mesh_topology):
    return _extract_topology_variables(
        dataset, mesh_topology, UGRID1D_TOPOLOGY_VARIABLES
    )


def get_ugrid2d_variables(dataset, mesh_topology):
    return _extract_topology_variables(
        dataset, mesh_topology, UGRID2D_TOPOLOGY_VARIABLES
    )


def cast(da: xr.DataArray, fill_value: Union[float, int], dtype: type) -> xr.DataArray:
    """
    Set the appropriate fill value and cast to dtype.
    """
    old_fill_value = da.attrs["_FillValue"]
    data = da.values
    # This returns a copy.
    if np.isnan(old_fill):
        is_fill = np.isnan(data)
    else:
        is_fill = data == old_fill_value
    data[is_fill] = fill_value
    data = data.astype(dtype, copy=True)
    return da.copy(data=data)


# Writing / composing:


def ugrid1d_dataset(
    node_x: FloatArray,
    node_y: FloatArray,
    fill_value: int,
    edge_node_connectivity: IntArray,
    name: str = None,
    attrs: Dict[str, str] = None,
) -> xr.Dataset:
    if name is None:
        name = UGRID1D_DEFAULT_NAME

    if attrs is None:
        attrs = UGrid.network1d_get_attributes(name)

    node_dimension = attrs["node_dimension"]
    edge_dimension = attrs["edge_dimension"]
    node_x_name, node_y_name = attrs["node_coordinates"].split()
    edge_node_name = attrs["edge_node_connectivity"]

    ds = xr.Dataset()
    ds[name] = xr.DataArray(
        data=0,
        attrs=attrs,
    )
    ds = ds.assign_coords(
        {
            node_x_name: xr.DataArray(
                data=node_x,
                dims=[node_dimension],
            )
        }
    )
    ds = ds.assign_coords(
        {
            node_y_name: xr.DataArray(
                data=node_y,
                dims=[node_dimension],
            )
        }
    )
    ds[edge_node_name] = xr.DataArray(
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
    name: str = None,
    attrs: Dict[str, str] = None,
) -> xr.Dataset:
    if name is None:
        name = UGRID2D_DEFAULT_NAME

    if attrs is None:
        attrs = UGrid.mesh2d_get_attributes(name)

    node_dimension = attrs["node_dimension"]
    face_dimension = attrs["face_dimension"]
    edge_dimension = attrs["edge_dimension"]
    node_x_name, node_y_name = attrs["node_coordinates"].split()
    edge_node_name = attrs["edge_node_connectivity"]
    face_node_name = attrs["face_node_connectivity"]

    ds = xr.Dataset()
    ds[name] = xr.DataArray(
        data=0,
        attrs=attrs,
    )
    ds = ds.assign_coords(
        {
            node_x_name: xr.DataArray(
                data=node_x,
                dims=[node_dimension],
            )
        }
    )
    ds = ds.assign_coords(
        {
            node_y_name: xr.DataArray(
                data=node_y,
                dims=[node_dimension],
            )
        }
    )
    ds[face_node_name] = xr.DataArray(
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
        ds[edge_node_name] = xr.DataArray(
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
