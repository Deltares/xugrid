"""
This module deals with extracting the relevant data from the UGRID attributes.

It takes some inspiration from: https://github.com/xarray-contrib/cf-xarray
"""

import warnings
from collections import ChainMap
from itertools import chain
from typing import Dict, List, Optional, Set, Tuple

import xarray as xr


class UgridDimensionError(Exception):
    pass


class UgridCoordinateError(Exception):
    pass


_DIM_NAMES = {
    1: ("node_dimension", "edge_dimension"),
    2: ("node_dimension", "face_dimension", "edge_dimension"),
}

_COORD_NAMES = {
    1: ("node_coordinates", "edge_coordinates"),
    2: ("node_coordinates", "face_coordinates", "edge_coordinates"),
}
_COORD_DIMS = {
    "node_coordinates": "node_dimension",
    "edge_coordinates": "edge_dimension",
    "face_coordinates": "face_dimension",
}

_CONNECTIVITY_NAMES = {
    1: ("edge_node_connectivity",),
    2: (
        "face_node_connectivity",
        "edge_node_connectivity",
        "face_edge_connectivity",
        "face_face_connectivity",
        "edge_face_connectivity",
        "boundary_node_connectivity",
    ),
}

_CONNECTIVITY_DIMS = {
    "face_node_connectivity": ("face_dimension", None),
    "edge_node_connectivity": ("edge_dimension", 2),
    "face_edge_connectivity": ("face_dimension", None),
    "face_face_connectivity": ("face_dimension", None),
    "edge_face_connectivity": ("edge_dimension", 2),
    "boundary_node_connectivity": ("boundary_edge_dimension", 2),
}

X_STANDARD_NAMES = ("projection_x_coordinate", "longitude")
Y_STANDARD_NAMES = ("projection_y_coordinate", "latitude")

PROJECTED = True
GEOGRAPHIC = False
DEFAULT_ATTRS = {
    "node_x": {
        PROJECTED: {
            "standard_name": "projection_x_coordinate",
        },
        GEOGRAPHIC: {
            "standard_name": "longitude",
        },
    },
    "node_y": {
        PROJECTED: {
            "standard_name": "projection_y_coordinate",
        },
        GEOGRAPHIC: {
            "standard_name": "latitude",
        },
    },
    "edge_x": {
        PROJECTED: {
            "standard_name": "projection_x_coordinate",
        },
        GEOGRAPHIC: {
            "standard_name": "longitude",
        },
    },
    "edge_y": {
        PROJECTED: {
            "standard_name": "projection_y_coordinate",
        },
        GEOGRAPHIC: {
            "standard_name": "latitude",
        },
    },
    "face_x": {
        PROJECTED: {
            "standard_name": "projection_x_coordinate",
        },
        GEOGRAPHIC: {
            "standard_name": "longitude",
        },
    },
    "face_y": {
        PROJECTED: {
            "standard_name": "projection_y_coordinate",
        },
        GEOGRAPHIC: {
            "standard_name": "latitude",
        },
    },
    "face_node_connectivity": {
        "cf_role": "face_node_connectivity",
        "start_index": 0,
        "_FillValue": -1,
    },
    "edge_node_connectivity": {
        "cf_role": "edge_node_connectivity",
        "start_index": 0,
        "_FillValue": -1,
    },
    "face_edge_connectivity": {
        "cf_role": "face_edge_connectivity",
        "start_index": 0,
        "_FillValue": -1,
    },
    "face_face_connectivity": {
        "cf_role": "face_face_connectivity",
        "start_index": 0,
        "_FillValue": -1,
    },
    "edge_face_connectivity": {
        "cf_role": "edge_face_connectivity",
        "start_index": 0,
        "_FillValue": -1,
    },
    "boundary_node_connectivity": {
        "cf_role": "boundary_node_connectivity",
        "start_index": 0,
        "_FillValue": -1,
    },
}


def default_topology_attrs(name: str, topology_dimension: int):
    if topology_dimension == 1:
        return {
            "cf_role": "mesh_topology",
            "long_name": "Topology data of 1D network",
            "topology_dimension": 1,
            "node_dimension": f"{name}_nNodes",
            "edge_dimension": f"{name}_nEdges",
            "edge_node_connectivity": f"{name}_edge_nodes",
            "node_coordinates": f"{name}_node_x {name}_node_y",
            "edge_coordinates": f"{name}_edge_x {name}_edge_y",
        }
    elif topology_dimension == 2:
        return {
            "cf_role": "mesh_topology",
            "long_name": "Topology data of 2D mesh",
            "topology_dimension": 2,
            "node_dimension": f"{name}_nNodes",
            "edge_dimension": f"{name}_nEdges",
            "face_dimension": f"{name}_nFaces",
            "max_face_nodes_dimension": f"{name}_nMax_face_nodes",
            "boundary_edge_dimension": f"{name}_nBoundary_edges",
            "edge_node_connectivity": f"{name}_edge_nodes",
            "face_node_connectivity": f"{name}_face_nodes",
            "face_edge_connectivity": f"{name}_face_edges",
            "edge_face_connectivity": f"{name}_edge_faces",
            "boundary_node_connectivity": f"{name}_boundary_nodes",
            "face_face_connectivity": f"{name}_face_faces",
            "node_coordinates": f"{name}_node_x {name}_node_y",
            "edge_coordinates": f"{name}_edge_x {name}_edge_y",
            "face_coordinates": f"{name}_face_x {name}_face_y",
        }
    else:
        raise ValueError(
            f"topology_dimensions should be 1 or 2, received {topology_dimension}"
        )


def _get_topology(ds: xr.Dataset) -> List[str]:
    return [
        var
        for var in ds.data_vars
        if ds.variables[var].attrs.get("cf_role") == "mesh_topology"
    ]


def _infer_xy_coords(
    ds: xr.Dataset, candidates: List[str]
) -> Tuple[List[str], List[str]]:
    # TODO: add argument for latitude / longitude?
    x = []
    y = []
    for candidate in candidates:
        stdname = ds[candidate].attrs.get("standard_name")
        if stdname in X_STANDARD_NAMES:
            x.append(candidate)
        elif stdname in Y_STANDARD_NAMES:
            y.append(candidate)

    if not x and not y:
        first = candidates[0]
        second = candidates[1]
        warnings.warn(
            f"No standard_name of {X_STANDARD_NAMES + Y_STANDARD_NAMES} in {candidates}.\n"
            f"Using {first} and {second} as projected x and y coordinates."
        )
        x.append(first)
        y.append(second)
    elif not x:
        raise UgridCoordinateError(
            "No standard_name of {X_STANDARD_NAMES} in {candidates}"
        )
    elif not y:
        raise UgridCoordinateError(
            "No standard_name of {Y_STANDARD_NAMES} in {candidates}"
        )

    return x, y


def _get_coordinates(
    ds: xr.Dataset, topologies: List[str]
) -> Dict[str, Dict[str, Tuple[List[str], List[str]]]]:
    topology_dict = {}
    for topology in topologies:
        attrs = ds[topology].attrs
        topodim = attrs["topology_dimension"]
        vardict = {}
        for name in _COORD_NAMES[topodim]:
            if name in attrs:
                candidates = [c for c in attrs[name].split(" ") if c in ds]
                if len(candidates) == 0:
                    warnings.warn(
                        f"the following variables are specified for UGRID {name}: "
                        f'"{attrs[name]}", but they are not present in the dataset'
                    )
                    continue
                if len(candidates) < 2:
                    raise UgridCoordinateError(
                        f"{topology}: at least two values required for UGRID {name},"
                        f' while only "{attrs[name]}" are specified.'
                    )
                vardict[name] = _infer_xy_coords(ds, candidates)

        topology_dict[topology] = vardict

    return topology_dict


def _infer_dims(
    ds: xr.Dataset,
    connectivities: Dict[str, str],
    coordinates: Dict[str, Dict[str, Tuple[List[str]]]],
    vardict: Dict[str, str],
) -> Dict[str, str]:
    """Infer dimensions based on connectivity and coordinates."""
    inferred = {}
    for role, varname in connectivities.items():
        expected_dims = _CONNECTIVITY_DIMS[role]
        var_dims = ds[varname].dims
        for key, dim in zip(expected_dims, var_dims):
            if isinstance(key, str):
                prev_dim = inferred.get(key)
                # Not specified: default order can be used to infer dimensions.
                if prev_dim is None:
                    inferred[key] = dim
                else:
                    if prev_dim not in var_dims:
                        raise UgridDimensionError(
                            f"{key}: {prev_dim} not in {role}: {varname}"
                            f" with dimensions: {var_dims}"
                        )
            elif isinstance(key, int):
                dim_size = ds.sizes[dim]
                if not dim_size == key:
                    raise UgridDimensionError(
                        f"Expected size {key} for dimension {dim} in variable "
                        f"{varname} with role {role}, found instead: "
                        f"{dim_size}"
                    )

            # If key is None, we don't do any checking.

    for role, varnames in coordinates.items():
        key = _COORD_DIMS[role]
        for varname in chain.from_iterable(varnames):
            var_dims = ds[varname].dims
            if len(var_dims) != 1:
                continue
            var_dim = var_dims[0]

            prev_dim = vardict.get(key) or inferred.get(key)
            if prev_dim is None:
                inferred[key] = var_dim
            else:
                if prev_dim != var_dim:
                    raise UgridDimensionError(
                        f"Conflicting names for {key}: {prev_dim} versus {var_dim}"
                    )

    return inferred


def _get_dimensions(
    ds: xr.Dataset,
    topologies: List[str],
    connectivity: Dict[str, Dict[str, str]],
    coordinates: Dict[str, Dict[str, Tuple[List[str]]]],
) -> Dict[str, Dict[str, str]]:
    """
    Get the dimensions from the topology attributes and infer them from
    connectivity arrays or coordinates.
    """
    topology_dict = {}
    for topology in topologies:
        attrs = ds[topology].attrs
        topodim = attrs["topology_dimension"]
        # dimensions are optionally required: only if the dimension order is
        # nonstandard in any of the connectivity variables.
        vardict = {k: attrs[k] for k in _DIM_NAMES[topodim] if k in attrs}
        inferred = _infer_dims(
            ds, connectivity[topology], coordinates[topology], vardict
        )
        topology_dict[topology] = {**vardict, **inferred}

    return topology_dict


def _get_connectivity(
    ds: xr.Dataset, topologies: List[str]
) -> Dict[str, Dict[str, str]]:
    topology_dict = {}
    for topology in topologies:
        attrs = ds[topology].attrs
        topodim = attrs["topology_dimension"]
        topology_dict[topology] = {
            k: attrs[k]
            for k in _CONNECTIVITY_NAMES[topodim]
            if (k in attrs) and (attrs[k] in ds)
        }
    return topology_dict


def _get_grid_mapping_names(
    ds: xr.Dataset,
    topologies: List[str],
    dimensions: Dict[str, Dict[str, str]],
) -> Dict[str, str | None]:
    topology_dict = {}
    for topology in topologies:
        topology_dict[topology] = None
        # The grid mapping should be specified per variable.
        # Check which variables have the relevant UGRID dimensions, and extract
        # the grid mapping.
        topo_dims = set(dimensions[topology].values())
        names = {
            var.attrs.get("grid_mapping") or var.encoding.get("grid_mapping")
            for var in ds.variables.values()
            if topo_dims & set(var.dims)
        } - {None}

        if names:
            # In principle, multiple coordinates are allowed to be specified
            # in the topology variable. Let's say there are two sets of coordinates
            # node_coordinates: "mesh2d_node_x1 mesh2d_node_y1 mesh2d_node_x2 mesh2d_node_y2"
            # In this case, we need a grid mapping for (x1, y1) and for (x2, y2),
            # but given that the grid mapping is defined on a data variable, there is no
            # way to link them correctly. Hence the ValueError.
            # See also: https://github.com/ugrid-conventions/ugrid-conventions/issues/64
            if len(names) > 1:
                raise ValueError(
                    f"Multiple grid mappings found for topology '{topology}': "
                    f"{names}. Variables on the same topology are expected to "
                    f"share a single coordinate reference system (CRS). "
                    f"Load the dataset with xarray.open_dataset() and modify "
                    f"the grid_mapping attributes before converting to a "
                    f"UgridDataset."
                )

            topology_dict[topology] = next(iter(names))

    return topology_dict


def _infer_projected(
    ds: xr.Dataset,
    topologies: List[str],
    coordinates: Dict[str, Dict[str, Tuple[List[str], List[str]]]],
) -> Dict[str, bool | None]:
    topology_dict = {}
    for topology in topologies:
        inferred = []
        for role, (x_vars, y_vars) in coordinates[topology].items():
            for x_varname, y_varname in zip(x_vars, y_vars):
                # Check x
                stdname = ds[x_varname].attrs.get("standard_name")
                if stdname == X_STANDARD_NAMES[0]:
                    inferred.append((x_varname, True))
                elif stdname == X_STANDARD_NAMES[1]:
                    inferred.append((x_varname, False))
                # Check y
                stdname = ds[y_varname].attrs.get("standard_name")
                if stdname == Y_STANDARD_NAMES[0]:
                    inferred.append((y_varname, True))
                elif stdname == Y_STANDARD_NAMES[1]:
                    inferred.append((y_varname, False))

        # In principle, a geocentric CRS like EPSG:4328 is neither projected
        # nor geographic, but it is very niche we cannot easily support it
        # within xugrid.
        values = {v for _, v in inferred}
        if len(values) == 0:
            projected = None
        elif len(values) == 1:
            projected = values.pop()
        else:
            details = ", ".join(
                f"{n}: {'projected' if v else 'geographic'}" for n, v in inferred
            )
            warnings.warn(
                f"Inconsistent standard_names across coordinates for topology "
                f"'{topology}': {details}. Returning None."
            )
            projected = None
        topology_dict[topology] = projected

    return topology_dict


@xr.register_dataset_accessor("ugrid_roles")
class UgridRolesAccessor:
    """
    Xarray Dataset "accessor" to retrieve the names of UGRID variables.

    Examples
    --------
    To get a list of the UGRID dummy variables in the dataset:

    >>> dataset.ugrid_roles.topology

    To get the names of the connectivity variables in the dataset:

    >>> dataset.ugrid_roles.connectivity

    Names can also be accessed directly through the topology:

    >>> dataset.ugrid_roles["mesh2d"]["node_dimension"]

    """

    def __init__(self, ds: xr.Dataset):
        self._ds = ds

    def __getitem__(self, key: str):
        if key not in self.topology:
            raise KeyError(key)
        return ChainMap(
            self.dimensions[key], self.coordinates[key], self.connectivity[key]
        )

    @property
    def topology(self) -> List[str]:
        """
        Get the names of the topology dummy variables, marked by a CF-role of
        ``mesh_topology``.

        Returns
        -------
        topology: List[str]
        """
        return _get_topology(self._ds)

    @property
    def coordinates(self) -> Dict[str, Dict[str, Tuple[List[str], List[str]]]]:
        """
        Get the names of the coordinate variables from the topology attributes.

        Returns a dictionary with the coordinates for the UGRID coordinates:

            * node coordinates
            * edge coordinates
            * face coordinates

        Multiple coordinates may be defined. The coordinates are grouped by
        their role (x or y).

        Returns
        -------
        coordinates: dict[str, dict[str, Tuple[List[str]]]]
        """
        return _get_coordinates(self._ds, self.topology)

    @property
    def dimensions(self) -> Dict[str, Dict[str, str]]:
        """
        Get the dimension names from the topology attributes and infer them
        from connectivity arrays or coordinates.

        Returns a dictionary with the UGRID dimensions per topology:

            * node dimension
            * edge dimension
            * face dimension

        Returns
        -------
        dimensions: dict[str, dict[str, str]]
        """
        return _get_dimensions(
            self._ds, self.topology, self.connectivity, self.coordinates
        )

    @property
    def connectivity(self) -> Dict[str, Dict[str, str]]:
        """
        Get the names of the variables containing the UGRID connectivity data.

            * face_node_connectivity
            * edge_node_connectivity
            * face_edge_connectivity
            * edge_face_connectivity

        Returns
        -------
        connectivity: Dict[str, Dict[str, str]]
        """
        return _get_connectivity(self._ds, self.topology)

    @property
    def grid_mapping_names(self) -> Dict[str, Optional[Set[str]]]:
        """
        Get the names of the grid mapping variables associated with each topology.

        Returns
        -------
        grid_mapping: dict[str, str | None]
        """
        return _get_grid_mapping_names(self._ds, self.topology, self.dimensions)

    @property
    def is_projected(self) -> Dict[str, bool | None]:
        """
        Infer whether each topology uses projected or geographic coordinates
        from the standard_name attributes of the coordinate variables.

        Returns
        -------
        projected: dict[str, bool | None]
            True if projected, False if geographic, None if indeterminate.
        """
        return _infer_projected(self._ds, self.topology, self.coordinates)

    def __repr__(self):
        dimensions = self.dimensions
        coordinates = self.coordinates
        connectivity = self.connectivity
        grid_mapping_names = self.grid_mapping_names
        is_projected = self.is_projected

        def make_text_section(subtitle, entries, vardict):
            tab = "    "
            rows = [f"{tab}{subtitle}"]
            for role in entries:
                if role in vardict:
                    rows += [f"{tab}{tab}{role}: {vardict[role]}"]
                else:
                    rows += [f"{tab}{tab}{role}: n/a"]
            rows.append("")
            return rows

        rows = []
        for topology in self.topology:
            topodim = self._ds[topology].attrs["topology_dimension"]
            rows += [f"UGRID {topodim}D Topology {topology}:"]
            rows += make_text_section(
                "Dimensions:", _DIM_NAMES[topodim], dimensions[topology]
            )
            rows += make_text_section(
                "Connectivity:", _CONNECTIVITY_NAMES[topodim], connectivity[topology]
            )
            rows += make_text_section(
                "Coordinates:", _COORD_NAMES[topodim], coordinates[topology]
            )

            # CRS summary line
            name = grid_mapping_names[topology]
            projected = is_projected[topology]
            if projected is True:
                crs_type = "projected"
            elif projected is False:
                crs_type = "geographic"
            else:
                crs_type = "unknown"
            name_str = name if name is not None else "n/a"
            rows += [
                f"    Coordinate Type: {crs_type}",
                f"Grid Mapping Name: {name_str}",
                "",
            ]

        return "\n".join(rows)
