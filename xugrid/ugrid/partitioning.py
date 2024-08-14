"""Create and merge partitioned UGRID topologies."""
from collections import defaultdict
from itertools import accumulate, chain
from typing import List

import numpy as np
import xarray as xr

from xugrid.constants import IntArray, IntDType
from xugrid.core.wrap import UgridDataArray, UgridDataset
from xugrid.ugrid.connectivity import renumber
from xugrid.ugrid.ugridbase import UgridType


def labels_to_indices(labels: IntArray) -> List[IntArray]:
    """
    Convert a 1D array of N labels into a N arrays of indices.

    E.g. [0, 1, 0, 2, 2] -> [[0, 2], [1], [3, 4]]
    """
    sorter = np.argsort(labels)
    split_indices = np.cumsum(np.bincount(labels)[:-1])
    indices = np.split(sorter, split_indices)
    for index in indices:
        index.sort()
    return indices


def partition_by_label(grid, obj, labels: IntArray):
    """
    Partition the grid and xarray object by integer labels.

    This function is used by UgridDataArray.partition_by_label and
    UgridDataset.partition_by_label.

    Parameters
    ----------
    grid: Ugrid1d, Ugrid2d
    obj: DataArray or Dataset
    labels: UgridDataArray of integers

    Returns
    -------
    partitions: List of (grid, obj)
    """
    if not isinstance(labels, UgridDataArray):
        raise TypeError(
            "labels must be a UgridDataArray, " f"received: {type(labels).__name__}"
        )
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError(f"labels must have integer dtype, received {labels.dtype}")

    if labels.grid != grid:
        raise ValueError("grid of labels does not match xugrid object")
    if labels.dims != (grid.core_dimension,):
        raise ValueError(
            f"Can only partition this topology by {grid.core_dimension}, found"
            f" the dimensions: {labels.dims}"
        )

    if isinstance(obj, xr.Dataset):
        obj_type = UgridDataset
    elif isinstance(obj, xr.DataArray):
        obj_type = UgridDataArray
    else:
        raise TypeError(
            f"Expected DataArray or Dataset, received: {type(obj).__name__}"
        )

    indices = labels_to_indices(labels.values)
    partitions = []
    for index in indices:
        new_grid, indexes = grid.topology_subset(index, return_index=True)
        new_obj = obj.isel(indexes, missing_dims="ignore")
        partitions.append(obj_type(new_obj, new_grid))

    return partitions


def merge_nodes(grids):
    node_x = np.hstack([grid.node_x for grid in grids])
    node_y = np.hstack([grid.node_y for grid in grids])
    node_xy = np.column_stack((node_x, node_y))
    _, index, inverse = np.unique(
        node_xy, axis=0, return_index=True, return_inverse=True
    )
    inverse = inverse.ravel()
    # We want to maintain order, so create an inverse index to the new numbers.
    inverse = renumber(index)[inverse]
    # Maintain order.
    index.sort()
    unique_nodes = node_xy[index]
    # Slice the indexes per partition.
    slices = (0,) + tuple(accumulate(grid.n_node for grid in grids))
    sections = np.searchsorted(index, slices[1:-1])
    indexes = np.split(index, sections)
    for partition_index, offset in zip(indexes, slices):
        partition_index -= offset
    return unique_nodes, indexes, inverse


def _merge_connectivity(gathered, slices):
    # Make sure identical edges are identified: [0, 1] == [1, 0]
    # As well as faces: [0, 1, 2] == [2, 1, 0]
    sorted = np.sort(gathered, axis=1)
    _, index = np.unique(sorted, axis=0, return_index=True)
    # Maintain order.
    index.sort()
    merged = gathered[index]
    # Slice the indexes per partition.
    sections = np.searchsorted(index, slices[1:-1])
    indexes = np.split(index, sections)
    for partition_index, offset in zip(indexes, slices):
        partition_index -= offset
    return merged, indexes


def merge_faces(grids, node_inverse, fill_value: int = -1):
    node_offsets = tuple(accumulate([0] + [grid.n_node for grid in grids]))
    n_face = [grid.n_face for grid in grids]
    n_max_node = max(grid.n_max_node_per_face for grid in grids)
    slices = (0,) + tuple(accumulate(n_face))

    all_faces = np.full((sum(n_face), n_max_node), fill_value, dtype=IntDType)
    for grid, face_offset, node_offset in zip(grids, slices, node_offsets):
        faces = grid.face_node_connectivity
        n_face, n_node_per_face = faces.shape
        valid = faces != grid.fill_value
        all_faces[face_offset : face_offset + n_face, :n_node_per_face][
            valid
        ] = node_inverse[faces[valid] + node_offset]

    return _merge_connectivity(all_faces, slices)


def merge_edges(grids, node_inverse):
    node_offsets = tuple(accumulate([0] + [grid.n_node for grid in grids]))
    n_edge = [grid.n_edge for grid in grids]
    slices = (0,) + tuple(accumulate(n_edge))

    all_edges = np.empty((sum(n_edge), 2), dtype=IntDType)
    for grid, edge_offset, offset in zip(grids, slices, node_offsets):
        edges = grid.edge_node_connectivity
        n_edge = len(edges)
        all_edges[edge_offset : edge_offset + n_edge] = node_inverse[edges + offset]

    return _merge_connectivity(all_edges, slices)


def validate_partition_topology(grouped: defaultdict[str, UgridType]) -> None:
    for name, grids in grouped.items():
        types = {type(grid) for grid in grids}
        if len(types) > 1:
            raise TypeError(
                f"All partition topologies with name {name} should be of the "
                f"same type, received: {types}"
            )

        griddims = list({tuple(grid.dimensions) for grid in grids})
        if len(griddims) > 1:
            raise ValueError(
                f"Dimension names on UGRID topology {name} do not match "
                f"across partitions: {griddims[0]} versus {griddims[1]}"
            )

    return None


def group_grids_by_name(partitions: list[UgridDataset]) -> defaultdict[str, UgridType]:
    grouped = defaultdict(list)
    for partition in partitions:
        for grid in partition.grids:
            grouped[grid.name].append(grid)

    validate_partition_topology(grouped)
    return grouped


def group_data_objects_by_gridname(
    partitions: list[UgridDataset]
) -> defaultdict[str, xr.Dataset]:
    # Convert to dataset for convenience
    data_objects = [partition.obj for partition in partitions]
    data_objects = [
        obj.to_dataset() if isinstance(obj, xr.DataArray) else obj
        for obj in data_objects
    ]

    grouped = defaultdict(list)
    for partition, obj in zip(partitions, data_objects):
        for grid in partition.grids:
            grouped[grid.name].append(obj)

    return grouped


def validate_partition_objects(
    objects_by_gridname: defaultdict[str, xr.Dataset]
) -> None:
    for data_objects in objects_by_gridname.values():
        allvars = list({tuple(sorted(ds.data_vars)) for ds in data_objects})
        unique_vars = set(chain(*allvars))
        # Check dimensions
        dims_per_var = [
            {ds[var].dims for ds in data_objects if var in ds.data_vars}
            for var in unique_vars
        ]
        for var, vardims in zip(unique_vars, dims_per_var):
            if len(vardims) > 1:
                vardims_ls = list(vardims)
                raise ValueError(
                    f"Dimensions for '{var}' do not match across partitions: "
                    f"{vardims_ls[0]} versus {vardims_ls[1]}"
                )
    return None


def separate_variables(
    objects_by_gridname: defaultdict[str, xr.Dataset], ugrid_dims: set[str]
):
    """Separate into UGRID variables grouped by dimension, and other variables."""
    validate_partition_objects(objects_by_gridname)

    def assert_single_dim(intersection):
        if len(intersection) > 1:
            raise ValueError(
                f"{var} contains more than one UGRID dimension: {intersection}"
            )

    def remove_item(tuple, index):
        return tuple[:index] + tuple[index + 1 :]

    def all_equal(iterator):
        first = next(iter(iterator))
        return all(element == first for element in iterator)

    # Group variables by UGRID dimension.
    grouped = defaultdict(set)  # UGRID associated vars
    other = defaultdict(set)  # other vars

    for gridname, data_objects in objects_by_gridname.items():
        variables = {
            varname: data
            for obj in data_objects
            for varname, data in obj.variables.items()
        }
        vardims = {varname: data.dims for varname, data in variables.items()}
        for var, dims in vardims.items():
            shapes = [obj[var].shape for obj in data_objects if var in obj]

            # Check if variable depends on UGRID dimension.
            intersection = ugrid_dims.intersection(dims)
            if intersection:
                assert_single_dim(intersection)
                # Now check whether the non-UGRID dimensions match.
                dim = intersection.pop()  # Get the single element in the set.
                axis = dims.index(dim)
                shapes = [remove_item(shape, axis) for shape in shapes]
                if all_equal(shapes):
                    grouped[dim].add(var)

            elif all_equal(shapes):
                other[gridname].add(var)

    return grouped, other


def merge_data_along_dim(
    data_objects: list[xr.Dataset],
    vars: list[str],
    merge_dim: str,
    indexes: list[np.array],
    merged_grid: UgridType,
) -> xr.Dataset:
    """
    Select variables from the data objects.
    Pad connectivity dims if needed.
    Concatenate along dim.
    """
    max_sizes = merged_grid.max_connectivity_sizes
    ugrid_connectivity_dims = set(max_sizes)

    to_merge = []
    for obj, index in zip(data_objects, indexes):
        # Check for presence of vars
        missing_vars = set(vars).difference(set(obj.variables.keys()))
        if missing_vars:
            raise ValueError(f"Missing variables: {missing_vars} in partition {obj}")

        selection = obj[vars].isel({merge_dim: index}, missing_dims="ignore")

        # Pad the ugrid connectivity dims (e.g. n_max_face_node_connectivity) if
        # needed.
        present_dims = ugrid_connectivity_dims.intersection(selection.dims)
        pad_width = {}
        for dim in present_dims:
            nmax = max_sizes[dim]
            size = selection.sizes[dim]
            if size != nmax:
                pad_width[dim] = (0, nmax - size)
        if pad_width:
            selection = selection.pad(pad_width=pad_width)

        to_merge.append(selection)

    return xr.concat(to_merge, dim=merge_dim)


def merge_partitions(partitions, merge_ugrid_chunks: bool = True):
    """
    Merge topology and data, partitioned along UGRID dimensions, into a single
    UgridDataset.

    UGRID topologies and variables are merged if they share a name. Topologies
    and variables must be present in *all* partitions. Dimension names must
    match.

    Variables are omitted from the merged result if non-UGRID dimensions do not
    match in size.

    Parameters
    ----------
    partitions : sequence of UgridDataset or UgridDataArray
    merge_ugrid_chunks: bool, default is True.
        Whether to merge chunks along the UGRID topology dimensions.

    Returns
    -------
    merged : UgridDataset
    """
    types = {type(obj) for obj in partitions}
    msg = "Expected UgridDataArray or UgridDataset, received: {}"
    if len(types) == 0:
        raise ValueError("Received empty partitions list, cannot be merged.")
    if len(types) > 1:
        type_names = [t.__name__ for t in types]
        raise TypeError(msg.format(type_names))
    obj_type = types.pop()
    if obj_type not in (UgridDataArray, UgridDataset):
        raise TypeError(msg.format(obj_type.__name__))

    # Collect grids
    grids = [grid for p in partitions for grid in p.grids]
    ugrid_dims = {dim for grid in grids for dim in grid.dimensions}
    grids_by_name = group_grids_by_name(partitions)

    data_objects_by_name = group_data_objects_by_gridname(partitions)
    vars_by_dim, other_vars_by_name = separate_variables(
        data_objects_by_name, ugrid_dims
    )

    # First, take identical non-UGRID variables from the first partition:
    merged = xr.Dataset()

    # Merge the UGRID topologies into one, and find the indexes to index into
    # the data to avoid duplicates.
    merged_grids = []
    for gridname, grids in grids_by_name.items():
        data_objects = data_objects_by_name[gridname]
        other_vars = other_vars_by_name[gridname]

        # First, merge the grid topology.
        grid = grids[0]
        merged_grid, indexes = grid.merge_partitions(grids)
        merged_grids.append(merged_grid)

        # Add other vars, unassociated with UGRID dimensions, to dataset.
        for obj in data_objects:
            other_vars_obj = set(other_vars).intersection(set(obj.data_vars))
            merged.update(obj[other_vars_obj])

        for dim, dim_indexes in indexes.items():
            vars = vars_by_dim[dim]
            if len(vars) == 0:
                continue
            merged_selection = merge_data_along_dim(
                data_objects, vars, dim, dim_indexes, merged_grid
            )
            merged.update(merged_selection)

    # Merge chunks along the UGRID dimensions.
    if merged.chunks and merge_ugrid_chunks:
        chunks = dict(merged.chunks)
        for dim in chunks:
            # Define a single chunk for each UGRID dimension.
            if dim in ugrid_dims:
                chunks[dim] = (merged.sizes[dim],)
        merged = merged.chunk(chunks)

    return UgridDataset(merged, merged_grids)
