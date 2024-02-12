"""Create and merge partitioned UGRID topologies."""
from collections import defaultdict
from itertools import accumulate
from typing import List

import numpy as np
import xarray as xr

from xugrid.constants import IntArray, IntDType
from xugrid.core.wrap import UgridDataArray, UgridDataset
from xugrid.ugrid.connectivity import renumber


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


def validate_partition_topology(grouped, n_partition: int):
    n = n_partition
    if False:
        raise ValueError(
            f"Expected {n} UGRID topologies for {n} partitions, received: " f"{grouped}"
        )

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


def group_grids_by_name(partitions):
    grouped = defaultdict(list)
    for partition in partitions:
        for grid in partition.grids:
            grouped[grid.name].append(grid)

    validate_partition_topology(grouped, len(partitions))
    return grouped


def group_data_objects_by_gridname(partitions):
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


def validate_partition_objects(data_objects):
    # Check presence of variables.
    # TODO: Groupby gridtype, then test if variables present all grids per type.
    allvars = list({tuple(sorted(ds.data_vars)) for ds in data_objects})
    if len(allvars) > 1:
        raise ValueError(
            "These variables are present in some partitions, but not in "
            f"others: {set(allvars[0]).symmetric_difference(allvars[1])}"
        )
    # Check dimensions
    for var in allvars.pop():
        vardims = list({ds[var].dims for ds in data_objects})
        if len(vardims) > 1:
            raise ValueError(
                f"Dimensions for {var} do not match across partitions: "
                f"{vardims[0]} versus {vardims[1]}"
            )


def separate_variables(objects_by_gridname, ugrid_dims):
    """Separate into UGRID variables grouped by dimension, and other variables."""
    # validate_partition_objects(data_objects)

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
    grouped = defaultdict(list)  # UGRID associated vars
    other = defaultdict(list)  # other vars

    for gridname, data_objects in objects_by_gridname.items():
        first = data_objects[0]
        variables = first.variables
        vardims = {var: tuple(first[var].dims) for var in variables}
        for var, da in variables.items():
            shapes = (obj[var].shape for obj in data_objects)

            # Check if variable depends on UGRID dimension.
            intersection = ugrid_dims.intersection(da.dims)
            if intersection:
                assert_single_dim(intersection)
                # Now check whether the non-UGRID dimensions match.
                dim = intersection.pop()  # Get the single element in the set.
                axis = vardims[var].index(dim)
                shapes = [remove_item(shape, axis) for shape in shapes]
                if all_equal(shapes):
                    grouped[dim].append(var)

            elif all_equal(shapes):
                other[gridname].append(var)

    return grouped, other


def maybe_pad_connectivity_dims_to_max(selection, merged_grid):
    nmax_dict = merged_grid.max_connectivity_dimensions
    nmax_dict = {
        key: value for key, value in nmax_dict.items() if key in selection[0].dims
    }
    if not nmax_dict:
        return selection

    pad_width_ls = [
        {dim: (0, nmax - obj.sizes[dim]) for dim, nmax in nmax_dict.items()}
        for obj in selection
    ]

    return [
        obj.pad(pad_width=pad_width) for obj, pad_width in zip(selection, pad_width_ls)
    ]


def merge_partitions(partitions):
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

    Returns
    -------
    merged : UgridDataset
    """
    types = {type(obj) for obj in partitions}
    msg = "Expected UgridDataArray or UgridDataset, received: {}"
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
    # TODO: make sure 1D variables also in vars_by_dim
    data_objects_by_name = group_data_objects_by_gridname(partitions)
    vars_by_dim, other_vars_by_name = separate_variables(
        data_objects_by_name, ugrid_dims
    )

    # First, take identical non-UGRID variables from the first partition:
    merged = xr.Dataset()  # data_objects[0][other_vars]

    # Merge the UGRID topologies into one, and find the indexes to index into
    # the data to avoid duplicates.
    merged_grids = []
    for grids, data_objects, other_vars in zip(
        grids_by_name.values(),
        data_objects_by_name.values(),
        other_vars_by_name.values(),
    ):
        # First, merge the grid topology.
        merged.update(data_objects[0][other_vars])
        grid = grids[0]
        # TODO: shortcut for length 1 merge_partitions
        merged_grid, indexes = grid.merge_partitions(grids)
        merged_grids.append(merged_grid)

        # Now remove duplicates, then concatenate along the UGRID dimension.
        for dim, dim_indexes in indexes.items():
            vars = vars_by_dim[dim]
            selection = [
                obj[vars].isel({dim: index}, missing_dims="ignore")
                for obj, index in zip(data_objects, dim_indexes)
            ]
            selection_padded = maybe_pad_connectivity_dims_to_max(
                selection, merged_grid
            )

            merged_selection = xr.concat(selection_padded, dim=dim)
            merged.update(merged_selection)

    return UgridDataset(merged, merged_grids)
