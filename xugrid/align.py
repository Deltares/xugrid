"""
Align UGRID topology with the xarray object and vice versa.

* Assign coordinates to the node, face, edge dimensions.
* Do an xarray operation
* Update UGRID topology as needed, or error for invalid topology.

"""

def ugrid_aligns(obj, grid):
    """
    Check whether the xarray object dimensions still align with the grid.
    """
    griddims = grid.dimensions
    shared_dims = set(griddims).intersection(obj.dims)
    if shared_dims:
        for dim in shared_dims:
            ugridsize = griddims[dim]
            objsize = obj[dim].size
            if ugridsize != objsize:
                raise ValueError(
                    f"conflicting sizes for dimension '{dim}': "
                    f"length {ugridsize} in UGRID topology and "
                    f"length {objsize} in xarray dimension"
                )
        return True
    else:
        return False


def filter_indexers(indexers, grids):
    indexers = indexers.copy()
    ugrid_indexers = []
    for grid in grids:
        ugrid_dims = set(grid.dimensions).intersection(indexers)
        if ugrid_dims:
            if len(ugrid_dims) > 1:
                raise ValueError(
                    "Can only index along one UGRID dimension at a time. "
                    f"Received for topology {grid.name}: {ugrid_dims}"
                )
            dim = ugrid_dims.pop()
            ugrid_indexers.append((grid, (dim, indexers.pop(dim))))
        else:
            ugrid_indexers.append((grid, None))
    return indexers, ugrid_indexers


def ugrid_sel(obj, ugrid_indexers):
    grids = []
    for (grid, indexer_args) in ugrid_indexers:
        if indexer_args is None:
            grids.append(grid)
        else:
            obj, new_grid = grid.isel(*indexer_args, obj)
            grids.append(new_grid)
    return obj, grids
