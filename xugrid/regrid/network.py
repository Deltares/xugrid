import xarray as xr

import xugrid as xu


class Network1d:
    def __init__(self, obj):
        # TODO: do not omit type check on grid!
        if isinstance(obj, (xr.DataArray, xr.Dataset)) and obj.ugrid.is_indexed:
            self.ugrid_topology = obj.ugrid.grid
        elif isinstance(obj, xu.Ugrid1d):
            self.ugrid_topology = obj
        else:
            options = {"Ugrid1d", "DataArray", "Dataset"}
            raise TypeError(
                f"Expected one of {options}, received: {type(obj).__name__}"
            )

    @property
    def ndim(self):
        return 1

    @property
    def dims(self):
        return (self.ugrid_topology.edge_dimension,)

    @property
    def shape(self):
        return (self.ugrid_topology.n_edge,)

    @property
    def size(self):
        return self.ugrid_topology.n_edge

    @property
    def length(self):
        return self.ugrid_topology.edge_length
