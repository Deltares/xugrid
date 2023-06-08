import xugrid as xu
import xarray as xr
import numpy as np
from xugrid import (
    BarycentricInterpolator,
    CentroidLocatorRegridder,
    OverlapRegridder,
    RelativeOverlapRegridder,
)

disk = xu.data.disk()["face_z"]

layer = xr.DataArray([1.0, 2.0, 3.0], coords={"layer": [1, 2, 3]}, dims=("layer",))
# Disk is first in multiplication, to ensure that object is promoted to UgridDataArray
disk_layered = disk * layer
disk_layered = disk_layered.transpose("layer", disk.ugrid.grid.face_dimension)

# quads_1
dx = 1.0
xmin, ymin, xmax, ymax = xu.data.disk().ugrid.total_bounds
x = np.arange(xmin, xmax, dx) + 0.5 * dx
y = np.arange(ymin, ymax, dx) + 0.5 * dx
da = xr.DataArray(
    data=np.full((y.size, x.size), np.nan),
    coords={"y": y, "x": x},
    dims=[
        "y",
        "x",
    ],
)
quads_1=xu.UgridDataArray.from_structured(da)

square = quads_1
regridder = CentroidLocatorRegridder(source=disk, target=square)
result = regridder.regrid(disk)
weights = regridder.weights
new_regridder = CentroidLocatorRegridder.from_weights(weights, target=square)
new_result = new_regridder.regrid(disk_layered)
assert new_result.sel(layer=1).equals(result)

assert np.nan_equal(new_result.sel(layer=1).values,result.values)
