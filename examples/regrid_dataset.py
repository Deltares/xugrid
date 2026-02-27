# %%

import numpy as np
import xarray as xr

import xugrid as xu

# %%

uda = xu.data.elevation_nl()

# %%

uds = xu.UgridDataset(grids=[uda.ugrid.grid])
xmin, ymin, xmax, ymax = uds.ugrid.total_bounds
x = np.arange(xmin, xmax, 1_000.0)
y = np.arange(ymax, ymin, -1_000.0)
coords = {"x": x, "y": y}
template = xr.DataArray(np.ones((y.size, x.size)), coords=coords, dims=("y", "x"))
# %%

uds["a"] = uda
uds["b"] = uda + 100.0

# %%

out = xu.BarycentricInterpolator(source=uda, target=template).regrid(uds)

# %%

out["a"].plot.imshow()
# %%

out["b"].plot.imshow()
# %%
