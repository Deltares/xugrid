# %% 

import xugrid as xu
import numpy as np
import matplotlib.pyplot as plt

# %%

uda = xu.data.elevation_nl()

# %%

uda[:] = np.nan
uda[2050] = -100.0
uda[4518] = 100.0
# %%

fig, ax = plt.subplots()
uda.ugrid.plot(ax=ax)
#uda.ugrid.grid.plot(ax=ax)
ax.set_aspect(1.0)

# %%

filled1 = uda.ugrid.laplace_interpolate(direct_solve=True)
# %%

fig, ax = plt.subplots()
filled1.ugrid.plot(ax=ax)
#uda.ugrid.grid.plot(ax=ax)
ax.set_aspect(1.0)
# %%

filled2 = uda.ugrid.laplace_interpolate(direct_solve=False, tol=1.0-7, maxiter=30)
# %%

diff = filled2 - filled1 
diff.ugrid.plot()

# %%
