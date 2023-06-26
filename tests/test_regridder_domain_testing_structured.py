"""
Regridding domain testing
========================

`Regridding`_ is the process of converting gridded data from one grid to
another grid. Xugrid provides tools for 2D and 3D regridding of structured
gridded data, represented as xarray objects, as well as (`layered`_)
unstructured gridded data, represented as xugrid objects.

Here we test the output domains of different regridding methods. Some methods will 
fill in certain cells with Nan's whereas other methods may give a value for the same cell. 
"""
# %%

import matplotlib.pyplot as plt
import xarray as xr

import xugrid as xu

def test_domain_structured():
    # %%
    # load a  sample dataset: a triangular grid with the surface
    # elevation of the Netherlands.

    uda = xu.data.elevation_nl()


    # %%
    # Xugrid provides several "regridder" classes which can convert gridded data
    # from one grid to another grid. Let's generate a 2d  mesh that
    # covers the entire Netherlands. The node positions are random with a uniform distribution.

    def create_grid(bounds, amount_of_cells):
        """
        Create a simple grid of triangles covering a rectangle.
        """
        import numpy as np

        xmin, ymin, xmax, ymax = bounds
        dx = (xmax - xmin) / amount_of_cells
        dy = (ymax - ymin) / amount_of_cells
        x = np.arange(xmin, xmax + dx, dx)
        y = np.arange(ymax, ymin - dy, -dy)
        return xr.DataArray(
            data=np.ones((1, len(y), len(x)), np.double),
            dims=[ "layer", "y", "x",],
            coords={
                "y": y,
                "x": x,
                "layer" : [1], 
                "dx": dx,
                "dy": -dy,
            },
        )

    def fill_grid(grid, fraction_nodata):
        """
    fills the grid with values, a fraction of which are nodata (approximate)
        """
        import numpy as np

        randoms = np.random.rand(*grid.shape)
        to_be_nan = randoms < fraction_nodata
        randoms[to_be_nan] = np.nan

        grid.values = randoms



    source_grid = create_grid(uda.ugrid.total_bounds, 200)
    fill_grid(source_grid, 0.20)


    target_grid =  create_grid(uda.ugrid.total_bounds,2000)




    # %%
    # BarycentricInterpolator
    # ----------------

    regridder = xu.BarycentricInterpolator(source=source_grid, target=target_grid)
    _ = regridder.regrid(source_grid)
    assert True
