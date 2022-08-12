import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from matplotlib.collections import LineCollection, PathCollection, PolyCollection
from matplotlib.image import AxesImage
from matplotlib.tri import TriContourSet

import xugrid
from xugrid.plot import plot


class TestPlot:
    @pytest.fixture(autouse=True)
    def setup(self):
        # Remove all matplotlib figures
        plt.close("all")

        self.ds = xugrid.data.disk()
        self.grid = self.ds.ugrid.grids[0]
        self.node = self.ds["node_z"]
        self.edge = self.ds["edge_z"]
        self.face = self.ds["face_z"]
        self.node_da = self.node.ugrid.obj
        self.edge_da = self.edge.ugrid.obj
        self.face_da = self.face.ugrid.obj

    def test_get_ugrid_dim(self):
        with pytest.raises(ValueError, match="Not a valid UGRID dimension"):
            da = xr.DataArray([0, 1], dims=["x"])
            plot.get_ugrid_dim(self.grid, da)

        assert plot.get_ugrid_dim(self.grid, self.node_da) == plot.NODE
        assert plot.get_ugrid_dim(self.grid, self.edge_da) == plot.EDGE
        assert plot.get_ugrid_dim(self.grid, self.face_da) == plot.FACE

    def test_plot_contour(self):
        with pytest.raises(ValueError, match="contour only supports"):
            plot.contour(self.grid, self.edge_da)
        with pytest.raises(ValueError, match="contour only supports"):
            self.edge.ugrid.plot.contour()

        assert isinstance(plot.contour(self.grid, self.node_da), TriContourSet)
        assert isinstance(self.node.ugrid.plot.contour(), TriContourSet)
        assert isinstance(plot.contour(self.grid, self.face_da), TriContourSet)
        assert isinstance(self.face.ugrid.plot.contour(), TriContourSet)

        # Check whether line is dashed or not?
        assert isinstance(self.face.ugrid.plot.contour(colors="black"), TriContourSet)

    def test_plot_contourf(self):
        with pytest.raises(ValueError, match="contourf only supports"):
            plot.contourf(self.grid, self.edge_da)
        with pytest.raises(ValueError, match="contourf only supports"):
            self.edge.ugrid.plot.contourf()

        assert isinstance(plot.contourf(self.grid, self.node_da), TriContourSet)
        assert isinstance(self.node.ugrid.plot.contourf(), TriContourSet)
        assert isinstance(plot.contourf(self.grid, self.face_da), TriContourSet)
        assert isinstance(self.face.ugrid.plot.contourf(), TriContourSet)

    def test_plot_imshow(self):
        with pytest.raises(ValueError, match="imshow only supports"):
            plot.imshow(self.grid, self.edge_da)
        with pytest.raises(ValueError, match="imshow only supports"):
            self.edge.ugrid.plot.imshow()

        with pytest.raises(ValueError, match="imshow only supports"):
            plot.imshow(self.grid, self.node_da)
        with pytest.raises(ValueError, match="imshow only supports"):
            self.node.ugrid.plot.imshow()

        # Reduce resolution for non-JITed test runs (e.g. for coverage)
        assert isinstance(
            plot.imshow(self.grid, self.face_da, resolution=1.0), AxesImage
        )
        assert isinstance(self.face.ugrid.plot.imshow(resolution=1.0), AxesImage)

        with pytest.raises(ValueError, match="kwarg is not available in xugrid"):
            self.face.ugrid.plot.imshow(resolution=1.0, size=4, aspect="equal")

        actual = plot.imshow(
            self.grid, self.face_da, resolution=1.0, extent=(0.0, 1.0, 0.0, 1.0)
        )
        assert isinstance(actual, AxesImage)
        actual = plot.imshow(
            self.grid,
            self.face_da,
            resolution=1.0,
            extent=(0.0, 1.0, 0.0, 1.0),
            origin="upper",
        )
        assert isinstance(actual, AxesImage)
        actual = plot.imshow(
            self.grid,
            self.face_da,
            resolution=1.0,
            extent=(0.0, 1.0, 1.0, 0.0),
            origin="lower",
        )
        assert isinstance(actual, AxesImage)

    def test_plot_line(self):
        with pytest.raises(ValueError, match="line only supports"):
            plot.line(self.grid, self.node_da)
        with pytest.raises(ValueError, match="line only supports"):
            plot.line(self.grid, self.face_da)

        assert isinstance(plot.line(self.grid), LineCollection)
        assert isinstance(plot.line(self.grid, self.edge_da), LineCollection)
        assert isinstance(self.node.ugrid.plot.line(), LineCollection)
        assert isinstance(self.edge.ugrid.plot.line(), LineCollection)
        assert isinstance(self.face.ugrid.plot.line(), LineCollection)

    def test_plot_pcolormesh(self):
        with pytest.raises(ValueError, match="pcolormesh only supports"):
            plot.pcolormesh(self.grid, self.edge_da)
        with pytest.raises(ValueError, match="pcolormesh only supports"):
            self.edge.ugrid.plot.pcolormesh()

        with pytest.raises(ValueError, match="pcolormesh only supports"):
            plot.pcolormesh(self.grid, self.node_da)
        with pytest.raises(ValueError, match="pcolormesh only supports"):
            self.node.ugrid.plot.pcolormesh()

        assert isinstance(plot.pcolormesh(self.grid, self.face_da), PolyCollection)
        assert isinstance(self.face.ugrid.plot.pcolormesh(), PolyCollection)

    def test_plot_surface(self):
        with pytest.raises(ValueError, match="surface only supports"):
            plot.surface(self.grid, self.edge_da)
        with pytest.raises(ValueError, match="surface only supports"):
            self.edge.ugrid.plot.surface()

        assert isinstance(plot.surface(self.grid, self.node_da), PolyCollection)
        assert isinstance(plot.surface(self.grid, self.face_da), PolyCollection)
        assert isinstance(self.node.ugrid.plot.surface(), PolyCollection)
        assert isinstance(self.face.ugrid.plot.surface(), PolyCollection)

        with pytest.raises(ValueError, match="If ax is passed to surface()"):
            _, ax = plt.subplots()
            self.face.ugrid.plot.surface(ax=ax)

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        assert isinstance(self.face.ugrid.plot.surface(ax=ax), PolyCollection)

    def test_plot_scatter(self):
        assert isinstance(plot.scatter(self.grid, self.node_da), PathCollection)
        assert isinstance(plot.scatter(self.grid, self.edge_da), PathCollection)
        assert isinstance(plot.scatter(self.grid, self.face_da), PathCollection)
        assert isinstance(self.node.ugrid.plot.scatter(), PathCollection)
        assert isinstance(self.edge.ugrid.plot.scatter(), PathCollection)
        assert isinstance(self.face.ugrid.plot.scatter(), PathCollection)

    def test_plot_tripcolor(self):
        with pytest.raises(ValueError, match="tripcolor only supports"):
            plot.tripcolor(self.grid, self.edge_da)
        with pytest.raises(ValueError, match="tripcolor only supports"):
            self.edge.ugrid.plot.tripcolor()

        with pytest.raises(ValueError, match="tripcolor only supports"):
            plot.tripcolor(self.grid, self.face_da)
        with pytest.raises(ValueError, match="tripcolor only supports"):
            self.face.ugrid.plot.tripcolor()

        assert isinstance(plot.tripcolor(self.grid, self.node_da), PolyCollection)
        assert isinstance(self.node.ugrid.plot.tripcolor(), PolyCollection)

    def test_plot(self):
        assert isinstance(self.node.ugrid.plot(), PolyCollection)
        assert isinstance(self.edge.ugrid.plot(), LineCollection)
        assert isinstance(self.face.ugrid.plot(), PolyCollection)

    def test_add_colorbar_exception(self):
        with pytest.raises(ValueError, match="cbar_ax and cbar_kwargs"):
            cbar_kwargs = {"location": "top"}
            self.face.ugrid.plot(add_colorbar=False, cbar_kwargs=cbar_kwargs)

    def test_default_size(self):
        with pytest.raises(ValueError, match="cannot provide both `size` and `ax`"):
            _, ax = plt.subplots()
            self.face.ugrid.plot(ax=ax, size=10)

        self.face.ugrid.plot(size=10, add_colorbar=False)
        fig = plt.gca().figure
        h = fig.get_figheight()
        w = fig.get_figwidth()
        assert np.allclose(w / h, 1.0)

        self.face.ugrid.plot(size=10)
        fig = plt.gca().figure
        h = fig.get_figheight()
        w = fig.get_figwidth()
        assert np.allclose(w / h, 1.26)
