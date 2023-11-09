"""
This module is a subset of the testing functions in xarray.tests.test_plot. The
reason is that we use a number of plot utils that are private to xarray.

Additionally, the _importorskip has been copied.

We heavily discourage editing this file. Any update should only consist of
copying updated parts of the xarray module.

Xarray is licensed under Apache License 2.0:
https://github.com/pydata/xarray/blob/main/LICENSE
"""
from __future__ import annotations

import contextlib
import importlib
import math

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal  # noqa: F401
from packaging.version import Version
from xarray import DataArray

from xugrid.plot.utils import (
    _color_palette,
    _determine_cmap_params,
    _maybe_gca,
    get_axis,
    label_from_attrs,
)

# import mpl and change the backend before other mpl imports
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError:
    pass


def _importorskip(
    modname: str, minversion: str | None = None
) -> tuple[bool, pytest.MarkDecorator]:
    try:
        mod = importlib.import_module(modname)
        has = True
        if minversion is not None:
            if Version(mod.__version__) < Version(minversion):
                raise ImportError("Minimum version not satisfied")
    except ImportError:
        has = False
    func = pytest.mark.skipif(not has, reason=f"requires {modname}")
    return has, func


has_matplotlib, requires_matplotlib = _importorskip("matplotlib")


@contextlib.contextmanager
def figure_context(*args, **kwargs):
    """Context manager which autocloses a figure (even if the test failed)"""

    try:
        yield None
    finally:
        plt.close("all")


def easy_array(shape, start=0, stop=1):
    """
    Make an array with desired shape using np.linspace

    shape is a tuple like (2, 3)
    """
    a = np.linspace(start, stop, num=math.prod(shape))
    return a.reshape(shape)


@requires_matplotlib
class PlotTestCase:
    @pytest.fixture(autouse=True)
    def setup(self):
        yield
        # Remove all matplotlib figures
        plt.close("all")

    def pass_in_axis(self, plotmethod, subplot_kw=None):
        fig, axs = plt.subplots(ncols=2, subplot_kw=subplot_kw)
        plotmethod(ax=axs[0])
        assert axs[0].has_data()

    def imshow_called(self, plotmethod):
        plotmethod()
        images = plt.gca().findobj(mpl.image.AxesImage)
        return len(images) > 0

    def contourf_called(self, plotmethod):
        plotmethod()

        # Compatible with mpl before (PathCollection) and after (QuadContourSet) 3.8
        def matchfunc(x):
            return isinstance(
                x, (mpl.collections.PathCollection, mpl.contour.QuadContourSet)
            )

        paths = plt.gca().findobj(matchfunc)
        return len(paths) > 0


@requires_matplotlib
class TestDiscreteColorMap:
    @pytest.fixture(autouse=True)
    def setUp(self):
        x = np.arange(start=0, stop=10, step=2)
        y = np.arange(start=9, stop=-7, step=-3)
        xy = np.dstack(np.meshgrid(x, y))
        distance = np.linalg.norm(xy, axis=2)
        self.darray = DataArray(distance, list(zip(("y", "x"), (y, x))))
        self.data_min = distance.min()
        self.data_max = distance.max()
        yield
        # Remove all matplotlib figures
        plt.close("all")

    def test_recover_from_seaborn_jet_exception(self) -> None:
        pal = _color_palette("jet", 4)
        assert isinstance(pal, np.ndarray)
        assert len(pal) == 4


@requires_matplotlib
class TestDetermineCmapParams:
    @pytest.fixture(autouse=True)
    def setUp(self) -> None:
        self.data = np.linspace(0, 1, num=100)

    def test_robust(self) -> None:
        cmap_params = _determine_cmap_params(self.data, robust=True)
        assert cmap_params["vmin"] == np.percentile(self.data, 2)
        assert cmap_params["vmax"] == np.percentile(self.data, 98)
        assert cmap_params["cmap"] == "viridis"
        assert cmap_params["extend"] == "both"
        assert cmap_params["levels"] is None
        assert cmap_params["norm"] is None

    def test_center(self) -> None:
        cmap_params = _determine_cmap_params(self.data, center=0.5)
        assert cmap_params["vmax"] - 0.5 == 0.5 - cmap_params["vmin"]
        assert cmap_params["cmap"] == "RdBu_r"
        assert cmap_params["extend"] == "neither"
        assert cmap_params["levels"] is None
        assert cmap_params["norm"] is None

    # EDIT: removed tests for xr.set_options.
    def test_nan_inf_are_ignored(self) -> None:
        cmap_params1 = _determine_cmap_params(self.data)
        data = self.data
        data[50:55] = np.nan
        data[56:60] = np.inf
        cmap_params2 = _determine_cmap_params(data)
        assert cmap_params1["vmin"] == cmap_params2["vmin"]
        assert cmap_params1["vmax"] == cmap_params2["vmax"]

    def test_integer_levels(self) -> None:
        data = self.data + 1

        # default is to cover full data range but with no guarantee on Nlevels
        for level in np.arange(2, 10, dtype=int):
            cmap_params = _determine_cmap_params(data, levels=level)
            assert cmap_params["vmin"] is None
            assert cmap_params["vmax"] is None
            assert cmap_params["norm"].vmin == cmap_params["levels"][0]
            assert cmap_params["norm"].vmax == cmap_params["levels"][-1]
            assert cmap_params["extend"] == "neither"

        # with min max we are more strict
        cmap_params = _determine_cmap_params(
            data, levels=5, vmin=0, vmax=5, cmap="Blues"
        )
        assert cmap_params["vmin"] is None
        assert cmap_params["vmax"] is None
        assert cmap_params["norm"].vmin == 0
        assert cmap_params["norm"].vmax == 5
        assert cmap_params["norm"].vmin == cmap_params["levels"][0]
        assert cmap_params["norm"].vmax == cmap_params["levels"][-1]
        assert cmap_params["cmap"].name == "Blues"
        assert cmap_params["extend"] == "neither"
        assert cmap_params["cmap"].N == 4
        assert cmap_params["norm"].N == 5

        cmap_params = _determine_cmap_params(data, levels=5, vmin=0.5, vmax=1.5)
        assert cmap_params["cmap"].name == "viridis"
        assert cmap_params["extend"] == "max"

        cmap_params = _determine_cmap_params(data, levels=5, vmin=1.5)
        assert cmap_params["cmap"].name == "viridis"
        assert cmap_params["extend"] == "min"

        cmap_params = _determine_cmap_params(data, levels=5, vmin=1.3, vmax=1.5)
        assert cmap_params["cmap"].name == "viridis"
        assert cmap_params["extend"] == "both"

    def test_list_levels(self) -> None:
        data = self.data + 1

        orig_levels = [0, 1, 2, 3, 4, 5]
        # vmin and vmax should be ignored if levels are explicitly provided
        cmap_params = _determine_cmap_params(data, levels=orig_levels, vmin=0, vmax=3)
        assert cmap_params["vmin"] is None
        assert cmap_params["vmax"] is None
        assert cmap_params["norm"].vmin == 0
        assert cmap_params["norm"].vmax == 5
        assert cmap_params["cmap"].N == 5
        assert cmap_params["norm"].N == 6

        for wrap_levels in [list, np.array, pd.Index, DataArray]:
            cmap_params = _determine_cmap_params(data, levels=wrap_levels(orig_levels))
            assert_array_equal(cmap_params["levels"], orig_levels)

    def test_divergentcontrol(self) -> None:
        neg = self.data - 0.1
        pos = self.data

        # Default with positive data will be a normal cmap
        cmap_params = _determine_cmap_params(pos)
        assert cmap_params["vmin"] == 0
        assert cmap_params["vmax"] == 1
        assert cmap_params["cmap"] == "viridis"

        # Default with negative data will be a divergent cmap
        cmap_params = _determine_cmap_params(neg)
        assert cmap_params["vmin"] == -0.9
        assert cmap_params["vmax"] == 0.9
        assert cmap_params["cmap"] == "RdBu_r"

        # Setting vmin or vmax should prevent this only if center is false
        cmap_params = _determine_cmap_params(neg, vmin=-0.1, center=False)
        assert cmap_params["vmin"] == -0.1
        assert cmap_params["vmax"] == 0.9
        assert cmap_params["cmap"] == "viridis"
        cmap_params = _determine_cmap_params(neg, vmax=0.5, center=False)
        assert cmap_params["vmin"] == -0.1
        assert cmap_params["vmax"] == 0.5
        assert cmap_params["cmap"] == "viridis"

        # Setting center=False too
        cmap_params = _determine_cmap_params(neg, center=False)
        assert cmap_params["vmin"] == -0.1
        assert cmap_params["vmax"] == 0.9
        assert cmap_params["cmap"] == "viridis"

        # However, I should still be able to set center and have a div cmap
        cmap_params = _determine_cmap_params(neg, center=0)
        assert cmap_params["vmin"] == -0.9
        assert cmap_params["vmax"] == 0.9
        assert cmap_params["cmap"] == "RdBu_r"

        # Setting vmin or vmax alone will force symmetric bounds around center
        cmap_params = _determine_cmap_params(neg, vmin=-0.1)
        assert cmap_params["vmin"] == -0.1
        assert cmap_params["vmax"] == 0.1
        assert cmap_params["cmap"] == "RdBu_r"
        cmap_params = _determine_cmap_params(neg, vmax=0.5)
        assert cmap_params["vmin"] == -0.5
        assert cmap_params["vmax"] == 0.5
        assert cmap_params["cmap"] == "RdBu_r"
        cmap_params = _determine_cmap_params(neg, vmax=0.6, center=0.1)
        assert cmap_params["vmin"] == -0.4
        assert cmap_params["vmax"] == 0.6
        assert cmap_params["cmap"] == "RdBu_r"

        # But this is only true if vmin or vmax are negative
        cmap_params = _determine_cmap_params(pos, vmin=-0.1)
        assert cmap_params["vmin"] == -0.1
        assert cmap_params["vmax"] == 0.1
        assert cmap_params["cmap"] == "RdBu_r"
        cmap_params = _determine_cmap_params(pos, vmin=0.1)
        assert cmap_params["vmin"] == 0.1
        assert cmap_params["vmax"] == 1
        assert cmap_params["cmap"] == "viridis"
        cmap_params = _determine_cmap_params(pos, vmax=0.5)
        assert cmap_params["vmin"] == 0
        assert cmap_params["vmax"] == 0.5
        assert cmap_params["cmap"] == "viridis"

        # If both vmin and vmax are provided, output is non-divergent
        cmap_params = _determine_cmap_params(neg, vmin=-0.2, vmax=0.6)
        assert cmap_params["vmin"] == -0.2
        assert cmap_params["vmax"] == 0.6
        assert cmap_params["cmap"] == "viridis"

        # regression test for GH3524
        # infer diverging colormap from divergent levels
        cmap_params = _determine_cmap_params(pos, levels=[-0.1, 0, 1])
        # specifying levels makes cmap a Colormap object
        assert cmap_params["cmap"].name == "RdBu_r"

    def test_norm_sets_vmin_vmax(self) -> None:
        vmin = self.data.min()
        vmax = self.data.max()

        for norm, extend, levels in zip(
            [
                mpl.colors.Normalize(),
                mpl.colors.Normalize(),
                mpl.colors.Normalize(vmin + 0.1, vmax - 0.1),
                mpl.colors.Normalize(None, vmax - 0.1),
                mpl.colors.Normalize(vmin + 0.1, None),
            ],
            ["neither", "neither", "both", "max", "min"],
            [7, None, None, None, None],
        ):
            test_min = vmin if norm.vmin is None else norm.vmin
            test_max = vmax if norm.vmax is None else norm.vmax

            cmap_params = _determine_cmap_params(self.data, norm=norm, levels=levels)
            assert cmap_params["vmin"] is None
            assert cmap_params["vmax"] is None
            assert cmap_params["norm"].vmin == test_min
            assert cmap_params["norm"].vmax == test_max
            assert cmap_params["extend"] == extend
            assert cmap_params["norm"] == norm


@requires_matplotlib
def test_get_axis_current() -> None:
    with figure_context():
        _, ax = plt.subplots()
        out_ax = get_axis()
        assert ax is out_ax


@requires_matplotlib
def test_maybe_gca() -> None:
    with figure_context():
        ax = _maybe_gca(aspect=1)

        assert isinstance(ax, mpl.axes.Axes)
        assert ax.get_aspect() == 1

    with figure_context():
        # create figure without axes
        plt.figure()
        ax = _maybe_gca(aspect=1)

        assert isinstance(ax, mpl.axes.Axes)
        assert ax.get_aspect() == 1

    with figure_context():
        existing_axes = plt.axes()
        ax = _maybe_gca(aspect=1)

        # re-uses the existing axes
        assert existing_axes == ax
        # kwargs are ignored when reusing axes
        assert ax.get_aspect() == "auto"


class TestPlot(PlotTestCase):
    @pytest.fixture(autouse=True)
    def setup_array(self) -> None:
        self.darray = DataArray(easy_array((2, 3, 4)))

    def test_accessor(self) -> None:
        from xarray.plot.accessor import DataArrayPlotAccessor

        assert DataArray.plot is DataArrayPlotAccessor
        assert isinstance(self.darray.plot, DataArrayPlotAccessor)

    def test_label_from_attrs(self) -> None:
        da = self.darray.copy()
        assert "" == label_from_attrs(da)

        da.name = 0
        assert "0" == label_from_attrs(da)

        da.name = "a"
        da.attrs["units"] = "a_units"
        da.attrs["long_name"] = "a_long_name"
        da.attrs["standard_name"] = "a_standard_name"
        assert "a_long_name [a_units]" == label_from_attrs(da)

        da.attrs.pop("long_name")
        assert "a_standard_name [a_units]" == label_from_attrs(da)
        da.attrs.pop("units")
        assert "a_standard_name" == label_from_attrs(da)

        da.attrs["units"] = "a_units"
        da.attrs.pop("standard_name")
        assert "a [a_units]" == label_from_attrs(da)

        da.attrs.pop("units")
        assert "a" == label_from_attrs(da)

        # Latex strings can be longer without needing a new line:
        long_latex_name = r"$Ra_s = \mathrm{mean}(\epsilon_k) / \mu M^2_\infty$"
        da.attrs = {"long_name": long_latex_name}
        assert label_from_attrs(da) == long_latex_name
