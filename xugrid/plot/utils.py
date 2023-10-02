"""
The full content is this module is copied from xarray.plot.utils.py. Unused
parts are removed, so this module only contains about a third of the original's
content.

Additionally, _easy_facetgrid has been copied from xarray.plot.facetgrid.

The reason is that these functions are all private methods. Hence, Xarray
provides no guarantees on breaking changes.

We heavily discourage editing this file. Any update should only consist of
copying updated parts of the xarray module.

Some minor edits are necessary. These are marked by an comment with "EDIT:".

Xarray is licensed under Apache License 2.0:
https://github.com/pydata/xarray/blob/main/LICENSE
"""
from __future__ import annotations

import importlib
import textwrap
from collections.abc import Callable, Hashable, Iterable, Mapping
from datetime import datetime
from inspect import getfullargspec
from typing import TYPE_CHECKING, Any, Literal, TypeVar, Union

import numpy as np
from xarray import DataArray, Dataset
from xarray.plot.facetgrid import FacetGrid

# EDIT: added these here
AspectOptions = Union[Literal["auto", "equal"], float, None]
ScaleOptions = Literal["linear", "symlog", "log", "logit", None]
T_DataArrayOrSet = TypeVar("T_DataArrayOrSet", bound=Union["Dataset", "DataArray"])
nc_time_axis_available = importlib.util.find_spec("nc_time_axis") is not None


try:
    import cftime
except ImportError:
    cftime = None


if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt: Any = None  # type: ignore

ROBUST_PERCENTILE = 2.0


__all__ = [
    "_add_colorbar",
    "_ensure_plottable",
    "_process_cmap_cbar_kwargs",
    "_update_axes",
    "get_axis",
    "label_from_attrs",
    "_easy_facetgrid",
]


def _determine_extend(calc_data, vmin, vmax):
    extend_min = calc_data.min() < vmin
    extend_max = calc_data.max() > vmax
    if extend_min and extend_max:
        return "both"
    elif extend_min:
        return "min"
    elif extend_max:
        return "max"
    else:
        return "neither"


def _build_discrete_cmap(cmap, levels, extend, filled):
    """
    Build a discrete colormap and normalization of the data.
    """
    import matplotlib as mpl

    if len(levels) == 1:
        levels = [levels[0], levels[0]]

    if not filled:
        # non-filled contour plots
        extend = "max"

    if extend == "both":
        ext_n = 2
    elif extend in ["min", "max"]:
        ext_n = 1
    else:
        ext_n = 0

    n_colors = len(levels) + ext_n - 1
    pal = _color_palette(cmap, n_colors)

    new_cmap, cnorm = mpl.colors.from_levels_and_colors(levels, pal, extend=extend)
    # copy the old cmap name, for easier testing
    new_cmap.name = getattr(cmap, "name", cmap)

    # copy colors to use for bad, under, and over values in case they have been
    # set to non-default values
    try:
        # matplotlib<3.2 only uses bad color for masked values
        bad = cmap(np.ma.masked_invalid([np.nan]))[0]
    except TypeError:
        # cmap was a str or list rather than a color-map object, so there are
        # no bad, under or over values to check or copy
        pass
    else:
        under = cmap(-np.inf)
        over = cmap(np.inf)

        new_cmap.set_bad(bad)

        # Only update under and over if they were explicitly changed by the user
        # (i.e. are different from the lowest or highest values in cmap). Otherwise
        # leave unchanged so new_cmap uses its default values (its own lowest and
        # highest values).
        if under != cmap(0):
            new_cmap.set_under(under)
        if over != cmap(cmap.N - 1):
            new_cmap.set_over(over)

    return new_cmap, cnorm


def _color_palette(cmap, n_colors):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    colors_i = np.linspace(0, 1.0, n_colors)
    if isinstance(cmap, (list, tuple)):
        # we have a list of colors
        cmap = ListedColormap(cmap, N=n_colors)
        pal = cmap(colors_i)
    elif isinstance(cmap, str):
        # we have some sort of named palette
        try:
            # is this a matplotlib cmap?
            cmap = plt.get_cmap(cmap)
            pal = cmap(colors_i)
        except ValueError:
            # ValueError happens when mpl doesn't like a colormap, try seaborn
            try:
                from seaborn import color_palette

                pal = color_palette(cmap, n_colors=n_colors)
            except (ValueError, ImportError):
                # or maybe we just got a single color as a string
                cmap = ListedColormap([cmap], N=n_colors)
                pal = cmap(colors_i)
    else:
        # cmap better be a LinearSegmentedColormap (e.g. viridis)
        pal = cmap(colors_i)

    return pal


# _determine_cmap_params is adapted from Seaborn:
# https://github.com/mwaskom/seaborn/blob/v0.6/seaborn/matrix.py#L158
# Used under the terms of Seaborn's license, see licenses/SEABORN_LICENSE.


def _determine_cmap_params(
    plot_data,
    vmin=None,
    vmax=None,
    cmap=None,
    center=None,
    robust=False,
    extend=None,
    levels=None,
    filled=True,
    norm=None,
    _is_facetgrid=False,
):
    """
    Use some heuristics to set good defaults for colorbar and range.

    Parameters
    ----------
    plot_data : Numpy array
        Doesn't handle xarray objects

    Returns
    -------
    cmap_params : dict
        Use depends on the type of the plotting function
    """
    import matplotlib as mpl

    if isinstance(levels, Iterable):
        levels = sorted(levels)

    calc_data = np.ravel(plot_data[np.isfinite(plot_data)])

    # Handle all-NaN input data gracefully
    if calc_data.size == 0:
        # Arbitrary default for when all values are NaN
        calc_data = np.array(0.0)

    # Setting center=False prevents a divergent cmap
    possibly_divergent = center is not False

    # Set center to 0 so math below makes sense but remember its state
    center_is_none = False
    if center is None:
        center = 0
        center_is_none = True

    # Setting both vmin and vmax prevents a divergent cmap
    if (vmin is not None) and (vmax is not None):
        possibly_divergent = False

    # Setting vmin or vmax implies linspaced levels
    user_minmax = (vmin is not None) or (vmax is not None)

    # vlim might be computed below
    vlim = None

    # save state; needed later
    vmin_was_none = vmin is None
    vmax_was_none = vmax is None

    if vmin is None:
        if robust:
            vmin = np.percentile(calc_data, ROBUST_PERCENTILE)
        else:
            vmin = calc_data.min()
    elif possibly_divergent:
        vlim = abs(vmin - center)

    if vmax is None:
        if robust:
            vmax = np.percentile(calc_data, 100 - ROBUST_PERCENTILE)
        else:
            vmax = calc_data.max()
    elif possibly_divergent:
        vlim = abs(vmax - center)

    if possibly_divergent:
        levels_are_divergent = (
            isinstance(levels, Iterable) and levels[0] * levels[-1] < 0
        )
        # kwargs not specific about divergent or not: infer defaults from data
        divergent = (
            ((vmin < 0) and (vmax > 0)) or not center_is_none or levels_are_divergent
        )
    else:
        divergent = False

    # A divergent map should be symmetric around the center value
    if divergent:
        if vlim is None:
            vlim = max(abs(vmin - center), abs(vmax - center))
        vmin, vmax = -vlim, vlim

    # Now add in the centering value and set the limits
    vmin += center
    vmax += center

    # now check norm and harmonize with vmin, vmax
    if norm is not None:
        if norm.vmin is None:
            norm.vmin = vmin
        else:
            if not vmin_was_none and vmin != norm.vmin:
                raise ValueError("Cannot supply vmin and a norm with a different vmin.")
            vmin = norm.vmin

        if norm.vmax is None:
            norm.vmax = vmax
        else:
            if not vmax_was_none and vmax != norm.vmax:
                raise ValueError("Cannot supply vmax and a norm with a different vmax.")
            vmax = norm.vmax

    # if BoundaryNorm, then set levels
    if isinstance(norm, mpl.colors.BoundaryNorm):
        levels = norm.boundaries

    # Choose default colormaps if not provided
    if cmap is None:
        # EDIT: replaced OPTIONS
        if divergent:
            cmap = "RdBu_r"
        else:
            cmap = "viridis"

    # Handle discrete levels
    if levels is not None:
        # EDIT: replaced is_scalar(levels) by np.issubdtype(type(levels), np.integer)
        # xarray uses a relatelively complicated is_scalar function here. For
        # the purposes of plotting, checking for an integer should do.
        if np.issubdtype(type(levels), np.integer):
            if user_minmax:
                levels = np.linspace(vmin, vmax, levels)
            elif levels == 1:
                levels = np.asarray([(vmin + vmax) / 2])
            else:
                # N in MaxNLocator refers to bins, not ticks
                ticker = mpl.ticker.MaxNLocator(levels - 1)
                levels = ticker.tick_values(vmin, vmax)
        vmin, vmax = levels[0], levels[-1]

    # GH3734
    if vmin == vmax:
        vmin, vmax = mpl.ticker.LinearLocator(2).tick_values(vmin, vmax)

    if extend is None:
        extend = _determine_extend(calc_data, vmin, vmax)

    if (levels is not None) and (not isinstance(norm, mpl.colors.BoundaryNorm)):
        cmap, newnorm = _build_discrete_cmap(cmap, levels, extend, filled)
        norm = newnorm if norm is None else norm

    # vmin & vmax needs to be None if norm is passed
    # TODO: always return a norm with vmin and vmax
    if norm is not None:
        vmin = None
        vmax = None

    return dict(
        vmin=vmin, vmax=vmax, cmap=cmap, extend=extend, levels=levels, norm=norm
    )


def get_axis(
    figsize: Iterable[float] | None = None,
    size: float | None = None,
    aspect: AspectOptions = None,
    ax: Axes | None = None,
    **subplot_kws: Any,
) -> Axes:
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plot.utils.get_axis")

    if figsize is not None:
        if ax is not None:
            raise ValueError("cannot provide both `figsize` and `ax` arguments")
        if size is not None:
            raise ValueError("cannot provide both `figsize` and `size` arguments")
        _, ax = plt.subplots(figsize=figsize, subplot_kw=subplot_kws)
        return ax

    if size is not None:
        if ax is not None:
            raise ValueError("cannot provide both `size` and `ax` arguments")
        if aspect is None or aspect == "auto":
            width, height = mpl.rcParams["figure.figsize"]
            faspect = width / height
        elif aspect == "equal":
            faspect = 1
        else:
            faspect = aspect
        figsize = (size * faspect, size)
        _, ax = plt.subplots(figsize=figsize, subplot_kw=subplot_kws)
        return ax

    if aspect is not None:
        raise ValueError("cannot provide `aspect` argument without `size`")

    if subplot_kws and ax is not None:
        raise ValueError("cannot use subplot_kws with existing ax")

    if ax is None:
        ax = _maybe_gca(**subplot_kws)

    return ax


def _maybe_gca(**subplot_kws: Any) -> Axes:
    import matplotlib.pyplot as plt

    # can call gcf unconditionally: either it exists or would be created by plt.axes
    f = plt.gcf()

    # only call gca if an active axes exists
    if f.axes:
        # can not pass kwargs to active axes
        return plt.gca()

    return plt.axes(**subplot_kws)


def _get_units_from_attrs(da: DataArray) -> str:
    """Extracts and formats the unit/units from a attributes."""
    # EDIT: removed pint support for now.
    # pint_array_type = DuckArrayModule("pint").type
    units = " [{}]"
    # if isinstance(da.data, pint_array_type):
    #    return units.format(str(da.data.units))
    if "units" in da.attrs:
        return units.format(da.attrs["units"])
    if "unit" in da.attrs:
        return units.format(da.attrs["unit"])
    return ""


def label_from_attrs(da: DataArray | None, extra: str = "") -> str:
    """Makes informative labels if variable metadata (attrs) follows
    CF conventions."""
    if da is None:
        return ""

    name: str = "{}"
    if "long_name" in da.attrs:
        name = name.format(da.attrs["long_name"])
    elif "standard_name" in da.attrs:
        name = name.format(da.attrs["standard_name"])
    elif da.name is not None:
        name = name.format(da.name)
    else:
        name = ""

    units = _get_units_from_attrs(da)

    # Treat `name` differently if it's a latex sequence
    if name.startswith("$") and (name.count("$") % 2 == 0):
        return "$\n$".join(
            textwrap.wrap(name + extra + units, 60, break_long_words=False)
        )
    else:
        return "\n".join(textwrap.wrap(name + extra + units, 30))


def _valid_other_type(
    x: ArrayLike, types: type[object] | tuple[type[object], ...]
) -> bool:
    """
    Do all elements of x have a type from types?
    """
    return all(isinstance(el, types) for el in np.ravel(x))


def _valid_numpy_subdtype(x, numpy_types):
    """
    Is any dtype from numpy_types superior to the dtype of x?
    """
    # If any of the types given in numpy_types is understood as numpy.generic,
    # all possible x will be considered valid.  This is probably unwanted.
    for t in numpy_types:
        assert not np.issubdtype(np.generic, t)

    return any(np.issubdtype(x.dtype, t) for t in numpy_types)


def _ensure_plottable(*args) -> None:
    """
    Raise exception if there is anything in args that can't be plotted on an
    axis by matplotlib.
    """
    numpy_types: tuple[type[object], ...] = (
        np.floating,
        np.integer,
        np.timedelta64,
        np.datetime64,
        np.bool_,
        np.str_,
    )
    other_types: tuple[type[object], ...] = (datetime,)
    cftime_datetime_types: tuple[type[object], ...] = (
        () if cftime is None else (cftime.datetime,)
    )
    other_types += cftime_datetime_types

    for x in args:
        if not (
            _valid_numpy_subdtype(np.asarray(x), numpy_types)
            or _valid_other_type(np.asarray(x), other_types)
        ):
            raise TypeError(
                "Plotting requires coordinates to be numeric, boolean, "
                "or dates of type numpy.datetime64, "
                "datetime.datetime, cftime.datetime or "
                f"pandas.Interval. Received data of type {np.asarray(x).dtype} instead."
            )
        if _valid_other_type(np.asarray(x), cftime_datetime_types):
            if nc_time_axis_available:
                # Register cftime datetypes to matplotlib.units.registry,
                # otherwise matplotlib will raise an error:
                import nc_time_axis  # noqa: F401
            else:
                raise ImportError(
                    "Plotting of arrays of cftime.datetime "
                    "objects or arrays indexed by "
                    "cftime.datetime objects requires the "
                    "optional `nc-time-axis` (v1.2.0 or later) "
                    "package."
                )


def _add_colorbar(primitive, ax, cbar_ax, cbar_kwargs, cmap_params):
    cbar_kwargs.setdefault("extend", cmap_params["extend"])
    if cbar_ax is None:
        cbar_kwargs.setdefault("ax", ax)
    else:
        cbar_kwargs.setdefault("cax", cbar_ax)

    # dont pass extend as kwarg if it is in the mappable
    if hasattr(primitive, "extend"):
        cbar_kwargs.pop("extend")

    fig = ax.get_figure()
    cbar = fig.colorbar(primitive, **cbar_kwargs)

    return cbar


def _update_axes(
    ax: Axes,
    xincrease: bool | None,
    yincrease: bool | None,
    xscale: ScaleOptions = None,
    yscale: ScaleOptions = None,
    xticks: ArrayLike | None = None,
    yticks: ArrayLike | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """
    Update axes with provided parameters
    """
    if xincrease is None:
        pass
    elif xincrease and ax.xaxis_inverted():
        ax.invert_xaxis()
    elif not xincrease and not ax.xaxis_inverted():
        ax.invert_xaxis()

    if yincrease is None:
        pass
    elif yincrease and ax.yaxis_inverted():
        ax.invert_yaxis()
    elif not yincrease and not ax.yaxis_inverted():
        ax.invert_yaxis()

    # The default xscale, yscale needs to be None.
    # If we set a scale it resets the axes formatters,
    # This means that set_xscale('linear') on a datetime axis
    # will remove the date labels. So only set the scale when explicitly
    # asked to. https://github.com/matplotlib/matplotlib/issues/8740
    if xscale is not None:
        ax.set_xscale(xscale)
    if yscale is not None:
        ax.set_yscale(yscale)

    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)


def _process_cmap_cbar_kwargs(
    func,
    data,
    cmap=None,
    colors=None,
    cbar_kwargs: Iterable[tuple[str, Any]] | Mapping[str, Any] | None = None,
    levels=None,
    _is_facetgrid=False,
    **kwargs,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Parameters
    ----------
    func : plotting function
    data : ndarray,
        Data values

    Returns
    -------
    cmap_params : dict
    cbar_kwargs : dict
    """
    if func.__name__ == "surface":
        # Leave user to specify cmap settings for surface plots
        kwargs["cmap"] = cmap
        return {
            k: kwargs.get(k, None)
            for k in ["vmin", "vmax", "cmap", "extend", "levels", "norm"]
        }, {}

    cbar_kwargs = {} if cbar_kwargs is None else dict(cbar_kwargs)

    if "contour" in func.__name__ and levels is None:
        levels = 7  # this is the matplotlib default

    # colors is mutually exclusive with cmap
    if cmap and colors:
        raise ValueError("Can't specify both cmap and colors.")

    # colors is only valid when levels is supplied or the plot is of type
    # contour or contourf
    if colors and (("contour" not in func.__name__) and (levels is None)):
        raise ValueError("Can only specify colors with contour or levels")

    # we should not be getting a list of colors in cmap anymore
    # is there a better way to do this test?
    if isinstance(cmap, (list, tuple)):
        raise ValueError(
            "Specifying a list of colors in cmap is deprecated. "
            "Use colors keyword instead."
        )

    cmap_kwargs = {
        "plot_data": data,
        "levels": levels,
        "cmap": colors if colors else cmap,
        "filled": func.__name__ != "contour",
    }

    cmap_args = getfullargspec(_determine_cmap_params).args
    cmap_kwargs.update((a, kwargs[a]) for a in cmap_args if a in kwargs)
    if not _is_facetgrid:
        cmap_params = _determine_cmap_params(**cmap_kwargs)
    else:
        cmap_params = {
            k: cmap_kwargs[k]
            for k in ["vmin", "vmax", "cmap", "extend", "levels", "norm"]
        }

    return cmap_params, cbar_kwargs


def _easy_facetgrid(
    data: T_DataArrayOrSet,
    plotfunc: Callable,
    kind: Literal["line", "dataarray", "dataset", "plot1d"],
    x: Hashable | None = None,
    y: Hashable | None = None,
    row: Hashable | None = None,
    col: Hashable | None = None,
    col_wrap: int | None = None,
    sharex: bool = True,
    sharey: bool = True,
    aspect: float | None = None,
    size: float | None = None,
    subplot_kws: dict[str, Any] | None = None,
    ax: Axes | None = None,
    figsize: Iterable[float] | None = None,
    **kwargs: Any,
) -> FacetGrid[T_DataArrayOrSet]:
    """
    Convenience method to call xarray.plot.FacetGrid from 2d plotting methods

    kwargs are the arguments to 2d plotting method
    """
    if ax is not None:
        raise ValueError("Can't use axes when making faceted plots.")
    if aspect is None:
        aspect = 1
    if size is None:
        size = 3
    elif figsize is not None:
        raise ValueError("cannot provide both `figsize` and `size` arguments")
    if kwargs.get("z") is not None:
        # 3d plots doesn't support sharex, sharey, reset to mpl defaults:
        sharex = False
        sharey = False

    g = FacetGrid(
        data=data,
        col=col,
        row=row,
        col_wrap=col_wrap,
        sharex=sharex,
        sharey=sharey,
        figsize=figsize,
        aspect=aspect,
        size=size,
        subplot_kws=subplot_kws,
    )

    if kind == "line":
        return g.map_dataarray_line(plotfunc, x, y, **kwargs)

    if kind == "dataarray":
        return g.map_dataarray(plotfunc, x, y, **kwargs)

    if kind == "plot1d":
        return g.map_plot1d(plotfunc, x, y, **kwargs)

    if kind == "dataset":
        return g.map_dataset(plotfunc, x, y, **kwargs)

    raise ValueError(
        f"kind must be one of `line`, `dataarray`, `dataset` or `plot1d`, got {kind}"
    )
