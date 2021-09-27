"""
This module is strongly inspired by / copied from xarray/plot/plot.py.
"""
import functools
from typing import Tuple

import numpy as np
import xarray as xr
from matplotlib.collections import LineCollection, PolyCollection
import matplotlib as mpl
from xarray.core.utils import UncachedAccessor
from xarray.plot.facetgrid import _easy_facetgrid
from xarray.plot.utils import (
    _add_colorbar,
    _ensure_plottable,
    _process_cmap_cbar_kwargs,
    _update_axes,
    get_axis,
    import_matplotlib_pyplot,
    label_from_attrs,
)

from .typing import FloatArray, FloatDType, IntArray

Triangulation = Tuple[Tuple[FloatArray, FloatArray], IntArray]


def override_signature(f):
    def wrapper(func):
        func.__wrapped__ = f

        return func

    return wrapper


def _plot2d(plotfunc):
    """
    Decorator for common 2d plotting logic
    Also adds the 2d plot method to class _PlotMethods
    """
    commondoc = """
    Parameters
    ----------
    topology : Union[Ugrid1d, Ugrid2d, Tuple[FloatArray, FloatArray], Triangulation]
        Mesh topology.
    darray : DataArray
        Must be two-dimensional, unless creating faceted plots.
    figsize : tuple, optional
        A tuple (width, height) of the figure in inches.
        Mutually exclusive with ``size`` and ``ax``.
    aspect : scalar, optional
        Aspect ratio of plot, so that ``aspect * size`` gives the *width* in
        inches. Only used if a ``size`` is provided.
    size : scalar, optional
        If provided, create a new figure for the plot with the given size:
        *height* (in inches) of each plot. See also: ``aspect``.
    ax : matplotlib axes object, optional
        Axes on which to plot. By default, use the current axes.
        Mutually exclusive with ``size`` and ``figsize``.
    row : string, optional
        If passed, make row faceted plots on this dimension name.
    col : string, optional
        If passed, make column faceted plots on this dimension name.
    col_wrap : int, optional
        Use together with ``col`` to wrap faceted plots.
    xticks, yticks : array-like, optional
        Specify tick locations for *x*- and *y*-axis.
    xlim, ylim : array-like, optional
        Specify *x*- and *y*-axis limits.
    xincrease : None, True, or False, optional
        Should the values on the *x* axis be increasing from left to right?
        If ``None``, use the default for the Matplotlib function.
    yincrease : None, True, or False, optional
        Should the values on the *y* axis be increasing from top to bottom?
        If ``None``, use the default for the Matplotlib function.
    add_colorbar : bool, optional
        Add colorbar to axes.
    add_labels : bool, optional
        Use xarray metadata to label axes.
    norm : matplotlib.colors.Normalize, optional
        If ``norm`` has ``vmin`` or ``vmax`` specified, the corresponding
        kwarg must be ``None``.
    vmin, vmax : float, optional
        Values to anchor the colormap, otherwise they are inferred from the
        data and other keyword arguments. When a diverging dataset is inferred,
        setting one of these values will fix the other by symmetry around
        ``center``. Setting both values prevents use of a diverging colormap.
        If discrete levels are provided as an explicit list, both of these
        values are ignored.
    cmap : matplotlib colormap name or colormap, optional
        The mapping from data values to color space. If not provided, this
        will be either be ``'viridis'`` (if the function infers a sequential
        dataset) or ``'RdBu_r'`` (if the function infers a diverging dataset).
        See :doc:`Choosing Colormaps in Matplotlib <matplotlib:tutorials/colors/colormaps>`
        for more information.
        If *seaborn* is installed, ``cmap`` may also be a
        `seaborn color palette <https://seaborn.pydata.org/tutorial/color_palettes.html>`_.
        Note: if ``cmap`` is a seaborn color palette and the plot type
        is not ``'contour'`` or ``'contourf'``, ``levels`` must also be specified.
    colors : str or array-like of color-like, optional
        A single color or a sequence of colors. If the plot type is not ``'contour'``
        or ``'contourf'``, the ``levels`` argument is required.
    center : float, optional
        The value at which to center the colormap. Passing this value implies
        use of a diverging colormap. Setting it to ``False`` prevents use of a
        diverging colormap.
    robust : bool, optional
        If ``True`` and ``vmin`` or ``vmax`` are absent, the colormap range is
        computed with 2nd and 98th percentiles instead of the extreme values.
    extend : {'neither', 'both', 'min', 'max'}, optional
        How to draw arrows extending the colorbar beyond its limits. If not
        provided, ``extend`` is inferred from ``vmin``, ``vmax`` and the data limits.
    levels : int or array-like, optional
        Split the colormap (``cmap``) into discrete color intervals. If an integer
        is provided, "nice" levels are chosen based on the data range: this can
        imply that the final number of levels is not exactly the expected one.
        Setting ``vmin`` and/or ``vmax`` with ``levels=N`` is equivalent to
        setting ``levels=np.linspace(vmin, vmax, N)``.
    infer_intervals : bool, optional
        Only applies to pcolormesh. If ``True``, the coordinate intervals are
        passed to pcolormesh. If ``False``, the original coordinates are used
        (this can be useful for certain map projections). The default is to
        always infer intervals, unless the mesh is irregular and plotted on
        a map projection.
    subplot_kws : dict, optional
        Dictionary of keyword arguments for Matplotlib subplots. Only used
        for 2D and faceted plots.
        (see :py:meth:`matplotlib:matplotlib.figure.Figure.add_subplot`).
    cbar_ax : matplotlib axes object, optional
        Axes in which to draw the colorbar.
    cbar_kwargs : dict, optional
        Dictionary of keyword arguments to pass to the colorbar
        (see :meth:`matplotlib:matplotlib.figure.Figure.colorbar`).
    **kwargs : optional
        Additional keyword arguments to wrapped Matplotlib function.
    Returns
    -------
    artist :
        The same type of primitive artist that the wrapped Matplotlib
        function returns.
    """

    # Build on the original docstring
    plotfunc.__doc__ = f"{plotfunc.__doc__}\n{commondoc}"

    # plotfunc and newplotfunc have different signatures:
    # - plotfunc: (x, y, z, ax, **kwargs)
    # - newplotfunc: (darray, x, y, **kwargs)
    # where plotfunc accepts numpy arrays, while newplotfunc accepts a DataArray
    # and variable names. newplotfunc also explicitly lists most kwargs, so we
    # need to shorten it
    def signature(topology, darray, **kwargs):
        pass

    @override_signature(signature)
    @functools.wraps(plotfunc)
    def newplotfunc(
        topology,
        darray=None,
        figsize=None,
        size=None,
        aspect=None,
        ax=None,
        row=None,
        col=None,
        col_wrap=None,
        xincrease=True,
        yincrease=True,
        add_colorbar=None,
        add_labels=True,
        vmin=None,
        vmax=None,
        cmap=None,
        center=None,
        robust=False,
        extend=None,
        levels=None,
        colors=None,
        subplot_kws=None,
        cbar_ax=None,
        cbar_kwargs=None,
        xticks=None,
        yticks=None,
        xlim=None,
        ylim=None,
        norm=None,
        **kwargs,
    ):
        # All 2d plots in xarray share this function signature.
        # Method signature below should be consistent.

        # Decide on a default for the colorbar before facetgrids
        if add_colorbar is None:
            add_colorbar = True
            if (
                darray is None
                or plotfunc.__name__ == "contour"
                or (plotfunc.__name__ == "surface" and cmap is None)
            ):
                add_colorbar = False

        if subplot_kws is None:
            subplot_kws = dict()

        if plotfunc.__name__ == "surface" and not kwargs.get("_is_facetgrid", False):
            if ax is None:
                # TODO: Importing Axes3D is no longer necessary in matplotlib >= 3.2.
                # Remove when minimum requirement of matplotlib is 3.2:
                from mpl_toolkits.mplot3d import Axes3D  # type: ignore  # noqa: F401

                # delete so it does not end up in locals()
                del Axes3D

                # Need to create a "3d" Axes instance for surface plots
                subplot_kws["projection"] = "3d"

            # In facet grids, shared axis labels don't make sense for surface plots
            sharex = False
            sharey = False

        # Handle facetgrids first
        if row or col:
            if darray is None:
                raise ValueError(
                    "Cannot create facetgrid with only topology and no data."
                )
            allargs = locals().copy()
            del allargs["darray"]
            del allargs["imshow_rgb"]
            allargs.update(allargs.pop("kwargs"))
            # Need the decorated plotting function
            allargs["plotfunc"] = globals()[plotfunc.__name__]
            return _easy_facetgrid(darray, kind="dataarray", **allargs)

        plt = import_matplotlib_pyplot()

        # For 3d plot, ensure given ax is a Axes3D object
        if (
            plotfunc.__name__ == "surface"
            and not kwargs.get("_is_facetgrid", False)
            and ax is not None
        ):
            import mpl_toolkits  # type: ignore

            if not isinstance(ax, mpl_toolkits.mplot3d.Axes3D):
                raise ValueError(
                    "If ax is passed to surface(), it must be created with "
                    'projection="3d"'
                )

        # darray may be None when plotting just edges (the mesh)
        if darray is not None:
            _ensure_plottable(darray.values)
            cmap_params, cbar_kwargs = _process_cmap_cbar_kwargs(
                plotfunc,
                darray,
                **locals(),
                _is_facetgrid=kwargs.pop("_is_facetgrid", False),
            )
        else:
            cmap_params = dict()
            cbar_kwargs = dict()

        if "contour" in plotfunc.__name__:
            # extend is a keyword argument only for contour and contourf, but
            # passing it to the colorbar is sufficient for imshow and
            # pcolormesh
            kwargs["extend"] = cmap_params["extend"]
            kwargs["levels"] = cmap_params["levels"]
            # if colors == a single color, matplotlib draws dashed negative
            # contours. we lose this feature if we pass cmap and not colors
            if isinstance(colors, str):
                cmap_params["cmap"] = dict()
                kwargs["colors"] = colors

        if "imshow" == plotfunc.__name__ and isinstance(aspect, str):
            # forbid usage of mpl strings
            raise ValueError("plt.imshow's `aspect` kwarg is not available in xarray")

        for key in ["cmap", "vmin", "vmax", "norm"]:
            kwargs[key] = cmap_params.get(key)

        ax = get_axis(figsize, size, aspect, ax, **subplot_kws)
        primitive = plotfunc(
            topology,
            darray,
            ax=ax,
            **kwargs,
        )
        
        # Try to get a 1:1 ratio between x and y coordinates by default. If
        # colorbar is present, we need to make room for it in the x-direction.
        # 1.26 is the magic number; the colorbar takes up 26% additional space
        # by default.
        if aspect is None:
            if add_colorbar:
                aspect = 1.26
            else:
                aspect = 1.0

        # Preserve height, adjust width if needed; Do not call
        # ax.set_aspect: this shrinks or grows the ax relative to the
        # colobar
        if size is None:
            _, size = ax.figure.get_size_inches()

        figsize = (size * aspect, size)
        ax.figure.set_size_inches(figsize)

        # Label the plot with metadata
        if darray is not None and add_labels:
            # TODO: grab x and y information from topology?
            # ax.set_xlabel(label_from_attrs(darray[xlab], xlab_extra))
            # ax.set_ylabel(label_from_attrs(darray[ylab], ylab_extra))
            ax.set_title(darray._title_for_slice())
            if plotfunc.__name__ == "surface":
                ax.set_zlabel(label_from_attrs(darray))

        if add_colorbar:
            if add_labels and "label" not in cbar_kwargs:
                cbar_kwargs["label"] = label_from_attrs(darray)
            cbar = _add_colorbar(primitive, ax, cbar_ax, cbar_kwargs, cmap_params)
        elif cbar_ax is not None or cbar_kwargs:
            # inform the user about keywords which aren't used
            raise ValueError(
                "cbar_ax and cbar_kwargs can't be used with add_colorbar=False."
            )

        # origin kwarg overrides yincrease
        if "origin" in kwargs:
            yincrease = None

        # Spatial x and y coordinates: no need for e.g. logarithm axes.
        xscale = None
        yscale = None
        _update_axes(
            ax, xincrease, yincrease, xscale, yscale, xticks, yticks, xlim, ylim
        )

        return primitive

    # For use as DataArray.plot.plotmethod
    @functools.wraps(newplotfunc)
    def plotmethod(
        topology,
        darray,
        figsize=None,
        size=None,
        aspect=None,
        ax=None,
        row=None,
        col=None,
        col_wrap=None,
        xincrease=True,
        yincrease=True,
        add_colorbar=None,
        add_labels=True,
        vmin=None,
        vmax=None,
        cmap=None,
        colors=None,
        center=None,
        robust=False,
        extend=None,
        levels=None,
        subplot_kws=None,
        cbar_ax=None,
        cbar_kwargs=None,
        xscale=None,
        yscale=None,
        xticks=None,
        yticks=None,
        xlim=None,
        ylim=None,
        norm=None,
        **kwargs,
    ):
        """
        The method should have the same signature as the function.
        This just makes the method work on Plotmethods objects,
        and passes all the other arguments straight through.
        """
        allargs = locals()
        allargs.update(kwargs)
        for arg in ["_PlotMethods_obj", "newplotfunc", "kwargs"]:
            del allargs[arg]
        return newplotfunc(**allargs)

    return newplotfunc


@_plot2d
def scatter(xy, z, ax, **kwargs):
    x, y = xy
    primitive = ax.scatter(x, y, z.values.ravel(), **kwargs)
    return primitive


@_plot2d
def tripcolor(triangulation, z, ax, **kwargs):
    primitive = ax.tripcolor(*triangulation, z.values.ravel(), **kwargs)
    return primitive


@_plot2d
def line(grid, z, ax, **kwargs):
    edge_nodes = grid.edge_node_connectivity
    n_edge = len(edge_nodes)
    edge_coords = np.empty((n_edge, 2, 2), dtype=FloatDType)
    node_0 = edge_nodes[:, 0]
    node_1 = edge_nodes[:, 1]
    edge_coords[:, 0, 0] = grid.node_x[node_0]
    edge_coords[:, 0, 1] = grid.node_y[node_0]
    edge_coords[:, 1, 0] = grid.node_x[node_1]
    edge_coords[:, 1, 1] = grid.node_y[node_1]

    norm = kwargs.pop("norm", None)
    vmin = kwargs.pop("vmin", None)
    vmax = kwargs.pop("vmax", None)

    collection = LineCollection(edge_coords, **kwargs)

    if z is not None:
        dim = z.dims[0]
        attrs = grid.mesh_topology.attrs
        if dim == attrs.get("edge_dimension", "edge"):
            collection.set_array(z.values)
            collection._scale_norm(norm, vmin, vmax)

    primitive = ax.add_collection(collection, autolim=False)
    
    xmin, ymin, xmax, ymax = grid.bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    return primitive


@_plot2d
def imshow(grid, z, ax, **kwargs):
    """
    Image plot of 2D DataArray.
    Wraps :py:func:`matplotlib:matplotlib.pyplot.imshow`.

    This rasterizes the grid before plotting. Pass a ``resolution`` keyword to
    control the rasterization resolution.
    """
    if "extent" not in kwargs:
        xmin, ymin, xmax, ymax = grid.bounds
        kwargs["extent"] = xmin, xmax, ymin, ymax
    else:
        if kwargs.get("origin", None) == "upper":
            xmin, xmax, ymin, ymax = kwargs["extent"]
        else:
            xmin, xmax, ymax, ymin = kwargs["extent"]
    
    dx = xmax - xmin
    dy = ymax - ymin
    
    # Check if a rasterization resolution is passed; Default to 500 raster
    # cells otherwise for the smallest axis.
    resolution = kwargs.get("resolution", None)
    if resolution is None:
        resolution = min(dx, dy) / 500

    _, _, index = grid.rasterize(resolution)
    img = z.values[index].astype(float)
    img[index == -1] = np.nan
    primitive = ax.imshow(img, **kwargs)
    return primitive


@_plot2d
def contour(triangulation, z, ax, **kwargs):
    """
    Filled contour plot of 2D UgridDataArray.
    Wraps :py:func:`matplotlib:matplotlib.pyplot.tricontour`.
    """
    primitive = ax.tricontour(*triangulation, z.values.ravel(), **kwargs)
    return primitive


@_plot2d
def contourf(triangulation, z, ax, **kwargs):
    """
    Filled contour plot of 2D UgridDataArray.
    Wraps :py:func:`matplotlib:matplotlib.pyplot.tricontourf`.
    """
    primitive = ax.tricontourf(*triangulation, z.values.ravel(), **kwargs)
    return primitive


@_plot2d
def pcolormesh(grid, z, ax, **kwargs):
    """
    Pseudocolor plot of 2D UgridDataArray.
    Wraps :py:func:`matplotlib:matplotlib.pyplot.
    """
    nodes = np.column_stack([grid.node_x, grid.node_y])
    faces = grid.face_node_connectivity
    vertices = nodes[faces]
    # Replace fill value; PolyCollection ignores NaN.
    vertices[faces == -1] = np.nan
    
    norm = kwargs.pop("norm", None)
    vmin = kwargs.pop("vmin", None)
    vmax = kwargs.pop("vmax", None)

    collection = PolyCollection(vertices, **kwargs)
    collection.set_array(z.values.ravel())
    collection._scale_norm(norm, vmin, vmax)
    primitive = ax.add_collection(collection, autolim=False)

    xmin, ymin, xmax, ymax = grid.bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    corners = (xmin, ymin), (xmax, ymax)
    ax.update_datalim(corners)

    return primitive


@_plot2d
def surface(triangulation, z, ax, **kwargs):
    """
    Surface plot of x-y UgridDataArray.
    Wraps :py:func:`matplotlib:mplot3d:plot_trisurf`.
    """
    primitive = ax.plot_trisurf(*triangulation, z.values.ravel(), **kwargs)
    return primitive


def plot(
    grid,
    darray,
    ax=None,
    **kwargs,
):
    """
    Default plot of DataArray using :py:mod:`matplotlib:matplotlib.pyplot`.

    Calls xarray plotting function based on the topology dimension of the data.

    =============== ===========================
    Dimension       Plotting function
    =============== ===========================
    Face            :py:func:`xugrid.plot.pcolormesh`
    Edge            :py:func:`xugrid.plot.line`
    Node            :py:func:`xugrid.plot.tripcolor`
    =============== ===========================

    Parameters
    ----------
    darray : DataArray
    grid: Union[Ugrid1d, Ugrid2d]
    **kwargs : optional
        Additional keyword arguments for Matplotlib.
    """
    dim = darray.dims[0]
    attrs = grid.mesh_topology.attrs
    if dim == attrs.get("face_dimension", "face"):
        plotfunc = pcolormesh
    elif dim == attrs.get("node_dimension", "node"):
        plotfunc = tripcolor
    elif dim == attrs.get("edge_dimension", "edge"):
        plotfunc = line
    else:
        raise ValueError("Data dimensions is not one of face, node, or edge dimension.")
    kwargs["ax"] = ax
    return plotfunc(grid, darray, **kwargs)


class _EdgePlot:
    __slots__ = ("_grid", "_da")

    def __init__(self, obj):
        self._grid = obj._grid
        self._da = obj._da

    def __call__(self, **kwargs):
        dim = self._da.dims[0]
        attrs = self._grid.mesh_topology.attrs
        if dim == attrs.get("edge_dimension", "edge"):
            z = self._da
        else:
            z = None
        return line(self._grid, z, **kwargs)

    @functools.wraps(line)
    def line(self, *args, **kwargs):
        return line(self._grid, self._da, *args, **kwargs)


class _FacePlot:
    __slots__ = ("_grid", "_da")

    def __init__(self, obj):
        self._grid = obj._grid
        self._da = obj._da

    def __call__(self, **kwargs):
        return pcolormesh(self._grid, self._da, **kwargs)

    @functools.wraps(imshow)
    def pcolormesh(self, *args, **kwargs):
        return pcolormesh(self._grid, self._da, *args, **kwargs)

    @functools.wraps(imshow)
    def imshow(self, *args, **kwargs):
        return imshow(self._grid, self._da, *args, **kwargs)

    @functools.wraps(contour)
    def contour(self, *args, **kwargs):
        triangulation, index = self._grid.centroid_triangulation
        z = self._da.isel(face=index)
        return contour(triangulation, z, *args, **kwargs)

    @functools.wraps(contourf)
    def contourf(self, *args, **kwargs):
        triangulation, index = self._grid.centroid_triangulation
        z = self._da.isel(face=index)
        return contourf(triangulation, z, *args, **kwargs)

    @functools.wraps(scatter)
    def scatter(self, *args, **kwargs):
        centroids = self._grid.centroids
        x = centroids[:, 0]
        y = centroids[:, 1]
        return scatter(x, y, self._da, *args, **kwargs)

    @functools.wraps(surface)
    def surface(self, *args, **kwargs):
        triangulation, index = self._grid.centroid_triangulation
        z = self._da.isel(face=index)
        return surface(triangulation, z, *args, **kwargs)


class _NodePlot:
    __slots__ = ("_grid", "_da")

    def __init__(self, obj):
        self._grid = obj._grid
        self._da = obj._da

    def __call__(self, **kwargs):
        return tripcolor(self._grid, self._da, **kwargs)

    @functools.wraps(tripcolor)
    def tripcolor(self, *args, **kwargs):
        triangulation, _ = self._grid.triangulation
        return tripcolor(triangulation, self._da, *args, **kwargs)

    @functools.wraps(scatter)
    def scatter(self, *args, **kwargs):
        x = self._grid.node_x
        y = self._grid.node_y
        return scatter(x, y, self._da, *args, **kwargs)

    @functools.wraps(contour)
    def contour(self, *args, **kwargs):
        triangulation, _ = self._grid.triangulation
        return contour(triangulation, self._da, *args, **kwargs)

    @functools.wraps(contourf)
    def contourf(self, *args, **kwargs):
        triangulation, _ = self._grid.triangulation
        return contour(triangulation, self._da, *args, **kwargs)

    @functools.wraps(surface)
    def surface(self, *args, **kwargs):
        triangulation, _ = self._grid.triangulation
        return surface(triangulation, self._da, *args, **kwargs)


class _PlotMethods:
    """
    Enables use of plot functions as attributes.
    For example UgridDataArray.plot.face.pcolormesh()
    """

    __slots__ = ("_grid", "_da")

    def __init__(self, obj):
        darray = obj.obj
        grid = obj.grid

        if isinstance(darray, xr.Dataset):
            raise NotImplementedError("Cannot plot Datasets into a facetgrid (yet)")
        if len(darray.dims) > 1:
            msg = (
                "Data contains more dimensions than just a topology dimension "
                f"(face, node, edge): {', '.join(darray.dims)}"
            )
            raise ValueError(msg)
        self._grid = grid
        self._da = darray

    face = UncachedAccessor(_FacePlot)
    edge = UncachedAccessor(_EdgePlot)
    node = UncachedAccessor(_NodePlot)

    def __call__(self, **kwargs):
        return plot(self._grid, self._da, **kwargs)
