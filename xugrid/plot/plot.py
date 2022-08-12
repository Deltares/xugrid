"""
This module is strongly inspired by / copied from xarray/plot/plot.py.
"""
import functools

import numpy as np
import xarray as xr
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

from ..typing import FloatDType

NODE = 0
EDGE = 1
FACE = 2
COORDS = [
    "node_coordinates",
    "edge_coordinates",
    "face_coordinates",
]


def get_ugrid_dim(grid, da) -> int:
    dim = da.dims[0]
    if dim == grid.node_dimension:
        return NODE
    elif dim == grid.edge_dimension:
        return EDGE
    elif dim == grid.face_dimension:
        return FACE
    else:
        allowed_dims = [grid.node_dimension, grid.edge_dimension, grid.face_dimension]
        raise ValueError(
            f"Not a valid UGRID dimension: {dim}," f"should be one of: {allowed_dims}"
        )


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
        pass  # pragma: no-cover

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
                cmap_params["cmap"] = None
                kwargs["colors"] = colors

        if "imshow" == plotfunc.__name__ and isinstance(aspect, str):
            # forbid usage of mpl strings
            raise ValueError(
                "plt.imshow's `aspect` string kwarg is not available in xugrid. "
                "Use a float instead."
            )

        for key in ["cmap", "vmin", "vmax", "norm"]:
            kwargs[key] = cmap_params.get(key)

        ax = get_axis(figsize, size, aspect, ax, **subplot_kws)
        primitive = plotfunc(
            topology,
            darray,
            ax=ax,
            **kwargs,
        )

        if size is not None:
            # Try to get a 1:1 ratio between x and y coordinates by default. If
            # colorbar is present, we need to make room for it in the
            # x-direction.  1.26 is the magic number; the colorbar takes up 26%
            # additional space by default.
            if aspect is None:
                xmin, xmax = ax.get_xlim()
                ymin, ymax = ax.get_ylim()
                aspect = (xmax - xmin) / (ymax - ymin)
                if add_colorbar:
                    aspect *= 1.26

            # Preserve height, adjust width if needed; Do not call
            # ax.set_aspect: this shrinks or grows the ax relative to the
            # colobar
            figsize = (size * aspect, size)
            ax.figure.set_size_inches(figsize)

        # Label the plot with metadata
        if darray is not None and add_labels:
            # TODO: grab x and y information from topology?
            ax.set_xlabel("x")
            ax.set_ylabel("y")
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

    return newplotfunc


@_plot2d
def scatter(grid, da, ax, **kwargs):
    dim = get_ugrid_dim(grid, da)
    x, y = getattr(grid, COORDS[dim]).T
    primitive = ax.scatter(x, y, c=da.values.ravel(), **kwargs)
    return primitive


@_plot2d
def tripcolor(grid, da, ax, **kwargs):
    dim = get_ugrid_dim(grid, da)
    if dim != NODE:
        raise ValueError("tripcolor only supports data on nodes")
    (x, y, triangles), _ = grid.triangulation
    primitive = ax.tripcolor(x, y, triangles, da.values.ravel(), **kwargs)
    return primitive


@_plot2d
def line(grid, da, ax, **kwargs):
    from matplotlib.collections import LineCollection

    if da is not None:
        dim = get_ugrid_dim(grid, da)
        if dim != EDGE:
            raise ValueError("line only supports data on edges")
    else:
        dim = None

    edge_nodes = grid.edge_node_connectivity
    n_edge = len(edge_nodes)
    edge_coords = np.empty((n_edge, 2, 2), dtype=FloatDType)
    node_0 = edge_nodes[:, 0]
    node_1 = edge_nodes[:, 1]
    edge_coords[:, 0, 0] = grid.node_x[node_0]
    edge_coords[:, 0, 1] = grid.node_y[node_0]
    edge_coords[:, 1, 0] = grid.node_x[node_1]
    edge_coords[:, 1, 1] = grid.node_y[node_1]

    # PolyCollection takes a norm, but not vmin, vmax.
    norm = kwargs.get("norm", None)
    vmin = kwargs.pop("vmin", None)
    vmax = kwargs.pop("vmax", None)

    collection = LineCollection(edge_coords, **kwargs)

    if dim == EDGE:
        collection.set_array(da.values)
        collection._scale_norm(norm, vmin, vmax)

    primitive = ax.add_collection(collection, autolim=False)

    xmin, ymin, xmax, ymax = grid.bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    return primitive


@_plot2d
def imshow(grid, da, ax, **kwargs):
    """
    Image plot of 2D DataArray.
    Wraps :py:func:`matplotlib:matplotlib.pyplot.imshow`.

    This rasterizes the grid before plotting. Pass a ``resolution`` keyword to
    control the rasterization resolution.
    """
    dim = get_ugrid_dim(grid, da)
    if dim != FACE:
        raise ValueError("imshow only supports data on faces")

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
    resolution = kwargs.pop("resolution", None)
    if resolution is None:
        resolution = min(dx, dy) / 500

    _, _, index = grid.rasterize(resolution)
    img = da.values[index].astype(float)
    img[index == -1] = np.nan
    primitive = ax.imshow(img, **kwargs)
    return primitive


@_plot2d
def contour(grid, da, ax, **kwargs):
    """
    Filled contour plot of 2D UgridDataArray.
    Wraps :py:func:`matplotlib:matplotlib.pyplot.tricontour`.
    """
    dim = get_ugrid_dim(grid, da)
    if dim == NODE:
        (x, y, triangles), _ = grid.triangulation
        z = da
    elif dim == FACE:
        (x, y, triangles), index = grid.centroid_triangulation
        z = da.isel({grid.face_dimension: index})
    else:
        raise ValueError("contour only supports data on nodes or faces")

    primitive = ax.tricontour(x, y, triangles, z.values.ravel(), **kwargs)
    return primitive


@_plot2d
def contourf(grid, da, ax, **kwargs):
    """
    Filled contour plot of 2D UgridDataArray.
    Wraps :py:func:`matplotlib:matplotlib.pyplot.tricontourf`.
    """
    dim = get_ugrid_dim(grid, da)
    if dim == NODE:
        (x, y, triangles), _ = grid.triangulation
        z = da
    elif dim == FACE:
        (x, y, triangles), index = grid.centroid_triangulation
        z = da.isel({grid.face_dimension: index})
    else:
        raise ValueError("contourf only supports data on nodes or faces")

    primitive = ax.tricontourf(x, y, triangles, z.values.ravel(), **kwargs)
    return primitive


@_plot2d
def pcolormesh(grid, da, ax, **kwargs):
    """
    Pseudocolor plot of 2D UgridDataArray.
    Wraps :py:func:`matplotlib:matplotlib.pyplot.
    """
    from matplotlib.collections import PolyCollection

    dim = get_ugrid_dim(grid, da)
    if dim != FACE:
        raise ValueError("pcolormesh only supports data on faces")

    nodes = grid.node_coordinates
    faces = grid.face_node_connectivity
    vertices = nodes[faces]
    # Replace fill value; PolyCollection ignores NaN.
    vertices[faces == -1] = np.nan

    # PolyCollection takes a norm, but not vmin, vmax.
    norm = kwargs.get("norm", None)
    vmin = kwargs.pop("vmin", None)
    vmax = kwargs.pop("vmax", None)

    collection = PolyCollection(vertices, **kwargs)
    collection.set_array(da.values.ravel())
    collection._scale_norm(norm, vmin, vmax)
    primitive = ax.add_collection(collection, autolim=False)

    xmin, ymin, xmax, ymax = grid.bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    corners = (xmin, ymin), (xmax, ymax)
    ax.update_datalim(corners)

    return primitive


@_plot2d
def surface(grid, da, ax, **kwargs):
    """
    Surface plot of x-y UgridDataArray.
    Wraps :py:func:`matplotlib:mplot3d:plot_trisurf`.
    """
    dim = get_ugrid_dim(grid, da)
    if dim == NODE:
        (x, y, triangles), _ = grid.triangulation
        z = da
    elif dim == FACE:
        (x, y, triangles), index = grid.centroid_triangulation
        z = da.isel({grid.face_dimension: index})
    else:
        raise ValueError("surface only supports data on nodes or faces")

    primitive = ax.plot_trisurf(x, y, triangles, z.values.ravel(), **kwargs)
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
    kwargs["ax"] = ax
    if grid.topology_dimension == 1:
        if dim == grid.edge_dimension:
            return line(grid, darray, **kwargs)
        elif dim == grid.node_dimension:
            return scatter(grid, darray, **kwargs)
        else:
            raise ValueError("Data dimensions is not one of node or edge dimension.")
    elif grid.topology_dimension == 2:
        if dim == grid.face_dimension:
            return pcolormesh(grid, darray, **kwargs)
        elif dim == grid.node_dimension:
            return tripcolor(grid, darray, **kwargs)
        elif dim == grid.edge_dimension:
            return line(grid, darray, **kwargs)
        else:
            raise ValueError(
                "Data dimensions is not one of face, node, or edge dimension."
            )
    else:
        raise ValueError("Topology dimension is not 1 or 2")


class _PlotMethods:
    """
    Enables use of plot functions as attributes.
    For example UgridDataArray.ugrid.plot.pcolormesh()
    """

    __slots__ = ("grid", "darray")

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
        self.grid = grid
        self.darray = darray

    def __call__(self, **kwargs):
        return plot(self.grid, self.darray, **kwargs)

    @functools.wraps(contour)
    def contour(self, *args, **kwargs):
        return contour(self.grid, self.darray, *args, **kwargs)

    @functools.wraps(contourf)
    def contourf(self, *args, **kwargs):
        return contourf(self.grid, self.darray, *args, **kwargs)

    @functools.wraps(imshow)
    def imshow(self, *args, **kwargs):
        return imshow(self.grid, self.darray, *args, **kwargs)

    @functools.wraps(line)
    def line(self, *args, **kwargs):
        if self.darray.dims[0] == self.grid.edge_dimension:
            z = self.darray
        else:
            z = None
        return line(self.grid, z, *args, **kwargs)

    @functools.wraps(pcolormesh)
    def pcolormesh(self, *args, **kwargs):
        return pcolormesh(self.grid, self.darray, *args, **kwargs)

    @functools.wraps(scatter)
    def scatter(self, *args, **kwargs):
        return scatter(self.grid, self.darray, *args, **kwargs)

    @functools.wraps(surface)
    def surface(self, *args, **kwargs):
        return surface(self.grid, self.darray, *args, **kwargs)

    @functools.wraps(tripcolor)
    def tripcolor(self, *args, **kwargs):
        return tripcolor(self.grid, self.darray, *args, **kwargs)
