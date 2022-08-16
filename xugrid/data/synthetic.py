import meshzoo
import numpy as np
import xarray as xr

import xugrid


def transform(vertices, minx, maxx, miny):
    """
    Transform vertices to fit within minx to maxx.
    Maintains x:y aspect ratio.
    """
    x, y = vertices.T
    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()
    dx = xmax - xmin
    dy = ymax - ymin
    new_dx = maxx - minx
    new_dy = dy / dx * new_dx
    x = (x - xmin) * new_dx / dx + minx
    y = (y - ymin) * new_dy / dy + miny
    return np.column_stack([x, y])


def disk():
    def function_z(x, y):
        """
        from https://matplotlib.org/stable/gallery/images_contours_and_fields/tricontour_smooth_user.html
        """
        r1 = np.sqrt((0.5 - x) ** 2 + (0.5 - y) ** 2)
        theta1 = np.arctan2(0.5 - x, 0.5 - y)
        r2 = np.sqrt((-x - 0.2) ** 2 + (-y - 0.2) ** 2)
        theta2 = np.arctan2(-x - 0.2, -y - 0.2)
        z = -(
            2 * (np.exp((r1 / 10) ** 2) - 1) * 30.0 * np.cos(7.0 * theta1)
            + (np.exp((r2 / 10) ** 2) - 1) * 30.0 * np.cos(11.0 * theta2)
            + 0.7 * (x**2 + y**2)
        )
        zmin = z.min()
        zmax = z.max()
        return (zmax - z) / (zmax - zmin) * 10.0

    vertices, triangles = meshzoo.disk(6, 8)
    vertices = transform(vertices, 0.0, 10.0, 0.0)
    grid = xugrid.Ugrid2d(
        node_x=vertices[:, 0],
        node_y=vertices[:, 1],
        fill_value=-1,
        face_node_connectivity=triangles,
    )

    ds = xr.Dataset()
    ds["node_z"] = xr.DataArray(
        data=function_z(*grid.node_coordinates.T),
        dims=[grid.node_dimension],
    )
    ds["face_z"] = xr.DataArray(
        data=function_z(*grid.face_coordinates.T),
        dims=[grid.face_dimension],
    )
    ds["edge_z"] = xr.DataArray(
        data=function_z(*grid.edge_coordinates.T),
        dims=[grid.edge_dimension],
    )
    return xugrid.UgridDataset(ds, grid)
