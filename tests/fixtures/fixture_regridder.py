import numpy as np
import pytest
import xarray as xr

import xugrid as xu
from xugrid.regrid.structured import StructuredGrid1d, StructuredGrid2d

# Testgrids
# --------
# grid a(x):               |______50_____|_____100_____|_____150_____|               -> source
# grid b(x):        |______25_____|______75_____|_____125_____|_____175_____|        -> target
# --------
# grid c(x):            |______40_____|______90_____|_____140_____|____190_____|     -> target
# --------
# grid d(x):              |__30__|__55__|__80_|__105__|                              -> target
# --------
# grid e(x):              |__30__|_____67.5___|__105__|                              -> target
# --------


@pytest.fixture(scope="function")
def disk():
    return xu.data.disk()["face_z"]


@pytest.fixture(scope="function")
def quads_0_25():
    dx = 0.25
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
    return xu.UgridDataArray.from_structured(da)


@pytest.fixture(scope="function")
def quads_1():
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
    return xu.UgridDataArray.from_structured(da)


@pytest.fixture(scope="function")
def grid_data_a():
    return xr.DataArray(
        data=np.arange(9).reshape((3, 3)),
        dims=["y", "x"],
        coords={
            "y": np.array([150, 100, 50]),
            "x": np.array([50, 100, 150]),
            "dx": 50.0,
            "dy": -50.0,
        },
    )


@pytest.fixture(scope="function")
def grid_data_a_layered():
    return xr.DataArray(
        data=np.arange(18).reshape((2, 3, 3)),
        dims=["layer", "y", "x"],
        coords={
            "layer": np.arange(2) + 1,
            "y": np.array([150, 100, 50]),
            "x": np.array([50, 100, 150]),
            "dx": 50.0,
            "dy": -50.0,
        },
    )


@pytest.fixture(scope="function")
def grid_data_b():
    return xr.DataArray(
        data=np.zeros(16).reshape((4, 4)),
        dims=["y", "x"],
        coords={
            "y": np.array([175, 125, 75, 25]),
            "x": np.array([25, 75, 125, 175]),
            "dx": 50.0,
            "dy": -50.0,
        },
    )


@pytest.fixture(scope="function")
def grid_data_c():
    return xr.DataArray(
        data=np.arange(16).reshape((4, 4)),
        dims=["y", "x"],
        coords={
            "y": np.array([175, 125, 75, 25]),
            "x": np.array([40, 90, 140, 190]),
            "dx": 50.0,
            "dy": -50.0,
        },
    )


@pytest.fixture(scope="function")
def grid_data_d():
    return xr.DataArray(
        data=np.arange(16).reshape((4, 4)),
        dims=["y", "x"],
        coords={
            "y": np.array([175, 125, 75, 25]),
            "x": np.array([30, 55, 80, 105]),
            "dx": 25.0,
            "dy": -50.0,
        },
    )


@pytest.fixture(scope="function")
def grid_data_e():
    return xr.DataArray(
        data=np.zeros((4, 3, 2)),
        dims=["y", "x", "nbounds"],
        coords={
            "y": np.array([175, 125, 75, 25]),
            "x": np.array([30, 67.5, 105]),
            "dx": 25,
            "dy": -50.0,
            "xbounds": (
                ["x", "nbounds"],
                np.column_stack(
                    (np.array([17.5, 42.5, 92.5]), np.array([42.5, 92.5, 117.5]))
                ),
            ),
            "nbounds": np.arange(2),
        },
    )


@pytest.fixture(scope="function")
def grid_data_a_1d(grid_data_a):
    return StructuredGrid1d(grid_data_a, "x")


@pytest.fixture(scope="function")
def grid_data_a_layered_1d(grid_data_a_layered):
    return StructuredGrid1d(grid_data_a_layered, "x")


@pytest.fixture(scope="function")
def grid_data_a_2d(grid_data_a):
    return StructuredGrid2d(grid_data_a, "x", "y")


@pytest.fixture(scope="function")
def grid_data_a_layered_2d(grid_data_a_layered):
    return StructuredGrid2d(grid_data_a_layered, "x", "y")


@pytest.fixture(scope="function")
def grid_data_b_1d(grid_data_b):
    return StructuredGrid1d(grid_data_b, "x")


@pytest.fixture(scope="function")
def grid_data_b_flipped_1d(grid_data_b):
    return StructuredGrid1d(grid_data_b, "y")


@pytest.fixture(scope="function")
def grid_data_c_1d(grid_data_c):
    return StructuredGrid1d(grid_data_c, "x")


@pytest.fixture(scope="function")
def grid_data_d_1d(grid_data_d):
    return StructuredGrid1d(grid_data_d, "x")


@pytest.fixture(scope="function")
def grid_data_b_2d(grid_data_b):
    return StructuredGrid2d(grid_data_b, "x", "y")


@pytest.fixture(scope="function")
def grid_data_c_2d(grid_data_c):
    return StructuredGrid2d(grid_data_c, "x", "y")


@pytest.fixture(scope="function")
def grid_data_e_1d(grid_data_e):
    return StructuredGrid1d(grid_data_e, "x")


@pytest.fixture(scope="function")
def expected_results_centroid():
    return xr.DataArray(
        data=np.array(
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                0,
                1,
                np.nan,
                np.nan,
                3,
                4,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ]
        ).reshape((4, 4)),
        dims=["y", "x"],
        coords={
            "y": np.array([175, 125, 75, 25]),
            "x": np.array([25, 75, 125, 175]),
            "dx": 50.0,
            "dy": -50.0,
        },
    )


@pytest.fixture(scope="function")
def expected_results_overlap():
    # --------
    # target | source            | ntarget | sum(source)/ntaget
    # 0      | 0                 | 1       | 0.0
    # 1      | 0 1               | 2       | 0.5
    # 2      |   1 2             | 2       | 1.5
    # 3      |     2             | 1       | 2.0
    # 4      | 0     3           | 2       | 1.5
    # 5      | 0 1   3 4         | 4       | 2.0
    # 6      |   1 2   4 5       | 4       | 3.0
    # 7      |     2     5       | 2       | 3.5
    # 8      |       3     6     | 2       | 4.5
    # 9      |       3 4   6 7   | 4       | 5.0
    # 10     |         4 5   7 8 | 4       | 6.0
    # 11     |           5     8 | 2       | 6.5
    # 12     |             6     | 1       | 6.0
    # 13     |             6 7   | 2       | 6.5
    # 14     |               7 8 | 2       | 7.5
    # 15     |                 8 | 1       | 8.0
    # --------
    return xr.DataArray(
        data=np.array(
            [
                0.0,
                0.5,
                1.5,
                2.0,
                1.5,
                2.0,
                3.0,
                3.5,
                4.5,
                5.0,
                6.0,
                6.5,
                6.0,
                6.5,
                7.5,
                8.0,
            ]
        ).reshape((4, 4)),
        dims=["y", "x"],
        coords={
            "y": np.array([175, 125, 75, 25]),
            "x": np.array([25, 75, 125, 175]),
            "dx": 50.0,
            "dy": -50.0,
        },
    )


@pytest.fixture(scope="function")
def expected_results_linear():
    # --------
    # target | source            | weights                | sum(source x weight)
    # 5      | 0 1   3 4         | 0.25 0.25 0.25 0.25    | 2.0
    # 6      |   1 2   4 5       | 0.25 0.25 0.25 0.25    | 3.0
    # 9      |       3 4   6 7   | 0.25 0.25 0.25 0.25    | 5.0
    # 10     |         4 5   7 8 | 0.25 0.25 0.25 0.25    | 6.0
    # --------
    return xr.DataArray(
        data=np.array(
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                2.0,
                3.0,
                np.nan,
                np.nan,
                5.0,
                6.0,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ]
        ).reshape((4, 4)),
        dims=["y", "x"],
        coords={
            "y": np.array([175, 125, 75, 25]),
            "x": np.array([25, 75, 125, 175]),
            "dx": 50.0,
            "dy": -50.0,
        },
    )
