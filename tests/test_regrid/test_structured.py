import numpy as np
import pytest
import xarray as xr

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
# grid e(x):              |__30__|____67.5____|__105__|                              -> target
# --------


def test_init_1d(grid_data_a_1d):
    assert isinstance(grid_data_a_1d, StructuredGrid1d)
    with pytest.raises(TypeError):
        StructuredGrid1d(1)


def test_init_2d(grid_data_a_2d):
    assert isinstance(grid_data_a_2d, StructuredGrid2d)
    with pytest.raises(TypeError):
        StructuredGrid2d(1)


def assert_expected_overlap(
    actual_source,
    actual_target,
    actual_weights,
    expected_source,
    expected_target,
    expected_weights,
):
    # Numpy 2.0 release has change sorting behavior of non-stable sorting:
    # https://numpy.org/doc/stable/release/2.0.0-notes.html#minor-changes-in-behavior-of-sorting-functions
    # So the comparison method must be robust to work for numpy <2.0 and >=2.0.
    actual_mapping = np.column_stack((actual_target, actual_source))
    expected_mapping = np.column_stack((expected_target, expected_source))
    actual, actual_sorter = np.unique(actual_mapping, axis=0, return_index=True)
    expected, expected_sorter = np.unique(expected_mapping, axis=0, return_index=True)
    assert np.array_equal(actual, expected)
    assert np.allclose(actual_weights[actual_sorter], expected_weights[expected_sorter])


def test_overlap_1d(
    grid_data_a_1d, grid_data_b_1d, grid_data_b_flipped_1d, grid_data_e_1d
):
    # --------
    # source   targets  weight
    # node 0   0, 1     25 m
    # node 1   1, 2     25 m
    # node 2   2, 3     25 m
    # --------
    assert_expected_overlap(
        *grid_data_a_1d.overlap(grid_data_b_1d, relative=False),
        np.array([0, 0, 1, 1, 2, 2]),
        np.array([0, 1, 1, 2, 2, 3]),
        np.array([25, 25, 25, 25, 25, 25]),
    )

    # flipped axis (y-axis)
    # --------
    # source   targets  weight
    # 0        0, 1     25 m
    # 1        1, 2     25 m
    # 2        2, 3     25 m
    # --------
    assert_expected_overlap(
        *grid_data_a_1d.overlap(grid_data_b_flipped_1d, relative=False),
        np.array([0, 0, 1, 1, 2, 2]),
        np.array([2, 3, 1, 2, 0, 1]),
        np.array([25, 25, 25, 25, 25, 25]),
    )

    # non-equidistant
    # --------
    # source   targets  weight
    # node 0   0, 1     17.5 m, 32.5 m
    # node 1   1, 2     17.5 m, 25.0 m
    # --------
    assert_expected_overlap(
        *grid_data_a_1d.overlap(grid_data_e_1d, relative=False),
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 1, 2]),
        np.array([17.5, 32.5, 17.5, 25.0]),
    )

    # relative
    # --------
    # source   targets  weight
    # node 0   0, 1     17.5 m, 32.5 m
    # node 1   1, 2     17.5 m, 25.0 m
    # --------
    assert_expected_overlap(
        *grid_data_a_1d.overlap(grid_data_e_1d, relative=True),
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 1, 2]),
        np.array([17.5 / 50.0, 32.5 / 50.0, 17.5 / 50.0, 25.0 / 50.0]),
    )


def test_overlap_2d(grid_data_a_2d, grid_data_b_2d):
    # --------
    # source   targets            weights
    # 0        0,   1,  4,  5     625 m
    # 1        1,   2,  5,  6     625 m
    # 2        2,   3,  6,  7     625 m
    # 3        4,   5,  8,  9     625 m
    # 4        5,   6,  9, 10     625 m
    # 5        6,   7, 10, 11     625 m
    # 6        8,   9, 12, 13     625 m
    # 7        9,  10, 13, 14     625 m
    # 8        10, 11, 14, 15     625 m
    # --------
    assert_expected_overlap(
        *grid_data_a_2d.overlap(grid_data_b_2d, relative=False),
        expected_source=np.array(
            [
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                3,
                3,
                3,
                3,
                4,
                4,
                4,
                4,
                5,
                5,
                5,
                5,
                6,
                6,
                6,
                6,
                7,
                7,
                7,
                7,
                8,
                8,
                8,
                8,
            ]
        ),
        expected_target=np.array(
            [
                0,
                4,
                5,
                1,
                2,
                6,
                5,
                1,
                2,
                3,
                7,
                6,
                8,
                9,
                5,
                4,
                9,
                5,
                10,
                6,
                10,
                11,
                7,
                6,
                9,
                8,
                12,
                13,
                10,
                14,
                13,
                9,
                10,
                11,
                14,
                15,
            ]
        ),
        expected_weights=np.full(36, 625.0),
    )


def test_locate_centroids_1d(
    grid_data_a_1d, grid_data_b_1d, grid_data_b_flipped_1d, grid_data_e_1d
):
    # --------
    # source   target  weight
    # 0        1       1
    # 1        2       1
    # --------
    assert_expected_overlap(
        *grid_data_a_1d.locate_centroids(grid_data_b_1d),
        np.array([0, 1]),
        np.array([1, 2]),
        np.ones(2),
    )

    # flipped axis (y-axis)
    # --------
    # source   target  weight
    # 0        1       1
    # 1        2       1
    # --------
    assert_expected_overlap(
        *grid_data_a_1d.locate_centroids(grid_data_b_flipped_1d),
        np.array([0, 1]),
        np.array([2, 1]),
        np.ones(2),
    )

    # non-equidistant
    # --------
    # source   target  weight
    # 0        0, 1    1, 1
    # 1        2       1
    # --------
    assert_expected_overlap(
        *grid_data_a_1d.locate_centroids(grid_data_e_1d),
        np.array([0, 0, 1]),
        np.array([0, 1, 2]),
        np.ones(3),
    )


def test_locate_centroids_2d(grid_data_a_2d, grid_data_b_2d):
    # --------
    # source   target  weight
    # 0        5       1
    # 1        6       1
    # 3        9       1
    # 4        10      1
    # --------
    assert_expected_overlap(
        *grid_data_a_2d.locate_centroids(grid_data_b_2d),
        np.array([0, 1, 3, 4]),
        np.array([5, 6, 9, 10]),
        np.ones(4),
    )


def test_linear_weights_1d(
    grid_data_a_1d,
    grid_data_b_1d,
    grid_data_b_flipped_1d,
    grid_data_c_1d,
    grid_data_d_1d,
    grid_data_e_1d,
):
    # --------
    # source   target  weight
    # 0   ->   1       50%
    # 1   ->   1       50%
    # 1   ->   2       50%
    # 2   ->   2       50%
    # --------
    assert_expected_overlap(
        *grid_data_a_1d.linear_weights(grid_data_b_1d),
        np.array([0, 1, 1, 2]),
        np.array([1, 1, 2, 2]),
        np.array([0.5, 0.5, 0.5, 0.5]),
    )

    # flipped axis (y-axis)
    # --------
    # source   target  weight
    # 0   ->   2       50%
    # 1   ->   2       50%
    # 1   ->   1       50%
    # 2   ->   1       50%
    # --------
    assert_expected_overlap(
        *grid_data_a_1d.linear_weights(grid_data_b_flipped_1d),
        np.array([2, 1, 1, 0]),
        np.array([1, 1, 2, 2]),
        np.array([0.5, 0.5, 0.5, 0.5]),
    )

    # --------
    # source   target  weight
    # 1        1       80%
    # 0        1       20%
    # 2        2       80%
    # 1        2       20%
    # --------
    assert_expected_overlap(
        *grid_data_a_1d.linear_weights(grid_data_c_1d),
        np.array([0, 0, 1, 0, 2, 1]),
        np.array([0, 0, 1, 1, 2, 2]),
        np.array([0.0, 1.0, 0.8, 0.2, 0.8, 0.2]),
    )

    # --------
    # source   target  weight
    # 0        1       90%
    # 1        1       10%
    # 0        2       40%
    # 1        2       60% *reversed in output
    # 1        3       90%
    # 2        3       10%
    # --------
    assert_expected_overlap(
        *grid_data_a_1d.linear_weights(grid_data_d_1d),
        np.array([0, 0, 0, 1, 1, 0, 1, 2]),
        np.array([0, 0, 1, 1, 2, 2, 3, 3]),
        np.array([0.0, 0.1, 0.9, 0.1, 0.6, 0.4, 0.9, 0.1]),
    )

    # non-equidistant
    # --------
    # source   target  weight
    # 0        1       65%
    # 1        1       35%
    # 1        2       90%
    # 2        2       10%
    # --------
    assert_expected_overlap(
        *grid_data_a_1d.linear_weights(grid_data_e_1d),
        np.array([0, 0, 0, 1, 1, 2]),
        np.array([0, 0, 1, 1, 2, 2]),
        np.array([0.0, 1.0, 0.65, 0.35, 0.9, 0.1]),
    )

    # 1-1 grid
    # --------
    # source   target  weight
    # 1        1       100%
    # 0        1       0%
    # 2        2       100%
    # 1        2       0%
    # --------
    assert_expected_overlap(
        *grid_data_b_1d.linear_weights(grid_data_b_1d),
        np.array([0, 0, 1, 0, 2, 1, 3, 2]),
        np.array([0, 0, 1, 1, 2, 2, 3, 3]),
        np.array([0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
    )


def test_linear_weights_2d(
    grid_data_a_2d, grid_data_a_layered_2d, grid_data_b_2d, grid_data_c_2d
):
    # --------
    # source   targets     weight
    # 5        0, 1, 3, 4  25%
    # 6        1, 2, 4, 5  25%
    # 9        3, 4, 6, 7  25%
    # 10       4, 5, 7, 8  25%
    # --------
    assert_expected_overlap(
        *grid_data_a_2d.linear_weights(grid_data_b_2d),
        np.array([3, 4, 1, 0, 5, 4, 1, 2, 6, 7, 4, 3, 8, 7, 4, 5]),
        np.array([5, 5, 5, 5, 6, 6, 6, 6, 9, 9, 9, 9, 10, 10, 10, 10]),
        np.array([0.25] * 16),
    )

    # --------
    # source   targets      weight
    # 4        0, 0, 3, 3   0%  50% 0%  50%
    # 5        0, 1, 3, 4   10%	40%	10%	40%
    # 6        1, 2, 4, 5   10%	40%	10%	40%
    # 8        3, 3, 6, 6   0%  50% 0%  50%
    # 9        3, 4, 6, 7   10%	40%	10%	40%
    # 10       4, 5, 7, 8   10%	40%	10%	40%
    # --------
    assert_expected_overlap(
        *grid_data_a_layered_2d.linear_weights(grid_data_c_2d),
        np.array(
            [0, 0, 3, 3, 1, 0, 3, 4, 5, 4, 2, 1, 3, 3, 6, 6, 4, 3, 7, 6, 8, 5, 4, 7]
        ),
        np.array(
            [4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10]
        ),
        np.array(
            [
                0.0,
                0.5,
                0.0,
                0.5,
                0.4,
                0.1,
                0.1,
                0.4,
                0.4,
                0.1,
                0.4,
                0.1,
                0.0,
                0.5,
                0.0,
                0.5,
                0.4,
                0.1,
                0.4,
                0.1,
                0.4,
                0.4,
                0.1,
                0.1,
            ]
        ),
    )

    # 1-1
    # --------
    # source   targets      weight
    # 0-15     0-15         0% 0% 0% 100% (or shuffled)
    # result should be 1:1 mapping
    # --------
    source, target, weights = grid_data_b_2d.linear_weights(grid_data_b_2d)
    expected_target = np.repeat(np.arange(16), 4)
    assert np.array_equal(target, expected_target)
    assert np.array_equal(np.unique(weights), [0, 1])
    check_source = source[weights != 0]
    assert np.array_equal(check_source, np.arange(16))


def test_nonscalar_dx():
    da = xr.DataArray(
        [1, 2, 3], coords={"x": [1, 2, 3], "dx": ("x", [1, 1, 1])}, dims=("x",)
    )
    grid = StructuredGrid1d(da, name="x")
    actual = xr.DataArray([1, 2, 3], coords=grid.coords, dims=grid.dims)
    assert actual.identical(da)


def test_directional_bounds():
    da = xr.DataArray([1, 2, 3], coords={"y": [1, 2, 3]}, dims=("y",))
    decreasing = da.isel(y=slice(None, None, -1))
    grid_inc = StructuredGrid1d(da, name="y")
    grid_dec = StructuredGrid1d(decreasing, name="y")
    assert grid_inc.flipped is False
    assert grid_dec.flipped is True
    assert np.array_equal(grid_inc.bounds, grid_dec.bounds)
    assert np.array_equal(
        grid_inc.directional_bounds, grid_dec.directional_bounds[::-1]
    )
