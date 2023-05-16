import numpy as np
import pytest
import xarray as xr

from xugrid.regrid.structured import StructuredGrid1d, StructuredGrid2d

from fixtures.fixture_regridder import (
    grid_data_a,
    grid_data_a_layered,
    grid_data_a_1d,
    grid_data_a_layered_1d,
    grid_data_b,
    grid_data_b_1d,
    grid_data_b_flipped_1d,
    grid_data_c,
    grid_data_d,
    grid_data_a_2d,
    grid_data_a_layered_2d,
    grid_data_b_2d,
    grid_data_c_1d,
    grid_data_c_2d,
    grid_data_d_1d,
)

# Testgrids
# --------
# grid a(x):               |______50_____|_____100_____|_____150_____|               -> source
# grid b(x):        |______25_____|______75_____|_____125_____|_____175_____|        -> target
# --------
# grid c(x):            |______40_____|______90_____|_____140_____|____190_____|     -> target
# --------
# grid d(x):              |__30__|__55__|__80_|__105__|                              -> target
# --------


def test_init_1d(grid_data_a_1d):
    assert isinstance(grid_data_a_1d, StructuredGrid1d)
    with pytest.raises(TypeError):
        StructuredGrid1d(1)


def test_init_2d(grid_data_a_2d):
    assert isinstance(grid_data_a_2d, StructuredGrid2d)
    with pytest.raises(TypeError):
        StructuredGrid2d(1)


def test_overlap_1d(grid_data_a_1d, grid_data_b_1d, grid_data_b_flipped_1d):
    # --------
    # source   targets  weight
    # node 0   0        25 m      -> should be not valid?
    # node 1   0, 1     25 m
    # node 2   1, 2     25 m
    # node 3   2        25 m      -> should be not valid?
    # --------
    source, target, weights = grid_data_a_1d.overlap(grid_data_b_1d, relative=False)
    sorter = np.argsort(source)
    assert np.array_equal(source[sorter], np.array([0, 0, 1, 1, 2, 2]))
    assert np.array_equal(target[sorter], np.array([0, 1, 1, 2, 2, 3]))
    assert np.array_equal(weights[sorter], np.array([25, 25, 25, 25, 25, 25]))

    # flipped axis (y-axis)
    # --------
    # source   targets  weight
    # node 0   2        25 m      -> should be not valid?
    # node 1   2, 1     25 m
    # node 2   1, 0     25 m
    # node 3   0        25 m      -> should be not valid?
    # --------
    source, target, weights = grid_data_a_1d.overlap(
        grid_data_b_flipped_1d, relative=False
    )
    sorter = np.argsort(source)
    assert np.array_equal(source[sorter], np.array([0, 0, 1, 1, 2, 2]))
    assert np.array_equal(target[sorter], np.array([2, 3, 1, 2, 0, 1]))
    assert np.array_equal(weights[sorter], np.array([25, 25, 25, 25, 25, 25]))


def test_overlap_2d(grid_data_a_2d, grid_data_b_2d):
    # --------
    # source   targets            weights
    # node 0   0,   1,  4,  5     625 m
    # node 1   1,   2,  5,  6     625 m
    # node 2   2,   3,  6,  7     625 m
    # node 3   4,   5,  8,  9     625 m
    # node 4   5,   6,  9, 10     625 m
    # node 5   6,   7, 10, 11     625 m
    # node 6   8,   9, 12, 13     625 m
    # node 7   9,  10, 13, 14     625 m
    # node 8   10, 11, 14, 15     625 m
    # --------
    source, target, weights = grid_data_a_2d.overlap(grid_data_b_2d, relative=False)
    sorter = np.argsort(source)
    assert np.array_equal(
        source[sorter],
        np.array(
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
    )
    assert np.array_equal(
        target[sorter],
        np.array(
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
    )
    assert np.array_equal(weights[sorter], np.array([625] * source.size))


def test_locate_centroids_1d(grid_data_a_1d, grid_data_b_1d, grid_data_b_flipped_1d):
    # --------
    # source   target  weight
    # 0        1       1
    # 1        2       1
    # --------
    source, target, weights = grid_data_a_1d.locate_centroids(grid_data_b_1d)
    sorter = np.argsort(source)
    assert np.array_equal(source[sorter], np.array([0, 1]))
    assert np.array_equal(target[sorter], np.array([1, 2]))
    assert np.allclose(weights[sorter], np.ones(2))

    # flipped axis (y-axis)
    # --------
    # source   target  weight
    # 0        1       1
    # 1        2       1
    # --------
    source, target, weights = grid_data_a_1d.locate_centroids(grid_data_b_flipped_1d)
    sorter = np.argsort(source)
    assert np.array_equal(source[sorter], np.array([0, 1]))
    assert np.array_equal(target[sorter], np.array([2, 1]))
    assert np.allclose(weights[sorter], np.ones(2))


def test_locate_centroids_2d(grid_data_a_2d, grid_data_b_2d):
    # --------
    # source   target  weight
    # 0        5       1
    # 1        6       1
    # 3        9       1
    # 4        10      1
    # --------
    source, target, weights = grid_data_a_2d.locate_centroids(grid_data_b_2d)
    sorter = np.argsort(source)
    assert np.array_equal(source[sorter], np.array([0, 1, 3, 4]))
    assert np.array_equal(target[sorter], np.array([5, 6, 9, 10]))
    assert np.allclose(weights[sorter], np.ones(4))


def test_linear_weights_1d(
    grid_data_a_1d,
    grid_data_a_layered_1d,
    grid_data_b_1d,
    grid_data_b_flipped_1d,
    grid_data_c_1d,
    grid_data_d_1d,
):
    # --------
    # source   target  weight
    # 0   ->   1       50%
    # 1   ->   1       50%
    # 1   ->   2       50%
    # 2   ->   2       50%
    # --------
    source, target, weights = grid_data_a_1d.linear_weights(grid_data_b_1d)
    sorter = np.argsort(target)
    assert np.array_equal(source[sorter], np.array([0, 1, 1, 2]))
    assert np.array_equal(target[sorter], np.array([1, 1, 2, 2]))
    assert np.allclose(weights[sorter], np.array([0.5, 0.5, 0.5, 0.5]))

    # flipped axis (y-axis)
    # --------
    # source   target  weight
    # 0   ->   2       50%
    # 1   ->   2       50%
    # 1   ->   1       50%
    # 2   ->   1       50%
    # --------
    source, target, weights = grid_data_a_1d.linear_weights(grid_data_b_flipped_1d)
    sorter = np.argsort(target)
    assert np.array_equal(source[sorter], np.array([1, 2, 0, 1]))
    assert np.array_equal(target[sorter], np.array([1, 1, 2, 2]))
    assert np.allclose(weights[sorter], np.array([0.5, 0.5, 0.5, 0.5]))

    # --------
    # source   target  weight
    # 1        1       20%
    # 0        1       80%
    # 2        2       20%
    # 1        2       80%
    # --------
    source, target, weights = grid_data_a_1d.linear_weights(grid_data_c_1d)
    sorter = np.argsort(target)
    assert np.array_equal(source[sorter], np.array([1, 0, 2, 1]))
    assert np.array_equal(target[sorter], np.array([1, 1, 2, 2]))
    assert np.allclose(weights[sorter], np.array([0.2, 0.8, 0.2, 0.8]))

    # --------
    # source   target  weight
    # 0        1       10%
    # 1        1       90%
    # 0        2       60% 
    # 1        2       40% *reversed in output
    # 1        3       10%    
    # 2        3       90%
    # --------
    source, target, weights = grid_data_a_1d.linear_weights(grid_data_d_1d)
    sorter = np.argsort(target)
    assert np.array_equal(source[sorter], np.array([0, 1, 1, 0, 1, 2]))
    assert np.array_equal(target[sorter], np.array([1, 1, 2, 2, 3, 3]))
    assert np.allclose(weights[sorter], np.array([0.1, 0.9, 0.4, 0.6, 0.1, 0.9]))


def test_linear_weights_2d(
    grid_data_a_2d, 
    grid_data_a_layered_2d, 
    grid_data_b_2d, 
    grid_data_c_2d
):
    # --------
    # source   targets     weight
    # 5        0, 1, 3, 4  25%
    # 6        1, 2, 4, 5  25%
    # 9        3, 4, 6, 7  25%
    # 10       4, 5, 7, 8  25%
    # --------
    source, target, weights = grid_data_a_2d.linear_weights(grid_data_b_2d)
    sorter = np.argsort(target)
    assert np.array_equal(
        source[sorter], np.array([0, 1, 3, 4, 1, 2, 4, 5, 3, 4, 6, 7, 4, 5, 7, 8])
    )
    assert np.array_equal(
        target[sorter], np.array([5, 5, 5, 5, 6, 6, 6, 6, 9, 9, 9, 9, 10, 10, 10, 10])
    )
    assert np.allclose(weights[sorter], np.array([0.25] * 16))

    # --------
    # source   targets      weight
    # 5        0, 1, 3, 4   10%	40%	10%	40%
    # 6        1, 2, 4, 5   10%	40%	10%	40%
    # 9        3, 4, 6, 7   10%	40%	10%	40%
    # 10       4, 5, 7, 8   10%	40%	10%	40%
    # --------
    source, target, weights = grid_data_a_layered_2d.linear_weights(grid_data_c_2d)
    sorter = np.argsort(target)
    assert np.array_equal(
        source[sorter], np.array([1, 0, 4, 3, 2, 1, 5, 4, 4, 3, 7, 6, 5, 4, 8, 7])                           
    )
    assert np.array_equal(
        target[sorter], np.array([5, 5, 5, 5, 6, 6, 6, 6, 9, 9, 9, 9, 10, 10, 10, 10])
    )
    assert np.allclose(weights[sorter], np.array([0.1, 0.4, 0.1, 0.4] * 4))
