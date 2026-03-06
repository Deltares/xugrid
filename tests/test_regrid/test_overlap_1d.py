import numpy as np

from xugrid.regrid.utils import overlap_1d


def test_minmax():
    assert overlap_1d.minmax(-1.0, 0.0, 2.0) == 0.0
    assert overlap_1d.minmax(3.0, 0.0, 2.0) == 2.0
    assert overlap_1d.minmax(1.0, 0.0, 2.0) == 1.0


def test_find_indices():
    a = np.arange(0.0, 11.0)[np.newaxis, :]
    b = np.arange(0.0, 12.5, 2.5)[np.newaxis, :]
    source_index = np.array([0])
    target_index = np.array([0])

    lower = overlap_1d.find_lower_indices(a, b, source_index, target_index)
    upper = overlap_1d.find_upper_indices(a, b, source_index, target_index)
    assert np.array_equal(lower, [[0, 2, 5, 7, 9]])
    assert np.array_equal(upper, [[1, 4, 6, 9, 11]])


def test_vectorized_overlap():
    bounds_a = np.array(
        [
            [0.0, 3.0],
            [0.0, 3.0],
        ]
    )
    bounds_b = np.array(
        [
            [1.0, 2.0],
            [1.0, 2.0],
        ]
    )
    actual = overlap_1d.vectorized_overlap(bounds_a, bounds_b)
    assert np.array_equal(actual, np.array([1.0, 1.0]))


def test_overlap_1d():
    source_bounds = np.array(
        [
            [0.0, 1.0],
            [2.0, 3.0],
            [np.nan, np.nan],
            [5.0, 6.0],
        ]
    )
    target_bounds = np.array(
        [
            [0.0, 10.0],
            [10.0, 20.0],
        ]
    )
    source, target, overlap = overlap_1d.overlap_1d(source_bounds, target_bounds)
    assert np.array_equal(source, [0, 1, 3])
    assert np.array_equal(target, [0, 0, 0])
    assert np.allclose(overlap, [1.0, 1.0, 1.0])

    target_bounds = np.array(
        [
            [0.0, 2.5],
            [np.nan, np.nan],
        ]
    )
    source, target, overlap = overlap_1d.overlap_1d(source_bounds, target_bounds)
    assert np.array_equal(source, [0, 1])
    assert np.array_equal(target, [0, 0])
    assert np.allclose(overlap, [1.0, 0.5])


def test_overlap_1d_nd():
    source_bounds = np.array(
        [
            [
                [0.0, 1.0],
                [2.0, 3.0],
                [np.nan, np.nan],
                [5.0, 6.0],
            ]
        ]
    )
    target_bounds = np.array(
        [
            [
                [0.0, 10.0],
                [10.0, 20.0],
            ],
            [
                [0.0, 2.5],
                [np.nan, np.nan],
            ],
        ]
    )
    source_index = np.array([0, 0])
    target_index = np.array([0, 1])
    source, target, overlap = overlap_1d.overlap_1d_nd(
        source_bounds, target_bounds, source_index, target_index
    )
    assert np.array_equal(source, [0, 1, 3, 0, 1])
    assert np.array_equal(target, [0, 0, 0, 2, 2])
    assert np.allclose(overlap, [1.0, 1.0, 1.0, 1.0, 0.5])
