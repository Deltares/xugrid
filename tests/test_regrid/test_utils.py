import numpy as np

from xugrid.regrid import utils


def test_create_linear_index():
    index_a = [0, 0, 1]
    index_b = [0]
    shape = (2, 1)
    actual = utils.create_linear_index((index_a, index_b), shape)
    expected = np.array([0, 0, 1])
    assert np.array_equal(actual, expected)

    index_a = [0, 0, 1, 1]
    index_b = [0, 1, 2]
    shape = (2, 3)
    actual = utils.create_linear_index((index_a, index_b), shape)
    expected = np.array([0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5])
    assert np.array_equal(actual, expected)


def test_create_weights():
    weights_a = [0.25, 0.25, 0.25, 0.25]
    weights_b = [0.5, 0.5]
    actual = utils.create_weights((weights_a, weights_b))
    expected = np.full(8, 0.125)
    assert np.allclose(actual, expected)


def test_broadcast():
    source_shape = (3, 2)
    target_shape = (6, 4)
    index_src_a = [0, 0, 1, 1]
    index_src_b = [0, 1]
    index_tgt_a = [2, 2, 3, 3]
    index_tgt_b = [2, 3]
    weights_a = [0.5, 0.5, 0.5, 0.5]
    weights_b = [0.5, 0.5]
    actual_src, actual_tgt, actual_weights = utils.broadcast(
        source_shape,
        target_shape,
        (index_src_a, index_src_b),
        (index_tgt_a, index_tgt_b),
        (weights_a, weights_b),
    )
    expected_src = np.array([0, 1, 0, 1, 2, 3, 2, 3])
    expected_tgt = np.array([10, 11, 10, 11, 14, 15, 14, 15])
    expected_weights = np.full(8, 0.25)
    assert np.array_equal(actual_src, expected_src)
    assert np.array_equal(actual_tgt, expected_tgt)
    assert np.allclose(actual_weights, expected_weights)


def test_alt_cumsum():
    a = np.ones(5)
    assert np.array_equal(utils.alt_cumsum(a), np.arange(5))

    a = np.array([1, 3, 4])
    assert np.array_equal(utils.alt_cumsum(a), [0, 1, 4])
