import numpy as np
import pytest

from xugrid.regrid import reduce


def forward_args():
    values = np.array([0.0, 1.0, 2.0, np.nan])
    weights = np.array([0.5, 0.5, 0.5, 0.5])
    work = np.empty_like(weights)
    return values, weights, work


def reverse_args():
    values = np.flip(np.array([0.0, 1.0, 2.0, np.nan]))
    weights = np.array([0.5, 0.5, 0.5, 0.5])
    work = np.empty_like(weights)
    return values, weights, work


@pytest.mark.parametrize("args", [forward_args(), reverse_args()])
def test_mean(args):
    actual = reduce.mean(*args)
    assert np.allclose(actual, 1.0)


@pytest.mark.parametrize("args", [forward_args(), reverse_args()])
def test_harmonic_mean(args):
    actual = reduce.harmonic_mean(*args)
    assert np.allclose(actual, 1.0 / (0.5 / 1.0 + 0.5 / 2.0))


@pytest.mark.parametrize("args", [forward_args(), reverse_args()])
def test_geometric_mean(args):
    actual = reduce.geometric_mean(*args)
    assert np.allclose(actual, np.sqrt(1.0 * 2.0))


@pytest.mark.parametrize("args", [forward_args(), reverse_args()])
def test_sum(args):
    actual = reduce.sum(*args)
    assert np.allclose(actual, 3.0)


@pytest.mark.parametrize("args", [forward_args(), reverse_args()])
def test_minimum(args):
    actual = reduce.minimum(*args)
    assert np.allclose(actual, 0.0)


@pytest.mark.parametrize("args", [forward_args(), reverse_args()])
def test_maximum(args):
    actual = reduce.maximum(*args)
    assert np.allclose(actual, 2.0)


@pytest.mark.parametrize("args", [forward_args(), reverse_args()])
def test_mode(args):
    # In case of ties, returns the last not-nan value.
    actual = reduce.mode(*args)
    assert ~np.isnan(actual)


@pytest.mark.parametrize("args", [forward_args(), reverse_args()])
def test_median(args):
    actual = reduce.median(*args)
    assert np.allclose(actual, 1.0)


@pytest.mark.parametrize("args", [forward_args(), reverse_args()])
def test_conductance(args):
    actual = reduce.conductance(*args)
    assert np.allclose(actual, 1.5)


@pytest.mark.parametrize("args", [forward_args(), reverse_args()])
def test_max_overlap(args):
    actual = reduce.max_overlap(*args)
    # It returns the last not-nan value.
    assert ~np.isnan(actual)


def test_max_overlap_extra():
    values = np.array([0.0, 1.0, 2.0, np.nan])
    weights = np.array([0.5, 1.5, 0.5, 2.5])
    workspace = np.empty_like(weights)
    args = (values, weights, workspace)
    actual = reduce.max_overlap(*args)
    assert np.allclose(actual, 1.0)


def test_mode_extra():
    values = np.array([0.0, 1.0, 1.0, 2.0, np.nan])
    weights = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    workspace = np.empty_like(weights)
    args = (values, weights, workspace)
    actual = reduce.mode(*args)
    assert np.allclose(actual, 1.0)
    # The weights shouldn't be mutated!
    assert np.allclose(weights, 0.5)

    values = np.array([1, 1, 3])
    weights = np.array([1.0, 1.0, 1.0])
    workspace = np.empty_like(weights)
    args = (values, weights, workspace)
    actual = reduce.mode(*args)
    assert np.allclose(actual, 1.0)

    values = np.array([4, 5, 6])
    weights = np.array([0.5, 0.5, 0.5])
    workspace = np.empty_like(weights)
    args = (values, weights, workspace)
    actual = reduce.mode(*args)
    # Returns last not-nan value
    assert np.allclose(actual, 6)
    assert np.allclose(weights, 0.5)


def test_percentile():
    # Simplified from:
    # https://github.com/numba/numba/blob/2001717f3321a5082c39c5787676320e699aed12/numba/tests/test_array_reductions.py#L396
    def func(x, p):
        p = np.atleast_1d(p)
        values = x.ravel()
        weights = np.empty_like(values)
        work = np.empty_like(values)
        return np.array([reduce.percentile(values, weights, work, pval) for pval in p])

    q_upper_bound = 100.0
    x = np.arange(8) * 0.5
    np.testing.assert_equal(func(x, 0), 0.0)
    np.testing.assert_equal(func(x, q_upper_bound), 3.5)
    np.testing.assert_equal(func(x, q_upper_bound / 2), 1.75)

    x = np.arange(12).reshape(3, 4)
    q = np.array((0.25, 0.5, 1.0)) * q_upper_bound
    np.testing.assert_equal(func(x, q), [2.75, 5.5, 11.0])

    x = np.arange(3 * 4 * 5 * 6).reshape(3, 4, 5, 6)
    q = np.array((0.25, 0.50)) * q_upper_bound
    np.testing.assert_equal(func(x, q).shape, (2,))

    q = np.array((0.25, 0.50, 0.75)) * q_upper_bound
    np.testing.assert_equal(func(x, q).shape, (3,))

    x = np.arange(12).reshape(3, 4)
    np.testing.assert_equal(func(x, q_upper_bound / 2), 5.5)

    np.testing.assert_equal(func(np.array([1, 2, 3]), 0), 1)

    a = np.array([2, 3, 4, 1])
    func(a, [q_upper_bound / 2])
    np.testing.assert_equal(a, np.array([2, 3, 4, 1]))


METHODS = [
    reduce.mean,
    reduce.harmonic_mean,
    reduce.geometric_mean,
    reduce.sum,
    reduce.minimum,
    reduce.maximum,
    reduce.mode,
    reduce.first_order_conservative,
    reduce.conductance,
    reduce.max_overlap,
    reduce.median,
]


@pytest.mark.parametrize("f", METHODS)
def test_weights_all_zeros(f):
    values = np.ones(5)
    weights = np.zeros(5)
    workspace = np.zeros(5)
    assert np.isnan(f(values, weights, workspace))


@pytest.mark.parametrize("f", METHODS)
def test_weights_all_nan(f):
    values = np.full(5, np.nan)
    weights = np.ones(5)
    workspace = np.zeros(5)
    assert np.isnan(f(values, weights, workspace))
