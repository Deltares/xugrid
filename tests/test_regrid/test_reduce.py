import numpy as np
import pytest

from xugrid.regrid import reduce


@pytest.fixture(scope="function")
def args():
    values = np.array([0.0, 1.0, 2.0, np.nan, 3.0, 4.0])
    indices = np.array([0, 1, 2, 3])
    weights = np.array([0.5, 0.5, 0.5, 0.5])
    return values, indices, weights


def test_mean(args):
    actual = reduce.mean(*args)
    assert np.allclose(actual, 1.0)


def test_harmonic_mean(args):
    actual = reduce.harmonic_mean(*args)
    assert np.allclose(actual, 1.0 / (0.5 / 1.0 + 0.5 / 2.0))


def test_geometric_mean(args):
    actual = reduce.geometric_mean(*args)
    assert np.allclose(actual, np.sqrt(1.0 * 2.0))


def test_sum(args):
    actual = reduce.sum(*args)
    assert np.allclose(actual, 3.0)


def test_minimum(args):
    actual = reduce.minimum(*args)
    assert np.allclose(actual, 0.0)


def test_maximum(args):
    actual = reduce.maximum(*args)
    assert np.allclose(actual, 2.0)


def test_mode(args):
    actual = reduce.mode(*args)
    assert np.allclose(actual, 0.0)

    values = np.array([0.0, 1.0, 1.0, 2.0, np.nan])
    indices = np.array([0, 1, 2, 3, 4])
    weights = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    args = (values, indices, weights)
    actual = reduce.mode(*args)
    assert np.allclose(actual, 1.0)


def test_median(args):
    actual = reduce.median(*args)
    assert np.allclose(actual, 1.0)


def test_conductance(args):
    actual = reduce.conductance(*args)
    assert np.allclose(actual, 1.5)


def test_max_overlap(args):
    actual = reduce.max_overlap(*args)
    assert np.allclose(actual, 0.0)

    values = np.array([0.0, 1.0, 2.0, np.nan])
    indices = np.array([0, 1, 2, 3])
    weights = np.array([0.5, 1.5, 0.5, 2.5])
    args = (values, indices, weights)
    actual = reduce.max_overlap(*args)
    assert np.allclose(actual, 1.0)
