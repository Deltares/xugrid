import numpy as np
import pytest

import xugrid
from xugrid.regrid.unstructured import UnstructuredGrid2d


@pytest.fixture(scope="function")
def circle():
    return UnstructuredGrid2d(xugrid.data.disk().grid)


def test_init():
    uds = xugrid.data.disk()
    assert isinstance(UnstructuredGrid2d(uds), UnstructuredGrid2d)
    assert isinstance(UnstructuredGrid2d(uds["face_z"]), UnstructuredGrid2d)
    assert isinstance(UnstructuredGrid2d(uds.ugrid.grid), UnstructuredGrid2d)
    with pytest.raises(TypeError):
        UnstructuredGrid2d(1)


def test_grid_properties(circle):
    assert circle.dims == ("mesh2d_nFaces",)
    assert circle.shape == (384,)
    assert circle.size == 384
    assert isinstance(circle.area, np.ndarray)
    assert circle.area.size == 384


@pytest.mark.parametrize("relative", [True, False])
def test_overlap(circle, relative):
    source, target, weights = circle.overlap(other=circle, relative=relative)
    valid = weights > 1.0e-5
    source = source[valid]
    target = target[valid]
    weights = weights[valid]
    sorter = np.argsort(source)
    assert np.array_equal(source[sorter], np.arange(circle.size))
    assert np.array_equal(target[sorter], np.arange(circle.size))
    if relative:
        assert np.allclose(weights[sorter], np.ones(circle.size))
    else:
        assert np.allclose(weights[sorter], circle.area)


def test_locate_centroids(circle):
    source, target, weights = circle.locate_inside(circle)
    sorter = np.argsort(source)
    assert np.array_equal(source[sorter], np.arange(circle.size))
    assert np.array_equal(target[sorter], np.arange(circle.size))
    assert np.allclose(weights[sorter], np.ones(circle.size))


def test_barycentric(circle):
    source, target, weights = circle.barycentric(circle)
    sorter = np.argsort(source)
    assert np.array_equal(source[sorter], np.arange(circle.size))
    assert np.array_equal(target[sorter], np.arange(circle.size))
    assert np.allclose(weights[sorter], np.ones(circle.size))
