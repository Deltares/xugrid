import numpy as np
import pytest

import xarray as xr
from xugrid.regrid.structured import StructuredGrid1d

# Testgrids
# --------
# grid a:         |______50_____|_____100_____|_____150_____|   
# grid b:  |______25_____|______75_____|_____125_____|_____175_____|
# --------

@pytest.fixture
def grid_a():
    return xr.DataArray(
        data=np.arange(9).reshape((3,3)),
        dims=["y", "x"],
        coords={
            "y": np.array([150, 100, 50]),
            "x": np.array([50, 100, 150]),
            "dx": 50.0,
            "dy": -50.0,
        },
    )
@pytest.fixture
def grid_b():
    return xr.DataArray(
        data=np.arange(16).reshape((4,4)),
        dims=["y", "x"],
        coords={
            "y": np.array([175, 125, 75, 25]),
            "x": np.array([25, 75, 125, 175]),
            "dx": 50.0,
            "dy": -50.0,
        },
    )
@pytest.fixture
def grid_a_1d(grid_a):
    return StructuredGrid1d(grid_a, "x")
    
@pytest.fixture
def grid_b_1d(grid_b):
    return StructuredGrid1d(grid_b, "x")


def test_init(grid_a):
    assert isinstance(StructuredGrid1d(grid_a, "x"), StructuredGrid1d)
    with pytest.raises(TypeError):
        StructuredGrid1d(1)
        
        

def test_overlap(grid_a_1d,grid_b_1d):
    # test overlap grid_b (target) with grid_a(source)
    # --------
    # node 0 -> node 0 (50%) = 25 m, sum = 25 m
    # node 1 -> node 0 (50%) = 25 m
    # node 1 -> node 1 (50%) = 25 m, sum = 50 m
    # node 2 -> node 1 (50%) = 25 m
    # node 2 -> node 2 (50%) = 25 m, sum = 50 m
    # node 3 -> node 2 (50%) = 25 m, sum = 25 m
    # --------
    source, target, weights = grid_b_1d.overlap(grid_a_1d, relative=False)
    sorter = np.argsort(source)
    assert np.array_equal(source[sorter], np.array([0,1,1,2,2,3]))
    assert np.array_equal(target[sorter], np.array([0,0,1,1,2,2]))
    assert np.array_equal(weights[sorter], np.array([25,25,25,25,25,25]))


def test_locate_centroids(grid_a_1d,grid_b_1d):
    # test centroids grid_b (target) with grid_a(source), left aligned
    # --------
    # node 0 -> none
    # node 1 -> node 0 
    # node 2 -> node 1 
    # node 3 -> node 2 
    # --------
    source, target, weights = grid_b_1d.locate_centroids(grid_a_1d)
    sorter = np.argsort(source)
    assert np.array_equal(source[sorter], np.array([1,2,3]))
    assert np.array_equal(target[sorter], np.array([0,1,2]))
    assert np.allclose(weights[sorter], np.ones(grid_a_1d.size))


def test_linear_weights(grid_a_1d,grid_b_1d):
    # test linear_weights grid_b (target) with grid_a(source)
    # --------
    # node 0 -> node 0 = 0.5
    # node 1 -> node 0 = 0.5
    # node 1 -> node 1 = 0.5
    # node 2 -> node 1 = 0.5
    # node 2 -> node 2 = 0.5
    # node 3 -> node 2 = 0.5
    # --------
    source, target, weights = grid_b_1d.linear_weights(grid_a_1d)
    sorter = np.argsort(target)
    assert np.array_equal(source[sorter], np.array([0,0,1,1,2,2]))
    assert np.array_equal(target[sorter], np.array([0,1,1,2,2,3]))
    assert np.allclose(weights[sorter], np.array([0.5,0.5,0.5,0.5,0.5,0.5]))


# def test_grid_properties(circle):
#     assert circle.dims == ("mesh2d_nFaces",)
#     assert circle.shape == (384,)
#     assert circle.size == 384
#     assert isinstance(circle.area, np.ndarray)
#     assert circle.area.size == 384
#
#
# @pytest.mark.parametrize("relative", [True, False])
# def test_overlap(circle, relative):
#     source, target, weights = circle.overlap(other=circle, relative=relative)
#     valid = weights > 1.0e-5
#     source = source[valid]
#     target = target[valid]
#     weights = weights[valid]
#     sorter = np.argsort(source)
#     assert np.array_equal(source[sorter], np.arange(circle.size))
#     assert np.array_equal(target[sorter], np.arange(circle.size))
#     if relative:
#         assert np.allclose(weights[sorter], np.ones(circle.size))
#     else:
#         assert np.allclose(weights[sorter], circle.area)
#
