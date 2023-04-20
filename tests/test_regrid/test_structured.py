import numpy as np
import pytest

import xarray as xr
from xugrid.regrid.structured import StructuredGrid1d,StructuredGrid2d

# Testgrids
# --------
# grid a(x):               |______50_____|_____100_____|_____150_____|   
# grid b(x):        |______25_____|______75_____|_____125_____|_____175_____|  
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
def grid_a_2d(grid_a):
    return StructuredGrid2d(grid_a, "x","y")
 
@pytest.fixture
def grid_b_1d(grid_b):
    return StructuredGrid1d(grid_b, "x")
@pytest.fixture
def grid_b_2d(grid_b):
    return StructuredGrid2d(grid_b, "x","y")

def test_init_1d(grid_a):
    assert isinstance(StructuredGrid1d(grid_a, "x"), StructuredGrid1d)
    with pytest.raises(TypeError):
        StructuredGrid1d(1)
        
def test_init_2d(grid_a):
    assert isinstance(StructuredGrid2d(grid_a, "x","y"), StructuredGrid2d)
    with pytest.raises(TypeError):
        StructuredGrid2d(1)
        
def test_overlap_1d(grid_a_1d,grid_b_1d):
    # test overlap grid_b (target) with grid_a(source)
    # --------
    # node 0 -> node 0 (50%) = 25 m should be not valid?
    # node 1 -> node 0 (50%) = 25 m
    # node 1 -> node 1 (50%) = 25 m
    # node 2 -> node 1 (50%) = 25 m
    # node 2 -> node 2 (50%) = 25 m
    # node 3 -> node 2 (50%) = 25 m should be not valid?
    # --------
    source, target, weights = grid_b_1d.overlap(grid_a_1d, relative=False)
    sorter = np.argsort(source)
    assert np.array_equal(source[sorter], np.array([0,1,1,2,2,3]))
    assert np.array_equal(target[sorter], np.array([0,0,1,1,2,2]))
    assert np.array_equal(weights[sorter], np.array([25,25,25,25,25,25]))

def test_overlap_2d(grid_a_1d,grid_b_1d):
    # test overlap grid_b (target) with grid_a(source)
    # --------
    # node 0 -> node 0 (50%) = 25 m should be not valid?
    # node 1 -> node 0 (50%) = 25 m
    # node 1 -> node 1 (50%) = 25 m
    # node 2 -> node 1 (50%) = 25 m
    # node 2 -> node 2 (50%) = 25 m
    # node 3 -> node 2 (50%) = 25 m should be not valid?
    # --------
    source, target, weights = grid_b_1d.overlap(grid_a_1d, relative=False)
    sorter = np.argsort(source)
    assert np.array_equal(source[sorter], np.array([0,1,1,2,2,3]))
    assert np.array_equal(target[sorter], np.array([0,0,1,1,2,2]))
    assert np.array_equal(weights[sorter], np.array([25,25,25,25,25,25]))
    
def test_locate_centroids_1d(grid_a_1d,grid_b_1d):
    # test centroids grid_b (target) with grid_a(source), left aligned
    # --------
    # node 0 -> not valid
    # node 1 -> node 0 
    # node 2 -> node 1 
    # node 3 -> not valid
    # --------
    source, target, weights = grid_b_1d.locate_centroids(grid_a_1d)
    sorter = np.argsort(source)
    assert np.array_equal(source[sorter], np.array([1,2]))
    assert np.array_equal(target[sorter], np.array([0,1]))
    assert np.allclose(weights[sorter], np.ones(2))

def test_locate_centroids_1d(grid_a_1d,grid_c_1d):
    # test centroids grid_b (target) with grid_a(source), left aligned
    # --------
    # node 0 -> not valid
    # node 1 -> node 0 
    # node 2 -> node 1 
    # node 3 -> not valid
    # --------
    source, target, weights = grid_c_1d.locate_centroids(grid_a_1d)
    sorter = np.argsort(source)
    assert np.array_equal(source[sorter], np.array([1,2]))
    assert np.array_equal(target[sorter], np.array([0,1]))
    assert np.allclose(weights[sorter], np.ones(2))
    
    
def test_linear_weights_1d(grid_a_1d,grid_b_1d):
    # test linear_weights grid_b (target) with grid_a(source)
    # --------
    # node 0 -> not valid
    # node 1 -> node 0 = 0.5
    # node 1 -> node 1 = 0.5
    # node 2 -> node 1 = 0.5
    # node 2 -> node 2 = 0.5
    # node 3 -> not valid
    # --------
    source, target, weights = grid_b_1d.linear_weights(grid_a_1d)
    sorter = np.argsort(target)
    assert np.array_equal(source[sorter], np.array([1,1,2,2]))
    assert np.array_equal(target[sorter], np.array([0,1,1,2]))
    assert np.allclose(weights[sorter], np.array([0.5,0.5,0.5,0.5]))
