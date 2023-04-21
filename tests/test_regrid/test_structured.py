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

def test_overlap_2d(grid_a_2d,grid_b_2d):
    # test overlap grid_b (target) with grid_a(source)
    # --------
    # node 0  -> node 0(25%) = 625 m 
    # node 1  -> node 0(25%) = 625 m 
    # node 1  -> node 1(25%) = 625 m 
    # node 2  -> node 1(25%) = 625 m 
    # node 2  -> node 2(25%) = 625 m 
    # node 3  -> node 2(25%) = 625 m 
    # node 4  -> node 0(25%) = 625 m 
    # node 4  -> node 3(25%) = 625 m 
    # node 5  -> node 0(25%) = 625 m 
    # node 5  -> node 1(25%) = 625 m 
    # node 5  -> node 3(25%) = 625 m 
    # node 5  -> node 4(25%) = 625 m 
    # node 6  -> node 1(25%) = 625 m 
    # node 6  -> node 2(25%) = 625 m 
    # node 6  -> node 4(25%) = 625 m 
    # node 6  -> node 5(25%) = 625 m 
    # node 7  -> node 2(25%) = 625 m 
    # node 7  -> node 5(25%) = 625 m 
    # node 8  -> node 3(25%) = 625 m 
    # node 8  -> node 6(25%) = 625 m 
    # node 9  -> node 3(25%) = 625 m 
    # node 9  -> node 4(25%) = 625 m 
    # node 9  -> node 6(25%) = 625 m 
    # node 9  -> node 7(25%) = 625 m 
    # node 10 -> node 4(25%) = 625 m 
    # node 10 -> node 5(25%) = 625 m 
    # node 10 -> node 7(25%) = 625 m 
    # node 10 -> node 8(25%) = 625 m 
    # node 11 -> node 5(25%) = 625 m 
    # node 11 -> node 8(25%) = 625 m 
    # node 12 -> node 6(25%) = 625 m 
    # node 13 -> node 6(25%) = 625 m 
    # node 13 -> node 7(25%) = 625 m 
    # node 14 -> node 7(25%) = 625 m 
    # node 14 -> node 8(25%) = 625 m 
    # node 15 -> node 8(25%) = 625 m 
    # --------
    source, target, weights = grid_b_2d.overlap(grid_a_2d, relative=False)
    sorter = np.argsort(source)
    assert np.array_equal(source[sorter], np.array([0,1,1,2,2,3,4,4,5,5,5,5,6,6,6,6,7,7,8,8,9,9,9,9,10,10,10,10,11,11,12,13,13,14,14,15]))
    assert np.array_equal(target[sorter], np.array([0,1,0,1,2,2,0,3,0,4,3,1,5,4,1,2,5,2,3,6,7,6,4,3,4,5,8,7,5,8,6,7,6,8,7,8]))
    assert np.array_equal(weights[sorter], np.array([625]*source.size))
    
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

def test_locate_centroids_2d(grid_a_2d,grid_b_2d):
    # test centroids grid_b (target) with grid_a(source), left aligned
    # --------
    # node 0 -> not valid
    # node 1 -> node 0 
    # node 2 -> node 1 
    # node 3 -> not valid
    # --------
    source, target, weights = grid_b_2d.locate_centroids(grid_a_2d)
    sorter = np.argsort(source)
    assert np.array_equal(source[sorter], np.array([1,2]))
    assert np.array_equal(target[sorter], np.array([0,1]))
    assert np.allclose(weights[sorter], np.ones(2))
    
def test_locate_centroids_2d(grid_a_2d,grid_b_2d):
    # test centroids grid_b (target) with grid_a(source), left aligned
    # --------
    # node 0-4   -> not valid
    # node 5     -> node 0
    # node 6     -> node 1
    # node 9     -> node 3
    # node 10    -> node 4
    # node 11-15 -> not valid

    # --------
    source, target, weights = grid_b_2d.locate_centroids(grid_a_2d)
    sorter = np.argsort(source)
    assert np.array_equal(source[sorter], np.array([5,6,9,10]))
    assert np.array_equal(target[sorter], np.array([0,1,3,4]))
    assert np.allclose(weights[sorter], np.ones(4))   
   
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

def test_linear_weights_2d(grid_a_2d,grid_b_2d):
    # test linear_weights grid_b (target) with grid_a(source)
    # node 0-4    ->  not valid
    # node 5      ->  node 0
    # node 5      ->  node 1
    # node 5      ->  node 3
    # node 5      ->  node 4
    # node 6      ->  node 1
    # node 6      ->  node 2
    # node 6      ->  node 4
    # node 6      ->  node 5
    # node 9      ->  node 3
    # node 9      ->  node 4
    # node 9      ->  node 6
    # node 9      ->  node 7
    # node 10     ->  node 4
    # node 10     ->  node 5
    # node 10     ->  node 7
    # node 10     ->  node 8
    # node 11-15  ->  not valid
    # --------
    source, target, weights = grid_b_2d.linear_weights(grid_a_2d)
    sorter = np.argsort(source)
    assert np.array_equal(source[sorter], np.array([5,5,5,5,6,6,6,6,9,9,9,9,10,10,10,10]))
    assert np.array_equal(target[sorter], np.array([0,1,3,4,1,2,4,5,3,4,6,7,4,5,7,8]))
    assert np.allclose(weights[sorter], np.array([0.25]*16))