import sys

# append the path of the xugrid package
sys.path.append("../xugrid")
import textwrap
from xugrid import Ugrid
from xugrid import UgridDataset
from xugrid import UgridDataArray
import xarray as xr
import numpy as np
import pytest
from create_test_datasets import create_trimesh_transient, create_hexmesh_steady_state

@pytest.fixture
def triangle_transient_dataset():
    return create_trimesh_transient(10,10, 5)

@pytest.fixture
def hexmesh_invariant_dataset():
    return create_hexmesh_steady_state(11,11)


def test_createUgrid(triangle_transient_dataset):
   ugrid = Ugrid(triangle_transient_dataset)
   assert not ugrid._cell_tree is None 

def test_removeTopology(triangle_transient_dataset):
   assert len(triangle_transient_dataset.coords) == 3 
   assert len(triangle_transient_dataset.variables) == 6
   ugrid = Ugrid(triangle_transient_dataset)
   triangle_transient_dataset = ugrid.remove_topology(triangle_transient_dataset)
   assert len(triangle_transient_dataset.coords) == 1 #removed node_x and node_y
   assert len(triangle_transient_dataset.variables) == 2 #removed face_nodes, mesh2d, node_x and node_y

def test_createDataSet(triangle_transient_dataset):
    ugds = UgridDataset(triangle_transient_dataset)
    
    #verify topology was removed
    assert len(ugds.ds.coords) == 1 
    assert len(ugds.variables) == 2 

    #create other initializing both dataset (topology removed) and grid
    other_ugds = UgridDataset(ugds.ds, ugds.grid)
    assert len(other_ugds.ds.coords) == 1
    assert len(other_ugds.ds.variables) == 2

    #create other initializing both dataset (topology not removed) and grid
    other_ugds2 = UgridDataset(triangle_transient_dataset, ugds.grid)
    assert len(other_ugds2.ds.coords) == 1
    assert len(other_ugds2.ds.variables) == 2

def test_createDataArray(triangle_transient_dataset):
    da = triangle_transient_dataset.variables['data_transient']
    ugds = UgridDataset(triangle_transient_dataset)

    #create data array from xr.DataArray and grid
    ugda = UgridDataArray(da, ugds.grid)
    
    #create data array from xr.dataset 
    ugda2 = UgridDataArray(triangle_transient_dataset)
    
    assert((not ugda is None) & (not ugda2 is None))

def test_DataSet_members_reachable(triangle_transient_dataset):
    ugds = UgridDataset(triangle_transient_dataset)
    
    #check we can access dataset attributes on the UgridDataset
    assert np.array_equal(ugds.ds.coords, ugds.coords) 
    assert ugds.ds.sizes == ugds.sizes 

    #check we can call dataset methods on the UgridDataset
    t1 = ugds.ds.to_dict()
    t2 = ugds.to_dict()
    assert t1 == t2 

def test_DataArray_members_reachable(triangle_transient_dataset):
    da = triangle_transient_dataset.variables['data_transient']
    ugds = UgridDataset(triangle_transient_dataset)

    #create data array from xr.DataArray and grid
    ugda = UgridDataArray(da, ugds.grid)
    
    #check we can access dataArray attributes on the UgridDataArray    
    assert  da.dims == ugda.dims 

    #check we can access dataArray methods on the UgridDataArray    
    assert da.mean() == ugda.mean() 

def test_select_faces_from_points(hexmesh_invariant_dataset):
    points_x = np.array([5, 55,67], np.float64)
    points_y = np.array([25,34,78], np.float64)

    uds = UgridDataset(hexmesh_invariant_dataset)
    timeseries_data = uds.ugrid.sel_points(points_x, points_y)
    assert isinstance(timeseries_data, UgridDataset ) 
    assert len(timeseries_data.ds.coords["face"]) ==3 
    #the grid has an x axis from 0 to 11 , and an y axist ranging from 0 to 33 approx,
    #  so only the first point (5,25) lies inside of it.
    assert timeseries_data.ds.coords["face"].values[0]==93
    assert timeseries_data.ds.coords["face"].values[1]==-1
    assert timeseries_data.ds.coords["face"].values[2]==-1
 

def test_create_time_series_from_dataset_from_points(triangle_transient_dataset):
    points_x = np.array([5, 55,67], np.float64)
    points_y = np.array([25,34,78], np.float64)

    uds = UgridDataset(triangle_transient_dataset)
    timeseries_data = uds.ugrid.sel_points(points_x, points_y)
    assert isinstance(timeseries_data, UgridDataset ) 
    assert len(timeseries_data.ds.coords["face"]) ==3 

def test_create_time_series_from_dataset_from_arrays(triangle_transient_dataset):

    uds = UgridDataset(triangle_transient_dataset)
    timeseries_data = uds.ugrid.sel(x=np.array([10,20,30]), y=np.array([11,21,31]))
    assert isinstance(timeseries_data, UgridDataset ) 
    assert len(timeseries_data.ds.coords["face"]) ==9 

def test_create_time_series_from_dataArray_from_points(triangle_transient_dataset):
    da = triangle_transient_dataset.data_vars['data_transient']
    ugds = UgridDataset(triangle_transient_dataset)

    #create data array from xr.DataArray and grid
    ugda = UgridDataArray(da, ugds.grid)
    points_x = np.array([5, 55,67], np.float64)
    points_y = np.array([25,34,78], np.float64)

    timeseries_data = ugda.ugrid.sel_points(points_x, points_y)
    assert isinstance(timeseries_data, UgridDataArray ) 
    assert len(timeseries_data.obj.coords["face"]) ==3 


def test_create_time_series_from_dataArray_from_arrays(triangle_transient_dataset):
    uds = UgridDataset(triangle_transient_dataset)
    timeseries_data = uds.ugrid.sel(x=np.array([10,20,30]), y=np.array([11,21,31]))
    assert isinstance(timeseries_data, UgridDataset ) 
    assert len(timeseries_data.ds.coords["face"]) ==9 

def test_create_time_series_from_dataArray_from_scalars(triangle_transient_dataset):
    uds = UgridDataset(triangle_transient_dataset)
    from_int_coords = uds.ugrid.sel(x=22, y=33) 
    from_float_coords = uds.ugrid.sel(x=22.0, y=33.0)  #ints
    assert isinstance(from_int_coords, UgridDataset ) 
    assert isinstance(from_float_coords, UgridDataset ) 
    assert len(from_int_coords.ds.coords["face"]) ==1  
    assert len(from_float_coords.ds.coords["face"]) ==1  


def test_dataset_chaining(triangle_transient_dataset):
   uds = UgridDataset(triangle_transient_dataset)
   points_x = np.array([5, 55,67], np.float64)
   points_y = np.array([25,34,78], np.float64)

   chained = uds.sel(time=slice("2018-01-01", "2018-01-03")).ugrid.sel_points(points_x, points_y)
   assert  chained.dims['time']== 3 
   assert chained.dims['face'] == 3 

def test_data_array_chaining(triangle_transient_dataset):
    da = triangle_transient_dataset.data_vars['data_transient']
    ugds = UgridDataset(triangle_transient_dataset)

    #create data array from xr.DataArray and grid
    ugda = UgridDataArray(da, ugds.grid)
    points_x = np.array([5, 55,67], np.float64)
    points_y = np.array([25,34,78], np.float64)  

    chained = ugda.sel(time=slice("2018-01-01", "2018-01-03")).ugrid.sel_points(points_x, points_y)
    assert chained['time'].sizes['time']== 3 
    assert chained['face'].sizes['face'] == 3 

def test_dataset_dataframe(triangle_transient_dataset):
   uds = UgridDataset(triangle_transient_dataset)
   points_x = np.array([5, 55,67], np.float64)
   points_y = np.array([25,34,78], np.float64)  
   chained = uds.sel(time=slice("2018-01-01", "2018-01-03")).ugrid.sel_points(points_x, points_y)
   df = chained.to_dataframe()
   unstacked = df.unstack(0)
   refresult= textwrap.dedent("""              data_transient                
   face                  22      63      150
   time                                     
   2018-01-01           66.0   189.0   450.0
   2018-01-02          198.0   567.0  1350.0
   2018-01-03          396.0  1134.0  2700.0""")
   assert str(unstacked) == refresult

def test_dataarray_dataframe(triangle_transient_dataset):
   da = triangle_transient_dataset.data_vars['data_transient']
   ugds = UgridDataset(triangle_transient_dataset)

    #create data array from xr.DataArray and grid
   ugda = UgridDataArray(da, ugds.grid)
   points_x = np.array([5, 55,67], np.float64)
   points_y = np.array([25,34,78], np.float64)   

   chained = ugda.sel(time=slice("2018-01-01", "2018-01-03")).ugrid.sel_points(points_x, points_y)
   df = chained.to_dataframe()
   unstacked = df.unstack(0)
   refresult= textwrap.dedent('''        data_transient                      
   time     2018-01-01 2018-01-02 2018-01-03
   face                                     
   22             66.0      198.0      396.0
   63            189.0      567.0     1134.0
   150           450.0     1350.0     2700.0''')
   assert str(unstacked) == refresult 
def test_create_time_series_from_dataArray_from_slices(triangle_transient_dataset):
   da = triangle_transient_dataset.data_vars['data_transient']
   ugds = UgridDataset(triangle_transient_dataset)

    #create data array from xr.DataArray and grid
   ugda = UgridDataArray(da, ugds.grid)

   #use slices with a step size in this example
   chained_slice = ugda.sel(time=slice("2018-01-01", "2018-01-03")).ugrid.sel(slice(1, 51, 10), slice(2, 52, 10))
   points_x = np.array([1,11,21,31,41], np.float64)
   points_y = np.array([2,12,22,32,42], np.float64) 
   chained_points = ugda.sel(time=slice("2018-01-01", "2018-01-03")).ugrid.sel_points(points_x, points_y)

   assert chained_slice.obj.equals(chained_points.obj)

def test_create_time_series_from_dataset_from_slices(triangle_transient_dataset):
   ugds = UgridDataset(triangle_transient_dataset)

   #use slices with a step size in this example

   chained_slice = ugds.sel(time=slice("2018-01-01", "2018-01-03")).ugrid.sel(slice(1, 51, 10), slice(2, 52, 10))
   points_x = np.array([1,11,21,31,41], np.float64)
   points_y = np.array([2,12,22,32,42], np.float64) 
   chained_points = ugds.sel(time=slice("2018-01-01", "2018-01-03")).ugrid.sel_points(points_x, points_y)

   assert chained_slice.ds.equals(chained_points.ds)

def test_boundingbox(triangle_transient_dataset):
   ugds = UgridDataset(triangle_transient_dataset)  
   chained_slice = ugds.sel(time=slice("2018-01-01", "2018-01-03")).ugrid.sel(slice(1, 51), slice(2, 52))
   #test a few element numbers
   chained_slice.ds.coords['face'].values   
   assert len(chained_slice.ds.coords['face']) == 18
   assert chained_slice.ds.coords['face'].values[0] == 14
   assert chained_slice.ds.coords['face'].values[5] == 19
   assert chained_slice.ds.coords['face'].values[6] == 24
   assert chained_slice.ds.coords['face'].values[-1] == 53

