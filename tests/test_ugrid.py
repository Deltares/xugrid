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

@pytest.fixture
def triangle_dataset():
    return xr.open_dataset(r"./tests/test_data/tri-time-test1.nc")

def test_createUgrid(triangle_dataset):
   ugrid = Ugrid(triangle_dataset)
   assert not ugrid._cell_tree is None 

def test_removeTopology(triangle_dataset):
   assert len(triangle_dataset.coords) == 3 
   assert len(triangle_dataset.variables) == 6
   ugrid = Ugrid(triangle_dataset)
   triangle_dataset = ugrid.remove_topology(triangle_dataset)
   assert len(triangle_dataset.coords) == 1 #removed node_x and node_y
   assert len(triangle_dataset.variables) == 2 #removed face_nodes, mesh2d, node_x and node_y

def test_createDataSet(triangle_dataset):
    ugds = UgridDataset(triangle_dataset)
    
    #verify topology was removed
    assert len(ugds.ds.coords) == 1 
    assert len(ugds.variables) == 2 

    #create other initializing both dataset (topology removed) and grid
    other_ugds = UgridDataset(ugds.ds, ugds.grid)
    assert len(other_ugds.ds.coords) == 1
    assert len(other_ugds.ds.variables) == 2

    #create other initializing both dataset (topology not removed) and grid
    other_ugds2 = UgridDataset(triangle_dataset, ugds.grid)
    assert len(other_ugds2.ds.coords) == 1
    assert len(other_ugds2.ds.variables) == 2

def test_createDataArray(triangle_dataset):
    da = triangle_dataset.variables['data']
    ugds = UgridDataset(triangle_dataset)

    #create data array from xr.DataArray and grid
    ugda = UgridDataArray(da, ugds.grid)
    
    #create data array from xr.dataset 
    ugda2 = UgridDataArray(triangle_dataset)
    
    assert((not ugda is None) & (not ugda2 is None))

def test_DataSet_members_reachable(triangle_dataset):
    ugds = UgridDataset(triangle_dataset)
    
    #check we can access dataset attributes on the UgridDataset
    assert np.array_equal(ugds.ds.coords, ugds.coords) 
    assert ugds.ds.sizes == ugds.sizes 

    #check we can call dataset methods on the UgridDataset
    t1 = ugds.ds.to_dict()
    t2 = ugds.to_dict()
    assert t1 == t2 

def test_DataArray_members_reachable(triangle_dataset):
    da = triangle_dataset.variables['data']
    ugds = UgridDataset(triangle_dataset)

    #create data array from xr.DataArray and grid
    ugda = UgridDataArray(da, ugds.grid)
    
    #check we can access dataArray attributes on the UgridDataArray    
    assert  da.dims == ugda.dims 

    #check we can access dataArray methods on the UgridDataArray    
    assert da.mean() == ugda.mean() 

def test_create_time_series_from_dataset_from_points(triangle_dataset):
    points_x = np.array([5, 55,67], np.float64)
    points_y = np.array([25,34,78], np.float64)

    uds = UgridDataset(triangle_dataset)
    timeseries_data = uds.ugrid.sel_points(points_x, points_y)
    assert isinstance(timeseries_data, UgridDataset ) 
    assert len(timeseries_data.ds.coords["face"]) ==3 

def test_create_time_series_from_dataset_from_arrays(triangle_dataset):

    uds = UgridDataset(triangle_dataset)
    timeseries_data = uds.ugrid.sel(x=np.array([10,20,30]), y=np.array([11,21,31]))
    assert isinstance(timeseries_data, UgridDataset ) 
    assert len(timeseries_data.ds.coords["face"]) ==9 

def test_create_time_series_from_dataArray_from_points(triangle_dataset):
    da = triangle_dataset.data_vars['data']
    ugds = UgridDataset(triangle_dataset)

    #create data array from xr.DataArray and grid
    ugda = UgridDataArray(da, ugds.grid)
    points_x = np.array([5, 55,67], np.float64)
    points_y = np.array([25,34,78], np.float64)

    timeseries_data = ugda.ugrid.sel_points(points_x, points_y)
    assert isinstance(timeseries_data, UgridDataArray ) 
    assert len(timeseries_data.obj.coords["face"]) ==3 


def test_create_time_series_from_dataArray_from_arrays(triangle_dataset):
    uds = UgridDataset(triangle_dataset)
    timeseries_data = uds.ugrid.sel(x=np.array([10,20,30]), y=np.array([11,21,31]))
    assert isinstance(timeseries_data, UgridDataset ) 
    assert len(timeseries_data.ds.coords["face"]) ==9 

def test_create_time_series_from_dataArray_from_scalars(triangle_dataset):
    uds = UgridDataset(triangle_dataset)
    from_int_coords = uds.ugrid.sel(x=22, y=33) 
    from_float_coords = uds.ugrid.sel(x=22.0, y=33.0)  #ints
    assert isinstance(from_int_coords, UgridDataset ) 
    assert isinstance(from_float_coords, UgridDataset ) 
    assert len(from_int_coords.ds.coords["face"]) ==1  
    assert len(from_float_coords.ds.coords["face"]) ==1  


def test_dataset_chaining(triangle_dataset):
   uds = UgridDataset(triangle_dataset)
   points_x = np.array([5, 55,67], np.float64)
   points_y = np.array([25,34,78], np.float64)

   chained = uds.sel(time=slice("2018-01-01", "2018-01-03")).ugrid.sel_points(points_x, points_y)
   assert  chained.dims['time']== 3 
   assert chained.dims['face'] == 3 

def test_data_array_chaining(triangle_dataset):
    da = triangle_dataset.data_vars['data']
    ugds = UgridDataset(triangle_dataset)

    #create data array from xr.DataArray and grid
    ugda = UgridDataArray(da, ugds.grid)
    points_x = np.array([5, 55,67], np.float64)
    points_y = np.array([25,34,78], np.float64)  

    chained = ugda.sel(time=slice("2018-01-01", "2018-01-03")).ugrid.sel_points(points_x, points_y)
    assert chained['time'].sizes['time']== 3 
    assert chained['face'].sizes['face'] == 3 

def test_dataset_dataframe(triangle_dataset):
   uds = UgridDataset(triangle_dataset)
   points_x = np.array([5, 55,67], np.float64)
   points_y = np.array([25,34,78], np.float64)  
   chained = uds.sel(time=slice("2018-01-01", "2018-01-03")).ugrid.sel_points(points_x, points_y)
   df = chained.to_dataframe()
   unstacked = df.unstack(0)
   refresult= textwrap.dedent("""                   data                    
   face             54        194       108
   time                                    
   2018-01-01  0.641110  0.987178  0.432133
   2018-01-02  1.449409  1.695637  0.662104
   2018-01-03  2.416647  2.178768  0.753665""")
   assert str(unstacked) == refresult

def test_dataarray_dataframe(triangle_dataset):
   da = triangle_dataset.data_vars['data']
   ugds = UgridDataset(triangle_dataset)

    #create data array from xr.DataArray and grid
   ugda = UgridDataArray(da, ugds.grid)
   points_x = np.array([5, 55,67], np.float64)
   points_y = np.array([25,34,78], np.float64)   

   chained = ugda.sel(time=slice("2018-01-01", "2018-01-03")).ugrid.sel_points(points_x, points_y)
   df = chained.to_dataframe()
   unstacked = df.unstack(0)
   refresult= textwrap.dedent("""              data                      
   time 2018-01-01 2018-01-02 2018-01-03
   face                                 
   54     0.641110   1.449409   2.416647
   194    0.987178   1.695637   2.178768
   108    0.432133   0.662104   0.753665""")
   assert str(unstacked) == refresult 
def test_create_time_series_from_dataArray_from_slices(triangle_dataset):
   da = triangle_dataset.data_vars['data']
   ugds = UgridDataset(triangle_dataset)

    #create data array from xr.DataArray and grid
   ugda = UgridDataArray(da, ugds.grid)

   #use slices with a step size in this example
   chained_slice = ugda.sel(time=slice("2018-01-01", "2018-01-03")).ugrid.sel(slice(1, 51, 10), slice(2, 52, 10))
   points_x = np.array([1,11,21,31,41], np.float64)
   points_y = np.array([2,12,22,32,42], np.float64) 
   chained_points = ugda.sel(time=slice("2018-01-01", "2018-01-03")).ugrid.sel_points(points_x, points_y)

   assert chained_slice.obj.equals(chained_points.obj)

def test_create_time_series_from_dataset_from_slices(triangle_dataset):
   ugds = UgridDataset(triangle_dataset)

   #use slices with a step size in this example

   chained_slice = ugds.sel(time=slice("2018-01-01", "2018-01-03")).ugrid.sel(slice(1, 51, 10), slice(2, 52, 10))
   points_x = np.array([1,11,21,31,41], np.float64)
   points_y = np.array([2,12,22,32,42], np.float64) 
   chained_points = ugds.sel(time=slice("2018-01-01", "2018-01-03")).ugrid.sel_points(points_x, points_y)

   assert chained_slice.ds.equals(chained_points.ds)

def test_boundingbox(triangle_dataset):
   ugds = UgridDataset(triangle_dataset)    
   chained_slice = ugds.sel(time=slice("2018-01-01", "2018-01-03")).ugrid.sel(slice(1, 51), slice(2, 52))
   #test a few element numbers
   assert len(chained_slice.ds.coords['face']) == 32
   assert chained_slice.ds.coords['face'].values[0] == 30
   assert chained_slice.ds.coords['face'].values[5] == 35
   assert chained_slice.ds.coords['face'].values[6] == 50
   assert chained_slice.ds.coords['face'].values[-1] == 197

