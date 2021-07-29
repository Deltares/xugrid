import sys
  
# append the path of the xugrid package
sys.path.append("../xugrid")

from xugrid import Ugrid
from xugrid import UgridDataset
from xugrid import UgridDataArray
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def test_createUgrid():

   ds = xr.open_dataset(r"D:\dev\imodPython_exampleFiles\tri-time-test1.nc")
   ugrid = Ugrid(ds)
   assert not ugrid._cell_tree is None 

def test_removeTopology():

   ds = xr.open_dataset(r"D:\dev\imodPython_exampleFiles\tri-time-test1.nc")
   assert len(ds.coords) == 3 
   assert len(ds.variables) == 6
   ugrid = Ugrid(ds)
   ds = ugrid.remove_topology(ds)
   assert len(ds.coords) == 1 #removed node_x and node_y
   assert len(ds.variables) == 2 #removed face_nodes, mesh2d, node_x and node_y

def test_createDataSet():
    ds = xr.open_dataset(r"D:\dev\imodPython_exampleFiles\tri-time-test1.nc")
    ugds = UgridDataset(ds)
    
    #verify topology was removed
    assert len(ugds.ds.coords) == 1 
    assert len(ugds.variables) == 2 

    #create other initializing both dataset (topology removed) and grid
    other_ugds = UgridDataset(ugds.ds, ugds.grid)
    assert len(other_ugds.ds.coords) == 1
    assert len(other_ugds.ds.variables) == 2

    #create other initializing both dataset (topology not removed) and grid
    other_ugds2 = UgridDataset(ds, ugds.grid)
    assert len(other_ugds2.ds.coords) == 1
    assert len(other_ugds2.ds.variables) == 2

def test_createDataArray():
    ds = xr.open_dataset(r"D:\dev\imodPython_exampleFiles\tri-time-test1.nc")
    da = ds.variables['data']
    ugds = UgridDataset(ds)

    #create data array from xr.DataArray and grid
    ugda = UgridDataArray(da, ugds.grid)
    
    #create data array from xr.dataset 
    ugda2 = UgridDataArray(ds)
    
    assert((not ugda is None) & (not ugda2 is None))

def test_DataSet_members_reachable():
    ds = xr.open_dataset(r"D:\dev\imodPython_exampleFiles\tri-time-test1.nc")
    ugds = UgridDataset(ds)
    
    #check we can access dataset attributes on the UgridDataset
    assert np.array_equal(ugds.ds.coords, ugds.coords) 
    assert ugds.ds.sizes == ugds.sizes 

    #check we can call dataset methods on the UgridDataset
    t1 = ugds.ds.to_dict()
    t2 = ugds.to_dict()
    assert t1 == t2 

def test_DataArray_members_reachable():
    ds = xr.open_dataset(r"D:\dev\imodPython_exampleFiles\tri-time-test1.nc")
    da = ds.variables['data']
    ugds = UgridDataset(ds)

    #create data array from xr.DataArray and grid
    ugda = UgridDataArray(da, ugds.grid)
    
    #check we can access dataArray attributes on the UgridDataArray    
    assert  da.dims == ugda.dims 

    #check we can access dataArray methods on the UgridDataArray    
    assert da.mean() == ugda.mean() 

def test_create_time_series_from_dataset_from_points():
    ds = xr.open_dataset(r"D:\dev\imodPython_exampleFiles\tri-time-test1.nc")
    points_x = np.array([5, 55,67], np.float64)
    points_y = np.array([25,34,78], np.float64)

    uds = UgridDataset(ds)
    timeseries_data = uds.ugrid.sel_points(points_x, points_y)
    assert isinstance(timeseries_data, UgridDataset ) 
    assert len(timeseries_data.ds.coords["face"]) ==3 

def test_create_time_series_from_dataset_from_arrays():
    ds = xr.open_dataset(r"D:\dev\imodPython_exampleFiles\tri-time-test1.nc")

    uds = UgridDataset(ds)
    timeseries_data = uds.ugrid.sel(x=np.array([10,20,30]), y=np.array([11,21,31]))
    assert isinstance(timeseries_data, UgridDataset ) 
    assert len(timeseries_data.ds.coords["face"]) ==9 

def test_create_time_series_from_dataArray_from_points():
    ds = xr.open_dataset(r"D:\dev\imodPython_exampleFiles\tri-time-test1.nc")
    da = ds.data_vars['data']
    ugds = UgridDataset(ds)

    #create data array from xr.DataArray and grid
    ugda = UgridDataArray(da, ugds.grid)
    points_x = np.array([5, 55,67], np.float64)
    points_y = np.array([25,34,78], np.float64)

    timeseries_data = ugda.ugrid.sel_points(points_x, points_y)
    assert isinstance(timeseries_data, UgridDataArray ) 
    assert len(timeseries_data.obj.coords["face"]) ==3 


def test_create_time_series_from_dataArray_from_arrays():
    ds = xr.open_dataset(r"D:\dev\imodPython_exampleFiles\tri-time-test1.nc")

    uds = UgridDataset(ds)
    timeseries_data = uds.ugrid.sel(x=np.array([10,20,30]), y=np.array([11,21,31]))
    assert isinstance(timeseries_data, UgridDataset ) 
    assert len(timeseries_data.ds.coords["face"]) ==9 

def test_create_time_series_from_dataArray_from_scalars():
    ds = xr.open_dataset(r"D:\dev\imodPython_exampleFiles\tri-time-test1.nc")

    uds = UgridDataset(ds)
    from_int_coords = uds.ugrid.sel(x=22, y=33) 
    from_float_coords = uds.ugrid.sel(x=22.0, y=33.0)  #ints
    assert isinstance(from_int_coords, UgridDataset ) 
    assert isinstance(from_float_coords, UgridDataset ) 
    assert len(from_int_coords.ds.coords["face"]) ==1  
    assert len(from_float_coords.ds.coords["face"]) ==1  


def test_dataset_chaining():
   ds = xr.open_dataset(r"D:\dev\imodPython_exampleFiles\tri-time-test1.nc")
   uds = UgridDataset(ds)
   points_x = np.array([5, 55,67], np.float64)
   points_y = np.array([25,34,78], np.float64)

   chained = uds.sel(time=slice("2018-01-01", "2018-01-03")).ugrid.sel_points(points_x, points_y)
   assert  chained.dims['time']== 3 
   assert chained.dims['face'] == 3 

def test_data_array_chaining():
    ds = xr.open_dataset(r"D:\dev\imodPython_exampleFiles\tri-time-test1.nc")
    da = ds.data_vars['data']
    ugds = UgridDataset(ds)

    #create data array from xr.DataArray and grid
    ugda = UgridDataArray(da, ugds.grid)
    points_x = np.array([5, 55,67], np.float64)
    points_y = np.array([25,34,78], np.float64)  

    chained = ugda.sel(time=slice("2018-01-01", "2018-01-03")).ugrid.sel_points(points_x, points_y)
    assert chained['time'].sizes['time']== 3 
    assert chained['face'].sizes['face'] == 3 

def test_dataset_dataframe():
   ds = xr.open_dataset(r"D:\dev\imodPython_exampleFiles\tri-time-test1.nc")
   uds = UgridDataset(ds)
   points_x = np.array([5, 55,67], np.float64)
   points_y = np.array([25,34,78], np.float64)  
   chained = uds.sel(time=slice("2018-01-01", "2018-01-03")).ugrid.sel_points(points_x, points_y)
   df = chained.to_dataframe()
   unstacked = df.unstack(0)
   refresult= """                data                    
face             54        194       108
time                                    
2018-01-01  0.641110  0.987178  0.432133
2018-01-02  1.449409  1.695637  0.662104
2018-01-03  2.416647  2.178768  0.753665"""
   assert str(unstacked) == refresult

def test_dataarray_dataframe():
   ds = xr.open_dataset(r"D:\dev\imodPython_exampleFiles\tri-time-test1.nc")
   da = ds.data_vars['data']
   ugds = UgridDataset(ds)

    #create data array from xr.DataArray and grid
   ugda = UgridDataArray(da, ugds.grid)
   points_x = np.array([5, 55,67], np.float64)
   points_y = np.array([25,34,78], np.float64)   

   chained = ugda.sel(time=slice("2018-01-01", "2018-01-03")).ugrid.sel_points(points_x, points_y)
   df = chained.to_dataframe()
   unstacked = df.unstack(0)
   refresult= """           data                      
time 2018-01-01 2018-01-02 2018-01-03
face                                 
54     0.641110   1.449409   2.416647
194    0.987178   1.695637   2.178768
108    0.432133   0.662104   0.753665"""
   assert str(unstacked) == refresult 
def test_create_time_series_from_dataArray_from_slices():
   ds = xr.open_dataset(r"D:\dev\imodPython_exampleFiles\tri-time-test1.nc")
   da = ds.data_vars['data']
   ugds = UgridDataset(ds)

    #create data array from xr.DataArray and grid
   ugda = UgridDataArray(da, ugds.grid)

   #use slices with a step size in this example
   chained_slice = ugda.sel(time=slice("2018-01-01", "2018-01-03")).ugrid.sel(slice(1, 51, 10), slice(2, 52, 10))
   points_x = np.array([1,11,21,31,41], np.float64)
   points_y = np.array([2,12,22,32,42], np.float64) 
   chained_points = ugda.sel(time=slice("2018-01-01", "2018-01-03")).ugrid.sel_points(points_x, points_y)

   assert chained_slice.obj.equals(chained_points.obj)

def test_create_time_series_from_dataset_from_slices():
   ds = xr.open_dataset(r"D:\dev\imodPython_exampleFiles\tri-time-test1.nc")
   ugds = UgridDataset(ds)

   #use slices with a step size in this example

   chained_slice = ugds.sel(time=slice("2018-01-01", "2018-01-03")).ugrid.sel(slice(1, 51, 10), slice(2, 52, 10))
   points_x = np.array([1,11,21,31,41], np.float64)
   points_y = np.array([2,12,22,32,42], np.float64) 
   chained_points = ugds.sel(time=slice("2018-01-01", "2018-01-03")).ugrid.sel_points(points_x, points_y)

   assert chained_slice.ds.equals(chained_points.ds)

def test_boundingbox():
   ds = xr.open_dataset(r"D:\dev\imodPython_exampleFiles\tri-time-test1.nc")
   ugds = UgridDataset(ds)    
   chained_slice = ugds.sel(time=slice("2018-01-01", "2018-01-03")).ugrid.sel(slice(1, 51), slice(2, 52))


test_boundingbox()
test_createUgrid()
test_removeTopology()
test_createDataSet()
test_DataSet_members_reachable()
test_createDataArray()
test_DataArray_members_reachable()
test_create_time_series_from_dataset_from_points()
test_create_time_series_from_dataset_from_arrays()
test_create_time_series_from_dataArray_from_points()
test_create_time_series_from_dataArray_from_arrays()
test_create_time_series_from_dataArray_from_scalars()
test_dataset_chaining()
test_data_array_chaining()
test_dataset_dataframe()
test_dataarray_dataframe()
test_dataarray_dataframe()
test_create_time_series_from_dataArray_from_slices()
test_create_time_series_from_dataset_from_slices()


   
   