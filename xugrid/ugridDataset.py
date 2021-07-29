from numpy import float64
from xugrid.ugrid import Ugrid
import xarray as xr
import pandas as pd
from imod.util import *
import types

from functools import wraps
def dataarray_wrapper(func, grid):
    @wraps(func)
    def wrapped(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, xr.DataArray):
            return UgridDataArray(func(*args, **kwargs), grid)
        else:
            return result
    return wrapped


def dataset_wrapper(func, grid):
    @wraps(func)
    def wrapped(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, xr.Dataset):
            return UgridDataset(func(*args, **kwargs), grid)
        elif isinstance(result, xr.DataArray):
            return UgridDataArray(func(*args, **kwargs), grid)
        else:
            return result
    return wrapped    

class UgridDataArray:
    """ This wraps a DataArray. """
   
    def __init__(self, obj: xr.DataArray, grid: Ugrid = None):
        self.obj = obj
        if grid is None:
           grid = Ugrid(obj)
        self.grid = grid

    def __getitem__(self, key):
        result = self.obj[key]
        if isinstance(result, xr.DataArray):
            return UgridDataArray(result, self.grid)
        else:
            return result
    
    def __setitem__(self, key, value):
        self.obj[key] = value        
        
    def __getattr__(self, attr):
        result = getattr(self.obj, attr)
        if isinstance(result, xr.DataArray):
            return UgridDataArray(result, self.grid)
        elif isinstance(result, types.MethodType):
            return dataarray_wrapper(result, self.grid)
        else:
            return result

    @property
    def ugrid(self):
        return UgridAccessor(self.obj, self.grid)



class UgridDataset:
    """ This wraps a Dataset. """

    def __init__(self, obj: xr.Dataset, grid: Ugrid = None):
        
        if grid ==None:
            self.grid = Ugrid(obj)
        else:
            self.grid = grid
        self.ds = self.grid.remove_topology(obj)

        
    def __getitem__(self, key):
        result = self.ds[key]
        if isinstance(result, xr.DataArray):
            return UgridDataArray(result, self.grid)
        elif isinstance(result, xr.Dataset):
            return UgridDataset(result, self.grid)
        else:
            return result
    
    def __setitem__(self, key, value):
        self.ds[key] = value

    def __getattr__(self, attr):
        """ Appropriately wrap result if necessary. """
        result = getattr(self.ds, attr)
        if isinstance(result, xr.DataArray):
            return UgridDataArray(result, self.grid)
        elif isinstance(result, xr.Dataset):
            return UgridDataset(result, self.grid)
        elif isinstance(result, types.MethodType):
            return dataset_wrapper(result, self.grid)
        else:
            return result

    
    
    @property
    def ugrid(self):
        return UgridAccessor(self.ds, self.grid)

    
class UgridAccessor:
    def __init__(self, obj: Union[xr.Dataset, xr.DataArray], grid: Ugrid):
        self.obj = obj
        self.grid = grid

    def plot(self):
        if self.grid._triangulation is None:
            self.grid.triangulation = mtri.Triangulation(
                x=self.grid.nodes[:, 0],
                y=self.grid.nodes[:, 1],
                triangles=self.grid.faces,
            )
        plt.tripcolor(self.grid.triangulation, self.obj.values.ravel())
    
    def object_from_face_indices(self, face_indices):
        result = self.obj.isel(face=face_indices)
        result.coords['face'] = face_indices
        if isinstance(self.obj, xr.DataArray):
            return UgridDataArray(result, self.grid)
        elif isinstance(self.obj, xr.Dataset):
            return UgridDataset(result, self.grid)
        else:
            raise Exception("illegal type in _sel_points")

        
    def sel_points(self, points_x, points_y):
        if  ( points_x is None) | (points_y is None):
             raise Exception('coordinate arrays cannot be empty')
        if points_x.shape != points_y.shape:
            raise Exception('coordinate arrays size does not match')
        if points_x.ndim !=1 :
            raise Exception('coordinate arrays must be 1d')    

        points= np.vstack([points_x, points_y]).transpose()
        face_indices = self.grid.locate_faces(points)
        return self.object_from_face_indices(face_indices)

    def _sel_slices(self, x: slice, y:slice):
        if (x.start == None) | (x.stop == None) | (y.start == None) | (y.stop == None):
            raise Exception("slice start and stop should not be None")
        elif ( x.start > x.stop) | ( y.start > y.stop):
            raise Exception("slice start should be smaller than its stop")
        elif ( not x.step is None) & ( not y.step is None):
            xcoords = [num for num in range( x.start, x.stop, x.step)]
            ycoords = [num for num in range( y.start, y.stop, y.step)]
            return self.sel_points(np.array(xcoords),np.array(ycoords))
        elif (x.step is None) & (y.step is None):
            face_indices =  self.grid.locate_faces_bounding_box(x.start, x.stop, y.start, y.stop)
            return self.object_from_face_indices(face_indices)
        else:
           raise Exception("slices should both have a stepsize, or neither should have a stepsize") 



    def sel(self, x=None, y= None):

        if isinstance(x, np.ndarray) &isinstance(y,  np.ndarray): 
            xv, yv = np.meshgrid(x, y)
            xv = xv.reshape(xv.size)
            yv = yv.reshape(yv.size)
            points = np.vstack([xv,yv]).transpose()
            return self.sel_points(points[:,0],points[:,1] )
        elif (isinstance(x, float)| isinstance(x, int)) & (isinstance(y, float)| isinstance(y, int)):
            return self.sel_points(np.array([x], float64),np.array([y], float64))
        elif  isinstance(x, slice ) & isinstance(y, slice):
            return self._sel_slices(x, y)
        else:
            raise Exception('argument mismatch')
