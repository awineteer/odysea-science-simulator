import numpy as np
import xarray as xr
import glob
import os
import pandas as pd

from odysim import utils
    
    
# class WebGriddedModel:
# TODO: Implement the LLC 4320 model through the xmitgcm llcreader interface
#       or some other cloud readable interface.

class GriddedModel:
    
    """
    
    A class that holds functions for loading and co-locating ocean/atmosphere model data.
    Ocean model data is expected to be gridded in lat/lon with individual files for each
    time step. 
    
    Take this code as a starting point and adapt to your own specific model as needed.
    
    """
    

    def __init__(self,model_folder='/u/bura-m0/hectorg/COAS/llc2160/HighRes/',
                 u_folder='U',v_folder='V',tau_x_folder='oceTAUX',tau_y_folder='oceTAUY',
                 u_varname='U',v_varname='V',tau_x_varname='oceTAUX',tau_y_varname='oceTAUY',
                 search_string = '/*.nc',preprocess=None,n_files=-1):

        """
        Initialize a GriddedModel object.
        
        Args:
            model_folder (str): Top-level folder for model data. Contains sub folders for each variable.
            u_folder (str): Sub-folder containing model U current data (East-West currents).
            v_folder (str): Sub-folder containing model V current data (North-South currents).
            tau_x_folder (str): Sub-folder containing model U wind stress current data (East-West wind stress).
            tau_y_folder (str): Sub-folder containing model V wind stress current data (North-South wind stress).
            u_varname (str): Variable name inside model netcdf files for U current.
            v_varname (str): Variable name inside model netcdf files for V current.
            tau_x_varname (str): Variable name inside model netcdf files for U wind stress.
            tau_y_varname (str): Variable name inside model netcdf files for V wind stress.
            search_string (str): File extension for model data files.
            preprocess (function): function to pass to xarray.open_mfdataset for preprocessing.
            n_files (int): number of files to load, 0:n_files. Used to reduce load if many files are available in the model folder.
            
        Returns:
            GriddedModel obect

        """


        u_search = os.path.join(model_folder, u_folder)
        v_search = os.path.join(model_folder, v_folder)
        tau_x_search = os.path.join(model_folder, tau_x_folder)
        tau_y_search = os.path.join(model_folder, tau_y_folder)
        
        u_files = np.sort(glob.glob(u_search + '/*.nc'))[0:n_files]
        v_files = np.sort(glob.glob(v_search + '/*.nc'))[0:n_files]
        tau_x_files = np.sort(glob.glob(tau_x_search + '/*.nc'))[0:n_files]
        tau_y_files = np.sort(glob.glob(tau_y_search + '/*.nc'))[0:n_files]

        self.U = xr.open_mfdataset(u_files,parallel=True,preprocess=preprocess)
        self.V = xr.open_mfdataset(v_files,parallel=True,preprocess=preprocess)
        self.TX = xr.open_mfdataset(tau_x_files,parallel=True,preprocess=preprocess)
        self.TY = xr.open_mfdataset(tau_y_files,parallel=True,preprocess=preprocess)

        self.u_varname = u_varname
        self.v_varname = v_varname
        self.tau_x_varname = tau_x_varname
        self.tau_y_varname = tau_y_varname

        
        
    def colocatePoints(self,lats,lons,times):
        
        
        """
        Colocate model data to a set of lat/lon/time query points. 
            Ensure that lat/lon/time points of query exist within the loaded model data.
        
        Args:
            lats (numpy.array): latitudes in degrees
            lons (numpy.array): longitudes in degrees
            times (numpy.array): times represented as np.datetime64

        Returns:
           Model data linearly interpolated to the lat/lon/time query points.
           
           u: colocated model u currents.
           v: colocated model v currents.
           tx: colocated model u wind stress.
           ty: colocated model v wind stress.

        """

        
        if len(times) == 0:
            return [],[]
            
        ds_u =  self.U.interp(time=xr.DataArray(times.flatten(), dims='z'),
                            lat=xr.DataArray(lats.flatten(), dims='z'),
                            lon=xr.DataArray(lons.flatten(), dims='z'),
                            method='linear')

        ds_v =  self.V.interp(time=xr.DataArray(times.flatten(), dims='z'),
                            lat=xr.DataArray(lats.flatten(), dims='z'),
                            lon=xr.DataArray(lons.flatten(), dims='z'),
                            method='linear')
        
        ds_tx =  self.TX.interp(time=xr.DataArray(times.flatten(), dims='z'),
                            lat=xr.DataArray(lats.flatten(), dims='z'),
                            lon=xr.DataArray(lons.flatten(), dims='z'),
                            method='linear')

        ds_ty =  self.TY.interp(time=xr.DataArray(times.flatten(), dims='z'),
                            lat=xr.DataArray(lats.flatten(), dims='z'),
                            lon=xr.DataArray(lons.flatten(), dims='z'),
                            method='linear')
        

        u=np.reshape(ds_u[self.u_varname].values,np.shape(lats))
        v=np.reshape(ds_v[self.v_varname].values,np.shape(lats))
        tx=np.reshape(ds_tx[self.tau_x_varname].values,np.shape(lats))
        ty=np.reshape(ds_ty[self.tau_y_varname].values,np.shape(lats))

        return u,v,tx,ty
        
        
    def colocateSwathCurrents(self,orbit):

        """
        Colocate model current data to a swath (2d continuous array) of lat/lon/time query points. 
            Ensure that lat/lon/time points of query exist within the loaded model data.
        
        Args:
            orbit (object): xarray dataset orbit generated via the orbit.getOrbit() call. 
        Returns:
           original orbit containing model data linearly interpolated to the orbit swath.
                   new data is contained in u_model, v_model

        """
        
        lats  = orbit['lat'].values.flatten()
        lons  = orbit['lon'].values.flatten()
        times = orbit['sample_time'].values.flatten()

        ds_u =  self.U.interp(time=xr.DataArray(times, dims='z'),
                            lat=xr.DataArray(lats, dims='z'),
                            lon=xr.DataArray(lons, dims='z'),
                            method='linear')

        ds_v =  self.V.interp(time=xr.DataArray(times, dims='z'),
                            lat=xr.DataArray(lats, dims='z'),
                            lon=xr.DataArray(lons, dims='z'),
                            method='linear')


        u_interp = np.reshape(ds_u[self.u_varname].values,np.shape(orbit['lat'].values))
        v_interp = np.reshape(ds_v[self.v_varname].values,np.shape(orbit['lat'].values))

        orbit = orbit.assign({'u_model': (['along_track', 'cross_track'], u_interp),
                              'v_model': (['along_track', 'cross_track'], v_interp)})

        return orbit
    
    def colocateSwathWinds(self,orbit):

        """
        Colocate model wind data to a swath (2d continuous array) of lat/lon/time query points. 
            Ensure that lat/lon/time points of query exist within the loaded model data.
        
        Args:
            orbit (object): xarray dataset orbit generated via the orbit.getOrbit() call. 
        Returns:
           original orbit containing model data linearly interpolated to the orbit swath.
                   new data is contained in u10_model, v10_model, tx_model, ty_model, wind_speed_model, wind_dir_model

        """
        
        lats  = orbit['lat'].values.flatten()
        lons  = orbit['lon'].values.flatten()
        times = orbit['sample_time'].values.flatten()

        ds_tx =  self.TX.interp(time=xr.DataArray(times, dims='z'),
                            lat=xr.DataArray(lats, dims='z'),
                            lon=xr.DataArray(lons, dims='z'),
                            method='linear')

        ds_ty =  self.TY.interp(time=xr.DataArray(times, dims='z'),
                            lat=xr.DataArray(lats, dims='z'),
                            lon=xr.DataArray(lons, dims='z'),
                            method='linear')


        tx_interp = np.reshape(ds_tx[self.tau_x_varname].values,np.shape(orbit['lat'].values))
        ty_interp = np.reshape(ds_ty[self.tau_y_varname].values,np.shape(orbit['lat'].values))

        wind_speed = utils.stressToWind(np.sqrt(tx_interp**2 + ty_interp**2))
        wind_dir = np.arctan2(tx_interp,ty_interp) # in rad
        u10 = wind_speed * np.sin(wind_dir)
        v10 = wind_speed * np.cos(wind_dir)

        
        orbit = orbit.assign({'u10_model': (['along_track', 'cross_track'], u10),
                              'v10_model': (['along_track', 'cross_track'], v10)})
        
        
        orbit = orbit.assign({'tx_model': (['along_track', 'cross_track'], tx_interp),
                              'ty_model': (['along_track', 'cross_track'], ty_interp)})

        orbit = orbit.assign({'wind_speed_model': (['along_track', 'cross_track'], wind_speed),
                              'wind_dir_model': (['along_track', 'cross_track'], wind_dir*180/np.pi)})
        
        
        return orbit
    
    
def addTimeDim(ds):
    
    """
    Helper function for open_mfdataset. Very specific to a set of model data used at JPL.
        You may need something similar, but probably not exactly this.
        Adds an extra time dimension to a xarray dataset as it is opened so that open_mfdataset
        can stack data along that dimension.
        Looks at the filename from the opened netcdf file to deterimine the time dimension to add.

    Args:
        ds (xarray dataset): dataset that is opened by open_mfdataset.
    Returns:
        ds (xarray dataset): Original dataset with added time dimension.

    """
    
    
    ds = ds.isel(time=0)
    fn = os.path.basename(ds.encoding["source"])
    time_str = fn.split('_')[-1].split('.')[0]
    time = pd.to_datetime(time_str, format='%Y%m%d%H')
    ds = ds.expand_dims(time=[time])

    #display(ds)
    return ds


def addTimeDimCoarse(ds):
    
    """
    Helper function for open_mfdataset. Very specific to a set of model data used at JPL.
        You may need something similar, but probably not exactly this.
        Adds an extra time dimension to a xarray dataset as it is opened so that open_mfdataset
        can stack data along that dimension.
        Looks at the filename from the opened netcdf file to deterimine the time dimension to add.

    Args:
        ds (xarray dataset): dataset that is opened by open_mfdataset.
    Returns:
        ds (xarray dataset): Original dataset with added time dimension.

    """
    
    fn = os.path.basename(ds.encoding["source"])
    time_str = fn.split('.')[0].split('_')[-1]

    time = pd.to_datetime(time_str, format='%Y%m%d%H')
    ds = ds.expand_dims(time=[time])

    return ds
