import xarray as xr
import numpy as np

#### THIS CODE IS NOT FINISHED ####


class OdyseaPoint:
    
    def __init__(self,config_fname=None):
        
       """
        Initialize an OdyseaPoint object. Eventaully, this will contain configuration etc. TODO..
        
        Args:
            config_fname (str): configuration file (not yet implemented)

        Returns:
           OdyseaSwath object

        """
        pass
        
        
    def loadSampling(self,fn):
                
        """
        Load a pre-computed sampling file that contains a lat/lon grid of time sampling
            during one orbit repeat cycle (4 days).
        
        Args:
            fn (str): filename for sampling file.
        Returns:
           None

        """
        
        self.sampling = xr.open_dataset(fn,decode_times=True)
        
        self.min_time = np.nanmin(self.sampling['sample_time'].values)
        self.max_time = np.nanmax(self.sampling['sample_time'].values)
        self.dt = self.max_time-self.min_time
    
    
    def getSamplingTimes(self,lats,lons,start_time,end_time):
        
        """
        Return sampling times for a given set of lat/lons between a given start/end time.
        
        Args:
            lats (np.array numeric): array of latitudes for sampling
            lons (np.array numeric): array of longitudes for sampling
            start_time (np.datetime64): start time for first orbit
            end_time (np.array): end time for last orbit (modulo down to orbital period). No partial orbits.
        Returns:
           repeated_times: a numpy array containing np.datetime64 objects for Odysea sampling times at the given lat/lons.

        """
        
        
        times = []

        ds = self.sampling.sel(lat=xr.DataArray(lats, dims='z'),
                               lon=xr.DataArray(lons, dims='z'),
                               method='nearest')
        
        
        for idx in range(len(lats)):
            t = ds['sample_time'].values[:,idx]
            t = t[np.isfinite(t)]
            times.append(t)

        
        n_repeats = np.ceil((end_time - start_time)/np.timedelta64(4,'D'))
                           
        n_points = len(lats)
        repeated_times = copy.copy(times)

        for r in np.arange(1,n_repeats):
            offset = np.timedelta64(4,'D') * r
            for pt_idx in range(n_points):
                
                add_times = times[pt_idx] + offset
                repeated_times[pt_idx] = np.append(repeated_times[pt_idx],add_times)

        
        orbit_start_offset = start_time - self.min_time
        for pt_idx in range(n_points):
            repeated_times[pt_idx] = repeated_times[pt_idx] + orbit_start_offset 
            repeated_times[pt_idx] = repeated_times[pt_idx][repeated_times[pt_idx] < end_time]

        return repeated_times
    

    
#     def getErrors(self,size=1,resolution=5000,etype='baseline'):
        
#         if etype=='baseline':
#             base_std = .354
            
#         elif etype=='low':
#             base_std = .247
            
#         elif etype=='threshold':
#             base_std= .505
        
#         n_samples = (resolution/5000)**2
#         std = base_std/np.sqrt(n_samples)
        
#         errors = np.random.normal(scale=std,size=size)
        
#         return errors
