import xarray as xr
import numpy as np
import pandas as pd
import yaml
from scipy.interpolate import UnivariateSpline
import scipy
import datetime
from pathlib import Path
import os
import importlib.resources as import_resources

from odysim.coordinates import *
from odysim import utils

    
def splineFactory(x,y,k=3,smoothing=None):
    spl = UnivariateSpline(x, y, k=k)
    if smoothing is not None:
        spl.set_smoothing_factor(smoothing)
        
    return spl



def getBearing(latitude,longitude):

    d = 1
    X = np.zeros(np.shape(latitude))
    Y = np.zeros(np.shape(latitude))

    X[d::] = np.cos(latitude[d::]) * np.sin(longitude[d::]-longitude[0:-d])
    Y[d::] = np.cos(latitude[0:-d]) * np.sin(latitude[d::]) - np.sin(latitude[0:-d]) * np.cos(latitude[d::]) * np.cos(longitude[d::]-longitude[0:-d])

    t = np.arctan2(X,Y)
    
    return t

def ecef_to_llh(x,y,z):
    """Go from ECEF geocentric to lat lon height.
    Parameters
    ----------
    x : float or array_like
        ECEF x coordinate.
    y : float or array_like
        ECEF y coordinate.
    z : float or array_like
        ECEF z coordinate.
    Returns
    -------
    lat,lon,h
    lat and lon are in radians. h is in the same units as x,y,z.
    """

    # Longitude

    lon = np.arctan2(y,x)

    # Geodetic Latitude

    projRad = np.sqrt(x*x + y*y)

    alpha = np.arctan2(
        z,(projRad*np.sqrt(1. - WGS84.ECCENTRICITY_SQ)))
    sa = np.sin(alpha);
    ca = np.cos(alpha);
    sa3 = sa*sa*sa;
    ca3 = ca*ca*ca;

    lat = np.arctan2(
        (z + WGS84.EP_SQUARED*WGS84.SEMIMINOR_AXIS*sa3),
        (projRad - WGS84.ECCENTRICITY_SQ*WGS84.SEMIMAJOR_AXIS*ca3))

    # height

    h = projRad/np.cos(lat) - WGS84.eastRad(lat)

    return lat,lon,h

class WGS84:
    """Definition of WGS84 ellipsoid and calculation of local radii."""

    #The Earth's constants
    SEMIMAJOR_AXIS = 6378137. # in meters

    # SEMIMAJOR_AXIS*sqrt(1-ECCENTRICITY_SQ)
    SEMIMINOR_AXIS = 6356752.3135930374

    ECCENTRICITY_SQ = 0.00669437999015

    #ECCENTRICITY_SQ/(1-ECCENTRICITY_SQ)
    EP_SQUARED = 0.0067394969488402

    CENTER_SCALE = 0.9996

    #Auxiliary Functions

    @staticmethod
    def eastRad(lat):
        """radius of curvature in the east direction (lat in radians)"""
        return WGS84.SEMIMAJOR_AXIS/np.sqrt(1. - WGS84.ECCENTRICITY_SQ*
                                       np.sin(lat)**2)

    @staticmethod
    def northRad(lat):
        """radius of curvature in the north direction (lat in radians)"""
        return (WGS84.SEMIMAJOR_AXIS*(1. - WGS84.ECCENTRICITY_SQ)/
            (1. - WGS84.ECCENTRICITY_SQ*np.sin(lat)**2)**1.5)

    @staticmethod
    def localRad(azimuth, lat):
        """Local radius of curvature along an azimuth direction measured
        clockwise from north.
        (azimuth and latitude in radians)"""
        return (WGS84.eastRad(lat)*WGS84.northRad(lat)/
                (WGS84.eastRad(lat)*np.cos(azimuth)**2 +
                 WGS84.northRad(lat)*np.sin(azimuth)**2))



class OdyseaSwath:
    
    def __init__(self,orbit_fname='orbit_out_590km_2020_2023.npz',config_fname='wacm_sampling_config.py'):
                
        """
        Initialize an OdyseaSwath object. Eventaully, this will contain configuration etc. TODO..
        
        Args:
            config_fname (str): configuration file (not yet implemented)

        Returns:
           OdyseaSwath object

        """
                
        if orbit_fname == 'orbit_out_590km_2020_2023.npz': 
            # the default fname needs the relative path to the installed dir
            from odysim import orbit_files

            try:
                orbit_fname = os.path.join(import_resources.files(orbit_files),orbit_fname)
            except:
                # for some reason, sometimes import_resources retruns a mutliplexedpath instead of a string!
                orbit_path = str(import_resources.files(orbit_files)).split("'")[1]
                orbit_fname = os.path.join(orbit_path,orbit_fname)

            
        if config_fname == 'wacm_sampling_config.py': 
            # the default fname needs the relative path to the installed dir
            import odysim
            config_fname = os.path.join(import_resources.files(odysim),config_fname)
            
        
            
        self.loadOrbitXYZ(fn=orbit_fname)
        self.config_fname=config_fname
    
    
    def getOrbitSwath(self,orbit_x,orbit_y,orbit_z,orbit_time_stamp,orbit_s,write=False):

        time_stamp_vector = orbit_time_stamp
        coarse_x = orbit_x
        coarse_y = orbit_y
        coarse_z = orbit_z
        coarse_s = orbit_s


        with open(self.config_fname, 'r') as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
            
        swath_width = cfg['SWATH_WIDTH']

        c_bins = np.arange(-swath_width/2,swath_width/2,cfg['IN_SWATH_RESOLUTION'])
        swath_blanking = (np.abs(c_bins) > np.nanmax(c_bins) - swath_width*cfg['SWATH_EDGE_GAP']) | (np.abs(c_bins) < swath_width*cfg['SWATH_CENTER_GAP'])

        h = np.zeros(np.shape(c_bins))

        Re = 6378 # km
        f = (Re+cfg['ORBIT_HEIGHT'])/Re # factor to convert resolution on the ground track to resolution of the satellite orbit track
        #s_pegs = np.arange(0,f*cfg['N_ALONG_TRACK'])*cfg['IN_SWATH_RESOLUTION'] # 8000 bins, 5km each.
        
        s_pegs = np.arange(np.nanmin(coarse_s)+.0001,np.nanmax(coarse_s)-.0001,cfg['IN_SWATH_RESOLUTION'] * f)
        
        ns,nc = len(s_pegs),len(c_bins)


        platform_s = coarse_s
        #platform_s = platform_s - np.nanmin(platform_s)

        #print(np.nanmax(platform_s))
        
        slat,slon,sh,s_time = [np.nan*np.zeros((ns,nc)) for _ in range(0,4)]

        pf_x_smoothed = splineFactory(platform_s,coarse_x)(s_pegs)
        pf_y_smoothed = splineFactory(platform_s,coarse_y)(s_pegs)
        pf_z_smoothed = splineFactory(platform_s,coarse_z)(s_pegs)

        pf_time_smoothed = splineFactory(platform_s,time_stamp_vector,smoothing=0)(s_pegs)

        pf_lat_smoothed,pf_lon_smoothed,pf_h_smoothed = ecef_to_llh(pf_x_smoothed,pf_y_smoothed,pf_z_smoothed)
        pf_bearing_smoothed = getBearing(pf_lat_smoothed,pf_lon_smoothed)

        for idx_s,s in enumerate(s_pegs):

            peg_lat = pf_lat_smoothed[idx_s] 
            peg_lon = pf_lon_smoothed[idx_s]
            peg_hdg = pf_bearing_smoothed[idx_s]
            peg_time = pf_time_smoothed[idx_s]
            peg_localRadius = WGS84.localRad(peg_hdg,peg_lat) 

            slat[idx_s,:], slon[idx_s,:], sh[idx_s,:] = sch_array_to_llh_array(0*c_bins.flatten(),c_bins.flatten(),
                                                                               h.flatten(),peg_lat,peg_lon,peg_hdg,peg_localRadius)
            s_time[idx_s,:] = peg_time

        mask = np.isfinite(slat+slon+s_time) 


        sample_time_track=s_time
        sample_lat_track=slat
        sample_lon_track=slon


        ds = xr.Dataset() 

        along_track_sz,cross_track_sz = np.shape(sample_time_track)

        sample_time_track[np.isnan(sample_time_track)] = 0

        
        
        encoding = {"lon": {"dtype": "int16", "zlib": True,'complevel': 9,'_FillValue':-9999,'scale_factor':.01,'add_offset':180},
                    "lat": {"dtype": "int16", "zlib": True,'complevel': 9,'_FillValue':-9999,'scale_factor':.01,'add_offset':90},
                    "sample_time": {"dtype": "float32", "zlib": True,'complevel': 9,'_FillValue':-9999, 'least_significant_digit':1},
                    "swath_blanking": {"dtype": "int16", "zlib": True,'complevel': 3,'_FillValue':-9999},
                   }

        ds = ds.assign_coords(coords={'along_track': (['along_track'], np.arange(0,along_track_sz)),
                                      'cross_track': (['cross_track'],  np.arange(0,cross_track_sz))})   


#         sample_time_track = np.moveaxis(sample_time_track,-1,0)
#         sample_lat_track  = np.moveaxis(sample_lat_track, -1,0)
#         sample_lon_track  = np.moveaxis(sample_lon_track, -1,0)

        nanmask = np.isnan(sample_time_track + sample_lat_track + sample_lon_track)

        sample_time_track[nanmask] = -9999
        sample_lat_track[nanmask]  = -9999
        sample_lon_track[nanmask]  = -9999
        

        sample_time_track_dt = [np.datetime64('1970-01-01') + np.timedelta64(int(stt*1000),'ms') for stt in sample_time_track.flatten()]
        sample_time_track_dt = np.reshape(sample_time_track_dt,np.shape(sample_time_track)).astype('datetime64[s]')
                                          
        ds = ds.assign({'sample_time': ([ 'along_track', 'cross_track'], np.array(sample_time_track_dt)),
                       'lat': (['along_track', 'cross_track'], np.array(sample_lat_track,dtype='float32')),
                       'lon': (['along_track', 'cross_track'], np.array(sample_lon_track,dtype='float32')),
                       'swath_blanking': (['cross_track'], swath_blanking)})

        ds['swath_blanking'].attrs['comment'] = 'Flagged in areas of the swath that are expected to have unacceptable error performance.'


        ds['lat'].attrs['valid_min'] = -90.00
        ds['lat'].attrs['valid_max'] = 90.00
        ds['lat'].attrs['long_name'] = 'latitude'
        ds['lat'].attrs['standard_name'] = 'latitude'
        ds['lat'].attrs['units'] = 'degrees_north'

        ds['lon'].attrs['valid_min'] = -180.00
        ds['lon'].attrs['valid_max'] = 180.00
        ds['lon'].attrs['long_name'] = 'longitude'
        ds['lon'].attrs['standard_name'] = 'longitude'
        ds['lon'].attrs['units'] = 'degrees_north'

        ds['sample_time'].attrs['long_name'] = 'Time of WaCM overpass.'
        ds['sample_time'].attrs['comments'] = 'Time of WaCM overpass in seconds since 1970.'
        ds['sample_time'].attrs['standard_name'] = 'time'
        #ds['sample_time'].attrs['units'] = 'seconds since 1970-01-01 00:00:00'

        ds.attrs['title'] = 'Odysea Simple Orbit Sampling V0.1'
        ds.attrs['project'] = 'Odysea'
        ds.attrs['summary'] = "Simplified orbit sampling assuming basic Odysea orbital parameters and viewing geometry. These data have been generated using only knowledge of along/cross track viewing geometry. No radar timing or antenna rotation was used."
        ds.attrs['references'] = 'Rodriguez 2018, Wineteer 2020'
        ds.attrs['institution'] = 'Jet Propulsion Laboratory (JPL)'
        ds.attrs['creator_name'] = "Alexander Wineteer"
        ds.attrs['version_id'] = '0.1'
        ds.attrs['date_created'] = str(datetime.datetime.now())
        ds.attrs['geospatial_lat_min'] = '-89.99N'
        ds.attrs['geospatial_lat_max'] = '89.99N'
        ds.attrs['geospatial_lon_min'] = '-180.00E'
        ds.attrs['geospatial_lon_max'] = '180.00E'
        ds.attrs['time_coverage_start'] = str(datetime.datetime.utcfromtimestamp(np.nanmin(sample_time_track[sample_time_track>0])))
        ds.attrs['time_coverage_end']   = str(datetime.datetime.utcfromtimestamp(np.nanmax(sample_time_track)))

    #     ds.attrs['Height [km]'] = orbiter.orbit.a.value - 6378.135
    #     ds.attrs['Semi-major Axis [km]'] = orbiter.orbit.a.value
    #     ds.attrs['Inclination [deg]'] = orbiter.orbit.inc.value*180/np.pi
    #     ds.attrs['LTAN'] = orbiter.ltan
    #     ds.attrs['Eccentricity'] = orbiter.orbit.ecc.value
    #     ds.attrs['Keplerian Period [min]'] = orbiter.orbit.period.value/60
    #     ds.attrs['Nodal Period [min]'] = orbiter.Tn/60           
    #     ds.attrs['Local Incidence'] = orbiter.r.theta_local        
    #     ds.attrs['Look Angle'] = orbiter.r.theta        
    #     ds.attrs['Swath Width'] = orbiter.r.swath_width        
    #     ds.attrs['Swath Width'] = orbiter.r.swath_width        



    #     fn_out = 'odysea_swath_simple_orbit_635km_46deg' + '.nc'

    #     if write:
    #         print('Writing to: ' + fn_out)
    #         ds.to_netcdf(cfg['output_folder'] + fn_out, format='NETCDF4',encoding=encoding)

        return ds
    
    def loadOrbitXYZ(self,fn='orbit_out_590km_2020_2023.npz'):
        
        orbit_out = np.load(fn)
        
        self.orbit_cut_points = orbit_out['orbit_cut_points']
        self.time_stamp_vector_coarse = orbit_out['time_stamp_vector']
        self.coarse_x = orbit_out['coarse_x']
        self.coarse_y = orbit_out['coarse_y']
        self.coarse_z = orbit_out['coarse_z']
        self.coarse_s = orbit_out['coarse_s']
        #self.orbit_cut_points = np.where(np.diff(np.signbit(self.coarse_lat)))[0][::2]

    

    def getOrbits(self,start_time,end_time,set_azimuth=True):
        
        """
        Return an iterator that contains xarray datasets, each dataset representing a single Odysea orbit,
            with the full iterator containing all orbits between start_time and end_time.
        
        Args:
            start_time (np.datetime64): start time for first orbit
            end_time (np.array): end time for last orbit (modulo down to orbital period). No partial orbits.
        Returns:
           ds: iterator containing orbit objects, each generated at the time of __next__ call.

        """
        
        start_time = start_time.timestamp()
        end_time = end_time.timestamp()

        
        start_index = np.where((self.time_stamp_vector_coarse > start_time) & (self.time_stamp_vector_coarse < end_time))[0][0]
        end_index = np.where((self.time_stamp_vector_coarse > start_time) & (self.time_stamp_vector_coarse < end_time))[0][-1]
        
        valid_orbit_cut_points = self.orbit_cut_points[(self.orbit_cut_points > start_index) & (self.orbit_cut_points < end_index)]

        for idx_orbit,orbit_start in enumerate(valid_orbit_cut_points):

            start_idx = orbit_start

            if (idx_orbit + 1 >= len(valid_orbit_cut_points)):
                break #end_idx = len(self.time_stamp_vector_coarse)
            else:
                end_idx = valid_orbit_cut_points[idx_orbit+1]

            ds = self.getOrbitSwath(self.coarse_x[start_idx:end_idx],
                                    self.coarse_y[start_idx:end_idx],
                                    self.coarse_z[start_idx:end_idx],
                                    self.time_stamp_vector_coarse[start_idx:end_idx],
                                    self.coarse_s[start_idx:end_idx],
                                    write=False)

            if set_azimuth:
                ds = self.setAzimuth(ds)
            
            
            
            yield ds

            
    def setAzimuth(self,orbit):
    
        """
        Set the azimuth and encoder angles for each grid cell from forward and backward looks. Set the along-track bearing.
        
        Args:
            orbit (xarray dataset): orbit dataset generated by getOrbits().next() or for loop.
        Returns:
            orbit (xarray dataset): original orbit dataset with added encoder_fore, encoder_aft,
                                    azimuth_fore, azimuth_aft, and bearing variables.

        """
   
        cross_track = orbit.cross_track.values - np.median(orbit.cross_track.values)
        
        encoder_fore,encoder_aft = utils.computeEncoderByXT(cross_track)
        encoder_fore = np.broadcast_to(encoder_fore, np.shape(orbit.lat.values))
        encoder_aft = np.broadcast_to(encoder_aft, np.shape(orbit.lat.values))

        
        platform_latitude = np.nanmean(orbit.lat.values[:,int(len(orbit.cross_track)/2 - 2):int(len(orbit.cross_track)/2 +2)],axis=1)
        platform_longitude = np.nanmean(orbit.lon.values[:,int(len(orbit.cross_track)/2 - 2):int(len(orbit.cross_track)/2 +2)],axis=1)
        
        bearing = utils.getBearing(platform_latitude*np.pi/180,platform_longitude*np.pi/180)
        bearing = utils.splineFactory(orbit.along_track.values,bearing)(orbit.along_track.values)
        bearing = scipy.signal.medfilt(bearing,9) # 9 element median to remove single-value outliers.

        azimuth_fore =  utils.normalizeTo180((encoder_fore + bearing[:,np.newaxis]))
        azimuth_aft =  utils.normalizeTo180((encoder_aft + bearing[:,np.newaxis]))


        orbit = orbit.assign({'encoder_fore': (['along_track', 'cross_track'], encoder_fore),
                              'encoder_aft' : (['along_track', 'cross_track'], encoder_aft)})
        
        
        orbit = orbit.assign({'azimuth_fore': (['along_track', 'cross_track'], azimuth_fore),
                              'azimuth_aft' : (['along_track', 'cross_track'], azimuth_aft)})
        
        
        orbit = orbit.assign({'bearing': (['along_track'], bearing)})
        
        
        return orbit