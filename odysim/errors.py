import numpy as np
from scipy.interpolate import RegularGridInterpolator
import warnings
warnings.filterwarnings('ignore')

from odysim import utils, uncertainty_tables
import os
import importlib.resources as import_resources


class OdyseaErrors:
    

    def __init__(self,type='cbe'):
        
        """
        Initialize an OdyseaErrors object. This contains various meshgrids and loads an error interpolator.
            The interpolator is indexed as a function of wind speed, wind direction, and cross-track location.
        
        Args:
            None

        Returns:
           OdyseaErrors object

        """

        if type == 'mev': 
            # the default fname needs the relative path to the installed dir
            lut_fn='odysea_sigma_vr_lut_height590km_look49deg_swath1500km_08_2_2024_MEV.npz'
        else:
            lut_fn='odysea_sigma_vr_lut_height590km_look49deg_swath1500km_08_2_2024_CBE.npz'

        try:
            lut_fn = os.path.join(import_resources.files(uncertainty_tables),lut_fn)
        except:
            # for some reason, sometimes import_resources retruns a mutliplexedpath instead of a string!
            lut_path = str(import_resources.files(uncertainty_tables)).split("'")[1]
            lut_fn = os.path.join(lut_path,lut_fn)

        
        # These ranges are defined by the size of the vradial_lut at creation.
        wind_dir_range = np.arange(-195,195,5)  
        wind_speed_range = np.arange(0,20,.1)
        encoder_angle_range = np.arange(-190,190,5)

        self.wind_dir_mesh,self.wind_speed_mesh,self.encoder_angle_mesh = np.meshgrid(wind_dir_range,wind_speed_range,encoder_angle_range)
        self.vradial_lut = np.load(lut_fn)['sigma_vr']
        self.vradial_interpolator = RegularGridInterpolator((wind_speed_range, wind_dir_range, encoder_angle_range), self.vradial_lut)
            
    
    def vradialSTDLookup(self,wind_speed,wind_dir,encoder_angle,azimuth):
        
        """
        Look up surface vradial standard deviation for a set of wind speed, direciton, encoder angle, and azimuth.
            Lookup is done using linear interpolation.
        
        Args:
            wind_speed (np.array): wind speeds in m/s
            wind_dir (np.array): wind directions in deg (towards)
            encoder_angle (np.array): encoder angle of radar in degrees
            azimuth (np.array): azimuth angle from North of radar in degrees

        Returns:
           vradial_std (np.array): Radial velocity standard deviation in m/s, interpolated to the set of inputs.

        """

        relative_azimuth = utils.normalizeTo180Jit(utils.normalizeTo360(wind_dir) - utils.normalizeTo360(azimuth))
        encoder_norm = utils.normalizeTo180Jit(encoder_angle)
        
        return np.reshape(self.vradial_interpolator((wind_speed.flatten(),relative_azimuth.flatten(),encoder_norm.flatten())),np.shape(encoder_norm))
    
    
    def setXYVelocitySTDForeAft(self,orbit):
        # check this.. not using encoder_angle_fore..? might not matter since theyre opposites..
        
        
        """
        Set velocity standard deviations in the cross and along track directions.
        
        Args:
            orbit (xarray dataset): orbit dataset already containing vradial standard deviations. 

        Returns:
            orbit (xarray dataset): original input dataset with added vx_std and vy_std variables in m/s.

        """
        
        std_vx = np.sqrt(orbit.vr_std_fore**2 + orbit.vr_std_aft**2)/(2*np.cos(orbit.encoder_aft*np.pi/180))
        std_vy = np.sqrt(orbit.vr_std_fore**2 + orbit.vr_std_aft**2)/(2*np.sin(orbit.encoder_aft*np.pi/180))

        orbit = orbit.assign({'vx_std': (['along_track', 'cross_track'], std_vx.values),
                              'vy_std': (['along_track', 'cross_track'], std_vy.values)})
        
        
        return orbit

    
    def setUVVelocitySTD(self,orbit):

        """
        Set velocity standard deviations in geographic coordinates (u/v).
        
        Args:
            orbit (xarray dataset): orbit dataset already containing vradial standard deviations. 

        Returns:
            orbit (xarray dataset): original input dataset with added u_std and v_std variables in m/s.

        """
        
        pf_v_dir = orbit.azimuth_fore - orbit.encoder_fore

        t = pf_v_dir*np.pi/180
        
        sigma_U = np.sqrt(np.sin(t)**2*orbit.vx_std**2 + np.cos(t)**2*orbit.vy_std**2) # neglecting 2absigmaab covariance term
        sigma_V = np.sqrt(np.cos(t)**2*orbit.vx_std**2 + np.sin(t)**2*orbit.vy_std**2) # neglecting 2absigmaab covariance term

        orbit = orbit.assign({'u_std': (['along_track', 'cross_track'], sigma_U.values),
                              'v_std': (['along_track', 'cross_track'], sigma_V.values)})
        
        
        return orbit
    
    
    def simulateCurrentSTD(self,orbit,wind_speed=7,wind_dir=0):
        
        orbit = self.setRadialVelocitySTD(orbit,wind_speed=wind_speed,wind_dir=wind_dir)
        orbit = self.setXYVelocitySTDForeAft(orbit)
        orbit = self.setUVVelocitySTD(orbit)
        
        return orbit

    
    def setRadialVelocitySTD(self,orbit,wind_speed=7,wind_dir=0):
        
        """
        Set radial veloicty errors by calling the lookup table. Wind speed LUT only goes up to 20.
        
        Args:
            orbit (xarray dataset): orbit dataset ideally containing wind speed and direction variables.  
            wind_speed (numeric): wind speed for lookup table. Default 7 m/s. Only used if wind_speed is not available in orbit.

        Returns:
            orbit (xarray dataset): original input dataset with added vx_std and vy_std variables in m/s.

        """
        
        try:
            wind_speed = np.copy(orbit.wind_speed_model.values)
            wind_dir = np.copy(orbit.wind_dir_model.values)
            
            wind_speed[wind_speed>19] = 19
            wind_speed[wind_speed<1] = 1

        except:
            print('Using wind speed from function args. Assign "wind_speed" and "wind_dir" variables to orbit dataset if desired.')
            wind_speed = wind_speed * np.ones(np.shape(orbit.encoder_fore.values))
            wind_dir = wind_dir * np.ones(np.shape(orbit.encoder_fore.values))

            
        nanmask = np.isnan(wind_speed + wind_dir)
        vradial_fore_std = np.nan*np.ones(np.shape(wind_speed))
        vradial_aft_std = np.nan*np.ones(np.shape(wind_speed))

        vradial_fore_std[~nanmask] = self.vradialSTDLookup(wind_speed[~nanmask],wind_dir[~nanmask],
                                                           np.copy(orbit.encoder_fore.values)[~nanmask],
                                                           np.copy(orbit.azimuth_fore.values)[~nanmask])
        
        vradial_aft_std[~nanmask] = self.vradialSTDLookup(wind_speed[~nanmask],wind_dir[~nanmask],
                                                          np.copy(orbit.encoder_aft.values)[~nanmask],
                                                          np.copy(orbit.azimuth_aft.values)[~nanmask])

        orbit = orbit.assign({'vr_std_fore': (['along_track', 'cross_track'], vradial_fore_std),
                              'vr_std_aft': (['along_track', 'cross_track'], vradial_aft_std)})
        
        return orbit
    
    
#     TODO
#     def getPointingErrors():
        
#         # given a simple RMS pointing knowledge error,
#         # simulate the effect on surface currents.
#         # this assumes calibration has been performed and this is the residual error
        
#         return .0015 # set to the micro-rad knowledge requirement; load from config eventually
    
    
    
    def setCurrentErrors(self,orbit,resolution=5000,etype='baseline',wind_speed=7,wind_dir=0):
        
        """
        Set u/v veloicty errors either with a simplified single-number standard deviation or
            by using a wind speed/direction, cross-swath dependent lookup table.
            The resolution can be set such that the simulated error is reduced by sqrt(num_samples)
            relative to the baseline 5000m resolution.
        
        Args:
            orbit (xarray dataset): orbit dataset ideally containing wind speed and direction variables if etype==simulated_baseline.  
            resolution (numeric): desired resolution at which errors are set. > 5000
            etype (str): error type. Select from 'baseline' (.354 m/s), 'low' (.247 m/s), 'threshold' (.505 m/s), or 
                        'simulated_baseline' (wind speed/dir, cross track dependent baseline)
            wind_speed (numeric): wind speed for lookup table. Default 7 m/s. Only needed for simulated error type. 
                                    Only used if wind_speed is not available in orbit.
            wind_dir (numeric): wind direction for lookup table. Default 0 deg. Only needed for simulated error type. 
                                    Only used if wind_dir is not available in orbit.

        Returns:
            orbit (xarray dataset): original input dataset with added u_error and v_error variables in m/s.

        """
        
        n_samples = (resolution/5000)**2

        # base std is the base speed std to be split between components

        if etype=='baseline':
            base_std = .354

        elif etype=='low':
            base_std = .247

        elif etype=='threshold':
            base_std= .505

        elif etype=='simulated_baseline':
            orbit = self.simulateCurrentSTD(orbit,wind_speed=wind_speed,wind_dir=wind_dir)

        if 'simulated' not in etype:
            # basic single number STD

            component_std = base_std/np.sqrt(2)
            component_std = component_std/np.sqrt(n_samples)

            size=np.shape(orbit['lat'])
            u_errors = np.random.normal(scale=component_std,size=size)
            v_errors = np.random.normal(scale=component_std,size=size)

        else:
            # using the radial veloicty lookup table errors

            u_errors = np.random.normal(scale=orbit['u_std'].values/np.sqrt(n_samples))
            v_errors = np.random.normal(scale=orbit['v_std'].values/np.sqrt(n_samples))


        orbit = orbit.assign({'u_error': (['along_track', 'cross_track'], u_errors),
                              'v_error': (['along_track', 'cross_track'], v_errors)})

        return orbit

    def setWindErrors(self,orbit,resolution=5000,etype='baseline'):


        """
        Set wind speed/dir errors either with a simplified single-number standard deviation.
            The resolution can be set such that the simulated error is reduced by sqrt(num_samples)
            relative to the baseline 5000m resolution.
        
        Args:
            orbit (xarray dataset): orbit dataset ideally containing wind speed and direction variables if etype==simulated_baseline.  
            resolution (numeric): desired resolution at which errors are set. > 5000
            etype (str): error type. Select from 'baseline', 'low', 'threshold'
 
        Returns:
            orbit (xarray dataset): original input dataset with added wind_speed_error, wind_dir_error, 
                                    wind_u_error, wind_v_error, stress_u_error, stress_v_error, and stress_mag_error variables.

        """
        
        if etype=='baseline':

            base_speed_std = 1
            base_speed_pct = .1

            base_dir_std_lowspd = 20
            base_dir_std_highspd = 15


        elif etype=='low':

            base_speed_std = .75
            base_speed_pct = .05

            base_dir_std_lowspd = 15
            base_dir_std_highspd = 10


        elif etype=='threshold':

            base_speed_std = 1.5
            bace_speed_pct = .15

            base_dir_std_lowspd = 25
            base_dir_std_highspd = 20


        speed_std = orbit.wind_speed_model.values * base_speed_pct
        speed_std[speed_std > base_speed_std] = base_speed_std

        dir_std = base_dir_std_lowspd * np.ones_like(orbit.wind_speed_model.values)
        dir_std[orbit.wind_speed_model.values > 10] = base_dir_std_highspd


        n_samples = (resolution/5000)**2
        dir_std = dir_std/np.sqrt(n_samples)
        speed_std = speed_std/np.sqrt(n_samples)

        speed_errors = np.random.normal(scale=speed_std)
        dir_errors = np.random.normal(scale=dir_std)

        orbit = orbit.assign({'wind_speed_error': (['along_track', 'cross_track'], speed_errors),
                              'wind_dir_error': (['along_track', 'cross_track'], dir_errors)})

        
        # propagate the speed/dir errors through to U/V errors
        u_wind_std,v_wind_std = utils.SDToUVErrors(orbit.wind_speed_model.values,orbit.wind_dir_model.values,speed_std,dir_std)

        u_wind_error = np.random.normal(scale=u_wind_std)
        v_wind_error = np.random.normal(scale=v_wind_std)

        orbit = orbit.assign({'wind_u_error': (['along_track', 'cross_track'], u_wind_error),
                              'wind_v_error': (['along_track', 'cross_track'], v_wind_error)})


        # now kludge some stress errors
        # Compute stress from noisy and original wind speed/dir fields; stress error is the difference.
        
        wind_speed_noisy = orbit.wind_speed_model.values + speed_errors
        wind_dir_noisy = orbit.wind_dir_model.values + dir_errors

        stress_mag_noisy = utils.windToStress(wind_speed_noisy)
        stress_u_noisy,stress_v_noisy = utils.windToStress(wind_speed_noisy,wind_dir_noisy)

        stress_mag = utils.windToStress(orbit.wind_speed_model.values)

        stress_mag_error = stress_mag - stress_mag_noisy
        u_stress_error = orbit.tx_model.values - stress_u_noisy
        v_stress_error = orbit.ty_model.values - stress_v_noisy

        orbit = orbit.assign({'stress_mag_error': (['along_track', 'cross_track'], stress_mag_error)})

        orbit = orbit.assign({'stress_u_error': (['along_track', 'cross_track'], u_stress_error),
                              'stress_v_error': (['along_track', 'cross_track'], v_stress_error)})


        return orbit

    
    def getPointwiseErrors(self,size=1,resolution=5000,etype='baseline'):
        
        # TODO: implement wind and more complicated current errors for pointwise.
        
        """
        Return an array of errors with the given etype. See setCurrentErrors.
        
        Args:
            size (numeric): size of error array to be returned  
            resolution (numeric): desired resolution at which errors are set. > 5000
            etype (str): error type. Select from 'baseline', 'low', 'threshold'
 
        Returns:
            errors (numeric): a numpy array of simulated errors.
        """
        
        if etype=='baseline':
            base_std = .354
            
        elif etype=='low':
            base_std = .247
            
        elif etype=='threshold':
            base_std= .505
        
        n_samples = (resolution/5000)**2
        std = base_std/np.sqrt(n_samples)
        
        errors = np.random.normal(scale=std,size=size)
        
        return errors
