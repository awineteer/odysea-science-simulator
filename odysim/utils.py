import numpy as np
import scipy
import os
#from numba import jit
import matplotlib.pyplot as plt
from cartopy import config
import cartopy.crs as ccrs
from scipy import stats

import os
import importlib.resources as import_resources
from odysim import cartopy_files

try:
    cf = import_resources.files(cartopy_files)
    os.environ["CARTOPY_USER_BACKGROUNDS"] = cf
except:
    # for some reason, sometimes import_resources retruns a mutliplexedpath instead of a string!
    cf = str(import_resources.files(cartopy_files)).split("'")[0]
    os.environ["CARTOPY_USER_BACKGROUNDS"] = cf



from scipy.interpolate import UnivariateSpline
    
def splineFactory(x,y,smoothing=.1):
    spl = UnivariateSpline(x, y)
    spl.set_smoothing_factor(.1)
    return spl

#@jit(nopython=True)
def signedAngleDiff(ang1,ang2):

    ang1 = np.asarray(ang1)
    ang2 = np.asarray(ang2)
    ang11 = normalizeTo360(ang1)
    ang22 = normalizeTo360(ang2)

    # ang11 = np.array(ang11)
    # ang21 = np.array(ang22)

    result = ang22 - ang11

    resultF = result.flatten()

    for ii in range(resultF.shape[0]):
        if resultF[ii] > 180:
            resultF[ii] = resultF[ii] - 360
        if resultF[ii] < -180:
            resultF[ii] = 360 + resultF[ii]

    result = resultF.reshape(np.shape(result))


    return result

def computeEncoderByXT(cross_track):
    """
    Compute the expected encoder angle from cross-track location
    Returns encoder_angle_fore, encoder_angle aft, the forward and backward looking samples. Degrees clockwise from the velocity vector.
    """
    
    encoder_angle_fore = normalizeTo180(90 - 180/np.pi*np.arccos(cross_track/np.max(cross_track)))

    encoder_angle_aft = normalizeTo180(180 - encoder_angle_fore)
 
    return encoder_angle_fore,encoder_angle_aft


def getBearing(platform_latitude,platform_longitude):

    d = 1
    X = np.zeros(np.shape(platform_latitude))
    Y = np.zeros(np.shape(platform_latitude))

    
    lon_diff = signedAngleDiff(platform_longitude[0:-d]*180/np.pi,platform_longitude[d::]*180/np.pi)*np.pi/180 # dont @ me
    
    
    X[d::] = np.cos(platform_latitude[d::]) * np.sin(lon_diff)
    Y[d::] = np.cos(platform_latitude[0:-d]) * np.sin(platform_latitude[d::]) - np.sin(platform_latitude[0:-d]) * np.cos(platform_latitude[d::]) * np.cos(lon_diff)

    pf_velocity_dir = np.arctan2(X,Y) * 180/np.pi

    return pf_velocity_dir


        
def localSTD(inpt,sigma):

    inpt[np.abs(inpt)>1] = np.nan
    inpt[np.isnan(inpt)] = 0

    u = scipy.ndimage.filters.gaussian_filter(np.copy(inpt), sigma=sigma)
    u2 = scipy.ndimage.filters.gaussian_filter(np.copy(inpt)**2, sigma=sigma)

    std = np.sqrt(u2 - u**2)

    return std
    

def normalizeTo180(angle):
    # note this strange logic was originally to use numba JIT
    for idx,ang in np.ndenumerate(angle):
    
        ang =  ang % 360
        ang = (ang + 360) % 360
        if ang > 180:
            ang -= 360
        angle[idx] = ang
        
    return angle


#@jit(nopython=True)
def normalizeTo180Jit(angle):

    for idx,ang in np.ndenumerate(angle):
    
        ang =  ang % 360
        ang = (ang + 360) % 360
        if ang > 180:
            ang -= 360
        angle[idx] = ang
        
    return angle


def normalizeTo360(angle):

    #angle2 = np.array(angle)

    angle2 = angle % 360

    return angle2


def fixLon(ds):

    ds['lon'].values = normalizeTo180(ds['lon'].values)

    return ds    


def toUTM(target_lon, target_lat):


    lonlat_epsg=4326  # This WGS 84 lat/lon
    xy_epsg=3857      # default google projection

    lonlat_epsg = lonlat_epsg
    lonlat_crs = CRS(lonlat_epsg)
    xy_epsg = xy_epsg
    xy_crs = CRS(xy_epsg)

    lonlat_to_xy = Transformer.from_crs(lonlat_crs,
                                         xy_crs,
                                         always_xy=True)

    target_x, target_y = lonlat_to_xy.transform(target_lon, target_lat)

    return target_x, target_y

def stressToWind(stress_magnitude):
    """
    Convert wind stress to wind field
    Ideally, this would be iterated to get Cd right
    """
    ### Assuming Large and Pond < 10 m/s
    cdl = 1.12e-3
    rho = 1.22  # density of the air

    wind_speed = np.sqrt(stress_magnitude/(rho*cdl))

    return wind_speed

 
def windToStress(wind_speed,wind_dir=None):
    """
    Convert wind to wind stress field
    Assumes current relative winds 
    """

    ### Assuming Large and Pond < 10 m/s
    cdl = 1.12e-3
    rho = 1.22  # density of the air
 
    #cd = (.49 + 0.065*wind_speed) * 10**-3
    #cd[wind_speed < 10] = cdl
    cd = cdl
    
    if wind_dir is None:
        stress_magnitude = rho * cd * wind_speed**2
        return stress_magnitude
    else:
        stress_magnitude = rho * cd * wind_speed**2
        stress_u = stress_magnitude * np.sin(wind_dir*np.pi/180)
        stress_v = stress_magnitude * np.cos(wind_dir*np.pi/180)
        return stress_u,stress_v

    
def SDToUVErrors(magnitude,direction,magnitude_error,direction_error):

    sin_dir = np.sin(direction*np.pi/180)
    cos_dir = np.cos(direction*np.pi/180)

    std_sin_dir = cos_dir*direction_error*np.pi/180
    std_cos_dir = sin_dir*direction_error*np.pi/180

    u_error = np.abs(sin_dir*magnitude)*np.sqrt((std_sin_dir/sin_dir)**2 + (magnitude_error/magnitude)**2)
    v_error = np.abs(cos_dir*magnitude)*np.sqrt((std_cos_dir/cos_dir)**2 + (magnitude_error/magnitude)**2)

    return u_error,v_error



def makePlot(lon,lat,data,vmin,vmax,cblabel,colormap,figsize=(20,10),bg=True,gridMe=False,is_err=False,globe=False,cb=True):

    
    if gridMe:
        
        mask = np.isfinite(lon+lat+data)

        
        lon_lin = np.arange(-180,180,0.25)
        lat_lin = np.arange(-90,90,0.25)

        lon_mesh,lat_mesh = np.meshgrid((lon_lin[1::]+lon_lin[0:-1])/2,(lat_lin[1::]+lat_lin[0:-1])/2)
        
        data, bin_edges, binnumber = scipy.stats.binned_statistic_dd([lon[mask],lat[mask]],values=data[mask],statistic='mean',bins=[lon_lin,lat_lin])
        
            
        data = data.T
        lon = lon_mesh
        lat = lat_mesh
    
    fig = plt.figure(figsize=figsize)
    if globe:
        ax = plt.subplot(111, projection=ccrs.Orthographic(-65, 15))
    else:
        ax = plt.subplot(111, projection=ccrs.PlateCarree())

    if bg:
        ax.background_img(name='BM', resolution='low')

    
    plt.pcolormesh(lon, lat, data,transform=ccrs.PlateCarree(),vmax=vmax,vmin=vmin,cmap=colormap)

    ax.coastlines()
    
    if cb:
        plt.colorbar(label=cblabel,orientation='horizontal',fraction=0.046, pad=0.04)

    fig.tight_layout()

    
    plt.show()