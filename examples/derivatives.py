import numpy as np
import scipy.stats
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib
import glob
from scipy import interpolate
from scipy import signal
from scipy import stats
from scipy import fftpack
from scipy.interpolate import UnivariateSpline
from astropy.convolution import (Gaussian2DKernel, CustomKernel,
                                 interpolate_replace_nans,
                                 convolve)
import scipy
#import plottingTools as pt
#import cmocean.cm as cmo


def get_kernel_sigma(delta_in,delta_out):
    """Get the Gaussian filter standard deviation desired
    to get an a kernel with a full width 1/2 power
    width of delta_out, when delta_in is the pixel size.
    """

    sigma_out = delta_out/np.sqrt(2*np.log(2))/2

    return sigma_out/delta_in # in pixels


def filt_single_fast(u,delta_out=5000,delta_in=5000.,keep_nan=True):
    """Filter the velocity components to the desired spatial
    resolution, given a data set where the mask has been set."""
    sigma = get_kernel_sigma(delta_in,delta_out)

    #u = convolve(u,kernel,nan_treatment='interpolate',preserve_nan=keep_nan)
    
    u_n = np.copy(u)
    nanmask = np.isnan(u)
    u_n[nanmask] = 0
    u_s = scipy.ndimage.filters.gaussian_filter(u,sigma)
    u_s[nanmask] = np.nan
    
    return u_s


def filt_single(u,delta_out=5000,delta_in=5000.,keep_nan=True):
    """Filter the velocity components to the desired spatial
    resolution, given a data set where the mask has been set."""
    sz = get_kernel_sigma(delta_in,delta_out)

    kernel = Gaussian2DKernel(x_stddev=sz)

    # plt.figure()
    # plt.imshow(kernel, interpolation='none', origin='lower')
    # plt.colorbar()
    # plt.show()
    u = convolve(u,kernel,nan_treatment='interpolate',preserve_nan=keep_nan)

    return u


def remove_islands(inpt,k_size):
    # max nan is max nans in a kernel
    kernel = np.ones((int(k_size),int(k_size)))
    conv = convolve(inpt,kernel,fill_value=np.nan,nan_treatment='fill')
    inpt[np.isnan(conv)] = np.nan
    return inpt

def velocity_derivatives(u,v,x,y,delta_in=5000):
    """Compute the velocity derivatives given u and v."""

    # Compute the derivatives
    du_dx = np.ones_like(u)*np.nan
    du_dy = np.ones_like(u)*np.nan
    #print(np.nanmean(x[:,2:]-x[:,:-2]))

    du_dx[:,1:-1] = (u[:,2:]-u[:,:-2])/(x[:,2:]-x[:,:-2])
    du_dy[1:-1,:] = (u[2:,:]-u[:-2,:])/(y[2:,:]-y[:-2,:])

    dv_dx = np.ones_like(v)*np.nan
    dv_dy = np.ones_like(v)*np.nan

    dv_dx[:,1:-1] = (v[:,2:]-v[:,:-2])/(x[:,2:]-x[:,:-2])
    dv_dy[1:-1,:] = (v[2:,:]-v[:-2,:])/(y[2:,:]-y[:-2,:])

    curl = dv_dx - du_dy
    div = du_dx + dv_dy
    strain_rate = np.sqrt((du_dx - dv_dy)**2 + (dv_dx + du_dy)**2)

    return curl,div
