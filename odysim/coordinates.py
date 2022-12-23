
import numpy as np
#from numba import jit

#@jit(nopython=True,cache=True)
def eastRad(lat):
    """radius of curvature in the east direction (lat in radians)"""
    #The Earth's constants
    SEMIMAJOR_AXIS = 6378137. # in meters
    ECCENTRICITY_SQ = 0.00669437999015

    return SEMIMAJOR_AXIS/np.sqrt(1. - ECCENTRICITY_SQ* np.sin(lat)**2)

#@jit(nopython=True,cache=True)
def northRad(lat):
    """radius of curvature in the north direction (lat in radians)"""
    SEMIMAJOR_AXIS = 6378137. # in meters
    ECCENTRICITY_SQ = 0.00669437999015
    return (SEMIMAJOR_AXIS*(1. - ECCENTRICITY_SQ)/
            (1. - ECCENTRICITY_SQ*np.sin(lat)**2)**1.5)

#@jit(nopython=True,cache=True)
def localRad(hdg, lat):
    """Local radius of curvature along heading
    (heading and latitude in radians)"""
    return (eastRad(lat)*northRad(lat)/(eastRad(lat)*np.cos(hdg)**2 + 
                                        northRad(lat)*np.sin(hdg)**2))


#@jit(nopython=True,cache=True)
def getPegPointVector(peg_lat,peg_lon):
    """Get vector from WGS-84 center to peg point in
    geocentric coordinates."""
    ECCENTRICITY_SQ = 0.00669437999015

    p = np.zeros(3,np.float64)

    # Calculate useful constants
    clt = np.cos(peg_lat);
    slt = np.sin(peg_lat);
    clo = np.cos(peg_lon);
    slo = np.sin(peg_lon);

    # east radius of curvature */
    eastRadius = eastRad(peg_lat);

    # displacement vector */
    p[0] = eastRadius*clt*clo;
    p[1] = eastRadius*clt*slo;
    p[2] = eastRadius*(1. - ECCENTRICITY_SQ)*slt;

    return p

#@jit(nopython=True,cache=True)
def getXYZ_to_GEO_affine(peg_lat,peg_lon,peg_hdg,peg_localRadius):
    """Function to compute the transformation matrix
    form xyz to geocentric"""

    m = np.zeros((3,3),np.float64)
    up = np.zeros(3,np.float64) #local up vector in geocentric coordinates */

    # Calculate useful constants
    clt = np.cos(peg_lat);
    slt = np.sin(peg_lat);
    clo = np.cos(peg_lon);
    slo = np.sin(peg_lon);
    chg = np.cos(peg_hdg);
    shg = np.sin(peg_hdg);

    # Fill in the rotation matrix
    m[0][0] = clt*clo;
    m[0][1] = -shg*slo - slt*clo*chg;
    m[0][2] = slo*chg - slt*clo*shg;
    m[1][0] = clt*slo;
    m[1][1] = clo*shg - slt*slo*chg;
    m[1][2] = -clo*chg - slt*slo*shg;
    m[2][0] = slt;
    m[2][1] = clt*chg;
    m[2][2] = clt*shg;

    #Find the vector from the center of the ellipsoid to the peg point */
    p = getPegPointVector(peg_lat,peg_lon);

    # Calculate the local upward vector in geocentric coordinates */
    up[0] = peg_localRadius*clt*clo;
    up[1] = peg_localRadius*clt*slo;
    up[2] = peg_localRadius*slt;

    #Calculate the translation vector for the sch -> xyz transformation
    ov = p - up

    return m, ov

#@jit(nopython=True,cache=True)
def getGEO_to_XYZ_affine(peg_lat,peg_lon,peg_hdg,peg_localRadius):
    """Function to compute the transformation matrix form
    geocentric to xyz"""

    # Call the forward transform
    m, ov = getXYZ_to_GEO_affine(peg_lat,peg_lon,peg_hdg,peg_localRadius)

    # Inverse rotation matrix is transpose
    a = m.transpose()

    # The translation hast to be rotated and its sign changed
    d = np.zeros(3,dtype=np.float64)
    d[0] = -(a[0][0]*ov[0] + a[0][1]*ov[1] + a[0][2]*ov[2]);
    d[1] = -(a[1][0]*ov[0] + a[1][1]*ov[1] + a[1][2]*ov[2]);
    d[2] = -(a[2][0]*ov[0] + a[2][1]*ov[1] + a[2][2]*ov[2]);

    return a, d

#@jit(nopython=True,cache=True)
def geo_array_to_xyz_array(v, peg_lat,peg_lon,peg_hdg,peg_localRadius):
    """Go from geocentric coordinates to xyz coordinates, with peg point peg_
    The geocentric vector is an array of shape (npoints,3).
    """

    npoints = v.shape[0]

    # Initialize the xyz point
    p = np.zeros((npoints,3),dtype=np.float64)

    # Get affine transformation
    a, d = getGEO_to_XYZ_affine(peg_lat,peg_lon,peg_hdg,peg_localRadius)

    # Apply affine transformation

    p[:,0] = a[0][0]*v[:,0] + a[0][1]*v[:,1] + a[0][2]*v[:,2] + d[0];
    p[:,1] = a[1][0]*v[:,0] + a[1][1]*v[:,1] + a[1][2]*v[:,2] + d[1];
    p[:,2] = a[2][0]*v[:,0] + a[2][1]*v[:,1] + a[2][2]*v[:,2] + d[2];

    return p;

#@jit(nopython=True,cache=True)
def xyz_array_to_sch_array(p_xyz,peg_localRadius):
    npoints = p_xyz.shape[0]
    s = np.zeros(npoints,dtype=np.float64)
    c = np.zeros(npoints,dtype=np.float64)
    h = np.zeros(npoints,dtype=np.float64)

    x = p_xyz[:,0]
    y = p_xyz[:,1]
    z = p_xyz[:,2]
    r = np.sqrt(x*x + y*y + z*z);
    h = r - peg_localRadius;
    c = peg_localRadius*np.arcsin(z/r);
    s = peg_localRadius*np.arctan(y/x);

    return s,c,h

#@jit(nopython=True,cache=True)
def sch_array_to_xyz_array(s,c,h,peg_localRadius):
    npoints = s.shape[0]
    p_xyz = np.zeros((npoints,3),dtype=np.float64)

    c_lat = c/peg_localRadius;
    s_lon = s/peg_localRadius;
    r = peg_localRadius + h;

    # From spherical to Cartesian
    p_xyz[:,0] = r*np.cos(c_lat)*np.cos(s_lon);
    p_xyz[:,1] = r*np.cos(c_lat)*np.sin(s_lon);
    p_xyz[:,2] = r*np.sin(c_lat);

    return p_xyz

#@jit(nopython=True,cache=True)
def xyz_array_to_geo_array(p_xyz,peg_lat,peg_lon,peg_hdg,peg_localRadius):
    """Go from xyz array to a geo array."""

    # Get affine transformation
    m, ov = getXYZ_to_GEO_affine(peg_lat,peg_lon,peg_hdg,peg_localRadius)

    # Test whether an array is being passed, or just a single point

    npoints = p_xyz.shape[0]
    p = np.zeros((npoints,3),dtype=np.float64)

    # Apply affine transformation

    x = p_xyz[:,0]
    y = p_xyz[:,1]
    z = p_xyz[:,2]

    p[:,0] = m[0][0]*x + m[0][1]*y + m[0][2]*z + ov[0];
    p[:,1] = m[1][0]*x + m[1][1]*y + m[1][2]*z + ov[1];
    p[:,2] = m[2][0]*x + m[2][1]*y + m[2][2]*z + ov[2];

    return p;

#@jit(nopython=True,cache=True)
def geo_array_to_llh_array(v):
    """Given a numpy 2D array of (x,y,z) geocenric vectors, return a 2D array
    of (lat,lon,h)"""
    SEMIMAJOR_AXIS = 6378137. # in meters
    ECCENTRICITY_SQ = 0.00669437999015
    #ECCENTRICITY_SQ/(1-ECCENTRICITY_SQ)
    EP_SQUARED = 0.0067394969488402
    # SEMIMAJOR_AXIS*sqrt(1-ECCENTRICITY_SQ)
    SEMIMINOR_AXIS = 6356752.3135930374

    npoints = v.shape[0]
    lat = np.zeros((npoints,),dtype=np.float64)
    lon = np.zeros((npoints,),dtype=np.float64)
    h = np.zeros((npoints,),dtype=np.float64)

    # Longitude
    lon = np.arctan2(v[:,1],v[:,0])

    x = v[:,0]
    y = v[:,1]
    z = v[:,2]

    # Geodetic Latitude
    projRad = np.sqrt(x**2 + y**2)

    alpha = np.arctan(
    z/(projRad*np.sqrt(1. - ECCENTRICITY_SQ)))
    sa = np.sin(alpha);
    ca = np.cos(alpha);
    sa3 = sa*sa*sa;
    ca3 = ca*ca*ca;

    lat = np.arctan(
    (z + EP_SQUARED*SEMIMINOR_AXIS*sa3)/
        (projRad - ECCENTRICITY_SQ*SEMIMAJOR_AXIS*ca3))

    # height
    h = projRad/np.cos(lat) - eastRad(lat)

    return np.degrees(lat),np.degrees(lon),h


#@jit(nopython=True,cache=True)
def llh_array_to_sch_array(lat,lon,h,peg_lat,peg_lon,peg_hdg,peg_localRadius):
    """Go from lat,lon,h arrays to s,c,h arrays."""
    ECCENTRICITY_SQ = 0.00669437999015

    lat = np.radians(lat)
    lon = np.radians(lon)

    npoints = lat.shape[0]

    eastRadius = eastRad(lat);

    # Go from LLH to geocentyric

    v = np.zeros((npoints,3),dtype=np.float64)

    v[:,0] = (eastRadius + h)*np.cos(lat)*np.cos(lon)
    v[:,1] = (eastRadius + h)*np.cos(lat)*np.sin(lon)
    v[:,2] = (eastRadius*(1. - ECCENTRICITY_SQ) + h)*np.sin(lat)

    # Go from geocentric to xyz

    p_xyz = geo_array_to_xyz_array(v,peg_lat,peg_lon,peg_hdg,peg_localRadius)

    # Go from xyz to sch

    s,c,h = xyz_array_to_sch_array(p_xyz,peg_localRadius)

    return s,c,h

def sch_array_to_llh_array(s,c,h,peg_lat,peg_lon,peg_hdg,peg_localRadius):
    """Go from s,c,h arrays to lat,lon,h arrays."""

    # go from sch to xyz

    p_xyz = sch_array_to_xyz_array(s,c,h,peg_localRadius)

    # go from xyz to geocentric

    p_geo = xyz_array_to_geo_array(p_xyz,peg_lat,peg_lon,peg_hdg,peg_localRadius)

    # go from geocentric to llh

    lat, lon, h = geo_array_to_llh_array(p_geo)

    return lat, lon, h