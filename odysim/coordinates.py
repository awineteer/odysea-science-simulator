
import numpy as np
#from numba import jit


class WGS84:
    """Definition of WGS84 ellipsoid and calculation of local radii."""
    # The Earth's constants
    SEMIMAJOR_AXIS = 6378137.  # in meters
    # SEMIMAJOR_AXIS*sqrt(1-ECCENTRICITY_SQ)
    SEMIMINOR_AXIS = 6356752.3135930374
    ECCENTRICITY_SQ = 0.00669437999015
    # ECCENTRICITY_SQ/(1-ECCENTRICITY_SQ)
    EP_SQUARED = 0.0067394969488402
    CENTER_SCALE = 0.9996

    # Auxiliary Functions
    @staticmethod
    def east_radius(lat):
        """radius of curvature in the east direction"""
        r = WGS84.SEMIMAJOR_AXIS / np.sqrt(1. - WGS84.ECCENTRICITY_SQ*np.sin(np.deg2rad(lat))**2)
        return r

    @staticmethod
    def north_radius(lat):
        """radius of curvature in the north direction"""
        r = WGS84.SEMIMAJOR_AXIS*(1. - WGS84.ECCENTRICITY_SQ) / (1. - WGS84.ECCENTRICITY_SQ*np.sin(np.deg2rad(lat))**2)**1.5
        return r

    @staticmethod
    def local_radius(azimuth, lat):
        """Local radius of curvature along an azimuth direction measured
        clockwise from north. Azimuth in radians
        """
        east_radius = WGS84.east_radius(lat)
        north_radius = WGS84.north_radius(lat)
        r = east_radius*north_radius / (east_radius*np.cos(azimuth)**2 + north_radius*np.sin(azimuth)**2)
        return r


def getPegPointVector(peg_lat, peg_lon):
    """Get vector from WGS-84 center to peg point in
    geocentric coordinates."""
    p = np.full((peg_lat.size, 3), np.nan)

    # Calculate useful constants
    clt = np.cos(np.deg2rad(peg_lat))
    slt = np.sin(np.deg2rad(peg_lat))
    clo = np.cos(np.deg2rad(peg_lon))
    slo = np.sin(np.deg2rad(peg_lon))

    # east radius of curvature */
    east_radius = WGS84.east_radius(peg_lat)

    # displacement vector */
    p[:, 0] = east_radius*clt*clo
    p[:, 1] = east_radius*clt*slo
    p[:, 2] = east_radius*(1. - WGS84.ECCENTRICITY_SQ)*slt

    return p


def getXYZ_to_GEO_affine(peg_lat, peg_lon, peg_hdg, peg_local_radius):
    """Function to compute the transformation matrix
    form xyz to geocentric"""

    m = np.full((peg_lat.size, 3, 3), np.nan)
    up = np.full((peg_lat.size, 3), np.nan)  # local up vector in geocentric coordinates

    # Calculate useful constants
    clt = np.cos(np.deg2rad(peg_lat))
    slt = np.sin(np.deg2rad(peg_lat))
    clo = np.cos(np.deg2rad(peg_lon))
    slo = np.sin(np.deg2rad(peg_lon))
    chg = np.cos(np.deg2rad(peg_hdg))
    shg = np.sin(np.deg2rad(peg_hdg))

    # Fill in the rotation matrix
    m[:, 0, 0] = clt*clo
    m[:, 0, 1] = -shg*slo - slt*clo*chg
    m[:, 0, 2] = slo*chg - slt*clo*shg
    m[:, 1, 0] = clt*slo
    m[:, 1, 1] = clo*shg - slt*slo*chg
    m[:, 1, 2] = -clo*chg - slt*slo*shg
    m[:, 2, 0] = slt
    m[:, 2, 1] = clt*chg
    m[:, 2, 2] = clt*shg

    #Find the vector from the center of the ellipsoid to the peg point */
    p = getPegPointVector(peg_lat, peg_lon)

    # Calculate the local upward vector in geocentric coordinates */
    up[:, 0] = peg_local_radius*clt*clo
    up[:, 1] = peg_local_radius*clt*slo
    up[:, 2] = peg_local_radius*slt

    #Calculate the translation vector for the sch -> xyz transformation
    ov = p - up

    return m, ov


def getGEO_to_XYZ_affine(peg_lat, peg_lon, peg_hdg, peg_local_radius):
    """Function to compute the transformation matrix form
    geocentric to xyz"""

    # Call the forward transform
    m, ov = getXYZ_to_GEO_affine(peg_lat, peg_lon, peg_hdg, peg_local_radius)

    # Inverse rotation matrix is transpose
    a = m.transpose((0, 2, 1))

    # The translation hast to be rotated and its sign changed
    d = np.zeros((peg_lat.size, 3))
    d[:, 0] = -(a[:, 0, 0]*ov[:, 0] + a[:, 0, 1]*ov[:, 1] + a[:, 0, 2]*ov[:, 2])
    d[:, 1] = -(a[:, 1, 0]*ov[:, 0] + a[:, 1, 1]*ov[:, 1] + a[:, 1, 2]*ov[:, 2])
    d[:, 2] = -(a[:, 2, 0]*ov[:, 0] + a[:, 2, 1]*ov[:, 1] + a[:, 2, 2]*ov[:, 2])

    return a, d


def geo_array_to_xyz_array(x, y, z, peg_lat, peg_lon, peg_hdg, peg_local_radius):
    """Go from geocentric coordinates to xyz coordinates, with peg point peg_
    The geocentric vector is an array of shape (npoints,3).
    """
    # Initialize the xyz point
    p = np.zeros((peg_lat.size, 3))

    # Get affine transformation
    a, d = getGEO_to_XYZ_affine(peg_lat, peg_lon, peg_hdg, peg_local_radius)

    # Apply affine transformation

    x_prime = a[:, 0, 0, np.newaxis]*x + a[:, 0, 1, np.newaxis]*y + a[:, 0, 2, np.newaxis]*z + d[:, 0, np.newaxis]
    y_prime = a[:, 1, 0, np.newaxis]*x + a[:, 1, 1, np.newaxis]*y + a[:, 1, 2, np.newaxis]*z + d[:, 1, np.newaxis]
    z_prime = a[:, 2, 0, np.newaxis]*x + a[:, 2, 1, np.newaxis]*y + a[:, 2, 2, np.newaxis]*z + d[:, 2, np.newaxis]

    return x_prime, y_prime, z_prime


def xyz_array_to_sch_array(x_prime, y_prime, z_prime, peg_local_radius):

    r = np.sqrt(x_prime**2 + y_prime**2 + z_prime**2)
    h = r - peg_local_radius
    c = peg_local_radius*np.arcsin(z_prime/r)
    s = peg_local_radius*np.arctan(y_prime/x_prime)

    return s, c, h


def sch_array_to_xyz_array(s, c, h, peg_local_radius):
    """Transform spherical cross-track height (s, c, h) coordinates to geocentric spheroid (x', y', z') coordinates."""

    c_lat = np.outer(1/peg_local_radius, c)
    s_lon = np.outer(1/peg_local_radius, s)
    r = np.add.outer(peg_local_radius, h)

    # Get geocentric x, y, z coordinates based on sphere approximation
    x_prime = r*np.cos(c_lat)*np.cos(s_lon)
    y_prime = r*np.cos(c_lat)*np.sin(s_lon)
    z_prime = r*np.sin(c_lat)

    return x_prime, y_prime, z_prime


def xyz_array_to_geo_array(x_prime, y_prime, z_prime, peg_lat, peg_lon, peg_hdg, peg_local_radius):
    """Go from xyz array to a geo array."""
    # Get affine transformation
    m, ov = getXYZ_to_GEO_affine(peg_lat, peg_lon, peg_hdg, peg_local_radius)

    # Apply affine transformation from spheroid geocentric coords to WGS84 ellipsoid geocentric coords
    x = m[:, 0, 0, np.newaxis]*x_prime + m[:, 0, 1, np.newaxis]*y_prime + m[:, 0, 2, np.newaxis]*z_prime + ov[:, 0, np.newaxis]
    y = m[:, 1, 0, np.newaxis]*x_prime + m[:, 1, 1, np.newaxis]*y_prime + m[:, 1, 2, np.newaxis]*z_prime + ov[:, 1, np.newaxis]
    z = m[:, 2, 0, np.newaxis]*x_prime + m[:, 2, 1, np.newaxis]*y_prime + m[:, 2, 2, np.newaxis]*z_prime + ov[:, 2, np.newaxis]

    return x, y, z


def geo_array_to_llh_array(x, y, z):
    """Given a numpy 2D array of (x,y,z) geocenric vectors, return a 2D array
    of (lat,lon,h)"""
    lon = np.arctan2(y, x)

    proj_rad = np.sqrt(x**2 + y**2)

    alpha = np.arctan(
        z/(proj_rad*np.sqrt(1. - WGS84.ECCENTRICITY_SQ))
    )
    sa = np.sin(alpha)
    ca = np.cos(alpha)

    lat = np.arctan(
    (z + WGS84.EP_SQUARED*WGS84.SEMIMINOR_AXIS*sa**3)/
        (proj_rad - WGS84.ECCENTRICITY_SQ*WGS84.SEMIMAJOR_AXIS*ca**3))

    # lat = np.rad2deg(lat)
    # lon = np.rad2deg(lon)

    # height
    h = proj_rad/np.cos(lat) - WGS84.east_radius(np.rad2deg(lat))

    return np.rad2deg(lat), np.rad2deg(lon), h


def llh_array_to_sch_array(lat, lon, h, peg_lat, peg_lon, peg_hdg, peg_local_radius):
    """Go from lat, lon, h arrays to s, c, h arrays."""
    lat = np.array(lat)
    lon = np.array(lon)
    h = np.array(h)
    peg_lat = np.array(peg_lat)
    peg_lon = np.array(peg_lon)
    peg_hdg = np.array(peg_hdg)
    peg_local_radius = np.array(peg_local_radius)

    east_radius = WGS84.east_radius(lat)
    # Go from LLH to geocentric
    x = (east_radius + h)*np.cos(np.deg2rad(lat))*np.cos(np.deg2rad(lon))
    y = (east_radius + h)*np.cos(np.deg2rad(lat))*np.sin(np.deg2rad(lon))
    z = (east_radius*(1. - WGS84.ECCENTRICITY_SQ) + h)*np.sin(np.deg2rad(lat))

    # Go from geocentric to xyz

    x_prime, y_prime, z_prime = geo_array_to_xyz_array(x, y, z, peg_lat, peg_lon, peg_hdg, peg_local_radius)

    # Go from xyz to sch

    s, c, h = xyz_array_to_sch_array(x_prime, y_prime, z_prime, peg_local_radius)

    return s, c, h


def sch_array_to_llh_array(s, c, h, peg_lat, peg_lon, peg_hdg, peg_local_radius):
    """Go from s,c,h arrays to lat,lon,h arrays."""

    # go from sch to xyz

    x_prime, y_prime, z_prime = sch_array_to_xyz_array(s, c, h, peg_local_radius)

    # go from xyz to geocentric

    x, y, z = xyz_array_to_geo_array(x_prime, y_prime, z_prime, peg_lat, peg_lon, peg_hdg, peg_local_radius)

    # go from geocentric to llh

    lat, lon, h = geo_array_to_llh_array(x, y, z)

    return lat, lon, h