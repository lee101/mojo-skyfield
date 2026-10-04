"""`skyfield.earthlib` in Mojo."""

import numpy as np

from ._lib import addr, lib, out_vec
from .constants import AU_M, ERAD, IERS_2010_INVERSE_EARTH_FLATTENING
from .functions import _flat, _follow_flat, _follow_vec, _lead_vec

earth_radius_au = ERAD / AU_M
one_minus_flattening = 1.0 - 1.0 / IERS_2010_INVERSE_EARTH_FLATTENING
one_minus_flattening_squared = one_minus_flattening * one_minus_flattening


def terra(latitude, longitude, elevation, gast):
    """Deprecated conversion from lat,lon,t -> GCRS; neglects polar motion."""
    lat, scalar = _flat(latitude)
    n = lat.size
    lon = _follow_flat(lat, longitude, n)
    elev = _follow_flat(lat, elevation, n)
    g = _follow_flat(lat, gast, n)
    pos, vel = out_vec(n), out_vec(n)
    lib().sk_terra(addr(lat), addr(lon), addr(elev), addr(g), n,
                   addr(pos), addr(vel))
    if scalar:
        return pos[:, 0], vel[:, 0]
    return pos, vel


def reverse_terra(xyz_au, gast, iterations=3):
    """Deprecated conversion from GCRS -> lat,lon,t; neglects polar motion."""
    v, ndim = _lead_vec(xyz_au)
    n = v.shape[1]
    g = _follow_flat(v[0], gast, n)
    lat, lon, elev = np.empty(n), np.empty(n), np.empty(n)
    lib().sk_reverse_terra(addr(v), addr(g), iterations, n, addr(lat), addr(lon),
                           addr(elev))
    if ndim == 1:
        return float(lat[0]), float(lon[0]), float(elev[0])
    return lat, lon, elev


def compute_limb_angle(position_au, observer_au):
    """Determine the angle of an object above or below the Earth's limb."""
    p, ndim = _lead_vec(position_au)
    n = p.shape[1]
    o = _follow_vec(p, observer_au)
    limb, nadir = np.empty(n), np.empty(n)
    lib().sk_compute_limb_angle(addr(p), addr(o), n, addr(limb), addr(nadir))
    if ndim == 1:
        return float(limb[0]), float(nadir[0])
    return limb, nadir


def sidereal_time(t):
    """Compute Greenwich Mean Sidereal Time (GMST) in hours at time ``t``."""
    jd_ut1, fraction_ut1, tdb_fraction, n = _sidereal_args(t)
    dst = np.empty(n)
    lib().sk_sidereal_time(addr(jd_ut1), addr(fraction_ut1), addr(tdb_fraction),
                           n, addr(dst))
    return dst


def _sidereal_args(t):
    whole = np.ascontiguousarray(np.asarray(t.whole, dtype=np.float64))
    ut1 = np.ascontiguousarray(np.asarray(t.ut1_fraction, dtype=np.float64))
    tdb = np.ascontiguousarray(np.asarray(t.tdb_fraction, dtype=np.float64))
    return whole, ut1, tdb, whole.size


def earth_rotation_angle(jd_ut1, fraction_ut1=0.0):
    """Return the value of the Earth Rotation Angle (theta) for a UT1 date."""
    jd, scalar = _flat(jd_ut1)
    n = jd.size
    frac = np.ascontiguousarray(
        np.broadcast_to(np.asarray(fraction_ut1, dtype=np.float64), (n,)))
    dst = np.empty(n)
    lib().sk_earth_rotation_angle(addr(jd), addr(frac), n, addr(dst))
    return float(dst[0]) if scalar else dst


def refraction(alt_degrees, temperature_C, pressure_mbar):
    """Given an observed altitude, estimate atmospheric refraction, in degrees."""
    alt, scalar = _flat(alt_degrees)
    dst = np.empty_like(alt)
    lib().sk_refraction(addr(alt), float(temperature_C), float(pressure_mbar),
                        alt.size, addr(dst))
    return float(dst[0]) if scalar else dst


def refract(alt_degrees, temperature_C, pressure_mbar):
    """Given an unrefracted `alt` determine where it will appear in the sky."""
    alt, scalar = _flat(alt_degrees)
    dst = np.empty_like(alt)
    lib().sk_refract(addr(alt), float(temperature_C), float(pressure_mbar),
                     alt.size, addr(dst))
    return float(dst[0]) if scalar else dst
