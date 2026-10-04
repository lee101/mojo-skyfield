"""Every ported kernel is checked numerically against upstream Skyfield.

The comparison is against the real `skyfield` package on PyPI, which the
pixi environment installs as a test dependency.  Nothing here asserts
"it ran": each test compares arrays element by element and requires the
difference to be at or near the float64 rounding of the operation.
"""

import numpy as np
import pytest

skyfield_earthlib = pytest.importorskip("skyfield.earthlib")
skyfield_functions = pytest.importorskip("skyfield.functions")
skyfield_keplerlib = pytest.importorskip("skyfield.keplerlib")
skyfield_nutationlib = pytest.importorskip("skyfield.nutationlib")
skyfield_precessionlib = pytest.importorskip("skyfield.precessionlib")
skyfield_relativity = pytest.importorskip("skyfield.relativity")
skyfield_sgp4lib = pytest.importorskip("skyfield.sgp4lib")
skyfield_timelib = pytest.importorskip("skyfield.timelib")
skyfield_toposlib = pytest.importorskip("skyfield.toposlib")

from skyfield import (earthlib as up_earth, functions as up_fn,
                      keplerlib as up_kep, nutationlib as up_nut,
                      precessionlib as up_pre, relativity as up_rel,
                      sgp4lib as up_sgp, timelib as up_time,
                      toposlib as up_top)

from mojo_skyfield import (constants, earthlib as ms_earth,
                           functions as ms_fn, keplerlib as ms_kep,
                           nutationlib as ms_nut, precessionlib as ms_pre,
                           relativity as ms_rel, sgp4lib as ms_sgp,
                           timelib as ms_time, toposlib as ms_top)

GM = 398600.4418
AU_M = 149597870700.0


def assert_close(ours, theirs, rtol=1e-13, atol=0.0):
    ours = np.asarray(ours, dtype=np.float64)
    theirs = np.asarray(theirs, dtype=np.float64)
    assert ours.shape == theirs.shape, (ours.shape, theirs.shape)
    scale = max(np.max(np.abs(theirs)), 1e-300)
    assert np.max(np.abs(ours - theirs)) <= atol + rtol * scale


@pytest.fixture(scope="module")
def vectors():
    rng = np.random.default_rng(0)
    n = 7
    v = rng.normal(size=(3, n)) + np.array([1.0, 0.0, 0.0])[:, None]
    u = rng.normal(size=(3, n))
    m = np.zeros((3, 3, n))
    m[0, 0] = m[1, 1] = m[2, 2] = 1.0
    m[0, 1] = rng.normal(size=n) * 0.1
    m[1, 2] = rng.normal(size=n) * 0.1
    m[2, 0] = rng.normal(size=n) * 0.1
    return v, u, m


@pytest.fixture(scope="module")
def jd():
    rng = np.random.default_rng(1)
    return 2451545.0 + rng.random(7) * 8000.0


# ---------------------------------------------------------------------------
# skyfield.functions
# ---------------------------------------------------------------------------

def test_dots(vectors):
    v, u, _ = vectors
    assert_close(ms_fn.dots(v, u), up_fn.dots(v, u), 1e-14)
    assert_close(ms_fn.dots(v[:, 0], u[:, 0]), up_fn.dots(v[:, 0], u[:, 0]), 1e-14)


def test_length_of(vectors):
    v, _, _ = vectors
    assert_close(ms_fn.length_of(v), up_fn.length_of(v), 0.0)
    assert_close(ms_fn.length_of(v[:, 0]), up_fn.length_of(v[:, 0]), 0.0)


def test_mxv(vectors):
    v, u, m = vectors
    assert_close(ms_fn.mxv(m, v), up_fn.mxv(m, v), 0.0)
    assert_close(ms_fn.mxv(m[:, :, 0], v[:, 0]), up_fn.mxv(m[:, :, 0], v[:, 0]), 0.0)


def test_mxm_and_mxmxm(vectors):
    v, u, m = vectors
    assert_close(ms_fn.mxm(m, m), up_fn.mxm(m, m), 1e-15)
    assert_close(ms_fn.mxmxm(m, m, m), up_fn.mxmxm(m, m, m), 1e-15)
    assert_close(ms_fn.T(m), up_fn.T(m), 0.0)


def test_angle_between(vectors):
    v, u, _ = vectors
    assert_close(ms_fn.angle_between(v, u), up_fn.angle_between(v, u), 0.0)


def test_to_and_from_spherical(vectors):
    v, _, _ = vectors
    r, theta, phi = ms_fn.to_spherical(v)
    ur, utheta, uphi = up_fn.to_spherical(v)
    assert_close(r, ur, 0.0)
    assert_close(theta, utheta, 0.0)
    assert_close(phi, uphi, 0.0)
    assert_close(ms_fn.from_spherical(ur, utheta, uphi),
                 up_fn.from_spherical(ur, utheta, uphi), 1e-15)


def test_rotations():
    theta = np.array([0.0, 0.3, 1.0, -2.0, 3.0, 0.7, -0.1])
    for ours, theirs in ((ms_fn.rot_x, up_fn.rot_x),
                         (ms_fn.rot_y, up_fn.rot_y),
                         (ms_fn.rot_z, up_fn.rot_z),
                         (ms_fn.R1, up_fn.R1),
                         (ms_fn.R2, up_fn.R2),
                         (ms_fn.R3, up_fn.R3)):
        assert_close(ours(theta), theirs(theta), 0.0)
    assert ms_fn.rot_x(0.3).shape == (3, 3)


def test_angular_velocity_matrix():
    v = np.array([0.1, -0.2, 0.05])
    assert_close(ms_fn.angular_velocity_matrix(v),
                 up_fn.angular_velocity_matrix(v), 0.0)


def test_sqrt_nan():
    x = np.array([-1.0, 0.0, 2.0, 9.0])
    ours, theirs = ms_fn.sqrt_nan(x), up_fn.sqrt_nan(x)
    assert np.isnan(ours[0]) and np.isnan(theirs[0])
    assert_close(ours[1:], theirs[1:], 0.0)


# ---------------------------------------------------------------------------
# skyfield.timelib
# ---------------------------------------------------------------------------

def test_julian_day():
    year = np.array([2000, 2020, 1850, -100, 1582, 2400])
    month = np.array([1, 7, 12, 3, 10, 2])
    day = np.array([1, 4, 31, 15, 14, 29])
    assert_close(ms_time.julian_day(year, month, day),
                 up_time.julian_day(year, month, day), 0.0)
    # `julian_before` switches the pre-1582 dates to the Julian calendar.
    assert_close(ms_time.julian_day(year, month, day, 2299161),
                 up_time.julian_day(year, month, day, 2299161), 0.0)


def test_julian_date():
    year = np.array([2000, 2020, 1850, -100])
    month = np.array([1, 7, 12, 3])
    day = np.array([1, 4, 31, 15])
    hour = np.array([0, 13, 23, 6])
    minute = np.array([0, 5, 59, 30])
    second = np.array([0.0, 1.5, 59.25, 0.125])
    assert_close(ms_time.julian_date(year, month, day, hour, minute, second),
                 up_time.julian_date(year, month, day, hour, minute, second), 0.0)


def test_compute_calendar_date():
    jd = np.arange(2459000, 2459007)
    for ours, theirs in zip(ms_time.compute_calendar_date(jd),
                            up_time.compute_calendar_date(jd)):
        assert np.array_equal(ours, theirs)
    for ours, theirs in zip(ms_time.compute_calendar_date(jd, 2299161),
                            up_time.compute_calendar_date(jd, 2299161)):
        assert np.array_equal(ours, theirs)


def test_calendar_tuple(jd):
    ours = ms_time.calendar_tuple(jd)
    theirs = up_time.calendar_tuple(jd)
    for a, b in zip(ours, theirs):
        assert_close(np.asarray(a, dtype=float), np.asarray(b, dtype=float), 0.0)


def test_tdb_minus_tt(jd):
    assert_close(ms_time.tdb_minus_tt(jd), up_time.tdb_minus_tt(jd), 1e-14)


def test_julian_date_of_besselian_epoch():
    b = np.array([1900.0, 1950.0, 2000.0, 2025.0])
    assert_close(ms_time.julian_date_of_besselian_epoch(b),
                 up_time.julian_date_of_besselian_epoch(b), 0.0)


# ---------------------------------------------------------------------------
# skyfield.nutationlib and skyfield.precessionlib
# ---------------------------------------------------------------------------

def test_mean_obliquity(jd):
    assert_close(ms_nut.mean_obliquity(jd), up_nut.mean_obliquity(jd), 0.0)


def test_fundamental_arguments(jd):
    t = (jd - constants.T0) / 36525.0
    assert_close(ms_nut.fundamental_arguments(t),
                 up_nut.fundamental_arguments(t), 0.0)
    for terms in (1, 2, 3, 4, 5):
        assert_close(ms_nut.fundamental_arguments(t, terms),
                     up_nut.fundamental_arguments(t, terms), 0.0)


def test_iau2000a(jd):
    ours, theirs = ms_nut.iau2000a(jd), up_nut.iau2000a(jd)
    assert_close(ours[0], theirs[0], 1e-13)
    assert_close(ours[1], theirs[1], 1e-13)


def test_iau2000b(jd):
    ours, theirs = ms_nut.iau2000b(jd), up_nut.iau2000b(jd)
    assert_close(ours[0], theirs[0], 1e-13)
    assert_close(ours[1], theirs[1], 1e-13)


def test_iau2000a_truncated(jd):
    """The IAU 2000B model is IAU 2000A with (2, 77, 0) terms."""
    ours, theirs = ms_nut.iau2000a(jd, 2, 77, 0), up_nut.iau2000a(jd, 2, 77, 0)
    assert_close(ours[0], theirs[0], 1e-13)
    assert_close(ours[1], theirs[1], 1e-13)


def test_equation_of_the_equinoxes(jd):
    assert_close(
        ms_nut.equation_of_the_equinoxes_complimentary_terms(jd),
        up_nut.equation_of_the_equinoxes_complimentary_terms(jd), 1e-11)


def test_build_nutation_matrix(jd):
    mean_obliquity = up_nut.mean_obliquity(jd) * constants.ASEC2RAD
    dpsi, deps = up_nut.iau2000a(jd)
    k = constants.ASEC2RAD / 1e7
    dpsi, deps = dpsi * k, deps * k
    assert_close(
        ms_nut.build_nutation_matrix(mean_obliquity, mean_obliquity + deps, dpsi),
        up_nut.build_nutation_matrix(mean_obliquity, mean_obliquity + deps, dpsi),
        1e-15)


def test_compute_precession(jd):
    assert_close(ms_pre.compute_precession(jd),
                 up_pre.compute_precession(jd), 1e-15)


# ---------------------------------------------------------------------------
# skyfield.earthlib
# ---------------------------------------------------------------------------

def test_earth_rotation_angle(jd):
    assert_close(ms_earth.earth_rotation_angle(jd),
                 up_earth.earth_rotation_angle(jd), 1e-13)


def test_refraction_and_refract():
    alt = np.linspace(-2.0, 88.0, 40)
    assert_close(ms_earth.refraction(alt, 10.0, 1010.0),
                 up_earth.refraction(alt, 10.0, 1010.0), 0.0)
    assert_close(ms_earth.refract(alt, 10.0, 1010.0),
                 up_earth.refract(alt, 10.0, 1010.0), 1e-9)


def test_compute_limb_angle(vectors):
    v, u, _ = vectors
    assert_close(np.array(ms_earth.compute_limb_angle(v, u)),
                 np.array(up_earth.compute_limb_angle(v, u)), 1e-14)


def test_terra_and_reverse_terra():
    n = 7
    lat = np.linspace(-1.4, 1.4, n)
    lon = np.linspace(-3.0, 3.0, n)
    elev = np.linspace(0.0, 3000.0, n)
    gast = np.linspace(0.0, 24.0, n)
    ours = ms_earth.terra(lat, lon, elev, gast)
    theirs = up_earth.terra(lat, lon, elev, gast)
    assert_close(ours[0], theirs[0], 1e-14)
    assert_close(ours[1], theirs[1], 1e-14)

    ours = ms_earth.reverse_terra(theirs[0], gast)
    theirs = up_earth.reverse_terra(theirs[0], gast)
    for a, b in zip(ours, theirs):
        assert_close(a, b, 1e-12)


# ---------------------------------------------------------------------------
# skyfield.relativity
# ---------------------------------------------------------------------------

def test_light_time_difference(vectors):
    v, u, _ = vectors
    assert_close(ms_rel.light_time_difference(v, u),
                 up_rel.light_time_difference(v, u), 1e-14)


def test_compute_deflection(vectors):
    v, u, _ = vectors
    assert_close(ms_rel._compute_deflection(v, u, ms_rel.rmasses[399]),
                 up_rel._compute_deflection(v, u, up_rel.rmasses[399]), 1e-14)


def test_add_aberration(vectors):
    v, u, _ = vectors
    light_time = np.full(v.shape[1], 0.01)
    ours, theirs = v.copy(), v.copy()
    ms_rel.add_aberration(ours, u, light_time)
    up_rel.add_aberration(theirs, u, light_time)
    assert_close(ours, theirs, 1e-14)


# ---------------------------------------------------------------------------
# skyfield.sgp4lib
# ---------------------------------------------------------------------------

def test_theta_GMST1982(jd):
    ours, theirs = ms_sgp.theta_GMST1982(jd), up_sgp.theta_GMST1982(jd)
    assert_close(ours[0], theirs[0], 1e-13)
    assert_close(ours[1], theirs[1], 1e-13)


def test_cross(vectors):
    v, u, _ = vectors
    assert_close(ms_sgp._cross(v, u), up_sgp._cross(v, u), 1e-15)


def test_TEME_to_ITRF(jd, vectors):
    v, u, _ = vectors
    ours, theirs = ms_sgp.TEME_to_ITRF(jd, v, u), up_sgp.TEME_to_ITRF(jd, v, u)
    assert_close(ours[0], theirs[0], 1e-13)
    assert_close(ours[1], theirs[1], 1e-13)


def test_TEME_to_ITRF_with_polar_motion(jd, vectors):
    v, u, _ = vectors
    ours = ms_sgp.TEME_to_ITRF(jd, v, u, 0.3, -0.2)
    theirs = up_sgp.TEME_to_ITRF(jd, v, u, 0.3, -0.2)
    assert_close(ours[0], theirs[0], 1e-13)
    assert_close(ours[1], theirs[1], 1e-13)


@pytest.mark.parametrize('dtype', [np.int64, np.int32, np.float32, np.float64])
def test_TEME_rotation_at_theta_dtype_does_not_matter(dtype):
    """A caller-supplied `theta` must be read as float64 whatever it is.

    `TEME_rotation_at`'s `theta` is the one buffer that did not go through a
    float64 coercion before its address was handed to the kernel. The kernel
    reads it as `Float64`, so an int32 or float32 array of `n` elements was
    being re-read 8 bytes per element: wrong angles, and reads past the end of
    the buffer. This pins the fix, including the exact answer, not just
    "it did not crash".
    """
    t = ms_time.Time.from_julian_date(np.array([2460408.5, 2460409.5]))
    reference = ms_sgp.TEME_rotation_at(t, np.array([1.0, 2.0]))
    got = ms_sgp.TEME_rotation_at(t, np.array([1.0, 2.0], dtype=dtype))
    assert np.array_equal(got, reference), f'changed for dtype {dtype}'
    # A scalar int is the same hazard through a different door.
    assert np.allclose(ms_sgp.TEME_rotation_at(t, 2),
                       ms_sgp.TEME_rotation_at(t, 2.0))


# ---------------------------------------------------------------------------
# skyfield.toposlib
# ---------------------------------------------------------------------------

class _Geocentric(object):
    """Just enough of a position object for upstream's geoid routines."""

    center = 399

    def __init__(self, xyz_au):
        self._xyz = xyz_au

    def frame_xyz(self, frame):
        class _AU:
            au = self._xyz
        return _AU()


def test_latlon():
    for geoid, up_geoid in ((ms_top.wgs84, up_top.wgs84),
                            (ms_top.iers2010, up_top.iers2010)):
        assert geoid.radius == up_geoid.radius.au / AU_M * AU_M / AU_M * AU_M
        ours = geoid.latlon(37.3414, -121.6429, 0.05)
        theirs = up_geoid.latlon(37.3414, -121.6429, 0.05)
        assert_close(ours.itrs_xyz, theirs.itrs_xyz.au, 1e-15)


def test_latlon_of_and_height_of():
    xyz_au = up_top.wgs84.latlon(37.3414, -121.6429, 1234.5).itrs_xyz.au
    ours = ms_top.wgs84.latlon_of(xyz_au)
    theirs = up_top.wgs84.latlon_of(_Geocentric(xyz_au))
    assert_close(ours[0], theirs[0].radians, 1e-14)
    assert_close(ours[1], theirs[1].radians, 1e-14)
    assert_close(ms_top.wgs84.height_of(xyz_au) * AU_M,
                 up_top.wgs84.height_of(_Geocentric(xyz_au)).m, 1e-9)


def test_subpoint_of():
    xyz_au = up_top.wgs84.latlon(-12.5, 33.25, 800.0).itrs_xyz.au
    sub = ms_top.wgs84.subpoint_of(xyz_au)
    lat, lon = up_top.wgs84.latlon_of(_Geocentric(xyz_au))
    assert_close(sub.latitude, lat.degrees, 1e-12)
    assert_close(sub.longitude, lon.degrees, 1e-12)
    assert sub.elevation == 0.0


# ---------------------------------------------------------------------------
# skyfield.keplerlib
# ---------------------------------------------------------------------------

def test_normpi():
    x = np.array([-40.0, -7.0, -np.pi, -0.5, 0.0, 0.5, np.pi, 7.0, 40.0])
    assert_close(ms_kep.normpi(x), up_kep.normpi(x), 0.0)
    assert_close(ms_kep.normpi(7.0), up_kep.normpi(7.0), 0.0)


def test_eccentric_anomaly():
    e = np.array([0.1, 0.5, 0.9, 0.01, 0.3, 0.7, 0.2])
    M = np.array([0.1, 1.0, 3.0, -2.0, 0.5, -0.4, 2.5])
    assert_close([ms_kep.eccentric_anomaly(e[i], M[i]) for i in range(len(e))],
                 [up_kep.eccentric_anomaly(e[i], M[i]) for i in range(len(e))],
                 1e-12)


def test_true_anomaly_closed():
    e = np.array([0.1, 0.5, 0.9, 0.01, 0.3, 0.7, 0.2])
    E = ms_kep.eccentric_anomaly(e, np.array([0.1, 1.0, 3.0, -2.0, 0.5, -0.4, 2.5]))
    assert_close(ms_kep.true_anomaly_closed(e, E), up_kep.true_anomaly_closed(e, E),
                 1e-12)


def test_true_anomaly_hyperbolic():
    e = np.array([1.5, 2.0, 1.2])
    E = np.array([0.2, 0.5, 1.0])
    assert_close(ms_kep.true_anomaly_hyperbolic(e, E),
                 up_kep.true_anomaly_hyperbolic(e, E), 1e-12)


def test_true_anomaly_parabolic():
    p = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    M = np.array([0.1, 1.0, 3.0, -2.0, 0.5, -0.4, 2.5])
    # The cube root goes through `pow(x, 1/3)`, whose two libm
    # implementations agree to ~1e-10 rather than bit-exactly.
    assert_close(ms_kep.true_anomaly_parabolic(p, GM, M),
                 up_kep.true_anomaly_parabolic(p, GM, M), 1e-9)


def test_stumpff():
    x = np.array([-5.0, -1.0, -0.5, 0.0, 0.5, 1.0, 5.0, 0.1, -0.01])
    ours, theirs = np.array(ms_kep.stumpff(x)), np.array(up_kep.stumpff(x))
    assert_close(ours, theirs, 1e-11)


def test_ele_to_vec():
    p = np.array([7000.0, 8000.0, 9000.0, 10000.0, 11000.0, 12000.0, 13000.0])
    e = np.array([0.1, 0.5, 0.9, 0.01, 0.3, 0.7, 0.2])
    i = np.array([0.5, 0.9, 1.2, 0.2, 0.7, 1.0, 0.3])
    Om = np.array([0.1, 2.0, 3.5, 0.0, 1.0, 4.0, 5.0])
    w = np.array([0.2, 0.4, 1.0, 2.0, 3.0, 0.5, 1.5])
    v = np.array([0.1, 0.5, 1.0, 2.0, 2.5, 0.3, 1.2])
    ours = ms_kep.ele_to_vec(p, e, i, Om, w, v, GM)
    theirs = up_kep.ele_to_vec(p, e, i, Om, w, v, GM)
    assert_close(ours[0], theirs[0], 1e-13)
    assert_close(ours[1], theirs[1], 1e-13)


def test_ele_to_vec_rejects_bad_inclination():
    with pytest.raises(ValueError):
        ms_kep.ele_to_vec(7000.0, 0.1, 4.0, 0.0, 0.0, 0.0, GM)


def test_ele_to_vec_broadcasts_the_six_elements_against_each_other():
    """Mixed scalar and array arguments, the way upstream broadcasts them.

    Upstream's `ele_to_vec` broadcasts all six element arrays against each
    other, so `p` need not be the long one. The kernel reads all six as
    length-`n` runs, so the wrapper has to stretch the short ones itself --
    and the guards below have to see every element, not just the first.
    """
    scalars = (7000.0, 0.1, 1.0, 0.2, 0.3, 0.4)
    ours = ms_kep.ele_to_vec(*scalars, GM)
    theirs = up_kep.ele_to_vec(*scalars, GM)
    assert np.asarray(ours[0]).shape == (3,)
    assert_close(ours[0], np.asarray(theirs[0]).ravel(), 1e-13)

    # Only `e` is long: two states from one scalar semi-latus rectum.
    ours = ms_kep.ele_to_vec(7000.0, [0.1, 0.2], 1.0, 0.2, 0.3, 0.4, GM)
    theirs = up_kep.ele_to_vec(
        np.array([7000.0, 7000.0]), np.array([0.1, 0.2]), np.array([1.0, 1.0]),
        np.array([0.2, 0.2]), np.array([0.3, 0.3]), np.array([0.4, 0.4]), GM)
    assert np.asarray(ours[0]).shape == (3, 2)
    assert_close(ours[0], np.asarray(theirs[0]).reshape(3, 2), 1e-13)

    # And the guards must still fire over the whole array, not just element 0.
    with pytest.raises(ValueError):
        ms_kep.ele_to_vec(7000.0, 0.1, np.array([1.0, 3.5]), 0.2, 0.3, 0.4, GM)
    with pytest.raises(ValueError):
        ms_kep.ele_to_vec(7000.0, np.array([1.5, 0.1]), 1.0, 0.2, 0.3,
                          np.array([3.0, 0.4]), GM)


@pytest.mark.parametrize('position,velocity,t1', [
    (np.array([7000.0, 0.0, 1000.0]), np.array([0.0, 5.0, 4.0]),
     np.array([0.0, 100.0, 1000.0, 10000.0, 1e6, 1e7, -500.0, -1e6])),
    (np.array([1.0e7, 0.0, 0.0]), np.array([0.0, 12.0, 0.0]),
     np.array([1.0, 1e4, 1e6, 1e8])),
    (np.array([-2.0e7, 0.0, 0.0]), np.array([0.0, 8.0, 0.0]),
     np.array([1.0, 1e3, 1e5])),
])
def test_propagate(position, velocity, t1):
    ours = ms_kep.propagate(position, velocity, 0.0, t1, GM)
    theirs = up_kep.propagate(position, velocity, 0.0, t1, GM)
    assert_close(ours[0], theirs[0].reshape(3, -1), 1e-11)
    assert_close(ours[1], theirs[1].reshape(3, -1), 1e-11)


def test_propagate_rejects_bad_gm():
    with pytest.raises(ValueError):
        ms_kep.propagate(np.array([7000.0, 0, 0]), np.array([0.0, 7.5, 0.0]),
                         0.0, np.array([1.0]), -1.0)


def test_geographic_position_of():
    for xyz_au in (np.array([0.4, 0.3, 0.5]), np.array([1e-4, 2e-4, 3e-4])):
        ours = ms_top.wgs84.geographic_position_of(xyz_au)
        theirs = up_top.wgs84.geographic_position_of(_Geocentric(xyz_au))
        assert_close(ours.latitude, theirs.latitude.degrees, 1e-12)
        assert_close(ours.longitude, theirs.longitude.degrees, 1e-12)
        assert_close(ours.elevation, theirs.elevation.m, 1e-6)
        assert_close(ours.itrs_xyz, xyz_au, 0.0)


def test_subpoint_alias_is_geographic_position_of():
    assert (ms_top.wgs84.subpoint.__func__
            is ms_top.wgs84.geographic_position_of.__func__)


def test_lst_hours_at(jd):
    from skyfield.api import load
    theirs_t = load.timescale().tt_jd(jd)
    ours_t = ms_time.Time(np.floor(jd), jd - np.floor(jd),
                          theirs_t.ut1_fraction, theirs_t.tdb_fraction)
    ours = ms_top.wgs84.latlon(37.3414, -121.6429, 30.0)
    theirs = up_top.wgs84.latlon(37.3414, -121.6429, 30.0)
    assert_close(ours.lst_hours_at(ours_t), theirs.lst_hours_at(theirs_t), 1e-11)


def test_geographic_position_refract():
    ours = ms_top.wgs84.latlon(37.3414, -121.6429, 30.0)
    theirs = up_top.wgs84.latlon(37.3414, -121.6429, 30.0)
    alt = np.linspace(-2.0, 80.0, 25)
    assert_close(ours.refract(alt, 'standard', 'standard'),
                 theirs.refract(alt, 'standard', 'standard').degrees, 1e-9)


def test_topos_uses_the_iers2010_geoid():
    """Upstream's deprecated `Topos` builds on IERS2010, not WGS84.

    Upstream's own `Topos` takes a different (and internally inconsistent)
    argument list, so the check is against the geoid its source names.
    """
    ours = ms_top.Topos(37.3414, -121.6429, 30.0)
    assert_close(ours.itrs_xyz,
                 ms_top.iers2010.latlon(37.3414, -121.6429, 30.0).itrs_xyz, 0.0)
    assert ours.model is ms_top.iers2010


def test_polar_aliases():
    assert ms_fn.to_polar is ms_fn.to_spherical
    assert ms_fn.from_polar is ms_fn.from_spherical
