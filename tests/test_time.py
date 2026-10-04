"""The `Time` object's reified properties, checked against upstream's.

Upstream builds a `Time` from a UTC calendar date plus the leap-second and
Delta-T tables.  Those tables are data lookups that this port does not
cover, so the tests below hand both implementations the same `whole`,
`ut1_fraction` and `tdb_fraction` and then compare everything computed
from them: sidereal time, nutation, precession and the ICRS rotation
matrix.
"""

import numpy as np
import pytest

skyfield_api = pytest.importorskip("skyfield.api")

from skyfield import framelib as up_frame
from skyfield.api import load

from mojo_skyfield import constants, framelib as ms_frame, timelib as ms_time

jd = np.array([2451545.0, 2455000.5, 2459000.25, 2460000.125, 2440000.75,
               2470000.5, 2458000.3])


@pytest.fixture(scope="module")
def times():
    ts = load.timescale()
    theirs = ts.tt_jd(jd)
    ours = ms_time.Time(np.floor(jd), jd - np.floor(jd),
                        theirs.ut1_fraction, theirs.tdb_fraction)
    return ours, theirs


def assert_close(a, b, rtol):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert a.shape == b.shape, (a.shape, b.shape)
    scale = max(np.max(np.abs(b)), 1e-300)
    assert np.max(np.abs(a - b)) <= rtol * scale


def test_low_precision_floats(times):
    ours, theirs = times
    assert_close(ours.tt, theirs.tt, 0.0)
    assert_close(ours.tdb, theirs.tdb, 0.0)
    assert_close(ours.ut1, theirs.ut1, 0.0)


def test_gmst(times):
    ours, theirs = times
    assert_close(ours.gmst, theirs.gmst, 1e-12)


def test_gast(times):
    ours, theirs = times
    assert_close(ours.gast, theirs.gast, 1e-11)


def test_nutation_and_precession_matrices(times):
    ours, theirs = times
    assert_close(ours.nutation_matrix(), theirs.nutation_matrix(), 1e-14)
    assert_close(ours.precession_matrix(), theirs.precession_matrix(), 1e-14)


def test_mean_and_true_obliquity(times):
    ours, theirs = times
    assert_close(ours._mean_obliquity_radians, theirs._mean_obliquity_radians,
                 1e-14)
    angles = theirs._nutation_angles_radians
    dpsi, deps = angles() if callable(angles) else angles
    assert_close(ours.true_obliquity,
                 theirs._mean_obliquity_radians + deps, 1e-13)


def test_ICRS_rotation_matrix(times):
    ours, theirs = times
    assert_close(ours.M, theirs.M, 1e-13)
    assert_close(ours.MT, theirs.MT, 1e-13)


def test_frame_bias_matches_upstream():
    assert np.array_equal(ms_frame.ICRS_to_J2000, up_frame.ICRS_to_J2000)


def test_itrs_and_tirs_rotation(times):
    ours, theirs = times
    assert_close(ms_frame.tirs.rotation_at(ours),
                 up_frame.tirs.rotation_at(theirs), 1e-12)
    assert_close(ms_frame.itrs.rotation_at(ours),
                 up_frame.itrs.rotation_at(theirs), 1e-12)


def test_polar_motion_rotation(times):
    from mojo_skyfield import functions as ms_fn
    xp = np.full(len(jd), 0.15)
    yp = np.full(len(jd), 0.25)
    ours, _ = times
    ours_pm = ours.polar_motion_matrix(xp, yp)
    # Upstream's `Time.polar_motion_matrix()`, given the same three angles.
    sprime, x, y = ours.polar_motion_angles(xp, yp)
    theirs_pm = ms_fn.mxmxm(ms_fn.rot_x(y * constants.ASEC2RAD),
                            ms_fn.rot_y(x * constants.ASEC2RAD),
                            ms_fn.rot_z(-sprime * constants.ASEC2RAD))
    assert_close(ours_pm, theirs_pm, 1e-13)
    # Upstream's itrs.rotation_at() is mxm(polar_motion, tirs.rotation_at(t)).
    assert_close(ms_frame.itrs.rotation_at(ours, polar_motion=ours_pm),
                 ms_fn.mxm(ours_pm, ms_frame.tirs.rotation_at(ours)), 1e-14)


def test_ecliptic_matrix(times):
    ours, theirs = times
    assert_close(ms_frame.build_ecliptic_matrix(ours),
                 up_frame.build_ecliptic_matrix(theirs), 1e-12)


def test_TEME_rotation_at(times):
    from mojo_skyfield import sgp4lib as ms_sgp
    from skyfield import sgp4lib as up_sgp
    ours, theirs = times
    assert_close(ms_sgp.TEME_rotation_at(ours),
                 up_sgp.TEME.rotation_at(theirs), 1e-12)


def test_ICRS_and_inertial_frame_are_identity(times):
    ours, theirs = times
    assert np.array_equal(ms_frame.ICRS.rotation_at(ours),
                          up_frame.ICRS.rotation_at(theirs))
    frame = ms_frame.InertialFrame(np.eye(3))
    assert np.array_equal(frame.rotation_at(ours), np.eye(3))


def test_equinox_frames_of_date(times):
    ours, theirs = times
    assert_close(ms_frame.mean_equinox_of_date.rotation_at(ours),
                 up_frame.mean_equator_and_equinox_of_date.rotation_at(theirs),
                 1e-12)
    assert_close(ms_frame.true_equinox_of_date.rotation_at(ours),
                 up_frame.true_equator_and_equinox_of_date.rotation_at(theirs),
                 1e-12)
    assert_close(ms_frame.ecliptic_frame.rotation_at(ours),
                 up_frame.ecliptic_frame.rotation_at(theirs), 1e-12)


def test_geographic_position_rotation_at(times):
    from mojo_skyfield import toposlib as ms_top
    from skyfield import toposlib as up_top
    ours, theirs = times
    ours_p = ms_top.wgs84.latlon(37.3414, -121.6429, 30.0)
    theirs_p = up_top.wgs84.latlon(37.3414, -121.6429, 30.0)
    assert_close(ours_p.rotation_at(ours), theirs_p.rotation_at(theirs), 1e-12)
    assert_close(ours_p._dRdt_times_RT_at(ours),
                 theirs_p._dRdt_times_RT_at(theirs), 1e-12)
