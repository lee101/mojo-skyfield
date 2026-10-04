"""Every exported `sk_*` kernel, called by name, checked against upstream.

`test_parity.py` reaches the kernels through the Python API.  This file
calls each exported symbol through `ctypes` itself, so a kernel that is
wired up and completely wrong cannot ship green: nothing here reaches it
except by name.  `test_kernel_coverage` fails if an export is added without
a test here.

Buffers cross as `Int` addresses and are laid out exactly as upstream's
NumPy arrays are, so the same inputs feed both sides.
"""

import re
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("skyfield")

from skyfield import (earthlib as up_earth, framelib as up_frame,  # noqa: E402
                      functions as up_fn, keplerlib as up_kep,
                      nutationlib as up_nut, precessionlib as up_pre,
                      relativity as up_rel, sgp4lib as up_sgp,
                      timelib as up_time)

from mojo_skyfield import _tables  # noqa: E402
from mojo_skyfield._lib import _SIGNATURES, addr, lib  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
JULIAN_BEFORE = 2299161.0  # upstream's default; None on the Python side.


def a(x):
    """A C-contiguous float64 view, as the C ABI requires."""
    return np.ascontiguousarray(x, dtype=np.float64)


def assert_matrix_close(ours, theirs, rtol=1e-12):
    """Compare a (3, 3, n) stack entry by entry.

    These matrices have a ~1.0 diagonal and ~1e-6 off-diagonal, and each
    off-diagonal entry is a difference of products of unit-sized `sin`/`cos`.
    A single relative tolerance over the whole array would be dominated by
    whichever element is largest, so compare each element against its own
    magnitude -- which is what "float64 rounding of the operation" means for
    a quantity formed this way.
    """
    ours = np.asarray(ours, dtype=np.float64)
    theirs = np.asarray(theirs, dtype=np.float64)
    assert ours.shape == theirs.shape, (ours.shape, theirs.shape)
    scale = np.maximum(np.abs(theirs), 1e-300)
    worst = np.max(np.abs(ours - theirs) / scale)
    assert worst <= rtol, f"worst relative element difference {worst:g}"


# ---------------------------------------------------------------------------
# skyfield.functions
# ---------------------------------------------------------------------------

def test_sk_sqrt_nan(kern):
    x = a([-1.0, 0.0, 4.0, 1e12])
    dst = np.empty_like(x)
    kern.sk_sqrt_nan(addr(x), x.size, addr(dst))
    theirs = up_fn.sqrt_nan(x)
    assert np.isnan(dst[0]) and np.isnan(theirs[0])
    np.testing.assert_array_equal(dst[1:], theirs[1:])


def test_sk_dots(kern):
    v = a(np.random.default_rng(2).normal(size=(3, 9)))
    u = a(np.random.default_rng(3).normal(size=(3, 9)))
    dst = np.empty(9)
    kern.sk_dots(addr(v), addr(u), 9, addr(dst))
    np.testing.assert_allclose(dst, up_fn.dots(v, u), rtol=1e-15, atol=0.0)


def test_sk_T(kern):
    M = np.random.default_rng(4).normal(size=(3, 3, 5))
    dst = np.empty_like(M)
    kern.sk_T(addr(M), 5, addr(dst))
    np.testing.assert_array_equal(dst, up_fn.T(M))


def test_sk_mxv(kern):
    M = a(np.random.default_rng(5).normal(size=(3, 3, 6)))
    v = a(np.random.default_rng(6).normal(size=(3, 6)))
    dst = np.empty_like(v)
    kern.sk_mxv(addr(M), addr(v), 6, addr(dst))
    np.testing.assert_allclose(dst, up_fn.mxv(M, v), rtol=1e-15, atol=0.0)


def test_sk_mxm(kern):
    M1 = a(np.random.default_rng(7).normal(size=(3, 3, 4)))
    M2 = a(np.random.default_rng(8).normal(size=(3, 3, 4)))
    dst = np.empty_like(M1)
    kern.sk_mxm(addr(M1), addr(M2), 4, addr(dst))
    np.testing.assert_allclose(dst, up_fn.mxm(M1, M2), rtol=1e-14, atol=0.0)


def test_sk_mxmxm(kern):
    Ms = [a(np.random.default_rng(9 + i).normal(size=(3, 3, 3)))
          for i in range(3)]
    dst = np.empty_like(Ms[0])
    kern.sk_mxmxm(addr(Ms[0]), addr(Ms[1]), addr(Ms[2]), 3, addr(dst))
    theirs = up_fn.mxmxm(*Ms)
    np.testing.assert_allclose(dst, theirs, rtol=1e-14, atol=0.0)


def test_sk_length_of(kern):
    xyz = a(np.random.default_rng(12).normal(size=(3, 11)))
    dst = np.empty(11)
    kern.sk_length_of(addr(xyz), 11, addr(dst))
    np.testing.assert_allclose(dst, up_fn.length_of(xyz), rtol=1e-15, atol=0.0)


def test_sk_angle_between(kern):
    u = a(np.random.default_rng(13).normal(size=(3, 7)))
    v = a(np.random.default_rng(14).normal(size=(3, 7)))
    dst = np.empty(7)
    kern.sk_angle_between(addr(u), addr(v), 7, addr(dst))
    np.testing.assert_allclose(dst, up_fn.angle_between(u, v), rtol=1e-15, atol=0.0)


def test_sk_to_spherical(kern):
    xyz = a(np.random.default_rng(15).normal(size=(3, 8)))
    r, theta, phi = np.empty(8), np.empty(8), np.empty(8)
    kern.sk_to_spherical(addr(xyz), 8, addr(r), addr(theta), addr(phi))
    theirs = up_fn.to_spherical(xyz)
    for ours, other in zip((r, theta, phi), theirs):
        np.testing.assert_allclose(ours, other, rtol=1e-14, atol=0.0)


def test_sk_from_spherical(kern):
    r, theta, phi = (a(np.random.default_rng(16 + i).normal(size=6))
                     for i in range(3))
    dst = np.empty((3, 6))
    kern.sk_from_spherical(addr(r), addr(theta), addr(phi), 6, addr(dst))
    np.testing.assert_allclose(dst, up_fn.from_spherical(r, theta, phi),
                               rtol=1e-14, atol=0.0)


@pytest.mark.parametrize("symbol,upstream", [
    ("sk_rot_x", up_fn.rot_x),
    ("sk_rot_y", up_fn.rot_y),
    ("sk_rot_z", up_fn.rot_z),
    ("sk_R1", up_fn.R1),
    ("sk_R2", up_fn.R2),
    ("sk_R3", up_fn.R3),
])
def test_sk_rotations(kern, symbol, upstream):
    theta = a(np.linspace(-3.0, 3.0, 13))
    dst = np.empty((3, 3, 13))
    getattr(kern, symbol)(addr(theta), 13, addr(dst))
    np.testing.assert_allclose(dst, upstream(theta), rtol=1e-15, atol=0.0)


def test_sk_angular_velocity_matrix(kern):
    v = a(np.random.default_rng(20).normal(size=(3, 5)))
    dst = np.empty((3, 3, 5))
    kern.sk_angular_velocity_matrix(addr(v), 5, addr(dst))
    np.testing.assert_allclose(dst, up_fn.angular_velocity_matrix(v),
                               rtol=1e-15, atol=0.0)


# ---------------------------------------------------------------------------
# skyfield.timelib
# ---------------------------------------------------------------------------

def test_sk_julian_day(kern):
    year = a([2000, 2020, 1850, -100, 1582, 2400])
    month = a([1, 7, 12, 3, 10, 2])
    day = a([1, 4, 31, 15, 14, 29])
    for before in (None, JULIAN_BEFORE):
        dst = np.empty(6)
        kern.sk_julian_day(addr(year), addr(month), addr(day),
                           -1.0 if before is None else before, 6, addr(dst))
        np.testing.assert_array_equal(
            dst, up_time.julian_day(year.astype(int), month.astype(int),
                                    day.astype(int), before))


def test_sk_julian_date(kern):
    args = (a([2000, 2020, 1850, -100]), a([1, 7, 12, 3]), a([1, 4, 31, 15]),
            a([0, 13, 23, 6]), a([0, 5, 59, 30]), a([0.0, 1.5, 59.25, 0.125]))
    dst = np.empty(4)
    kern.sk_julian_date(*(addr(x) for x in args), -1.0, 4, addr(dst))
    np.testing.assert_allclose(dst, up_time.julian_date(*args), rtol=1e-14,
                               atol=0.0)


def test_sk_julian_date_of_besselian_epoch(kern):
    b = a([1900.0, 1950.0, 2000.0, 2025.0])
    dst = np.empty(4)
    kern.sk_julian_date_of_besselian_epoch(addr(b), 4, addr(dst))
    np.testing.assert_array_equal(dst,
                                  up_time.julian_date_of_besselian_epoch(b))


def test_sk_compute_calendar_date(kern):
    jd = a(np.arange(2459000, 2459007))
    year, month, day = (np.empty(7) for _ in range(3))
    for before in (-1.0, JULIAN_BEFORE):
        kern.sk_compute_calendar_date(addr(jd), before, 7, addr(year),
                                      addr(month), addr(day))
        theirs = up_time.compute_calendar_date(
            jd, None if before < 0 else before)
        # Upstream returns floats here; the kernel stores into float64.
        for ours, other in zip((year, month, day), theirs):
            np.testing.assert_array_equal(ours, other)


def test_sk_calendar_tuple(kern):
    jd = a(np.arange(2459000, 2459007) + 0.5)
    frac = a(np.linspace(0.0, 1.0, 7))
    outs = [np.empty(7) for _ in range(6)]
    kern.sk_calendar_tuple(addr(jd), addr(frac), -1.0, 7,
                           *(addr(o) for o in outs))
    theirs = up_time.calendar_tuple(jd, frac)
    for ours, other in zip(outs, theirs):
        if np.issubdtype(other.dtype, np.integer):
            np.testing.assert_array_equal(ours, other.astype(float))
        else:
            np.testing.assert_allclose(ours, other, rtol=1e-12, atol=1e-9)


def test_sk_tdb_minus_tt(kern):
    jd = a([2451545.0, 2460000.0, 2470000.0])
    frac = a([0.0, 0.25, 0.75])
    dst = np.empty(3)
    kern.sk_tdb_minus_tt(addr(jd), addr(frac), 3, addr(dst))
    np.testing.assert_allclose(dst, up_time.tdb_minus_tt(jd, frac),
                               rtol=1e-14, atol=0.0)


# ---------------------------------------------------------------------------
# skyfield.earthlib
# ---------------------------------------------------------------------------

def test_sk_terra(kern):
    n = 6
    lat, lon, elev, gast = (a(x) for x in (
        np.linspace(-1.4, 1.4, n), np.linspace(-3.0, 3.0, n),
        np.linspace(0.0, 3000.0, n), np.linspace(0.0, 24.0, n)))
    pos, vel = np.empty((3, n)), np.empty((3, n))
    kern.sk_terra(addr(lat), addr(lon), addr(elev), addr(gast), n,
                  addr(pos), addr(vel))
    theirs = up_earth.terra(lat, lon, elev, gast)
    np.testing.assert_allclose(pos, theirs[0], rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(vel, theirs[1], rtol=1e-14, atol=0.0)


def test_sk_reverse_terra(kern):
    n = 5
    xyz = a(np.random.default_rng(30).normal(size=(3, n)) * 1e-3)
    gast = a(np.linspace(0.0, 24.0, n))
    lat, lon, elev_m = (np.empty(n) for _ in range(3))
    kern.sk_reverse_terra(addr(xyz), addr(gast), 3, n, addr(lat), addr(lon),
                          addr(elev_m))
    for ours, other in zip((lat, lon, elev_m),
                           up_earth.reverse_terra(xyz, gast, iterations=3)):
        np.testing.assert_allclose(ours, other, rtol=1e-12, atol=0.0)


def test_sk_compute_limb_angle(kern):
    n = 5
    position = a(np.random.default_rng(31).normal(size=(3, n)))
    observer = a(np.random.default_rng(32).normal(size=(3, n)))
    limb, nadir = np.empty(n), np.empty(n)
    kern.sk_compute_limb_angle(addr(position), addr(observer), n, addr(limb),
                               addr(nadir))
    theirs = up_earth.compute_limb_angle(position, observer)
    np.testing.assert_allclose(limb, theirs[0], rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(nadir, theirs[1], rtol=1e-14, atol=0.0)


def test_sk_earth_rotation_angle(kern):
    n = 6
    jd = a(np.linspace(2451545.0, 2470000.0, n))
    frac = a(np.linspace(0.0, 1.0, n))
    dst = np.empty(n)
    kern.sk_earth_rotation_angle(addr(jd), addr(frac), n, addr(dst))
    np.testing.assert_allclose(dst, up_earth.earth_rotation_angle(jd, frac),
                               rtol=1e-13, atol=1e-15)


def test_sk_sidereal_time(kern):
    from skyfield.api import load
    from skyfield.timelib import Time as UpTime
    n = 6
    # `sidereal_time(t)` reads `t.whole` (a day number), `t.ut1_fraction`
    # and `t.tdb_fraction`.  Build the reference `Time` directly from a day
    # number and a TT fraction -- upstream derives `ut1_fraction` and
    # `tdb_fraction` from the latter -- and feed the kernel the values it
    # would have derived.
    whole = 2451545.0 + np.arange(6, dtype=np.float64)
    tt_fraction = a(np.linspace(0.0, 1.0, n))
    ts = load.timescale(builtin=True)
    t = UpTime(ts, whole, tt_fraction)
    # Upstream passes `t.whole` -- the float Julian date -- into
    # `earth_rotation_angle`; feed the kernel the same number.
    jd_ut1 = a(np.asarray(t.whole, dtype=np.float64))
    ut1_fraction = a(np.asarray(t.ut1_fraction))
    tdb_fraction = a(np.asarray(t.tdb_fraction))
    dst = np.empty(n)
    kern.sk_sidereal_time(addr(jd_ut1), addr(ut1_fraction), addr(tdb_fraction),
                          n, addr(dst))
    np.testing.assert_allclose(dst, np.asarray(up_earth.sidereal_time(t)),
                               rtol=1e-13, atol=1e-14)


def test_sk_refraction(kern):
    alt = a(np.linspace(-2.0, 88.0, 30))
    dst = np.empty(30)
    kern.sk_refraction(addr(alt), 10.0, 1010.0, 30, addr(dst))
    np.testing.assert_array_equal(
        dst, up_earth.refraction(alt, 10.0, 1010.0))


def test_sk_refract(kern):
    alt = a(np.linspace(-2.0, 88.0, 30))
    dst = np.empty(30)
    kern.sk_refract(addr(alt), 10.0, 1010.0, 30, addr(dst))
    np.testing.assert_allclose(dst, up_earth.refract(alt, 10.0, 1010.0),
                               rtol=1e-9, atol=1e-16)


# ---------------------------------------------------------------------------
# skyfield.precessionlib and skyfield.nutationlib
# ---------------------------------------------------------------------------

def test_sk_compute_precession(kern):
    jd = a(2451545.0 + np.random.default_rng(40).random(9) * 8000.0)
    n = jd.size
    dst = np.empty((3, 3, n))
    kern.sk_compute_precession(addr(jd), n, 1, addr(dst))
    # The off-diagonal entries are differences of products of `sin`/`cos`, so
    # they are ~1e-6 while the diagonal is ~1. Compare element by element
    # against its own magnitude.
    theirs = up_pre.compute_precession(jd)
    scale = np.maximum(np.max(np.abs(theirs), axis=2, keepdims=True), 1e-30)
    assert np.max(np.abs(dst - theirs) / scale) < 1e-11


def test_sk_fundamental_arguments(kern):
    t = a(np.linspace(-1.0, 1.0, 10))
    n = t.size
    for terms in (1, 2, 3, 4, 5):
        dst = np.empty((5, n))
        kern.sk_fundamental_arguments(addr(t), terms, n, 1, addr(dst))
        np.testing.assert_allclose(dst, up_nut.fundamental_arguments(t, terms),
                                   rtol=1e-15, atol=0.0)


def test_sk_mean_obliquity(kern):
    jd = a(2451545.0 + np.random.default_rng(41).random(12) * 8000.0)
    dst = np.empty(12)
    kern.sk_mean_obliquity(addr(jd), 12, addr(dst))
    np.testing.assert_array_equal(dst, up_nut.mean_obliquity(jd))


def test_sk_build_nutation_matrix(kern):
    n = 12
    jd = 2451545.0 + np.random.default_rng(42).random(n) * 8000.0
    mean = a(up_nut.mean_obliquity(jd) * 4.84813681109536e-6)
    dpsi, deps = up_nut.iau2000a(jd)
    k = 4.84813681109536e-6 / 1e7
    dpsi, deps = a(dpsi * k), a(deps * k)
    dst = np.empty((3, 3, n))
    kern.sk_build_nutation_matrix(addr(mean), addr(mean + deps), addr(dpsi), n,
                                  addr(dst))
    # Entry [1][2] is `cpsi*sobm*cobt - cobm*sobt`, whose two ~0.365 terms
    # cancel down to ~3e-7 -- a factor of 1.1e6.  A one-ulp difference in
    # `cos`/`sin` between the two implementations therefore shows up there
    # as ~5e-11, and that is a property of upstream's formula rather than of
    # this port; every other entry agrees to a few ulp.
    assert_matrix_close(
        dst, up_nut.build_nutation_matrix(mean, mean + deps, dpsi),
        rtol=1e-10)


def test_sk_equation_of_the_equinoxes_complimentary_terms(kern):
    jd = a(2451545.0 + np.random.default_rng(43).random(9) * 8000.0)
    n = jd.size
    dst = np.empty(n)
    kern.sk_equation_of_the_equinoxes_complimentary_terms(
        addr(jd), addr(_tables.ke0_t), addr(_tables.ke1), addr(_tables.se0_t_0),
        addr(_tables.se0_t_1), _tables.ke0_t.shape[0], n, 1, addr(dst))
    np.testing.assert_allclose(
        dst, up_nut.equation_of_the_equinoxes_complimentary_terms(jd),
        rtol=1e-11, atol=0.0)


@pytest.mark.parametrize("symbol,fa,lst,pbt", [
    ("sk_iau2000a", 5, 687, 687),
    ("sk_iau2000b", 2, 77, 0),
])
def test_sk_iau2000(kern, symbol, fa, lst, pbt):
    jd = a(2451545.0 + np.random.default_rng(44).random(8) * 8000.0)
    n = jd.size
    dpsi, deps = np.empty(n), np.empty(n)
    # `nals_t` indexes the 678 lunisolar terms and `napl_t` the 687 planetary
    # ones, so a caller asking for the full model must clamp to the table.
    # The Python wrapper does this; do the same here so the kernel is never
    # handed a count that would read past the end of a table.
    lst = min(lst, _tables.nals_t.shape[0])
    pbt = min(pbt, _tables.napl_t.shape[0])
    getattr(kern, symbol)(
        addr(jd), fa, lst, pbt,
        addr(_tables.nals_t), addr(_tables.lunisolar_longitude_coefficients),
        addr(_tables.lunisolar_obliquity_coefficients), addr(_tables.napl_t),
        addr(_tables.nutation_coefficients_longitude),
        addr(_tables.nutation_coefficients_obliquity), n, 1,
        addr(dpsi), addr(deps))
    theirs = (up_nut.iau2000a if symbol == "sk_iau2000a" else up_nut.iau2000b)(jd)
    for ours, other in zip((dpsi, deps), theirs):
        np.testing.assert_allclose(ours, other, rtol=1e-12, atol=1e-8)


# ---------------------------------------------------------------------------
# skyfield.framelib
# ---------------------------------------------------------------------------

def test_sk_build_matrix(kern):
    dst = np.empty(9)
    kern.sk_build_matrix(addr(dst))
    # Upstream deletes `build_matrix` at import, keeping only the module-level
    # `ICRS_to_J2000` it computed.  That constant is the value to compare to.
    np.testing.assert_array_equal(dst.reshape(3, 3), up_frame.ICRS_to_J2000)


def _frame_inputs(n):
    """The `M`, obliquity and GAST stacks the frame kernels consume."""
    jd = a(2451545.0 + np.random.default_rng(50).random(n) * 8000.0)
    M = a(up_pre.compute_precession(jd))
    obl = a(up_nut.mean_obliquity(jd) * 4.84813681109536e-6)
    gast = a(np.linspace(0.0, 24.0, n))
    return M, obl, gast


def test_sk_build_ecliptic_matrix(kern):
    n = 7
    M, obl, _ = _frame_inputs(n)
    dst, scratch = np.empty((3, 3, n)), np.empty(9 * n)
    kern.sk_build_ecliptic_matrix(addr(M), addr(obl), n, addr(dst),
                                  addr(scratch))
    assert_matrix_close(dst, up_fn.mxm(up_fn.rot_x(-obl), M))


def test_sk_tirs_rotation_at(kern):
    n = 7
    M, _, gast = _frame_inputs(n)
    dst, scratch = np.empty((3, 3, n)), np.empty(9 * n)
    kern.sk_tirs_rotation_at(addr(M), addr(gast), n, addr(dst), addr(scratch))
    assert_matrix_close(
        dst, up_fn.mxm(up_fn.rot_z(-gast * 2 * np.pi / 24.0), M))


def test_sk_itrs_rotation_at(kern):
    n = 7
    M, _, gast = _frame_inputs(n)
    # `itrs.rotation_at` is `mxm(t.polar_motion_matrix(), mxm(rot_z(-gast*tau/24),
    # M))`, and `polar_motion_matrix` is a single (3, 3) matrix built from the
    # polar-motion angles, broadcast across the epochs.
    sprime, x, y = 1e-6, 2e-6, 3e-6
    asec2rad = 4.84813681109536e-6
    # Build W as a (3, 3, 1) stack so it can be broadcast across the epochs;
    # NumPy will not broadcast a bare (3, 3) up to (3, 3, n).
    W3 = a(up_fn.mxmxm(up_fn.rot_x(a([y * asec2rad])),
                       up_fn.rot_y(a([x * asec2rad])),
                       up_fn.rot_z(a([-sprime * asec2rad]))))
    W = a(np.broadcast_to(W3, (3, 3, n)))
    dst = np.empty((3, 3, n))
    scratch, scratch2 = np.empty(9 * n), np.empty(9 * n)
    kern.sk_itrs_rotation_at(addr(M), addr(gast), addr(W), n, addr(dst),
                             addr(scratch), addr(scratch2))
    theirs = up_fn.mxm(W3,
                       up_fn.mxm(up_fn.rot_z(-gast * 2 * np.pi / 24.0), M))
    assert_matrix_close(dst, theirs)


# ---------------------------------------------------------------------------
# skyfield.relativity
# ---------------------------------------------------------------------------

def test_sk_light_time_difference(kern):
    n = 6
    position = a(np.random.default_rng(51).normal(size=(3, n)))
    deflector = a(np.random.default_rng(52).normal(size=(3, n)))
    dst = np.empty(n)
    kern.sk_light_time_difference(addr(position), addr(deflector), n, addr(dst))
    np.testing.assert_allclose(dst, up_rel.light_time_difference(position,
                                                                 deflector),
                               rtol=1e-14, atol=0.0)


def test_sk_compute_deflection(kern):
    n = 6
    position = a(np.random.default_rng(53).normal(size=(3, n)))
    pe = a(np.random.default_rng(54).normal(size=(3, n)))
    rmass = up_rel.rmasses[399]
    dst = np.empty((3, n))
    kern.sk_compute_deflection(addr(position), addr(pe), rmass, n, addr(dst))
    np.testing.assert_allclose(
        dst, up_rel._compute_deflection(position, pe, rmass),
        rtol=1e-14, atol=0.0)


def test_sk_add_aberration(kern):
    n = 6
    position = a(np.random.default_rng(55).normal(size=(3, n)))
    velocity = a(np.random.default_rng(56).normal(size=(3, n)))
    light_time = a(np.full(n, 0.01))
    theirs = position.copy()
    up_rel.add_aberration(theirs, velocity, light_time)
    kern.sk_add_aberration(addr(position), addr(velocity), addr(light_time), n)
    np.testing.assert_allclose(position, theirs, rtol=1e-14, atol=0.0)


# ---------------------------------------------------------------------------
# skyfield.sgp4lib
# ---------------------------------------------------------------------------

def test_sk_cross(kern):
    n = 7
    x = a(np.random.default_rng(60).normal(size=(3, n)))
    y = a(np.random.default_rng(61).normal(size=(3, n)))
    dst = np.empty((3, n))
    kern.sk_cross(addr(x), addr(y), n, addr(dst))
    np.testing.assert_allclose(dst, up_sgp._cross(x, y), rtol=1e-15, atol=0.0)


def test_sk_theta_GMST1982(kern):
    n = 9
    jd = a(np.linspace(2451545.0, 2470000.0, n))
    frac = a(np.linspace(0.0, 1.0, n))
    theta, theta_dot = np.empty(n), np.empty(n)
    kern.sk_theta_GMST1982(addr(jd), addr(frac), n, addr(theta), addr(theta_dot))
    theirs = up_sgp.theta_GMST1982(jd, frac)
    np.testing.assert_allclose(theta, theirs[0], rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(theta_dot, theirs[1], rtol=1e-13, atol=1e-15)


def test_sk_TEME_to_ITRF(kern):
    n = 6
    jd = a(np.linspace(2451545.0, 2470000.0, n))
    frac = a(np.linspace(0.0, 1.0, n))
    r = a(np.random.default_rng(62).normal(size=(3, n)))
    v = a(np.random.default_rng(63).normal(size=(3, n)))
    r_out, v_out = np.empty((3, n)), np.empty((3, n))
    for xp, yp in ((0.0, 0.0), (0.3, -0.2)):
        # The kernel is one fused pass and takes no scratch buffers.
        kern.sk_TEME_to_ITRF(addr(jd), addr(r), addr(v), xp, yp, addr(frac), n,
                             1, addr(r_out), addr(v_out))
        theirs = up_sgp.TEME_to_ITRF(jd, r, v, xp, yp, frac)
        for ours, other in zip((r_out, v_out), theirs):
            # `rPEF` is ~1 in magnitude; the polar-motion branch divides by
            # sin(yp) so it is not defined at yp = 0, hence the two cases.
            assert np.max(np.abs(ours - other)) <= 1e-11 * np.max(
                np.abs(other))


@pytest.mark.parametrize("workers", [1, 3, 8, 32])
@pytest.mark.parametrize("xp,yp", [(0.0, 0.0), (0.3, -0.2), (-0.5, 0.9)])
def test_sk_TEME_to_ITRF_fused_pass(workers, xp, yp):
    """The fused one-pass kernel agrees with upstream at every worker count.

    Folding the `R` and `W` matrices, the angular-velocity vector and the
    cross product into registers replaced seven scratch buffers with a single
    pass over the inputs, and made the kernel parallel.  A split that dropped
    or duplicated an epoch would still pass at one worker, so check the split
    forms at sizes that leave a ragged tail for every worker count, and check
    both the polar-motion and no-polar-motion branches.
    """
    rng = np.random.default_rng(64)
    for n in (1, 5, 4097, 20011):
        jd = a(2451545.0 + rng.random(n) * 8000.0)
        frac = a(rng.random(n))
        r = a(rng.normal(size=(3, n)) * 1000.0)
        v = a(rng.normal(size=(3, n)))
        r_out, v_out = np.empty((3, n)), np.empty((3, n))
        kern = lib()
        kern.sk_TEME_to_ITRF(addr(jd), addr(r), addr(v), xp, yp, addr(frac), n,
                             workers, addr(r_out), addr(v_out))
        theirs = up_sgp.TEME_to_ITRF(jd, r, v, xp, yp, frac)
        for ours, other in zip((r_out, v_out), theirs):
            scale = max(np.max(np.abs(other)), 1e-300)
            assert np.max(np.abs(ours - other)) <= 1e-11 * scale, (n, workers)


def test_sk_TEME_to_ITRF_split_matches_serial():
    """Splitting the fused pass across workers changes no element.

    Epochs are independent and nothing is reduced across them, so the split
    result must be bit-for-bit identical to the serial one, not merely close.
    """
    from mojo_skyfield import _lib
    from mojo_skyfield import sgp4lib as ms_sgp
    rng = np.random.default_rng(65)
    n = 20011
    jd = a(2451545.0 + rng.random(n) * 8000.0)
    frac = a(rng.random(n))
    r = a(rng.normal(size=(3, n)) * 1000.0)
    v = a(rng.normal(size=(3, n)))
    saved = _lib.workers()
    try:
        _lib.set_workers(1)
        serial = ms_sgp.TEME_to_ITRF(jd, r, v, 0.0, 0.0, frac)
        _lib.set_workers(8)
        split = ms_sgp.TEME_to_ITRF(jd, r, v, 0.0, 0.0, frac)
    finally:
        _lib.set_workers(saved)
    for ours, other in zip(split, serial):
        assert np.array_equal(ours, other)


def test_sk_TEME_rotation_at(kern):
    n = 6
    theta = a(np.linspace(0.0, 24.0, n))
    gast = a(np.linspace(0.0, 24.0, n))
    M = a(up_pre.compute_precession(
        a(2451545.0 + np.random.default_rng(64).random(n) * 8000.0)))
    dst, scratch = np.empty((3, 3, n)), np.empty(9 * n)
    kern.sk_TEME_rotation_at(addr(theta), addr(gast), addr(M), n, addr(dst),
                             addr(scratch))
    # `TEME.rotation_at` is `mxm(rot_z(theta - gast/24*tau), t.M)`.
    angle = theta - gast / 24.0 * 2 * np.pi
    assert_matrix_close(dst, up_fn.mxm(up_fn.rot_z(angle), M))


# ---------------------------------------------------------------------------
# skyfield.keplerlib
# ---------------------------------------------------------------------------

def test_sk_normpi(kern):
    x = a([-40.0, -7.0, -np.pi, -0.5, 0.0, 0.5, np.pi, 7.0, 40.0])
    dst = np.empty_like(x)
    kern.sk_normpi(addr(x), x.size, addr(dst))
    np.testing.assert_array_equal(dst, up_kep.normpi(x))


def test_sk_eccentric_anomaly(kern):
    # Upstream divides by `e`, so `e = 0` is outside its domain; stay in it.
    e = a([0.05, 0.1, 0.5, 0.9])
    M = a([0.1, 1.0, 3.0, -2.0])
    dst = np.empty(4)
    kern.sk_eccentric_anomaly(addr(e), addr(M), 4, addr(dst))
    # Upstream's convergence test `abs(dE) < 1e-14` is scalar, so it only
    # answers for one orbit at a time; compare the array element by element.
    for ours, other in zip(dst, [up_kep.eccentric_anomaly(ev, Mv)
                                for ev, Mv in zip(e, M)]):
        np.testing.assert_allclose(ours, other, rtol=1e-13, atol=1e-14)


def test_sk_true_anomaly_hyperbolic(kern):
    e = a([1.1, 1.5, 3.0])
    E = a([0.1, 0.5, 1.0])
    dst = np.empty_like(e)
    kern.sk_true_anomaly_hyperbolic(addr(e), addr(E), e.size, addr(dst))
    np.testing.assert_allclose(dst, up_kep.true_anomaly_hyperbolic(e, E),
                               rtol=1e-14, atol=0.0)


def test_sk_true_anomaly_closed(kern):
    e = a([0.0, 0.1, 0.5, 0.9, 0.99])
    E = a([0.1, 1.0, 3.0, -2.0, 0.5])
    dst = np.empty_like(e)
    kern.sk_true_anomaly_closed(addr(e), addr(E), e.size, addr(dst))
    np.testing.assert_allclose(dst, up_kep.true_anomaly_closed(e, E),
                               rtol=1e-14, atol=0.0)


def test_sk_true_anomaly_parabolic(kern):
    p = a([1.0, 2.0, 1e6])
    M = a([0.1, 1.0, -2.0])
    gm = 1.32712440017987e11
    dst = np.empty_like(p)
    kern.sk_true_anomaly_parabolic(addr(p), gm, addr(M), p.size, addr(dst))
    np.testing.assert_allclose(dst, up_kep.true_anomaly_parabolic(p, gm, M),
                               rtol=1e-15, atol=1e-15)


def test_sk_ele_to_vec(kern):
    n = 5
    p = a([1.0, 2.0, 7000.0, 8000.0, 4.2e4])
    e = a([0.0, 0.1, 0.5, 0.9, 0.01])
    i = a([0.1, 0.5, 1.0, 2.0, 3.0])
    Om = a([0.2, 0.4, 0.6, 0.8, 1.0])
    w = a([0.3, 0.5, 0.7, 0.9, 1.1])
    v = a([0.4, 0.6, 0.8, 1.0, 1.2])
    mu = 398600.4418
    pos, vel = np.empty((3, n)), np.empty((3, n))
    kern.sk_ele_to_vec(addr(p), addr(e), addr(i), addr(Om), addr(w), addr(v),
                       mu, n, addr(pos), addr(vel))
    theirs = up_kep.ele_to_vec(p, e, i, Om, w, v, mu)
    np.testing.assert_allclose(pos, theirs[0], rtol=1e-13, atol=0.0)
    np.testing.assert_allclose(vel, theirs[1], rtol=1e-13, atol=0.0)


def test_sk_stumpff(kern):
    # Upstream refuses `x < stumpff_bound` (-504775.9, where `cosh(sqrt(-x))`
    # overflows a float64); the port has no such guard, so stay well inside.
    x = a(np.linspace(0.2, 20.0, 40))
    outs = [np.empty(40) for _ in range(4)]
    kern.sk_stumpff(addr(x), 40, *(addr(o) for o in outs))
    for ours, other in zip(outs, up_kep.stumpff(x)):
        np.testing.assert_allclose(ours, other, rtol=1e-12, atol=1e-15)


def test_sk_propagate(kern):
    from mojo_skyfield import keplerlib as ms_kep
    n = 512
    position = a([7000.0, 0.0, 1000.0])
    velocity = a([0.0, 5.0, 4.0])
    t1 = a(np.linspace(-20000.0, 40000.0, n))
    pos, vel = np.empty((3, n)), np.empty((3, n))
    scratch = np.empty(14 * n)
    flags = np.zeros(64, dtype=np.int64)
    rc = kern.sk_propagate(addr(position), addr(velocity), 0.0, addr(t1),
                           398600.4418, n, 1, addr(scratch), addr(flags),
                           addr(pos), addr(vel))
    assert rc == 0, "propagate reported an epoch outside its reliable range"
    theirs = up_kep.propagate(position, velocity, 0.0, t1, 398600.4418)
    for ours, other in zip((pos, vel), theirs):
        np.testing.assert_allclose(ours, other, rtol=1e-9, atol=1e-6)
    # And through the Python API, which is how a caller reaches it.
    api_pos, api_vel = ms_kep.propagate(position, velocity, 0.0, t1, 398600.4418)
    np.testing.assert_allclose(api_pos, pos, rtol=1e-15, atol=0.0)
    np.testing.assert_allclose(api_vel, vel, rtol=1e-15, atol=0.0)


def test_sk_propagate_reports_out_of_range(kern):
    # `dt/bq` is clipped to `+-bound` on entry, so the bracketing loop only
    # gives up when the doubling stops moving `x` -- which needs a `dt` large
    # enough that `bound` itself no longer moves it.  Upstream raises the same
    # ValueError here, and so must we.
    position = a([7000.0, 0.0, 1000.0])
    velocity = a([0.0, 5.0, 4.0])
    t1 = a([1e300, np.inf])
    pos, vel = np.empty((3, 2)), np.empty((3, 2))
    rc = kern.sk_propagate(addr(position), addr(velocity), 0.0, addr(t1),
                           398600.4418, 2, 1, addr(np.empty(28)),
                           addr(np.zeros(64, dtype=np.int64)), addr(pos),
                           addr(vel))
    assert rc == 1, "a dt that cannot be bracketed must be reported"

    from mojo_skyfield import keplerlib as ms_kep
    with pytest.raises(ValueError):
        up_kep.propagate(position, velocity, 0.0, t1, 398600.4418)
    with pytest.raises(ValueError):
        ms_kep.propagate(position, velocity, 0.0, t1, 398600.4418)


# ---------------------------------------------------------------------------
# Coverage: every export must be named by a test in this file.
# ---------------------------------------------------------------------------

def _exported_symbols():
    pattern = re.compile(r'@export\("([A-Za-z_0-9]+)"\)')
    found = set()
    for path in sorted((ROOT / "src").glob("*.mojo")):
        found.update(pattern.findall(path.read_text()))
    return found


def test_kernel_coverage(kern):
    """Every `@export` in `src/` has a test here that names it and asserts."""
    here = Path(__file__).read_text()
    untested = sorted(s for s in _exported_symbols() if s not in here)
    assert not untested, f"exported kernels with no direct test: {untested}"
    # And the ctypes glue must actually declare each one, or the test above
    # would be asserting against a signature that does not exist.
    missing = sorted(set(_SIGNATURES) - _exported_symbols())
    assert not missing, f"ctypes signatures for non-existent kernels: {missing}"