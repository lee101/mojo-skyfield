"""Mojo port of Skyfield's ephemeris and satellite-propagation kernels.

Read this file side by side with the upstream Python it came from.  The
port is taken from, and within each section the functions keep upstream's
names, argument order, branch structure and arithmetic.  The arithmetic is
upstream's; what is not upstream's is the loop around it.  A per-epoch loop
is split across cores with `max.algorithm.parallelize`, and a few bodies
were reshaped so that a worker can run one without shared state:
`_iau2000a_one`, `_precession_at`, `_propagate_slice` and `_stumpff_one`
exist for that reason.  `_stumpff_one` also stops multiplying each factorial
denominator out from scratch, and the Stumpff series' signs live in its
coefficients so the sum is unchanged.

Passes that upstream walks the array twice to do are fused into one here,
which is why a few of the comment blocks below describe the fusion rather
than restating upstream line for line.  ``TEME_to_ITRF`` is the extreme case:
every one of its intermediates is only three numbers long, so they are held
in registers and the kernel takes no ``scratch`` buffers at all.

Array layout
------------
Upstream is NumPy.  Every |xyz| vector has shape (3, n) and every 3x3
rotation matrix has shape (3, 3, n), with the leading axis contiguous, so
element (i, j) lives at ``i * n + j`` and matrix element (i, k, j) at
``(i * 3 + k) * n + j``.  The same addresses arrive from the ctypes glue,
so a caller can hand upstream-shaped arrays straight through.

Where upstream composes array expressions (``mxm(rot_z(a), t.M)``) the
port keeps the composition and takes the intermediate matrix as a
caller-supplied ``scratch`` buffer: Mojo cannot allocate inside an
``abi("C")`` function it must not leak from, and it will not convert a
``List`` to a ``Pointer``.  ``propagate`` takes its fourteen work arrays the
same way, because a parallel worker has to write through a pointer it was
given.  Every other temporary is a scalar local, and a scalar local does not
need a buffer: that is what ``TEME_to_ITRF`` and ``_teme_to_itrf_one`` rely
on.
"""

from max.algorithm import parallelize
from std.math import (
    abs, acos, asin, atan, atan2, cbrt, cos, cosh, exp, floor, hypot, isfinite,
    log, max, min, nan, sin, sinh, sqrt, tan, tanh, trunc,
)

# ----------------------------------------------------------------------------
# skyfield.constants
# ----------------------------------------------------------------------------

comptime AU_M: Float64 = 149597870700.0
comptime AU_KM: Float64 = 149597870.700
comptime ASEC360: Float64 = 1296000.0
comptime DAY_S: Float64 = 86400.0
comptime ASEC2RAD: Float64 = 4.848136811095359935899141e-6
comptime DEG2RAD: Float64 = 0.017453292519943296
comptime RAD2DEG: Float64 = 57.295779513082321
comptime pi: Float64 = 3.141592653589793
comptime tau: Float64 = 6.283185307179586476925287
comptime C: Float64 = 299792458.0
comptime ANGVEL: Float64 = 7.2921150e-5
comptime ERAD: Float64 = 6378136.6
comptime IERS_2010_INVERSE_EARTH_FLATTENING: Float64 = 298.25642
comptime GS: Float64 = 1.32712440017987e+20
comptime T0: Float64 = 2451545.0
comptime C_AUDAY: Float64 = 173.14463267424034

comptime _AVOID_DIVIDE_BY_ZERO: Float64 = 2.2250738585072014e-308

comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]

# ----------------------------------------------------------------------------
# Parallel dispatch
# ----------------------------------------------------------------------------

# Elements per worker below which a launch costs more than the work saves.
comptime _PAR_MIN: Int = 2048


def _plan(n: Int, workers: Int) -> Tuple[Int, Int]:
    """Split a length-`n` loop into `(num_work_items, num_workers)` chunks.

    `workers` is the core count the caller passed down, usually
    `os.cpu_count()`.  Below `_PAR_MIN` elements per worker a launch costs
    more than the work saves, so the loop stays serial.
    """
    var w = workers
    if w < 2 or n < _PAR_MIN * 2:
        return (1, 1)
    var want = n // _PAR_MIN
    if want < w:
        w = want
    return (w, w)


def _clear_flags(flags: IPtr, workers: Int):
    var i = 0
    while i < workers:
        flags.unsafe_store(i, Int64(0))
        i += 1


def _any_flag(flags: IPtr, workers: Int) -> Bool:
    var i = 0
    while i < workers:
        if flags.unsafe_load(i) != 0:
            return True
        i += 1
    return False


# ----------------------------------------------------------------------------
# Python scalar semantics, needed because the ported code is array code
# ----------------------------------------------------------------------------

def clip(x: Float64, lo: Float64, hi: Float64) -> Float64:
    """`numpy.clip()` for a scalar."""
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def pymod(a: Float64, b: Float64) -> Float64:
    """Python's `%` on floats: the result carries the sign of `b`."""
    return a - floor(a / b) * b


def pyfmod(a: Float64, b: Float64) -> Float64:
    """C's `fmod`, which is what `numpy.fmod` gives: the sign of `a`."""
    return a - trunc(a / b) * b


def floordiv(a: Float64, b: Float64) -> Float64:
    """Python's `//` on integers, carried in Float64."""
    return floor(a / b)


# ----------------------------------------------------------------------------
# skyfield.functions
# ----------------------------------------------------------------------------

def sqrt_nan(x: Ptr, size: Int, dst: Ptr):
    """Return the square root of ``n``, or ``nan`` if ``n < 0``."""
    var i = 0
    while i < size:
        var v = x.unsafe_load(i)
        dst.unsafe_store(i, sqrt(abs(v)) if v >= 0.0 else nan[DType.float64]())
        i += 1


def dots(v: Ptr, u: Ptr, n: Int, dst: Ptr):
    """Given one or more vectors in `v` and `u`, return their dot products."""
    var j = 0
    while j < n:
        var s = v.unsafe_load(0 * n + j) * u.unsafe_load(0 * n + j)
        s += v.unsafe_load(1 * n + j) * u.unsafe_load(1 * n + j)
        s += v.unsafe_load(2 * n + j) * u.unsafe_load(2 * n + j)
        dst.unsafe_store(j, s)
        j += 1


def T(M: Ptr, n: Int, dst: Ptr):
    """Swap the first two dimensions of an array."""
    var j = 0
    while j < n:
        var i = 0
        while i < 3:
            var k = 0
            while k < 3:
                dst.unsafe_store((i * 3 + k) * n + j,
                                 M.unsafe_load((k * 3 + i) * n + j))
                k += 1
            i += 1
        j += 1


def mxv(M: Ptr, v: Ptr, n: Int, dst: Ptr):
    """Matrix times vector: multiply an NxN matrix by a vector."""
    var j = 0
    while j < n:
        var i = 0
        while i < 3:
            var s = M.unsafe_load((i * 3 + 0) * n + j) * v.unsafe_load(0 * n + j)
            s += M.unsafe_load((i * 3 + 1) * n + j) * v.unsafe_load(1 * n + j)
            s += M.unsafe_load((i * 3 + 2) * n + j) * v.unsafe_load(2 * n + j)
            dst.unsafe_store(i * n + j, s)
            i += 1
        j += 1


def mxm(M1: Ptr, M2: Ptr, n: Int, dst: Ptr):
    """Matrix times matrix: multiply two NxN matrices."""
    var j = 0
    while j < n:
        var i = 0
        while i < 3:
            var k = 0
            while k < 3:
                var s = M1.unsafe_load((i * 3 + 0) * n + j) * M2.unsafe_load((0 * 3 + k) * n + j)
                s += M1.unsafe_load((i * 3 + 1) * n + j) * M2.unsafe_load((1 * 3 + k) * n + j)
                s += M1.unsafe_load((i * 3 + 2) * n + j) * M2.unsafe_load((2 * 3 + k) * n + j)
                dst.unsafe_store((i * 3 + k) * n + j, s)
                k += 1
            i += 1
        j += 1


def mxmxm(M1: Ptr, M2: Ptr, M3: Ptr, n: Int, dst: Ptr):
    """Matrix times matrix times matrix: multiply 3 NxN matrices together."""
    var j = 0
    while j < n:
        var i = 0
        while i < 3:
            var l = 0
            while l < 3:
                var s = 0.0
                var k = 0
                while k < 3:
                    var m = 0
                    while m < 3:
                        s += (M1.unsafe_load((i * 3 + k) * n + j)
                              * M2.unsafe_load((k * 3 + m) * n + j)
                              * M3.unsafe_load((m * 3 + l) * n + j))
                        m += 1
                    k += 1
                dst.unsafe_store((i * 3 + l) * n + j, s)
                l += 1
            i += 1
        j += 1


def length_of(xyz: Ptr, n: Int, dst: Ptr):
    """Given a 3-element array |xyz|, return its length."""
    var j = 0
    while j < n:
        var x = xyz.unsafe_load(0 * n + j)
        var y = xyz.unsafe_load(1 * n + j)
        var z = xyz.unsafe_load(2 * n + j)
        dst.unsafe_store(j, sqrt(x * x + y * y + z * z))
        j += 1


def length_of_one(xyz: Ptr, n: Int, j: Int) -> Float64:
    """`length_of()` for a single vector, so callers need no scratch."""
    var x = xyz.unsafe_load(0 * n + j)
    var y = xyz.unsafe_load(1 * n + j)
    var z = xyz.unsafe_load(2 * n + j)
    return sqrt(x * x + y * y + z * z)


def angle_between(u: Ptr, v: Ptr, n: Int, dst: Ptr):
    """Given two vectors `v` and `u`, return the radian angle separating them.

    This formula is from Section 12 of:
    https://people.eecs.berkeley.edu/~wkahan/Mindless.pdf
    """
    var j = 0
    while j < n:
        var lu = length_of_one(u, n, j)
        var lv = length_of_one(v, n, j)
        var diff = 0.0
        var total = 0.0
        var i = 0
        while i < 3:
            var a = u.unsafe_load(i * n + j) * lv
            var b = v.unsafe_load(i * n + j) * lu
            diff += (a - b) * (a - b)
            total += (a + b) * (a + b)
            i += 1
        dst.unsafe_store(j, 2.0 * atan2(sqrt(diff), sqrt(total)))
        j += 1


def to_spherical(xyz: Ptr, n: Int, d_r: Ptr, d_theta: Ptr, d_phi: Ptr):
    """Convert |xyz| to spherical coordinates (r,theta,phi)."""
    length_of(xyz, n, d_r)
    var j = 0
    while j < n:
        var x = xyz.unsafe_load(0 * n + j)
        var y = xyz.unsafe_load(1 * n + j)
        var z = xyz.unsafe_load(2 * n + j)
        d_theta.unsafe_store(j, atan2(z, hypot(x, y)))
        d_phi.unsafe_store(j, pymod(atan2(y, x), tau))
        j += 1


def from_spherical(r: Ptr, theta: Ptr, phi: Ptr, n: Int, dst: Ptr):
    """Convert (r,theta,phi) to Cartesian coordinates |xyz|."""
    var j = 0
    while j < n:
        var rxy = r.unsafe_load(j) * cos(theta.unsafe_load(j))
        dst.unsafe_store(0 * n + j, rxy * cos(phi.unsafe_load(j)))
        dst.unsafe_store(1 * n + j, rxy * sin(phi.unsafe_load(j)))
        dst.unsafe_store(2 * n + j, r.unsafe_load(j) * sin(theta.unsafe_load(j)))
        j += 1


def _rot_x_at(dst: Ptr, n: Int, j: Int, th: Float64):
    var c = cos(th)
    var s = sin(th)
    var zero = th * 0.0
    var one = zero + 1.0
    dst.unsafe_store((0 * 3 + 0) * n + j, one)
    dst.unsafe_store((0 * 3 + 1) * n + j, zero)
    dst.unsafe_store((0 * 3 + 2) * n + j, zero)
    dst.unsafe_store((1 * 3 + 0) * n + j, zero)
    dst.unsafe_store((1 * 3 + 1) * n + j, c)
    dst.unsafe_store((1 * 3 + 2) * n + j, -s)
    dst.unsafe_store((2 * 3 + 0) * n + j, zero)
    dst.unsafe_store((2 * 3 + 1) * n + j, s)
    dst.unsafe_store((2 * 3 + 2) * n + j, c)


def _rot_y_at(dst: Ptr, n: Int, j: Int, th: Float64):
    var c = cos(th)
    var s = sin(th)
    var zero = th * 0.0
    var one = zero + 1.0
    dst.unsafe_store((0 * 3 + 0) * n + j, c)
    dst.unsafe_store((0 * 3 + 1) * n + j, zero)
    dst.unsafe_store((0 * 3 + 2) * n + j, s)
    dst.unsafe_store((1 * 3 + 0) * n + j, zero)
    dst.unsafe_store((1 * 3 + 1) * n + j, one)
    dst.unsafe_store((1 * 3 + 2) * n + j, zero)
    dst.unsafe_store((2 * 3 + 0) * n + j, -s)
    dst.unsafe_store((2 * 3 + 1) * n + j, zero)
    dst.unsafe_store((2 * 3 + 2) * n + j, c)


def _rot_z_at(dst: Ptr, n: Int, j: Int, th: Float64):
    var c = cos(th)
    var s = sin(th)
    var zero = th * 0.0
    var one = zero + 1.0
    dst.unsafe_store((0 * 3 + 0) * n + j, c)
    dst.unsafe_store((0 * 3 + 1) * n + j, -s)
    dst.unsafe_store((0 * 3 + 2) * n + j, zero)
    dst.unsafe_store((1 * 3 + 0) * n + j, s)
    dst.unsafe_store((1 * 3 + 1) * n + j, c)
    dst.unsafe_store((1 * 3 + 2) * n + j, zero)
    dst.unsafe_store((2 * 3 + 0) * n + j, zero)
    dst.unsafe_store((2 * 3 + 1) * n + j, zero)
    dst.unsafe_store((2 * 3 + 2) * n + j, one)


def rot_x(theta: Ptr, n: Int, dst: Ptr):
    var j = 0
    while j < n:
        _rot_x_at(dst, n, j, theta.unsafe_load(j))
        j += 1


def rot_y(theta: Ptr, n: Int, dst: Ptr):
    var j = 0
    while j < n:
        _rot_y_at(dst, n, j, theta.unsafe_load(j))
        j += 1


def rot_z(theta: Ptr, n: Int, dst: Ptr):
    var j = 0
    while j < n:
        _rot_z_at(dst, n, j, theta.unsafe_load(j))
        j += 1


def R1(theta: Ptr, n: Int, dst: Ptr):
    """Left-handed rotation about x, per the Explanatory Supplement."""
    var j = 0
    while j < n:
        _rot_x_at(dst, n, j, -theta.unsafe_load(j))
        j += 1


def R2(theta: Ptr, n: Int, dst: Ptr):
    """Left-handed rotation about y, per the Explanatory Supplement."""
    rot_y(theta, n, dst)


def R3(theta: Ptr, n: Int, dst: Ptr):
    """Left-handed rotation about z, per the Explanatory Supplement."""
    var j = 0
    while j < n:
        _rot_z_at(dst, n, j, -theta.unsafe_load(j))
        j += 1


def angular_velocity_matrix(angular_velocity_vector: Ptr, n: Int, dst: Ptr):
    var j = 0
    while j < n:
        var x = angular_velocity_vector.unsafe_load(0 * n + j)
        var y = angular_velocity_vector.unsafe_load(1 * n + j)
        var z = angular_velocity_vector.unsafe_load(2 * n + j)
        var zero = x * 0.0
        dst.unsafe_store((0 * 3 + 0) * n + j, zero)
        dst.unsafe_store((0 * 3 + 1) * n + j, -z)
        dst.unsafe_store((0 * 3 + 2) * n + j, y)
        dst.unsafe_store((1 * 3 + 0) * n + j, z)
        dst.unsafe_store((1 * 3 + 1) * n + j, zero)
        dst.unsafe_store((1 * 3 + 2) * n + j, -x)
        dst.unsafe_store((2 * 3 + 0) * n + j, -y)
        dst.unsafe_store((2 * 3 + 1) * n + j, x)
        dst.unsafe_store((2 * 3 + 2) * n + j, zero)
        j += 1


# ----------------------------------------------------------------------------
# skyfield.timelib
# ----------------------------------------------------------------------------

def julian_day(year: Ptr, month: Ptr, day: Ptr, julian_before: Float64, n: Int, dst: Ptr):
    """Given a calendar date, return a Julian day integer.

    `julian_before` of -1.0 stands for upstream's `None`, i.e. "always use
    the proleptic Gregorian calendar".
    """
    var j = 0
    while j < n:
        var yy = year.unsafe_load(j)
        var mo = month.unsafe_load(j)
        var dd = day.unsafe_load(j)

        # Support months <1 and >12 by overflowing cleanly into adjacent years.
        var y = floordiv(mo - 1.0, 12.0)
        var m = pymod(mo - 1.0, 12.0)
        yy += y
        m += 1.0

        # See the Explanatory Supplement to the Astronomical Almanac 15.11.
        var janfeb = 1.0 if m <= 2.0 else 0.0
        var g = yy + 4716.0 - janfeb
        var f = pymod(m + 9.0, 12.0)
        var e = floordiv(1461.0 * g, 4.0) + dd - 1402.0
        var J = e + floordiv(153.0 * f + 2.0, 5.0)

        var mask = 1.0 if julian_before < 0.0 else (1.0 if J >= julian_before else 0.0)
        J += (38.0 - floordiv(floordiv(g + 184.0, 100.0) * 3.0, 4.0)) * mask
        dst.unsafe_store(j, J)
        j += 1


def julian_date(year: Ptr, month: Ptr, day: Ptr, hour: Ptr, minute: Ptr,
                second: Ptr, julian_before: Float64, n: Int, dst: Ptr):
    """Given a proleptic Gregorian calendar date and time, build a Julian date."""
    julian_day(year, month, day, julian_before, n, dst)
    var j = 0
    while j < n:
        var frac = (second.unsafe_load(j) + minute.unsafe_load(j) * 60.0
                    + hour.unsafe_load(j) * 3600.0) / DAY_S
        dst.unsafe_store(j, dst.unsafe_load(j) - 0.5 + frac)
        j += 1


def julian_date_of_besselian_epoch(b: Ptr, n: Int, dst: Ptr):
    var j = 0
    while j < n:
        dst.unsafe_store(j, 2415020.31352 + (b.unsafe_load(j) - 1900.0) * 365.242198781)
        j += 1


def _compute_calendar_date_one(jd: Float64, julian_before: Float64) -> Tuple[
        Float64, Float64, Float64]:
    """Convert one Julian day integer into a (year, month, day) calendar date."""
    var use_gregorian = 1.0
    if julian_before >= 0.0 and jd < julian_before:
        use_gregorian = 0.0

    # See the Explanatory Supplement to the Astronomical Almanac 15.11.
    var f = jd + 1401.0
    f += use_gregorian * (floordiv(floordiv(4.0 * jd + 274277.0, 146097.0) * 3.0, 4.0) - 38.0)
    var e = 4.0 * f + 3.0
    var g = floordiv(pymod(e, 1461.0), 4.0)
    var h = 5.0 * g + 2.0
    var day = floordiv(pymod(h, 153.0), 5.0) + 1.0
    var month = pymod(floordiv(h, 153.0) + 2.0, 12.0) + 1.0
    var year = floordiv(e, 1461.0) - 4716.0 + floordiv(14.0 - month, 12.0)
    return (year, month, day)


def compute_calendar_date(jd_integer: Ptr, julian_before: Float64, n: Int,
                          d_year: Ptr, d_month: Ptr, d_day: Ptr):
    """Convert Julian day ``jd_integer`` into a calendar (year, month, day)."""
    var j = 0
    while j < n:
        var ymd = _compute_calendar_date_one(jd_integer.unsafe_load(j), julian_before)
        d_year.unsafe_store(j, ymd[0])
        d_month.unsafe_store(j, ymd[1])
        d_day.unsafe_store(j, ymd[2])
        j += 1


def calendar_tuple(jd_float: Ptr, fraction: Ptr, julian_before: Float64, n: Int,
                   d_year: Ptr, d_month: Ptr, d_day: Ptr, d_hour: Ptr,
                   d_minute: Ptr, d_second: Ptr):
    """Return a (year, month, day, hour, minute, second.fraction) tuple."""
    var j = 0
    while j < n:
        var jd = jd_float.unsafe_load(j)
        var whole1 = floordiv(jd, 1.0)
        var fraction1 = pymod(jd, 1.0)
        var rem = fraction1 + fraction.unsafe_load(j) + 0.5
        var whole = whole1 + floordiv(rem, 1.0)
        var frac = pymod(rem, 1.0)
        var ymd = _compute_calendar_date_one(whole, julian_before)
        var second = frac * 86400.0
        var minute = floordiv(second, 60.0)
        second = pymod(second, 60.0)
        var hour = floordiv(minute, 60.0)
        minute = pymod(minute, 60.0)
        d_year.unsafe_store(j, ymd[0])
        d_month.unsafe_store(j, ymd[1])
        d_day.unsafe_store(j, ymd[2])
        d_hour.unsafe_store(j, hour)
        d_minute.unsafe_store(j, minute)
        d_second.unsafe_store(j, second)
        j += 1


def tdb_minus_tt(jd_tdb: Ptr, fraction_tdb: Ptr, n: Int, dst: Ptr):
    """Computes how far TDB is in advance of TT, given TDB."""
    var j = 0
    while j < n:
        var t = (jd_tdb.unsafe_load(j) - T0 + fraction_tdb.unsafe_load(j)) / 36525.0

        # USNO Circular 179, eq. 2.6.
        var d = 0.001657 * sin(628.3076 * t + 6.2401)
        d += 0.000022 * sin(575.3385 * t + 4.2970)
        d += 0.000014 * sin(1256.6152 * t + 6.1969)
        d += 0.000005 * sin(606.9777 * t + 4.0212)
        d += 0.000005 * sin(52.9691 * t + 0.4444)
        d += 0.000002 * sin(21.3299 * t + 5.5431)
        d += 0.000010 * t * sin(628.3076 * t + 4.2490)
        dst.unsafe_store(j, d)
        j += 1


# ----------------------------------------------------------------------------
# skyfield.earthlib
# ----------------------------------------------------------------------------

comptime earth_radius_au: Float64 = ERAD / AU_M
comptime one_minus_flattening: Float64 = 1.0 - 1.0 / IERS_2010_INVERSE_EARTH_FLATTENING
comptime one_minus_flattening_squared: Float64 = one_minus_flattening * one_minus_flattening


def terra(latitude: Ptr, longitude: Ptr, elevation: Ptr, gast: Ptr, n: Int,
          d_pos: Ptr, d_vel: Ptr):
    """Deprecated conversion from lat,lon,t -> GCRS; neglects polar motion."""
    var j = 0
    while j < n:
        var phi = latitude.unsafe_load(j)
        var lon = longitude.unsafe_load(j)
        var elev = elevation.unsafe_load(j)
        var g = gast.unsafe_load(j)
        var zero = g * 0.0

        var sinphi = sin(phi)
        var cosphi = cos(phi)
        var c = 1.0 / sqrt(cosphi * cosphi + sinphi * sinphi * one_minus_flattening_squared)
        var s = one_minus_flattening_squared * c
        var ach = earth_radius_au * c + elev
        var ash = earth_radius_au * s + elev

        # Compute local sidereal time factors at the observer's longitude.
        var stlocl = 15.0 * DEG2RAD * g + lon
        var sinst = sin(stlocl)
        var cosst = cos(stlocl)

        # Compute position vector components in au.
        var ac = ach * cosphi
        var acsst = ac * sinst
        var accst = ac * cosst
        d_pos.unsafe_store(0 * n + j, accst)
        d_pos.unsafe_store(1 * n + j, acsst)
        d_pos.unsafe_store(2 * n + j, zero + ash * sinphi)

        # Compute velocity vector components in au/day.
        var k = ANGVEL * DAY_S
        d_vel.unsafe_store(0 * n + j, -acsst * k)
        d_vel.unsafe_store(1 * n + j, accst * k)
        d_vel.unsafe_store(2 * n + j, zero)
        j += 1


def reverse_terra(xyz_au: Ptr, gast: Ptr, iterations: Int, n: Int,
                  d_lat: Ptr, d_lon: Ptr, d_elevation_m: Ptr):
    """Deprecated conversion from GCRS -> lat,lon,t; neglects polar motion."""
    var a = ERAD / AU_M
    var f = 1.0 / IERS_2010_INVERSE_EARTH_FLATTENING
    var e2 = 2.0 * f - f * f
    var j = 0
    while j < n:
        var x = xyz_au.unsafe_load(0 * n + j)
        var y = xyz_au.unsafe_load(1 * n + j)
        var z = xyz_au.unsafe_load(2 * n + j)
        var R = sqrt(x * x + y * y)

        var lon = pymod(atan2(y, x) - 15.0 * DEG2RAD * gast.unsafe_load(j) - pi, tau) - pi
        var lat = atan2(z, R)

        var C = 1.0
        var i = 0
        while i < iterations:
            i += 1
            C = 1.0 / sqrt(1.0 - e2 * (sin(lat) * sin(lat)))
            lat = atan2(z + a * C * e2 * sin(lat), R)
        d_lat.unsafe_store(j, lat)
        d_lon.unsafe_store(j, lon)
        d_elevation_m.unsafe_store(j, ((R / cos(lat)) - a * C) * AU_M)
        j += 1


def compute_limb_angle(position_au: Ptr, observer_au: Ptr, n: Int,
                       d_limb_angle: Ptr, d_nadir_angle: Ptr):
    """Determine the angle of an object above or below the Earth's limb."""
    var j = 0
    while j < n:
        var px = position_au.unsafe_load(0 * n + j)
        var py = position_au.unsafe_load(1 * n + j)
        var pz = position_au.unsafe_load(2 * n + j)
        var ox = observer_au.unsafe_load(0 * n + j)
        var oy = observer_au.unsafe_load(1 * n + j)
        var oz = observer_au.unsafe_load(2 * n + j)

        # Compute the distance to the object and the distance to the observer.
        var disobj = sqrt(px * px + py * py + pz * pz)
        var disobs = sqrt(ox * ox + oy * oy + oz * oz)

        # Compute apparent angular radius of Earth's limb.
        var aprad = asin(clip(earth_radius_au / disobs, -1.0, 1.0))

        # Compute zenith distance of Earth's limb.
        var zdlim = pi - aprad

        # Compute zenith distance of observed object.
        var coszd = clip((px * ox + py * oy + pz * oz) / (disobj * disobs), -1.0, 1.0)
        var zdobj = acos(coszd)

        # Angle of object wrt limb is difference in zenith distances.
        d_limb_angle.unsafe_store(j, (zdlim - zdobj) * RAD2DEG)

        # Nadir angle of object as a fraction of angular radius of limb.
        d_nadir_angle.unsafe_store(j, (pi - zdobj) / aprad)
        j += 1


def _earth_rotation_angle_one(jd_ut1: Float64, fraction_ut1: Float64) -> Float64:
    """`earth_rotation_angle()` for a single date.

    Uses the expression from the note to IAU Resolution B1.8 of 2000.
    """
    var th = 0.7790572732640 + 0.00273781191135448 * (jd_ut1 - T0 + fraction_ut1)
    return pymod(pymod(th, 1.0) + pymod(jd_ut1, 1.0) + fraction_ut1, 1.0)


def earth_rotation_angle(jd_ut1: Ptr, fraction_ut1: Ptr, n: Int, dst: Ptr):
    """Return the value of the Earth Rotation Angle (theta) for a UT1 date."""
    var j = 0
    while j < n:
        dst.unsafe_store(j, _earth_rotation_angle_one(jd_ut1.unsafe_load(j),
                                                      fraction_ut1.unsafe_load(j)))
        j += 1


def sidereal_time(jd_ut1: Ptr, fraction_ut1: Ptr, tdb_fraction: Ptr, n: Int, dst: Ptr):
    """Compute Greenwich Mean Sidereal Time (GMST) in hours at time ``t``."""
    var j = 0
    while j < n:
        var theta = _earth_rotation_angle_one(jd_ut1.unsafe_load(j),
                                              fraction_ut1.unsafe_load(j))
        var t = (jd_ut1.unsafe_load(j) - T0 + tdb_fraction.unsafe_load(j)) / 36525.0

        # The equinox method.  See Circular 179, Section 2.6.2.
        # Precession-in-RA terms in mean sidereal time taken from third
        # reference, eq. (42), with coefficients in arcseconds.
        var st = 0.014506 + ((((-0.0000000368 * t - 0.000029956) * t - 0.00000044) * t
                              + 1.3915817) * t + 4612.156534) * t
        dst.unsafe_store(j, pymod(st / 54000.0 + theta * 24.0, 24.0))
        j += 1


def _refraction_one(alt: Float64, temperature_C: Float64,
                    pressure_mbar: Float64) -> Float64:
    var r = 0.016667 / tan((alt + 7.31 / (alt + 4.4)) * DEG2RAD)
    var d = r * (0.28 * pressure_mbar / (temperature_C + 273.0))
    if -1.0 <= alt and alt <= 89.9:
        return d
    return 0.0


def refraction(alt_degrees: Ptr, temperature_C: Float64, pressure_mbar: Float64,
               n: Int, dst: Ptr):
    """Given an observed altitude, estimate atmospheric refraction, in degrees."""
    var j = 0
    while j < n:
        dst.unsafe_store(j, _refraction_one(alt_degrees.unsafe_load(j),
                                            temperature_C, pressure_mbar))
        j += 1


def refract(alt_degrees: Ptr, temperature_C: Float64, pressure_mbar: Float64,
            n: Int, dst: Ptr):
    """Given an unrefracted `alt` determine where it will appear in the sky."""
    var j = 0
    while j < n:
        dst.unsafe_store(j, alt_degrees.unsafe_load(j))
        j += 1
    while True:
        var converged = True
        j = 0
        while j < n:
            var alt1 = dst.unsafe_load(j)
            var alt = alt_degrees.unsafe_load(j) + _refraction_one(
                alt1, temperature_C, pressure_mbar)
            dst.unsafe_store(j, alt)
            var diff = abs(alt - alt1)
            if not isfinite(diff):
                diff = 0.0
            if diff > 3.0e-5:
                converged = False
            j += 1
        if converged:
            break


# ----------------------------------------------------------------------------
# skyfield.precessionlib
# ----------------------------------------------------------------------------

def _precession_at(jd: Float64) -> Tuple[
        Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64,
        Float64]:
    """One epoch of `compute_precession()`, as its nine matrix elements."""
    var eps0 = 84381.406

    # 't' is time in TDB centuries.
    var t = (jd - T0) / 36525.0

    # Numerical coefficients of psi_a, omega_a, and chia, along with
    # epsilon_0, the obliquity at J2000.0, are 4-angle formulation from
    # Capitaine et al. (2003), eqs. (4), (37), & (39).
    var psia = ((((-0.0000000951 * t + 0.000132851) * t - 0.00114045) * t
                 - 1.0790069) * t + 5038.481507) * t

    var omegaa = ((((0.0000003337 * t - 0.000000467) * t - 0.00772503) * t
                   + 0.0512623) * t - 0.025754) * t + eps0

    var chia = ((((-0.0000000560 * t + 0.000170663) * t - 0.00121197) * t
                 - 2.3814292) * t + 10.556403) * t

    eps0 = eps0 * ASEC2RAD
    psia = psia * ASEC2RAD
    omegaa = omegaa * ASEC2RAD
    chia = chia * ASEC2RAD

    var sa = sin(eps0)
    var ca = cos(eps0)
    var sb = sin(-psia)
    var cb = cos(-psia)
    var sc = sin(-omegaa)
    var cc = cos(-omegaa)
    var sd = sin(chia)
    var cd = cos(chia)

    # Compute elements of precession rotation matrix equivalent to
    # R3(chi_a) R1(-omega_a) R3(-psi_a) R1(epsilon_0).
    return (cd * cb - sb * sd * cc,
            cd * sb * ca + sd * cc * cb * ca - sa * sd * sc,
            cd * sb * sa + sd * cc * cb * sa + ca * sd * sc,
            -sd * cb - sb * cd * cc,
            -sd * sb * ca + cd * cc * cb * ca - sa * cd * sc,
            -sd * sb * sa + cd * cc * cb * sa + ca * cd * sc,
            sb * sc, -sc * cb * ca - sa * cc, -sc * cb * sa + cc * ca)


def compute_precession(jd_tdb: Ptr, n: Int, max_workers: Int, dst: Ptr):
    """Return the rotation matrices for precessing to an array of epochs."""
    var (items, workers) = _plan(n, max_workers)

    def work(i: Int) {imm}:
        var j = i * n // workers
        var hi = (i + 1) * n // workers
        while j < hi:
            var m = _precession_at(jd_tdb.unsafe_load(j))
            dst.unsafe_store((0 * 3 + 0) * n + j, m[0])
            dst.unsafe_store((0 * 3 + 1) * n + j, m[1])
            dst.unsafe_store((0 * 3 + 2) * n + j, m[2])
            dst.unsafe_store((1 * 3 + 0) * n + j, m[3])
            dst.unsafe_store((1 * 3 + 1) * n + j, m[4])
            dst.unsafe_store((1 * 3 + 2) * n + j, m[5])
            dst.unsafe_store((2 * 3 + 0) * n + j, m[6])
            dst.unsafe_store((2 * 3 + 1) * n + j, m[7])
            dst.unsafe_store((2 * 3 + 2) * n + j, m[8])
            j += 1

    if workers > 1:
        parallelize(work, items, workers)
    else:
        work(0)


# ----------------------------------------------------------------------------
# skyfield.nutationlib
# ----------------------------------------------------------------------------

comptime _TENTH_USEC_2_RAD: Float64 = ASEC2RAD / 1e7
# Sine and cosine coefficients for t^1.
comptime se1_0: Float64 = -0.87e-6
comptime se1_1: Float64 = +0.00e-6


def _fa(k: Int, i: Int) -> Float64:
    """`fa0` through `fa4` from upstream: the t^k coefficient of argument i."""
    if k == 4:
        if i == 0:
            return -0.00024470
        if i == 1:
            return -0.00001149
        if i == 2:
            return 0.00000417
        if i == 3:
            return -0.00003169
        return -0.00005939
    if k == 3:
        if i == 0:
            return 0.051635
        if i == 1:
            return 0.000136
        if i == 2:
            return -0.001037
        if i == 3:
            return 0.006593
        return 0.007702
    if k == 2:
        if i == 0:
            return 31.8792
        if i == 1:
            return -0.5532
        if i == 2:
            return -12.7512
        if i == 3:
            return -6.3706
        return 7.4722
    if k == 1:
        if i == 0:
            return 1717915923.2178
        if i == 1:
            return 129596581.0481
        if i == 2:
            return 1739527262.8478
        if i == 3:
            return 1602961601.2090
        return -6962890.5431
    if i == 0:
        return 485868.249036
    if i == 1:
        return 1287104.79305
    if i == 2:
        return 335779.526232
    if i == 3:
        return 1072260.70369
    return 450160.398036


def _fundamental_arguments(t: Float64, terms: Int) -> Tuple[
        Float64, Float64, Float64, Float64, Float64]:
    """Compute the fundamental arguments (mean elements) of Sun and Moon.

    The five arguments are always produced; `terms` selects how many of the
    polynomial orders get applied, exactly as upstream's slice of
    ``(fa4, fa3, fa2, fa1)`` does.  They come back as scalars rather than
    through a list because every caller wants all five, in registers, once.
    """
    # Upstream slices `(fa4, fa3, fa2, fa1)[-terms + 1:]`, and a negative
    # start is clamped to zero by Python, so `terms` above 5 keeps the whole
    # tuple rather than reaching past it.
    var start = 0
    if terms > 1 and terms < 5:
        start = 5 - terms
    var k = 4 - start
    var a0 = _fa(k, 0) * t
    var a1 = _fa(k, 1) * t
    var a2 = _fa(k, 2) * t
    var a3 = _fa(k, 3) * t
    var a4 = _fa(k, 4) * t
    k -= 1
    while k >= 1:
        a0 = (a0 + _fa(k, 0)) * t
        a1 = (a1 + _fa(k, 1)) * t
        a2 = (a2 + _fa(k, 2)) * t
        a3 = (a3 + _fa(k, 3)) * t
        a4 = (a4 + _fa(k, 4)) * t
        k -= 1
    return (pyfmod(a0 + _fa(0, 0), ASEC360) * ASEC2RAD,
            pyfmod(a1 + _fa(0, 1), ASEC360) * ASEC2RAD,
            pyfmod(a2 + _fa(0, 2), ASEC360) * ASEC2RAD,
            pyfmod(a3 + _fa(0, 3), ASEC360) * ASEC2RAD,
            pyfmod(a4 + _fa(0, 4), ASEC360) * ASEC2RAD)


def fundamental_arguments(t: Ptr, terms: Int, n: Int, max_workers: Int,
                       dst: Ptr):
    """Compute the fundamental arguments (mean elements) of Sun and Moon."""
    var (items, workers) = _plan(n, max_workers)

    def work(i: Int) {imm}:
        var j = i * n // workers
        var hi = (i + 1) * n // workers
        while j < hi:
            var (a0, a1, a2, a3, a4) = _fundamental_arguments(
                t.unsafe_load(j), terms)
            dst.unsafe_store(0 * n + j, a0)
            dst.unsafe_store(1 * n + j, a1)
            dst.unsafe_store(2 * n + j, a2)
            dst.unsafe_store(3 * n + j, a3)
            dst.unsafe_store(4 * n + j, a4)
            j += 1

    if workers > 1:
        parallelize(work, items, workers)
    else:
        work(0)


def mean_obliquity(jd_tdb: Ptr, n: Int, dst: Ptr):
    """Return the mean obliquity of the ecliptic in arcseconds."""
    var j = 0
    while j < n:
        # Compute time in Julian centuries from epoch J2000.0.
        var t = (jd_tdb.unsafe_load(j) - T0) / 36525.0

        # Compute the mean obliquity in arcseconds.  Use expression from the
        # reference's eq. (39) with obliquity at J2000.0 taken from eq. (37)
        # or Table 8.
        dst.unsafe_store(j, (((( -0.0000000434 * t - 0.000000576) * t + 0.00200340) * t
                              - 0.0001831) * t - 46.836769) * t + 84381.406)
        j += 1


def build_nutation_matrix(mean_obliquity_radians: Ptr, true_obliquity_radians: Ptr,
                          psi_radians: Ptr, n: Int, dst: Ptr):
    """Generate the nutation rotation matrix, given three nutation parameters."""
    var j = 0
    while j < n:
        var cobm = cos(mean_obliquity_radians.unsafe_load(j))
        var sobm = sin(mean_obliquity_radians.unsafe_load(j))
        var cobt = cos(true_obliquity_radians.unsafe_load(j))
        var sobt = sin(true_obliquity_radians.unsafe_load(j))
        var cpsi = cos(psi_radians.unsafe_load(j))
        var spsi = sin(psi_radians.unsafe_load(j))

        dst.unsafe_store((0 * 3 + 0) * n + j, cpsi)
        dst.unsafe_store((0 * 3 + 1) * n + j, -spsi * cobm)
        dst.unsafe_store((0 * 3 + 2) * n + j, -spsi * sobm)
        dst.unsafe_store((1 * 3 + 0) * n + j, spsi * cobt)
        dst.unsafe_store((1 * 3 + 1) * n + j, cpsi * cobm * cobt + sobm * sobt)
        dst.unsafe_store((1 * 3 + 2) * n + j, cpsi * sobm * cobt - cobm * sobt)
        dst.unsafe_store((2 * 3 + 0) * n + j, spsi * sobt)
        dst.unsafe_store((2 * 3 + 1) * n + j, cpsi * cobm * sobt - sobm * cobt)
        dst.unsafe_store((2 * 3 + 2) * n + j, cpsi * sobm * sobt + cobm * cobt)
        j += 1


def _eqeq_one(t: Float64, ke0_t: IPtr, ke1: IPtr, se0_t_0: Ptr, se0_t_1: Ptr,
               nke: Int, mut fa: List[Float64]) -> Float64:
    """One epoch of the complementary terms of the equation of the equinoxes.

    `fa` is scratch owned by the caller, so each worker can keep its own
    copy of the fourteen intermediate arguments.
    """
    # Mean Anomaly of the Moon.
    fa[0] = ((485868.249036 + (715923.2178 + (31.8792 + (0.051635 + (-0.00024470) * t) * t) * t) * t)
             * ASEC2RAD + pyfmod(1325.0 * t, 1.0) * tau)

    # Mean Anomaly of the Sun.
    fa[1] = ((1287104.793048 + (1292581.0481 + (-0.5532 + (0.000136 + (-0.00001149) * t) * t) * t) * t)
             * ASEC2RAD + pyfmod(99.0 * t, 1.0) * tau)

    # Mean Longitude of the Moon minus Mean Longitude of the Ascending
    # Node of the Moon.
    fa[2] = ((335779.526232 + (295262.8478 + (-12.7512 + (-0.001037 + 0.00000417 * t) * t) * t) * t)
             * ASEC2RAD + pyfmod(1342.0 * t, 1.0) * tau)

    # Mean Elongation of the Moon from the Sun.
    fa[3] = ((1072260.703692 + (1105601.2090 + (-6.3706 + (0.006593 + (-0.00003169) * t) * t) * t) * t)
             * ASEC2RAD + pyfmod(1236.0 * t, 1.0) * tau)

    # Mean Longitude of the Ascending Node of the Moon.
    fa[4] = ((450160.398036 + (-482890.5431 + (7.4722 + (0.007702 + (-0.00005939) * t) * t) * t) * t)
             * ASEC2RAD + pyfmod(-5.0 * t, 1.0) * tau)

    fa[5] = 4.402608842 + 2608.7903141574 * t
    fa[6] = 3.176146697 + 1021.3285546211 * t
    fa[7] = 1.753470314 + 628.3075849991 * t
    fa[8] = 6.203480913 + 334.0612426700 * t
    fa[9] = 0.599546497 + 52.9690962641 * t
    fa[10] = 0.874016757 + 21.3299104960 * t
    fa[11] = 5.481293872 + 7.4781598567 * t
    fa[12] = 5.311886287 + 3.8133035638 * t
    fa[13] = (0.024381750 + 0.00000538691 * t) * t
    var i = 0
    while i < 14:
        fa[i] = pymod(fa[i], tau)
        i += 1

    # Evaluate the complementary terms.
    var angle = 0.0
    i = 0
    while i < 14:
        angle += Float64(ke1.unsafe_load(i)) * fa[i]
        i += 1
    var c_terms = se1_0 * sin(angle)
    c_terms += se1_1 * cos(angle)
    c_terms *= t

    var m = 0
    while m < nke:
        var arg = 0.0
        i = 0
        while i < 14:
            arg += Float64(ke0_t.unsafe_load(m * 14 + i)) * fa[i]
            i += 1
        c_terms += se0_t_0.unsafe_load(m) * sin(arg)
        c_terms += se0_t_1.unsafe_load(m) * cos(arg)
        m += 1

    return c_terms * ASEC2RAD


def equation_of_the_equinoxes_complimentary_terms(jd_tt: Ptr, ke0_t: IPtr, ke1: IPtr,
                                                  se0_t_0: Ptr, se0_t_1: Ptr,
                                                  nke: Int, n: Int,
                                                  max_workers: Int, dst: Ptr):
    """Compute the complementary terms of the equation of the equinoxes."""
    var (items, workers) = _plan(n, max_workers)

    def work(i: Int) {imm}:
        # One scratch list per worker: the fourteen arguments are per-epoch
        # and nothing here is shared between epochs.
        var fa = List[Float64]()
        var k = 0
        while k < 14:
            fa.append(0.0)
            k += 1
        var j = i * n // workers
        var hi = (i + 1) * n // workers
        while j < hi:
            var t = (jd_tt.unsafe_load(j) - T0) / 36525.0
            dst.unsafe_store(j, _eqeq_one(t, ke0_t, ke1, se0_t_0, se0_t_1, nke, fa))
            j += 1

    if workers > 1:
        parallelize(work, items, workers)
    else:
        work(0)


def _anomaly_constant(i: Int) -> Float64:
    # Mean anomaly of the Moon.
    if i == 0:
        return 2.35555598
    # Mean anomaly of the Sun.
    if i == 1:
        return 6.24006013
    # Mean argument of the latitude of the Moon.
    if i == 2:
        return 1.627905234
    # Mean elongation of the Moon from the Sun.
    if i == 3:
        return 5.198466741
    # Mean longitude of the ascending node of the Moon.
    if i == 4:
        return 2.18243920
    # Planetary longitudes, Mercury through Neptune (Souchay et al. 1999).
    if i == 5:
        return 4.402608842
    if i == 6:
        return 3.176146697
    if i == 7:
        return 1.753470314
    if i == 8:
        return 6.203480913
    if i == 9:
        return 0.599546497
    if i == 10:
        return 0.874016757
    if i == 11:
        return 5.481293871
    if i == 12:
        return 5.321159000
    # General accumulated precession in longitude (gets multiplied by t).
    return 0.02438175


def _anomaly_coefficient(i: Int) -> Float64:
    if i == 0:
        return 8328.6914269554
    if i == 1:
        return 628.301955
    if i == 2:
        return 8433.466158131
    if i == 3:
        return 7771.3771468121
    if i == 4:
        return -33.757045
    if i == 5:
        return 2608.7903141574
    if i == 6:
        return 1021.3285546211
    if i == 7:
        return 628.3075849991
    if i == 8:
        return 334.0612426700
    if i == 9:
        return 52.9690962641
    if i == 10:
        return 21.3299104960
    if i == 11:
        return 7.4781598567
    if i == 12:
        return 3.8127774000
    return 0.00000538691


def _iau2000a_one(jd: Float64, fundamental_argument_terms: Int, lunisolar_terms: Int,
                  planetary_terms: Int, nals_t: IPtr, lls: Ptr, los: Ptr,
                  napl_t: IPtr, ncl: Ptr, nco: Ptr) -> Tuple[Float64, Float64]:
    """One epoch of `iau2000a()`, returning (delta-psi, delta-epsilon).

    The five fundamental arguments and the fourteen planetary anomalies live
    in scalars rather than in a list: each term's dot product reads them
    1374 times, and a list read is a load plus a bounds check.
    """
    # Interval between fundamental epoch J2000.0 and given date.
    var t = (jd - T0) / 36525.0

    # Compute fundamental arguments from Simon et al. (1994), in radians.
    var (a0, a1, a2, a3, a4) = _fundamental_arguments(
        t, fundamental_argument_terms)

    # ** Luni-solar nutation **
    # Summation of luni-solar nutation series.
    var dpsi = 0.0
    var deps = 0.0
    var m = 0
    while m < lunisolar_terms:
        var r = m * 5
        var arg = Float64(nals_t.unsafe_load(r)) * a0
        arg += Float64(nals_t.unsafe_load(r + 1)) * a1
        arg += Float64(nals_t.unsafe_load(r + 2)) * a2
        arg += Float64(nals_t.unsafe_load(r + 3)) * a3
        arg += Float64(nals_t.unsafe_load(r + 4)) * a4
        var sarg = sin(arg)
        var carg = cos(arg)
        var row = m * 3
        dpsi += sarg * lls.unsafe_load(row)
        dpsi += sarg * lls.unsafe_load(row + 1) * t
        dpsi += carg * lls.unsafe_load(row + 2)
        deps += carg * los.unsafe_load(row)
        deps += carg * los.unsafe_load(row + 1) * t
        deps += sarg * los.unsafe_load(row + 2)
        m += 1

    # Compute and add in planetary components.
    if planetary_terms > 0:
        var pa0 = _anomaly_coefficient(0) * t + _anomaly_constant(0)
        var pa1 = _anomaly_coefficient(1) * t + _anomaly_constant(1)
        var pa2 = _anomaly_coefficient(2) * t + _anomaly_constant(2)
        var pa3 = _anomaly_coefficient(3) * t + _anomaly_constant(3)
        var pa4 = _anomaly_coefficient(4) * t + _anomaly_constant(4)
        var pa5 = _anomaly_coefficient(5) * t + _anomaly_constant(5)
        var pa6 = _anomaly_coefficient(6) * t + _anomaly_constant(6)
        var pa7 = _anomaly_coefficient(7) * t + _anomaly_constant(7)
        var pa8 = _anomaly_coefficient(8) * t + _anomaly_constant(8)
        var pa9 = _anomaly_coefficient(9) * t + _anomaly_constant(9)
        var pa10 = _anomaly_coefficient(10) * t + _anomaly_constant(10)
        var pa11 = _anomaly_coefficient(11) * t + _anomaly_constant(11)
        var pa12 = _anomaly_coefficient(12) * t + _anomaly_constant(12)
        # General accumulated precession in longitude (gets multiplied by t).
        var pa13 = (_anomaly_coefficient(13) * t + _anomaly_constant(13)) * t
        m = 0
        while m < planetary_terms:
            var r = m * 14
            var arg = Float64(napl_t.unsafe_load(r)) * pa0
            arg += Float64(napl_t.unsafe_load(r + 1)) * pa1
            arg += Float64(napl_t.unsafe_load(r + 2)) * pa2
            arg += Float64(napl_t.unsafe_load(r + 3)) * pa3
            arg += Float64(napl_t.unsafe_load(r + 4)) * pa4
            arg += Float64(napl_t.unsafe_load(r + 5)) * pa5
            arg += Float64(napl_t.unsafe_load(r + 6)) * pa6
            arg += Float64(napl_t.unsafe_load(r + 7)) * pa7
            arg += Float64(napl_t.unsafe_load(r + 8)) * pa8
            arg += Float64(napl_t.unsafe_load(r + 9)) * pa9
            arg += Float64(napl_t.unsafe_load(r + 10)) * pa10
            arg += Float64(napl_t.unsafe_load(r + 11)) * pa11
            arg += Float64(napl_t.unsafe_load(r + 12)) * pa12
            arg += Float64(napl_t.unsafe_load(r + 13)) * pa13
            var sarg = sin(arg)
            var carg = cos(arg)
            dpsi += sarg * ncl.unsafe_load(m * 2)
            dpsi += carg * ncl.unsafe_load(m * 2 + 1)
            deps += sarg * nco.unsafe_load(m * 2)
            deps += carg * nco.unsafe_load(m * 2 + 1)
            m += 1

    return (dpsi, deps)


def iau2000a(jd_tt: Ptr, fundamental_argument_terms: Int, lunisolar_terms: Int,
             planetary_terms: Int, nals_t: IPtr, lunisolar_longitude_coefficients: Ptr,
             lunisolar_obliquity_coefficients: Ptr, napl_t: IPtr,
             nutation_coefficients_longitude: Ptr, nutation_coefficients_obliquity: Ptr,
             n: Int, max_workers: Int, d_psi: Ptr, d_eps: Ptr):
    """Compute Earth nutation based on the IAU 2000A nutation model.

    Returns delta-psi and delta-epsilon in tenths of a micro-arcsecond.
    """
    var (items, workers) = _plan(n, max_workers)

    def work(i: Int) {imm}:
        var j = i * n // workers
        var hi = (i + 1) * n // workers
        while j < hi:
            var r = _iau2000a_one(
                jd_tt.unsafe_load(j), fundamental_argument_terms, lunisolar_terms,
                planetary_terms, nals_t, lunisolar_longitude_coefficients,
                lunisolar_obliquity_coefficients, napl_t,
                nutation_coefficients_longitude, nutation_coefficients_obliquity)
            d_psi.unsafe_store(j, r[0])
            d_eps.unsafe_store(j, r[1])
            j += 1

    if workers > 1:
        parallelize(work, items, workers)
    else:
        work(0)


def iau2000b(jd_tt: Ptr, fundamental_argument_terms: Int, lunisolar_terms: Int,
             planetary_terms: Int, nals_t: IPtr, lunisolar_longitude_coefficients: Ptr,
             lunisolar_obliquity_coefficients: Ptr, napl_t: IPtr,
             nutation_coefficients_longitude: Ptr, nutation_coefficients_obliquity: Ptr,
             n: Int, max_workers: Int, d_psi: Ptr, d_eps: Ptr):
    """Compute Earth nutation based on the faster IAU 2000B nutation model."""
    var (items, workers) = _plan(n, max_workers)

    # The two IAU 2000B offsets are folded into the same pass that sums the
    # series, rather than walked over the whole output array again.
    def work(i: Int) {imm}:
        var j = i * n // workers
        var hi = (i + 1) * n // workers
        while j < hi:
            var r = _iau2000a_one(
                jd_tt.unsafe_load(j), fundamental_argument_terms, lunisolar_terms,
                planetary_terms, nals_t, lunisolar_longitude_coefficients,
                lunisolar_obliquity_coefficients, napl_t,
                nutation_coefficients_longitude, nutation_coefficients_obliquity)
            d_psi.unsafe_store(j, r[0] + -0.000135e7)
            d_eps.unsafe_store(j, r[1] + 0.000388e7)
            j += 1

    if workers > 1:
        parallelize(work, items, workers)
    else:
        work(0)


# ----------------------------------------------------------------------------
# skyfield.framelib
# ----------------------------------------------------------------------------

def build_matrix(dst: Ptr):
    """The ICRS frame-bias matrix, `B` in IERS (2003) Conventions, Chapter 5."""
    # 'xi0', 'eta0', and 'da0' are ICRS frame biases in arcseconds taken
    # from IERS (2003) Conventions, Chapter 5.
    var xi0 = -0.0166170 * ASEC2RAD
    var eta0 = -0.0068192 * ASEC2RAD
    var da0 = -0.01460 * ASEC2RAD

    # Compute elements of rotation matrix.
    var yx = -da0
    var zx = xi0
    var xy = da0
    var zy = eta0
    var xz = -xi0
    var yz = -eta0

    # Include second-order corrections to diagonal elements.
    var xx = 1.0 - 0.5 * (yx * yx + zx * zx)
    var yy = 1.0 - 0.5 * (yx * yx + zy * zy)
    var zz = 1.0 - 0.5 * (zy * zy + zx * zx)

    dst.unsafe_store(0, xx)
    dst.unsafe_store(1, xy)
    dst.unsafe_store(2, xz)
    dst.unsafe_store(3, yx)
    dst.unsafe_store(4, yy)
    dst.unsafe_store(5, yz)
    dst.unsafe_store(6, zx)
    dst.unsafe_store(7, zy)
    dst.unsafe_store(8, zz)


def build_ecliptic_matrix(M: Ptr, true_obliquity: Ptr, n: Int, dst: Ptr,
                          scratch: Ptr):
    """Build the matrix to rotate an ICRF vector into ecliptic coordinates."""
    var j = 0
    while j < n:
        _rot_x_at(scratch, n, j, -true_obliquity.unsafe_load(j))
        j += 1
    mxm(scratch, M, n, dst)


def tirs_rotation_at(M: Ptr, gast: Ptr, n: Int, dst: Ptr, scratch: Ptr):
    """The TIRS rotation: mxm(rot_z(-gast * tau / 24), t.M)."""
    var j = 0
    while j < n:
        _rot_z_at(scratch, n, j, -gast.unsafe_load(j) * tau / 24.0)
        j += 1
    mxm(scratch, M, n, dst)


def itrs_rotation_at(M: Ptr, gast: Ptr, polar_motion: Ptr, n: Int, dst: Ptr,
                     scratch: Ptr, scratch2: Ptr):
    """The ITRS rotation: polar motion, then the TIRS rotation."""
    tirs_rotation_at(M, gast, n, scratch2, scratch)
    mxm(polar_motion, scratch2, n, dst)


# ----------------------------------------------------------------------------
# skyfield.relativity
# ----------------------------------------------------------------------------

def light_time_difference(position: Ptr, deflector_position: Ptr, n: Int, dst: Ptr):
    """When did the light from ``position`` pass closest to a given deflector?"""
    var j = 0
    while j < n:
        # From 'pos1', form unit vector 'u1' in direction of star or light
        # source.
        var d = length_of_one(position, n, j) + _AVOID_DIVIDE_BY_ZERO
        var u1x = position.unsafe_load(0 * n + j) / d
        var u1y = position.unsafe_load(1 * n + j) / d
        var u1z = position.unsafe_load(2 * n + j) / d

        # Light-time returned is the projection of vector 'pos_obs' onto the
        # unit vector 'u1' (formed from 'pos1'), divided by the speed of light.
        var dot = u1x * deflector_position.unsafe_load(0 * n + j)
        dot += u1y * deflector_position.unsafe_load(1 * n + j)
        dot += u1z * deflector_position.unsafe_load(2 * n + j)
        dst.unsafe_store(j, dot / C_AUDAY)
        j += 1


def _compute_deflection(position: Ptr, pe: Ptr, rmass: Float64, n: Int, dst: Ptr):
    """Compute how much a mass will deflect a position."""
    var j = 0
    while j < n:
        var px = position.unsafe_load(0 * n + j)
        var py = position.unsafe_load(1 * n + j)
        var pz = position.unsafe_load(2 * n + j)
        var ex = pe.unsafe_load(0 * n + j)
        var ey = pe.unsafe_load(1 * n + j)
        var ez = pe.unsafe_load(2 * n + j)

        var pq = Tuple[Float64, Float64, Float64](
            px + ex,
            py + ey,
            pz + ez,
        )

        # Compute vector magnitudes and unit vectors.
        var pm = length_of_one(position, n, j)
        var qm = sqrt(pq[0] * pq[0] + pq[1] * pq[1] + pq[2] * pq[2])
        var em = sqrt(ex * ex + ey * ey + ez * ez)

        var phx = px / (pm if pm != 0.0 else 1.0)
        var phy = py / (pm if pm != 0.0 else 1.0)
        var phz = pz / (pm if pm != 0.0 else 1.0)
        var qhx = pq[0] / (qm if qm != 0.0 else 1.0)
        var qhy = pq[1] / (qm if qm != 0.0 else 1.0)
        var qhz = pq[2] / (qm if qm != 0.0 else 1.0)
        var ehx = ex / (em if em != 0.0 else 1.0)
        var ehy = ey / (em if em != 0.0 else 1.0)
        var ehz = ez / (em if em != 0.0 else 1.0)

        # Compute dot products of vectors.
        var pdotq = phx * qhx + phy * qhy + phz * qhz
        var qdote = qhx * ehx + qhy * ehy + qhz * ehz
        var edotp = ehx * phx + ehy * phy + ehz * phz

        # If gravitating body is observed object, or is on a straight line
        # toward or away from observed object to within 1 arcsec, deflection
        # is set to zero.
        var flag = 1.0
        if abs(edotp) > 0.99999999999:
            flag = 0.0

        # Compute scalar factors.
        var fac1 = 2.0 * GS / (C * C * em * AU_M * rmass)
        var fac2 = 1.0 + qdote

        dst.unsafe_store(0 * n + j, flag * fac1 * (pdotq * ehx - edotp * qhx) / fac2 * pm)
        dst.unsafe_store(1 * n + j, flag * fac1 * (pdotq * ehy - edotp * qhy) / fac2 * pm)
        dst.unsafe_store(2 * n + j, flag * fac1 * (pdotq * ehz - edotp * qhz) / fac2 * pm)
        j += 1


def add_aberration(position: Ptr, velocity: Ptr, light_time: Ptr, n: Int):
    """Correct a relative position vector for aberration of light (in place)."""
    var j = 0
    while j < n:
        var lt = light_time.unsafe_load(j)
        var px = position.unsafe_load(0 * n + j)
        var py = position.unsafe_load(1 * n + j)
        var pz = position.unsafe_load(2 * n + j)
        var vx = velocity.unsafe_load(0 * n + j)
        var vy = velocity.unsafe_load(1 * n + j)
        var vz = velocity.unsafe_load(2 * n + j)

        var p1mag = lt * C_AUDAY
        var ve = length_of_one(velocity, n, j)
        var beta = ve / C_AUDAY
        var dot = px * vx + py * vy + pz * vz
        var cosd = dot / (p1mag * ve + _AVOID_DIVIDE_BY_ZERO)
        var gammai = sqrt(1.0 - beta * beta)
        var p = beta * cosd
        var q = (1.0 + p / (1.0 + gammai)) * lt
        var r = 1.0 + p

        position.unsafe_store(0 * n + j, (px * gammai + q * vx) / r)
        position.unsafe_store(1 * n + j, (py * gammai + q * vy) / r)
        position.unsafe_store(2 * n + j, (pz * gammai + q * vz) / r)
        j += 1


# ----------------------------------------------------------------------------
# skyfield.sgp4lib
# ----------------------------------------------------------------------------

def _cross(a: Ptr, b: Ptr, n: Int, dst: Ptr):
    """The cross product, spelled the way upstream spells it."""
    var j = 0
    while j < n:
        dst.unsafe_store(0 * n + j,
                         a.unsafe_load(1 * n + j) * b.unsafe_load(2 * n + j)
                         - a.unsafe_load(2 * n + j) * b.unsafe_load(1 * n + j))
        dst.unsafe_store(1 * n + j,
                         a.unsafe_load(2 * n + j) * b.unsafe_load(0 * n + j)
                         - a.unsafe_load(0 * n + j) * b.unsafe_load(2 * n + j))
        dst.unsafe_store(2 * n + j,
                         a.unsafe_load(0 * n + j) * b.unsafe_load(1 * n + j)
                         - a.unsafe_load(1 * n + j) * b.unsafe_load(0 * n + j))
        j += 1


def _cross9(a0: Float64, a1: Float64, a2: Float64, b0: Float64, b1: Float64,
            b2: Float64) -> Tuple[Float64, Float64, Float64]:
    """The cross product `a x b`, one component at a time."""
    return (a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0)


def _cross_one(a: Ptr, b: Ptr, n: Int, j: Int) -> Tuple[Float64, Float64, Float64]:
    """`_cross()` for a single pair of vectors."""
    return _cross9(a.unsafe_load(0 * n + j), a.unsafe_load(1 * n + j),
                   a.unsafe_load(2 * n + j), b.unsafe_load(0 * n + j),
                   b.unsafe_load(1 * n + j), b.unsafe_load(2 * n + j))


def _theta_GMST1982_one(jd_ut1: Float64, fraction_ut1: Float64) -> Tuple[
        Float64, Float64]:
    """`theta_GMST1982()` for a single date: the angle, and its rate."""
    var t = (jd_ut1 - T0 + fraction_ut1) / 36525.0
    var g = 67310.54841 + (8640184.812866 + (0.093104 + (-6.2e-6) * t) * t) * t
    var dg = 8640184.812866 + (0.093104 * 2.0 + (-6.2e-6 * 3.0) * t) * t
    return (pymod(pymod(jd_ut1, 1.0) + fraction_ut1 + pyfmod(g / DAY_S, 1.0), 1.0) * tau,
            (1.0 + dg / (DAY_S * 36525.0)) * tau)


def theta_GMST1982(jd_ut1: Ptr, fraction_ut1: Ptr, n: Int, d_theta: Ptr,
                  d_theta_dot: Ptr):
    """Return the angle of Greenwich Mean Standard Time 1982 given the JD.

    From AIAA 2006-6753 Appendix C.
    """
    var j = 0
    while j < n:
        var th = _theta_GMST1982_one(jd_ut1.unsafe_load(j),
                                     fraction_ut1.unsafe_load(j))
        d_theta.unsafe_store(j, th[0])
        d_theta_dot.unsafe_store(j, th[1])
        j += 1


def _teme_to_itrf_one(jd_ut1: Float64, fraction_ut1: Float64, rx: Float64,
                      ry: Float64, rz: Float64, vx: Float64, vy: Float64,
                      vz: Float64, xp: Float64, yp: Float64) -> Tuple[
        Float64, Float64, Float64, Float64, Float64, Float64]:
    """One epoch of `TEME_to_ITRF()`, as (rITRF |xyz|, vITRF |xyz|).

    Upstream builds the whole (3, 3, n) stack of `R = rot_z(-theta)`, then
    walks the arrays to apply it, then builds the angular velocity vector into
    another stack, then crosses, then accumulates, then copies.  Every one of
    those intermediates is three numbers, so they live in registers here and
    the inputs are read once.  The arithmetic, and its order, is upstream's.
    """
    var th = _theta_GMST1982_one(jd_ut1, fraction_ut1)
    # `R = rot_z(-theta)`, applied to (x, y, z) as upstream's `mxv` would.
    var c = cos(-th[0])
    var s = sin(-th[0])
    var rx0 = c * rx + (-s) * ry
    var ry0 = s * rx + c * ry
    var rz0 = rz

    # `vPEF = mxv(R, vTEME) + _cross(angular_velocity, rPEF)` with
    # `angular_velocity = (0, 0, -theta_dot)`: the cross product of that with
    # rPEF is `(-theta_dot * ry0, theta_dot * rx0, 0)`.
    var vx0 = c * vx + (-s) * vy
    var vy0 = s * vx + c * vy
    var vz0 = vz
    vx0 += th[1] * ry0
    vy0 += -th[1] * rx0

    if xp == 0.0 and yp == 0.0:
        # No polar motion: upstream copies the PEF vectors straight across.
        return (rx0, ry0, rz0, vx0, vy0, vz0)

    # `W = rot_x(yp) @ rot_y(xp)` and `rITRF = mxv(W, rPEF)`.
    # `rot_x(yp)` is `(1, 0, 0 / 0, cX, -sX / 0, sX, cX)` and `rot_y(xp)` is
    # `(cY, 0, sY / 0, 1, 0 / -sY, 0, cY)`; each element below is the `mxm`
    # sum over the middle index, written out so the product is auditable
    # against `skyfield/functions.py` without the two (3, 3, n) stacks.
    var cX = cos(yp)
    var sX = sin(yp)
    var cY = cos(xp)
    var sY = sin(xp)

    var w00 = cY
    var w01 = 0.0
    var w02 = sY
    var w10 = (-sX) * (-sY)
    var w11 = cX
    var w12 = (-sX) * cY
    var w20 = cX * (-sY)
    var w21 = sX
    var w22 = cX * cY

    return (w00 * rx0 + w01 * ry0 + w02 * rz0,
            w10 * rx0 + w11 * ry0 + w12 * rz0,
            w20 * rx0 + w21 * ry0 + w22 * rz0,
            w00 * vx0 + w01 * vy0 + w02 * vz0,
            w10 * vx0 + w11 * vy0 + w12 * vz0,
            w20 * vx0 + w21 * vy0 + w22 * vz0)


def TEME_to_ITRF(jd_ut1: Ptr, rTEME: Ptr, vTEME: Ptr, xp: Float64, yp: Float64,
                 fraction_ut1: Ptr, n: Int, max_workers: Int, d_rITRF: Ptr,
                 d_vITRF: Ptr):
    """Rotate SGP4's TEME vectors into the ITRS frame.

    One pass over the inputs, with the `R` and `W` matrices, the angular
    velocity vector and the cross product held in registers per epoch instead
    of in caller-supplied (3, 3, n) and (3, n) stacks.  `TEME_to_ITRF` is
    therefore the one kernel here that needs no `scratch` buffers at all, and
    the scratch arguments it used to take are gone.
    """
    var (items, workers) = _plan(n, max_workers)

    def work(i: Int) {imm}:
        var j = i * n // workers
        var hi = (i + 1) * n // workers
        while j < hi:
            var r = _teme_to_itrf_one(
                jd_ut1.unsafe_load(j), fraction_ut1.unsafe_load(j),
                rTEME.unsafe_load(0 * n + j), rTEME.unsafe_load(1 * n + j),
                rTEME.unsafe_load(2 * n + j), vTEME.unsafe_load(0 * n + j),
                vTEME.unsafe_load(1 * n + j), vTEME.unsafe_load(2 * n + j),
                xp, yp)
            d_rITRF.unsafe_store(0 * n + j, r[0])
            d_rITRF.unsafe_store(1 * n + j, r[1])
            d_rITRF.unsafe_store(2 * n + j, r[2])
            d_vITRF.unsafe_store(0 * n + j, r[3])
            d_vITRF.unsafe_store(1 * n + j, r[4])
            d_vITRF.unsafe_store(2 * n + j, r[5])
            j += 1

    if workers > 1:
        parallelize(work, items, workers)
    else:
        work(0)




def TEME_rotation_at(theta: Ptr, gast: Ptr, M: Ptr, n: Int, dst: Ptr, scratch: Ptr):
    """`TEME.rotation_at(t)`: mxm(rot_z(theta - gast / 24 * tau), t.M)."""
    var j = 0
    while j < n:
        _rot_z_at(scratch, n, j,
                  theta.unsafe_load(j) - gast.unsafe_load(j) / 24.0 * tau)
        j += 1
    mxm(scratch, M, n, dst)


# ----------------------------------------------------------------------------
# skyfield.keplerlib
# ----------------------------------------------------------------------------

comptime dpmax: Float64 = 1.7976931348623157e308

# The mid-range Stumpff series is nine terms of x^k times (-1)^k/(2k+2)! and
# (-1)^k/(2k+3)!.  Upstream multiplies each denominator out from scratch on
# every evaluation; these are the same reciprocals with the sign already in
# them, so the accumulation order and the sign flips are unchanged.
comptime A2: Float64 = 1.0 / 2.0
comptime B2: Float64 = -1.0 / 24.0
comptime C2: Float64 = 1.0 / 720.0
comptime D2: Float64 = -1.0 / 40320.0
comptime E2: Float64 = 1.0 / 3628800.0
comptime F2: Float64 = -1.0 / 479001600.0
comptime G2: Float64 = 1.0 / 87178291200.0
comptime H2: Float64 = -1.0 / 20922789888000.0
comptime I2: Float64 = 1.0 / 6402373705728000.0
comptime A3: Float64 = 1.0 / 6.0
comptime B3: Float64 = -1.0 / 120.0
comptime C3: Float64 = 1.0 / 5040.0
comptime D3: Float64 = -1.0 / 362880.0
comptime E3: Float64 = 1.0 / 39916800.0
comptime F3: Float64 = -1.0 / 6227020800.0
comptime G3: Float64 = 1.0 / 1307674368000.0
comptime H3: Float64 = -1.0 / 355687428096000.0
comptime I3: Float64 = 1.0 / 121645100408832000.0


def normpi(x: Ptr, n: Int, dst: Ptr):
    """Return `x` wrapped into the interval [-pi, pi)."""
    var j = 0
    while j < n:
        dst.unsafe_store(j, pymod(x.unsafe_load(j) + pi, tau) - pi)
        j += 1


def _sign(x: Float64) -> Float64:
    if x > 0.0:
        return 1.0
    if x < 0.0:
        return -1.0
    return 0.0


def eccentric_anomaly(e: Ptr, M: Ptr, n: Int, dst: Ptr):
    """Iterate to solve Kepler's equation to find the eccentric anomaly.

    See arXiv:2108.03215.  Upstream raises if the ten iterations fail to
    converge; here the last iterate is returned instead.
    """
    var j = 0
    while j < n:
        var ev = e.unsafe_load(j)
        var sign_M = _sign(normpi_one(M.unsafe_load(j)))
        var Mv = normpi_one(M.unsafe_load(j)) * sign_M
        var ebar = 0.25 * pi / ev - 1.0
        var E = 0.5 * pi * ebar * (_sign(ebar) * sqrt(1.0 + Mv / (ev * ebar * ebar)) - 1.0)
        for _ in range(10):
            var f1 = 1.0 - ev * cos(E)
            var f2 = ev * sin(E)
            var f = E - f2 - Mv
            var dE = f * f1 / (f1 * f1 - 0.5 * f * f2)
            E -= dE
            if abs(dE) < 1e-14:
                break
        dst.unsafe_store(j, E * sign_M)
        j += 1


def normpi_one(x: Float64) -> Float64:
    return pymod(x + pi, tau) - pi


def true_anomaly_hyperbolic(e: Ptr, E: Ptr, n: Int, dst: Ptr):
    """True anomaly from eccentricity and eccentric anomaly, hyperbolic orbits."""
    var j = 0
    while j < n:
        var ev = e.unsafe_load(j)
        dst.unsafe_store(j, 2.0 * atan(sqrt((ev + 1.0) / (ev - 1.0))
                                      * tanh(E.unsafe_load(j) / 2.0)))
        j += 1


def true_anomaly_closed(e: Ptr, E: Ptr, n: Int, dst: Ptr):
    """True anomaly from eccentricity and eccentric anomaly, closed orbits."""
    var j = 0
    while j < n:
        var ev = e.unsafe_load(j)
        dst.unsafe_store(j, 2.0 * atan(sqrt((1.0 + ev) / (1.0 - ev))
                                      * tan(E.unsafe_load(j) / 2.0)))
        j += 1


def true_anomaly_parabolic(p: Ptr, gm: Float64, M: Ptr, n: Int, dst: Ptr):
    """True anomaly from semi-latus rectum, gm, and mean anomaly."""
    var j = 0
    while j < n:
        var pv = p.unsafe_load(j)
        # from http://www.bogan.ca/orbits/kepler/orbteqtn.html
        var delta_t = sqrt(2.0 * pv * pv * pv / gm) * M.unsafe_load(j)
        var periapsis_distance = pv / 2.0
        var A = 3.0 / 2.0 * sqrt(gm / (2.0 * periapsis_distance
                                        * periapsis_distance * periapsis_distance)) * delta_t
        # Upstream is `(A*A + 1)**(1/3)`.  `pow(x, y)` with a fractional `y`
        # is `exp(y * log(x))` in this toolchain, which is ~1e-10 relative
        # rather than correctly rounded; `cbrt()` is a real cube root and
        # agrees with libm to the last bit.  The near-cancellation in
        # `B - 1/B` below turns that 1e-10 into a 3e-10 error in the
        # anomaly, which is two orders past float64's 1e-16.
        var B = cbrt(A * A + 1.0 + A)
        dst.unsafe_store(j, 2.0 * atan(B - 1.0 / B))
        j += 1


def ele_to_vec(p: Ptr, e: Ptr, i: Ptr, Om: Ptr, w: Ptr, v: Ptr, mu: Float64,
               n: Int, d_position: Ptr, d_velocity: Ptr):
    """Calculates state vectors from orbital elements.

    Upstream also raises if the true anomaly is out of range for a
    hyperbolic orbit, or the inclination is outside [0, pi]; those checks
    are left to the Python wrapper, which raises the same errors.
    """
    var j = 0
    while j < n:
        var pv = p.unsafe_load(j)
        var ev = e.unsafe_load(j)
        var iv = i.unsafe_load(j)
        var Omv = Om.unsafe_load(j)
        var wv = w.unsafe_load(j)
        var vv = v.unsafe_load(j)

        var r = pv / (1.0 + ev * cos(vv))
        var h = sqrt(pv * mu)
        var u = vv + wv

        var X = r * (cos(Omv) * cos(u) - sin(Omv) * sin(u) * cos(iv))
        var Y = r * (sin(Omv) * cos(u) + cos(Omv) * sin(u) * cos(iv))
        var Z = r * (sin(iv) * sin(u))

        var X_dot = X * h * ev / (r * pv) * sin(vv) - h / r * (cos(Omv) * sin(u)
                                                               + sin(Omv) * cos(u) * cos(iv))
        var Y_dot = Y * h * ev / (r * pv) * sin(vv) - h / r * (sin(Omv) * sin(u)
                                                               - cos(Omv) * cos(u) * cos(iv))
        var Z_dot = Z * h * ev / (r * pv) * sin(vv) + h / r * sin(iv) * cos(u)

        d_position.unsafe_store(0 * n + j, X)
        d_position.unsafe_store(1 * n + j, Y)
        d_position.unsafe_store(2 * n + j, Z)
        d_velocity.unsafe_store(0 * n + j, X_dot)
        d_velocity.unsafe_store(1 * n + j, Y_dot)
        d_velocity.unsafe_store(2 * n + j, Z_dot)
        j += 1




def _stumpff_one(xv: Float64) -> Tuple[Float64, Float64, Float64, Float64]:
    """One Stumpff function evaluation, from SPICE toolkit src/spicelib/stmp03.f."""
    var z = sqrt(abs(xv))
    var c0: Float64
    var c1: Float64
    var c2: Float64
    var c3: Float64

    if xv < -1.0:
        c0 = cosh(z)
        c1 = sinh(z) / z
    elif xv > 1.0:
        c0 = cos(z)
        c1 = sin(z) / z
    else:
        # mid: numerators run [x, -x, x, ...] raised to powers 0, 1, 2, ...
        # c2 = sum((-x)^k / (2k+2)!) and c3 = sum((-x)^k / (2k+3)!)
        # `xp` is x^k by repeated multiplication and the signs live in the
        # coefficients, so the sum is upstream's to the last bit.
        c2 = A2
        c3 = A3
        var xp = xv
        c2 += B2 * xp
        c3 += B3 * xp
        xp *= xv
        c2 += C2 * xp
        c3 += C3 * xp
        xp *= xv
        c2 += D2 * xp
        c3 += D3 * xp
        xp *= xv
        c2 += E2 * xp
        c3 += E3 * xp
        xp *= xv
        c2 += F2 * xp
        c3 += F3 * xp
        xp *= xv
        c2 += G2 * xp
        c3 += G3 * xp
        xp *= xv
        c2 += H2 * xp
        c3 += H3 * xp
        xp *= xv
        c2 += I2 * xp
        c3 += I3 * xp
        c1 = 1.0 - xv * c3
        c0 = 1.0 - xv * c2
        return (c0, c1, c2, c3)

    c2 = (1.0 - c0) / xv
    c3 = (1.0 - c1) / xv
    return (c0, c1, c2, c3)


def stumpff(x: Ptr, n: Int, d_c0: Ptr, d_c1: Ptr, d_c2: Ptr, d_c3: Ptr):
    """Calculates Stumpff functions.

    Based on the function toolkit/src/spicelib/stmp03.f from the SPICE
    toolkit, which can be downloaded from
    naif.jpl.nasa.gov/naif/toolkit_FORTRAN.html
    """
    var j = 0
    while j < n:
        var c = _stumpff_one(x.unsafe_load(j))
        d_c0.unsafe_store(j, c[0])
        d_c1.unsafe_store(j, c[1])
        d_c2.unsafe_store(j, c[2])
        d_c3.unsafe_store(j, c[3])
        j += 1


def propagate(position: Ptr, velocity: Ptr, t0: Float64, t1: Ptr, gm: Float64,
              n: Int, max_workers: Int, scratch: Ptr, flags: IPtr,
              d_position: Ptr, d_velocity: Ptr) -> Int:
    """Propagates a position and velocity vector with an array of times.

    Based on the function toolkit/src/spicelib/prop2b.f from the SPICE
    toolkit.  This is upstream's algorithm with the ``#orbits`` batch axis
    dropped: a single orbit turns the per-orbit quantities into scalars,
    and the loop over time is untouched.  Returns 1 if some time falls
    outside the range that can reliably be propagated, which the Python
    wrapper turns into upstream's ``ValueError``.

    `scratch` is fourteen length-`n` arrays back to back and `flags` is one
    integer per worker.  Both come from the caller because a parallel worker
    writes through them and a worker cannot be handed a list it owns; one
    flat buffer also lets a worker's whole slice of epochs sit in cache.

    There is a single launch, not one per bisection round: each worker runs
    the entire bracketing and bisection for its own contiguous slice of
    epochs, which is what keeps the cost down.  `max.algorithm.parallelize`
    starts its workers afresh on every call, so spreading the 130-odd inner
    loops over the cores costs more in thread startup than the cores return.
    """
    # The initial position and velocity are a single |xyz| pair each, so the
    # per-orbit arrays below all have one element.
    var r0 = 0.0
    var rv = 0.0
    var h2 = 0.0
    var eqvec = List[Float64]()
    var i = 0
    while i < 3:
        eqvec.append(0.0)
        i += 1

    r0 = length_of_one(position, 1, 0)
    rv = (position.unsafe_load(0) * velocity.unsafe_load(0)
          + position.unsafe_load(1) * velocity.unsafe_load(1)
          + position.unsafe_load(2) * velocity.unsafe_load(2))
    var hv = _cross_one(position, velocity, 1, 0)
    h2 = (hv[0] * hv[0] + hv[1] * hv[1] + hv[2] * hv[2])

    # eqvec = _cross(velocity, hvec)/gm + -position/r0
    var vh = _cross9(velocity.unsafe_load(0), velocity.unsafe_load(1),
                     velocity.unsafe_load(2), hv[0], hv[1], hv[2])
    eqvec[0] = vh[0] / gm - position.unsafe_load(0) / r0
    eqvec[1] = vh[1] / gm - position.unsafe_load(1) / r0
    eqvec[2] = vh[2] / gm - position.unsafe_load(2) / r0
    var e = sqrt(eqvec[0] * eqvec[0] + eqvec[1] * eqvec[1] + eqvec[2] * eqvec[2])
    var q = h2 / (gm * (1.0 + e))

    var f = 1.0 - e
    var b = sqrt(q / gm)

    var br0 = b * r0
    var b2rv = b * b * rv
    var bq = b * q
    var qovr0 = q / r0

    var maxc = max(max(abs(br0), abs(b2rv)), max(abs(bq), abs(qovr0 / bq)))

    var hyperbolic = f < 0.0
    var bound: Float64
    if hyperbolic:
        var fixed = log(dpmax / 2.0) - log(maxc)
        var rootf = sqrt(-f)
        var logf = log(-f)
        bound = min(fixed / rootf, (fixed + 1.5 * logf) / rootf)
    else:
        bound = exp((log(1.5) + log(dpmax) - log(maxc)) / 3.0)

    var dt = scratch
    var x = scratch.unsafe_offset(n)
    var kfun = scratch.unsafe_offset(2 * n)
    var lower = scratch.unsafe_offset(3 * n)
    var upper = scratch.unsafe_offset(4 * n)
    var lcount = scratch.unsafe_offset(5 * n)
    var mostc = scratch.unsafe_offset(6 * n)
    var not_done = scratch.unsafe_offset(7 * n)
    var past = scratch.unsafe_offset(8 * n)
    var future = scratch.unsafe_offset(9 * n)
    var c0 = scratch.unsafe_offset(10 * n)
    var c1 = scratch.unsafe_offset(11 * n)
    var c2 = scratch.unsafe_offset(12 * n)
    var c3 = scratch.unsafe_offset(13 * n)

    var (items, workers) = _plan(n, max_workers)
    _clear_flags(flags, workers)

    def work(i: Int) {imm}:
        var lo = i * n // workers
        var hi = (i + 1) * n // workers
        flags.unsafe_store(i, Int64(_propagate_slice(
            lo, hi, n, t0, t1, position, velocity, f, br0, b2rv, bq, qovr0,
            bound, dt, x, kfun, lower, upper, lcount, mostc, not_done, past,
            future, c0, c1, c2, c3, d_position, d_velocity)))

    if workers > 1:
        parallelize(work, items, workers)
    else:
        work(0)
    return 1 if _any_flag(flags, workers) else 0


def _propagate_slice(lo: Int, hi: Int, n: Int, t0: Float64, t1: Ptr,
                     position: Ptr, velocity: Ptr, f: Float64, br0: Float64,
                     b2rv: Float64, bq: Float64, qovr0: Float64,
                     bound: Float64, dt: Ptr, x: Ptr, kfun: Ptr, lower: Ptr,
                     upper: Ptr, lcount: Ptr, mostc: Ptr, not_done: Ptr,
                     past: Ptr, future: Ptr, c0: Ptr, c1: Ptr, c2: Ptr,
                     c3: Ptr, d_position: Ptr, d_velocity: Ptr) -> Int:
    """Propagate epochs `lo` through `hi - 1`, and return 1 if one of them
    fell outside the range that can reliably be propagated."""
    var j = lo
    while j < hi:
        var d = t1.unsafe_load(j) - t0
        dt.unsafe_store(j, d)
        var xv = clip(d / bq, -bound, bound)
        x.unsafe_store(j, xv)
        kfun.unsafe_store(j, 0.0)
        lcount.unsafe_store(j, 0.0)
        mostc.unsafe_store(j, 1000.0)
        not_done.unsafe_store(j, 0.0)
        var is_past = d < 0.0
        past.unsafe_store(j, 1.0 if is_past else 0.0)
        future.unsafe_store(j, 0.0 if is_past else (1.0 if d > 0.0 else 0.0))
        lower.unsafe_store(j, xv if is_past else 0.0)
        upper.unsafe_store(j, 0.0 if is_past else xv)
        c0.unsafe_store(j, 0.0)
        c1.unsafe_store(j, 0.0)
        c2.unsafe_store(j, 0.0)
        c3.unsafe_store(j, 0.0)
        j += 1

    # Bracket the root by doubling the interval that has not worked yet.
    # Upstream runs the past loop and the future loop separately; no epoch is
    # in both, and each doubles its own bracket until it stops moving, so one
    # loop reaches the same answer.  The "is any epoch still bracketing" test
    # rides along in the Stumpff pass that produces `kfun` instead of walking
    # the whole slice a second time to ask it.
    var any = _kepler_slice(lo, hi, f, br0, b2rv, bq, x, dt, past, future, kfun,
                            c0, c1, c2, c3)
    while any:
        any = False
        j = lo
        while j < hi:
            var old = x.unsafe_load(j)
            if past.unsafe_load(j) == 1.0:
                upper.unsafe_store(j, lower.unsafe_load(j))
                lower.unsafe_store(j, lower.unsafe_load(j) * 2.0)
                var nv = clip(lower.unsafe_load(j), -bound, bound)
                x.unsafe_store(j, nv)
                if nv == old:
                    return 1
                any = True
            elif future.unsafe_load(j) == 1.0:
                lower.unsafe_store(j, upper.unsafe_load(j))
                upper.unsafe_store(j, upper.unsafe_load(j) * 2.0)
                var nv = clip(upper.unsafe_load(j), -bound, bound)
                x.unsafe_store(j, nv)
                if nv == old:
                    return 1
                any = True
            j += 1
        if not any:
            break
        any = _kepler_slice(lo, hi, f, br0, b2rv, bq, x, dt, past, future, kfun,
                            c0, c1, c2, c3)

    j = lo
    while j < hi:
        var lo2 = lower.unsafe_load(j)
        var up = upper.unsafe_load(j)
        var mid = (up + lo2) / 2.0 if lo2 <= up else up
        x.unsafe_store(j, mid)
        not_done.unsafe_store(j, 1.0 if (lo2 < mid and mid < up) else 0.0)
        j += 1

    # Bisection, giving up on any element that has run 64 iterations.  The
    # "is anything still bisecting" test is the array the update writes, so it
    # is folded into the update rather than walked once more per round.
    while True:
        _ = _kepler_slice(lo, hi, f, br0, b2rv, bq, x, dt, past, future, kfun,
                          c0, c1, c2, c3)
        any = False
        j = lo
        while j < hi:
            if not_done.unsafe_load(j) == 1.0:
                var xv = x.unsafe_load(j)
                var k = kfun.unsafe_load(j)
                var d = dt.unsafe_load(j)
                var high = k > d
                var low = k < d
                var lc = lower.unsafe_load(j)
                var up = upper.unsafe_load(j)
                if high or not low:
                    up = xv
                if low or not high:
                    lc = xv

                var mc = mostc.unsafe_load(j)
                var cnt = lcount.unsafe_load(j)
                if mc > 64.0 and up != 0.0 and lc != 0.0:
                    mc = 64.0
                    cnt = 0.0

                var mid = up if lc > up else (up + lc) / 2.0
                x.unsafe_store(j, mid)
                lower.unsafe_store(j, lc)
                upper.unsafe_store(j, up)
                cnt += 1.0
                mostc.unsafe_store(j, mc)
                lcount.unsafe_store(j, cnt)
                var nd = 1.0 if (lc < mid and mid < up and cnt < mc) else 0.0
                not_done.unsafe_store(j, nd)
                if nd == 1.0:
                    any = True
            j += 1
        if not any:
            break

    _ = _kepler_slice(lo, hi, f, br0, b2rv, bq, x, dt, past, future, kfun,
                      c0, c1, c2, c3)

    j = lo
    while j < hi:
        var xv = x.unsafe_load(j)
        var g1 = c1.unsafe_load(j)
        var g2 = c2.unsafe_load(j)
        var g3 = c3.unsafe_load(j)
        var br = br0 * c0.unsafe_load(j) + xv * (b2rv * g1 + xv * bq * g2)

        var pc = 1.0 - qovr0 * xv * xv * g2
        var vc = dt.unsafe_load(j) - bq * xv * xv * xv * g3
        var pcdot = -qovr0 / br * xv * g1
        var vcdot = 1.0 - bq / br * xv * xv * g2

        var c = 0
        while c < 3:
            var p0 = position.unsafe_load(c)
            var v0 = velocity.unsafe_load(c)
            d_position.unsafe_store(c * n + j, pc * p0 + vc * v0)
            d_velocity.unsafe_store(c * n + j, pcdot * p0 + vcdot * v0)
            c += 1
        j += 1
    return 0


def _kepler_slice(lo: Int, hi: Int, f: Float64, br0: Float64, b2rv: Float64,
                  bq: Float64, x: Ptr, dt: Ptr, past: Ptr, future: Ptr,
                  kfun: Ptr, c0: Ptr, c1: Ptr, c2: Ptr, c3: Ptr) -> Bool:
    """Evaluate the Stumpff series for epochs `lo` through `hi - 1`.

    Returns whether any of them still has to have its bracket widened, which
    is the SPICE `kpast`/`kfuture` test.  It is answered here because this is
    the pass that produces `kfun`; upstream walks the whole array again to ask
    the same question.
    """
    var bracketing = False
    var j = lo
    while j < hi:
        # `x`, `kfun` and the c-arrays all live in `propagate`'s scratch.
        var xv = x.unsafe_load(j)
        var c = _stumpff_one(f * xv * xv)
        var g1 = c[1]
        var g2 = c[2]
        var g3 = c[3]
        c0.unsafe_store(j, c[0])
        c1.unsafe_store(j, g1)
        c2.unsafe_store(j, g2)
        c3.unsafe_store(j, g3)
        var k = xv * (br0 * g1 + xv * (b2rv * g2 + xv * bq * g3))
        kfun.unsafe_store(j, k)
        if past.unsafe_load(j) == 1.0:
            if k > dt.unsafe_load(j):
                bracketing = True
        elif future.unsafe_load(j) == 1.0:
            if k < dt.unsafe_load(j):
                bracketing = True
        j += 1
    return bracketing
