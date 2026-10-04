"""`skyfield.timelib` in Mojo.

The calendar arithmetic, the TDB-TT offset, and the `Time` object's
reified properties.  Upstream's `Time` also carries the leap-second and
Delta-T tables that turn a UTC calendar date into a Julian date; those are
plain data lookups that were not ported, so a `Time` here is built from
Julian dates directly.  See the README's coverage section.
"""

import numpy as np

from ._lib import addr, lib, sentinel
from .constants import ASEC2RAD, DAY_S, T0
from .functions import _flat


def julian_day(year, month=1, day=1, julian_before=None):
    """Given a calendar date, return a Julian day integer.

    Uses the proleptic Gregorian calendar unless ``julian_before`` is set
    to a specific Julian day, in which case the Julian calendar is used
    for dates older than that.
    """
    year, scalar, n = _lead_int(year)
    month, _, _ = _match(month, year, n)
    day, _, _ = _match(day, year, n)
    dst = np.empty(n)
    lib().sk_julian_day(addr(year), addr(month), addr(day),
                        sentinel(julian_before), n, addr(dst))
    return float(dst[0]) if scalar else dst


def julian_date(year, month=1, day=1, hour=0, minute=0, second=0.0):
    """Given a proleptic Gregorian calendar date and time, build a Julian date."""
    year, scalar, n = _lead_int(year)
    month, _, _ = _match(month, year, n)
    day, _, _ = _match(day, year, n)
    hour, _, _ = _match(hour, year, n)
    minute, _, _ = _match(minute, year, n)
    second, _, _ = _match(second, year, n)
    dst = np.empty(n)
    lib().sk_julian_date(addr(year), addr(month), addr(day), addr(hour),
                         addr(minute), addr(second), -1.0, n, addr(dst))
    return float(dst[0]) if scalar else dst


def julian_date_of_besselian_epoch(b):
    b, scalar = _flat(b)
    dst = np.empty_like(b)
    lib().sk_julian_date_of_besselian_epoch(addr(b), b.size, addr(dst))
    return float(dst[0]) if scalar else dst


def compute_calendar_date(jd_integer, julian_before=None):
    """Convert Julian day ``jd_integer`` into a calendar (year, month, day)."""
    jd, scalar = _flat(jd_integer)
    n = jd.size
    year, month, day = np.empty(n), np.empty(n), np.empty(n)
    lib().sk_compute_calendar_date(addr(jd), sentinel(julian_before), n,
                                   addr(year), addr(month), addr(day))
    if scalar:
        return int(year[0]), int(month[0]), int(day[0])
    return year.astype(np.int64), month.astype(np.int64), day.astype(np.int64)


calendar_date = compute_calendar_date


def calendar_tuple(jd_float, fraction=0.0, julian_before=None):
    """Return a (year, month, day, hour, minute, second.fraction) tuple."""
    jd, scalar = _flat(jd_float)
    n = jd.size
    fraction = np.ascontiguousarray(
        np.broadcast_to(np.asarray(fraction, dtype=np.float64), (n,)))
    out = [np.empty(n) for _ in range(6)]
    lib().sk_calendar_tuple(addr(jd), addr(fraction), sentinel(julian_before), n,
                            addr(out[0]), addr(out[1]), addr(out[2]),
                            addr(out[3]), addr(out[4]), addr(out[5]))
    if scalar:
        return (int(out[0][0]), int(out[1][0]), int(out[2][0]), int(out[3][0]),
                int(out[4][0]), float(out[5][0]))
    return (out[0].astype(np.int64), out[1].astype(np.int64),
            out[2].astype(np.int64), out[3].astype(np.int64),
            out[4].astype(np.int64), out[5])


def tdb_minus_tt(jd_tdb, fraction_tdb=0.0):
    """Computes how far TDB is in advance of TT, given TDB."""
    jd, scalar = _flat(jd_tdb)
    n = jd.size
    fraction = np.ascontiguousarray(
        np.broadcast_to(np.asarray(fraction_tdb, dtype=np.float64), (n,)))
    dst = np.empty(n)
    lib().sk_tdb_minus_tt(addr(jd), addr(fraction), n, addr(dst))
    return float(dst[0]) if scalar else dst


def _lead_int(a):
    arr = np.asarray(a, dtype=np.float64)
    scalar = arr.ndim == 0
    return np.ascontiguousarray(arr).reshape(-1), scalar, arr.size


def _match(a, like, n):
    return np.ascontiguousarray(
        np.broadcast_to(np.asarray(a, dtype=np.float64), (n,))), False, n


class Time(object):
    """A single moment in history, or an array of several moments.

    Upstream builds a `Time` from a `Timescale` and a calendar date; the
    leap-second and Delta-T tables behind that step are not part of this
    port, so a `Time` here is built from Julian dates.  Every property
    upstream computes from nutation, precession and sidereal time is
    present, in Mojo.
    """

    def __init__(self, whole, tt_fraction=0.0, ut1_fraction=None,
                 tdb_fraction=None):
        self.whole, whole_scalar, self.n = _lead_int(whole)
        self.tt_fraction = np.ascontiguousarray(
            np.broadcast_to(np.asarray(tt_fraction, dtype=np.float64), (self.n,)))
        if ut1_fraction is None:
            # Without a Delta-T table the default is dut1 = 0.
            self.ut1_fraction = self.tt_fraction.copy()
        else:
            self.ut1_fraction = np.ascontiguousarray(
                np.broadcast_to(np.asarray(ut1_fraction, dtype=np.float64),
                                (self.n,)))
        if tdb_fraction is None:
            offset = tdb_minus_tt(self.whole, self.tt_fraction)
            self.tdb_fraction = (self.tt_fraction + offset / DAY_S
                                 if not whole_scalar
                                 else float(self.tt_fraction[0] + offset / DAY_S))
            if whole_scalar:
                self.tdb_fraction = np.array([self.tdb_fraction])
        else:
            self.tdb_fraction = np.ascontiguousarray(
                np.broadcast_to(np.asarray(tdb_fraction, dtype=np.float64),
                                (self.n,)))
        self._cache = {}

    @classmethod
    def from_julian_date(cls, jd, ut1_fraction=None):
        """Build a `Time` from a Terrestrial Time Julian date."""
        jd, scalar = _flat(jd)
        whole = np.floor(jd)
        fraction = jd - whole
        return cls(whole, fraction, ut1_fraction)

    def __len__(self):
        return self.n

    def __repr__(self):
        if self.n == 1:
            return f"<Time tt={float(self.tt[0])!r}>"
        return f"<Time tt={self.tt[0]!r} ... {self.tt[-1]!r} len={self.n}>"

    @property
    def tt(self):
        return self.whole + self.tt_fraction

    @property
    def tdb(self):
        return self.whole + self.tdb_fraction

    @property
    def ut1(self):
        return self.whole + self.ut1_fraction

    @property
    def _mean_obliquity_radians(self):
        return mean_obliquity(self.tdb) * ASEC2RAD

    def _nutation_angles_radians(self):
        from . import nutationlib
        return nutationlib.iau2000a_radians(self)

    @property
    def gmst(self):
        """Greenwich Mean Sidereal Time (GMST) in hours."""
        from . import earthlib
        return earthlib.sidereal_time(self)

    @property
    def gast(self):
        """Greenwich Apparent Sidereal Time (GAST) in hours."""
        from . import earthlib, nutationlib
        d_psi, _ = nutationlib.iau2000a_radians(self)
        c_terms = nutationlib.equation_of_the_equinoxes_complimentary_terms(
            self.tt)
        eq_eq = d_psi * np.cos(self._mean_obliquity_radians) + c_terms
        gmst = self.gmst
        return (gmst + eq_eq / (2.0 * np.pi) * 24.0) % 24.0

    def precession_matrix(self):
        """Compute the 3x3 precession matrix P for this date."""
        from . import precessionlib
        return precessionlib.compute_precession(self.tdb)

    def nutation_matrix(self):
        """Compute the 3x3 nutation matrix N for this date."""
        from . import nutationlib
        d_psi, d_eps = nutationlib.iau2000a_radians(self)
        mean_obliquity = self._mean_obliquity_radians
        true_obliquity = mean_obliquity + d_eps
        return nutationlib.build_nutation_matrix(mean_obliquity, true_obliquity,
                                                 d_psi)

    def polar_motion_angles(self, xp, yp):
        """Return ``(sprime, x, y)`` for a pair of IERS polar angles."""
        sprime = -47.0e-6 * (self.tdb - T0) / 36525.0
        return sprime, np.broadcast_to(np.asarray(xp, dtype=np.float64),
                                        (self.n,)), \
            np.broadcast_to(np.asarray(yp, dtype=np.float64), (self.n,))

    def polar_motion_matrix(self, xp, yp):
        """Compute the 3x3 polar motion matrix for this date."""
        from . import framelib
        sprime, x, y = self.polar_motion_angles(xp, yp)
        return framelib.polar_motion_matrix(
            sprime, np.ascontiguousarray(x), np.ascontiguousarray(y))

    @property
    def M(self):
        """3x3 rotation matrix: ICRS -> equinox of this date."""
        from . import framelib
        return framelib.mxmxm(self.nutation_matrix(),
                              self.precession_matrix(), framelib.ICRS_to_J2000)

    @property
    def MT(self):
        """3x3 rotation matrix: equinox of this date -> ICRS."""
        from . import framelib
        return framelib.T(self.M)

    @property
    def P(self):
        return self.precession_matrix()

    @property
    def N(self):
        return self.nutation_matrix()

    @property
    def true_obliquity(self):
        _, d_eps = self._nutation_angles_radians()
        return self._mean_obliquity_radians + d_eps


from .nutationlib import mean_obliquity  # noqa: E402  (used by Time)
