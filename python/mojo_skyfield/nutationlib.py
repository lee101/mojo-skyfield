"""`skyfield.nutationlib` in Mojo."""

import numpy as np

from . import _tables
from ._lib import addr, lib, out_mat, workers
from .functions import _flat, _follow_flat

ASEC2RAD = 4.848136811095359935899141e-6
T0 = 2451545.0
_TENTH_USEC_2_RAD = ASEC2RAD / 1e7


def iau2000a_radians(t, fundamental_argument_terms=5, lunisolar_terms=687,
                     planetary_terms=687):
    """Return the IAU 2000A angles delta-psi and delta-epsilon in radians."""
    d_psi, d_eps = iau2000a(t.tt, fundamental_argument_terms, lunisolar_terms,
                            planetary_terms)
    return d_psi * _TENTH_USEC_2_RAD, d_eps * _TENTH_USEC_2_RAD


def iau2000b_radians(t):
    """Return the IAU 2000B angles delta-psi and delta-epsilon in radians."""
    d_psi, d_eps = iau2000b(t.tt)
    return d_psi * _TENTH_USEC_2_RAD, d_eps * _TENTH_USEC_2_RAD


def build_nutation_matrix(mean_obliquity_radians, true_obliquity_radians,
                          psi_radians):
    """Generate the nutation rotation matrix, given three nutation parameters."""
    psi, scalar = _flat(psi_radians)
    n = psi.size
    mean_obliquity = _follow_flat(psi, mean_obliquity_radians, n)
    true_obliquity = _follow_flat(psi, true_obliquity_radians, n)
    dst = out_mat(n)
    lib().sk_build_nutation_matrix(addr(mean_obliquity), addr(true_obliquity),
                                   addr(psi), n, addr(dst))
    return dst[:, :, 0] if scalar else dst


def mean_obliquity(jd_tdb):
    """Return the mean obliquity of the ecliptic in arcseconds."""
    jd, scalar = _flat(jd_tdb)
    dst = np.empty_like(jd)
    lib().sk_mean_obliquity(addr(jd), jd.size, addr(dst))
    return float(dst[0]) if scalar else dst


def equation_of_the_equinoxes_complimentary_terms(jd_tt):
    """Compute the complementary terms of the equation of the equinoxes."""
    jd, scalar = _flat(jd_tt)
    dst = np.empty_like(jd)
    lib().sk_equation_of_the_equinoxes_complimentary_terms(
        addr(jd), addr(_tables.ke0_t), addr(_tables.ke1),
        addr(_tables.se0_t_0), addr(_tables.se0_t_1),
        _tables.ke0_t.shape[0], jd.size, workers(), addr(dst))
    return float(dst[0]) if scalar else dst


def fundamental_arguments(t, terms=5):
    """Compute the fundamental arguments (mean elements) of Sun and Moon."""
    t, scalar = _flat(t)
    n = t.size
    dst = np.empty((5, n))
    lib().sk_fundamental_arguments(addr(t), terms, n, workers(), addr(dst))
    # Upstream drops the trailing axis for a scalar `t`; keep that shape.
    return dst[:, 0] if scalar else dst


def iau2000a(jd_tt, fundamental_argument_terms=5, lunisolar_terms=687,
             planetary_terms=687):
    """Compute Earth nutation based on the IAU 2000A nutation model.

    ``jd_tt`` - Terrestrial Time: Julian date float, or NumPy array of floats

    Returns a tuple ``(delta_psi, delta_epsilon)`` measured in tenths of
    a micro-arcsecond.
    """
    return _iau2000a("sk_iau2000a", jd_tt, fundamental_argument_terms,
                     lunisolar_terms, planetary_terms)


def iau2000b(jd_tt):
    """Compute Earth nutation based on the faster IAU 2000B nutation model.

    Returns a tuple ``(delta_psi, delta_epsilon)`` measured in tenths of
    a micro-arcsecond.  The result agrees with ``iau2000a()`` to within a
    milliarcsecond between the years 1995 and 2020.
    """
    return _iau2000a("sk_iau2000b", jd_tt, 2, 77, 0)


def _iau2000a(kernel, jd_tt, fundamental_argument_terms, lunisolar_terms,
              planetary_terms):
    jd, scalar = _flat(jd_tt)
    n = jd.size
    # `nals_t[:cutoff]` and friends: NumPy slicing, replicated here so the
    # kernel never reads past the end of a table.
    ls = min(lunisolar_terms, _tables.nals_t.shape[0])
    pl = min(planetary_terms, _tables.napl_t.shape[0])
    dpsi, deps = np.empty(n), np.empty(n)
    getattr(lib(), kernel)(
        addr(jd), fundamental_argument_terms, ls, pl,
        addr(_tables.nals_t), addr(_tables.lunisolar_longitude_coefficients),
        addr(_tables.lunisolar_obliquity_coefficients), addr(_tables.napl_t),
        addr(_tables.nutation_coefficients_longitude),
        addr(_tables.nutation_coefficients_obliquity), n, workers(),
        addr(dpsi), addr(deps))
    if scalar:
        return float(dpsi[0]), float(deps[0])
    return dpsi, deps
