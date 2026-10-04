"""`skyfield.sgp4lib` in Mojo.

Upstream's `EarthSatellite` delegates the actual orbit integration to the
separate `sgp4` package; what lives here is the TEME-to-ITRF frame
transform, which this port covers in full.  See the README's coverage
section.
"""

import numpy as np

from ._lib import addr, lib, out_mat, out_vec, workers
from .functions import _flat, _follow_vec, _lead_mat, _lead_vec

_zero_zero_minus_one = np.array((0.0, 0.0, -1.0))


def _cross(a, b):
    """The cross product, spelled the way upstream spells it."""
    a, ndim = _lead_vec(a)
    n = a.shape[1]
    b = _follow_vec(a, b)
    dst = out_vec(n)
    lib().sk_cross(addr(a), addr(b), n, addr(dst))
    return dst[:, 0] if ndim == 1 else dst


cross = _cross


def theta_GMST1982(jd_ut1, fraction_ut1=0.0):
    """Return the angle of Greenwich Mean Standard Time 1982 given the JD.

    From AIAA 2006-6753 Appendix C.
    """
    jd, scalar = _flat(jd_ut1)
    n = jd.size
    frac = np.ascontiguousarray(
        np.broadcast_to(np.asarray(fraction_ut1, dtype=np.float64), (n,)))
    theta, theta_dot = np.empty(n), np.empty(n)
    lib().sk_theta_GMST1982(addr(jd), addr(frac), n, addr(theta),
                            addr(theta_dot))
    if scalar:
        return float(theta[0]), float(theta_dot[0])
    return theta, theta_dot


def TEME_to_ITRF(jd_ut1, rTEME, vTEME, xp=0.0, yp=0.0, fraction_ut1=0.0):
    """Rotate SGP4's TEME vectors into the ITRS frame."""
    r, ndim = _lead_vec(rTEME)
    n = r.shape[1]
    v = _follow_vec(r, vTEME)
    jd, _ = _flat(jd_ut1)
    jd = np.ascontiguousarray(np.broadcast_to(jd, (n,)))
    frac = np.ascontiguousarray(
        np.broadcast_to(np.asarray(fraction_ut1, dtype=np.float64), (n,)))
    rITRF, vITRF = out_vec(n), out_vec(n)
    lib().sk_TEME_to_ITRF(
        addr(jd), addr(r), addr(v), float(xp), float(yp), addr(frac), n,
        workers(), addr(rITRF), addr(vITRF))
    if ndim == 1:
        return rITRF[:, 0], vITRF[:, 0]
    return rITRF, vITRF


def TEME_rotation_at(t, theta=None):
    """`TEME.rotation_at(t)`: mxm(rot_z(theta - gast / 24 * tau), t.M)."""
    M, _ = _lead_mat(t.M)
    n = M.shape[2]
    if theta is None:
        theta, _ = theta_GMST1982(t.whole, t.ut1_fraction)
    # `theta` defaults to this function's own float64 output; a caller may
    # override it.  Either way the dtype has to be pinned before the address
    # goes out: the kernel reads the buffer as `Float64`, so an int32 or
    # float32 array would be re-read 8 bytes per element -- silently wrong
    # angles, and past the end of the buffer.  Upstream takes no `theta` at
    # all, so there is no upstream shape to match here beyond `n`.
    theta, _ = _flat(np.broadcast_to(theta, (n,)))
    gast, _ = _flat(np.broadcast_to(_flat(t.gast)[0], (n,)))
    dst = out_mat(n)
    scratch = np.empty(9 * n)
    lib().sk_TEME_rotation_at(addr(theta), addr(gast), addr(M), n, addr(dst),
                               addr(scratch))
    return dst[:, :, 0] if n == 1 else dst
