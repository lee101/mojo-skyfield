"""`skyfield.precessionlib` in Mojo."""

from ._lib import addr, lib, out_mat, workers
from .functions import _flat


def compute_precession(jd_tdb):
    """Return the rotation matrices for precessing to an array of epochs.

    `jd_tdb` - array of TDB Julian dates

    The array returned has the shape `(3, 3, n)` where `n` is the number
    of dates that have been provided as input.
    """
    jd, scalar = _flat(jd_tdb)
    dst = out_mat(jd.size)
    lib().sk_compute_precession(addr(jd), jd.size, workers(), addr(dst))
    return dst[:, :, 0] if scalar else dst
