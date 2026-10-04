"""`skyfield.framelib` in Mojo."""

import numpy as np

from ._lib import addr, lib, out_mat
from .constants import ANGVEL, ASEC2RAD, DAY_S
# `T` is re-exported deliberately: `timelib.Time.MT` calls `framelib.T`, and
# upstream's `framelib` is the module that carries it.
from .functions import T, _flat, _lead_mat, mxm, mxmxm


def build_matrix():
    """The ICRS frame-bias matrix `B`, from IERS (2003) Conventions, Chapter 5."""
    dst = np.empty(9)
    lib().sk_build_matrix(addr(dst))
    return dst.reshape(3, 3)


ICRS_to_J2000 = build_matrix()

_itrs_angvel_matrix = np.array((
    (0.0, DAY_S * ANGVEL, 0.0),
    (-DAY_S * ANGVEL, 0.0, 0.0),
    (0.0, 0.0, 0.0),
))


def polar_motion_matrix(sprime, x, y):
    """Compute the 3x3 polar motion matrix from ``(sprime, x, y)``."""
    from .functions import rot_x, rot_y, rot_z
    return mxmxm(rot_x(y * ASEC2RAD), rot_y(x * ASEC2RAD),
                 rot_z(-sprime * ASEC2RAD))


def build_ecliptic_matrix(t):
    """Build the matrix to rotate an ICRF vector into ecliptic coordinates."""
    M, _ = _lead_mat(t.M)
    n = M.shape[2]
    true_obliquity, _ = _flat(t.true_obliquity)
    dst = out_mat(n)
    scratch = np.empty(9 * n)
    lib().sk_build_ecliptic_matrix(addr(M), addr(true_obliquity), n, addr(dst),
                                   addr(scratch))
    return dst[:, :, 0] if n == 1 else dst


class ICRS(object):
    """The International Celestial Reference System (ICRS)."""

    @staticmethod
    def rotation_at(t):
        return _identity


_identity = np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)], dtype=np.float64)


class mean_equinox_of_date(object):
    """The coordinate frame of Earth's mean equator and equinox."""

    @staticmethod
    def rotation_at(t):
        return mxm(t.precession_matrix(), ICRS_to_J2000)


class true_equinox_of_date(object):
    """The dynamical frame of Earth's true equator and true equinox of date."""

    @staticmethod
    def rotation_at(t):
        return t.M


class tirs(object):
    """The Terrestrial Intermediate Reference System (TIRS).

    Like the ITRS this frame accounts for precession and nutation, but
    neglects polar motion and the TIO locator.
    """

    @staticmethod
    def rotation_at(t):
        M, _ = _lead_mat(t.M)
        n = M.shape[2]
        gast, _ = _flat(t.gast)
        dst = out_mat(n)
        scratch = np.empty(9 * n)
        lib().sk_tirs_rotation_at(addr(M), addr(gast), n, addr(dst),
                                  addr(scratch))
        return dst[:, :, 0] if n == 1 else dst

    @staticmethod
    def _dRdt_times_RT_at(t):
        return _itrs_angvel_matrix


class itrs(object):
    """The International Terrestrial Reference System (ITRS)."""

    @staticmethod
    def rotation_at(t, polar_motion=None):
        if polar_motion is None:
            return tirs.rotation_at(t)
        M, _ = _lead_mat(t.M)
        n = M.shape[2]
        gast, _ = _flat(t.gast)
        W, _ = _lead_mat(polar_motion)
        # `W` is the only caller-supplied array here, and the kernel reads it
        # as a (3, 3, n) stack with stride n -- so a bare (3, 3) or a
        # shallower stack would be read out of bounds.  Broadcast it up.
        W = np.ascontiguousarray(np.broadcast_to(W, (3, 3, n)))
        dst = out_mat(n)
        scratch = np.empty(9 * n)
        scratch2 = np.empty(9 * n)
        lib().sk_itrs_rotation_at(addr(M), addr(gast), addr(W), n, addr(dst),
                                  addr(scratch), addr(scratch2))
        return dst[:, :, 0] if n == 1 else dst

    @staticmethod
    def _dRdt_times_RT_at(t):
        return _itrs_angvel_matrix


class ecliptic_frame(object):
    """Reference frame of the true ecliptic and equinox of date."""

    @staticmethod
    def rotation_at(t):
        return build_ecliptic_matrix(t)


class InertialFrame(object):
    def __init__(self, matrix):
        self._matrix = matrix

    def rotation_at(self, t):
        return self._matrix
