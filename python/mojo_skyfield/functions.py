"""`skyfield.functions` in Mojo.

Upstream names, upstream argument order, upstream array shapes: a |xyz|
vector is ``(3,)`` or ``(3, n)`` and a 3x3 rotation matrix is ``(3, 3)``
or ``(3, 3, n)``, exactly as NumPy lays them out.  The first argument sets
``n``; the others are broadcast to match, as NumPy would.
"""

import numpy as np

from ._lib import addr, lib, out_mat, out_vec, squeeze

_AVOID_DIVIDE_BY_ZERO = np.finfo(np.float64).tiny


def _lead_vec(a):
    """`a` as a (3, n) C-contiguous float64 array, plus its original ndim."""
    arr = np.asarray(a, dtype=np.float64)
    ndim = arr.ndim
    if ndim == 1:
        arr = arr.reshape(3, 1)
    elif ndim != 2 or arr.shape[0] != 3:
        raise ValueError(f"expected shape (3,) or (3, n), got {arr.shape}")
    return np.ascontiguousarray(arr), ndim


def _follow_vec(like, a):
    """`a` broadcast to `like`'s shape, C-contiguous."""
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 1 and arr.shape[0] == 3:
        arr = arr.reshape(3, 1)
    return np.ascontiguousarray(np.broadcast_to(arr, like.shape))


def _lead_mat(a):
    arr = np.asarray(a, dtype=np.float64)
    ndim = arr.ndim
    if ndim == 2:
        arr = arr.reshape(3, 3, 1)
    elif ndim != 3 or arr.shape[:2] != (3, 3):
        raise ValueError(f"expected shape (3, 3) or (3, 3, n), got {arr.shape}")
    return np.ascontiguousarray(arr), ndim



def _flat(a):
    """`a` as a 1-D C-contiguous array, plus a flag for scalar input."""
    arr = np.asarray(a, dtype=np.float64)
    scalar = arr.ndim == 0
    return np.ascontiguousarray(arr).reshape(-1), scalar


def _rot(kernel, theta):
    theta, scalar = _flat(theta)
    n = theta.size
    dst = out_mat(n)
    getattr(lib(), kernel)(addr(theta), n, addr(dst))
    return dst[:, :, 0] if scalar else dst


def sqrt_nan(n):
    """Return the square root of ``n``, or ``nan`` if ``n < 0``."""
    a, scalar = _flat(n)
    dst = np.empty_like(a)
    lib().sk_sqrt_nan(addr(a), a.size, addr(dst))
    return squeeze(dst, scalar)


def dots(v, u):
    """Given one or more vectors in `v` and `u`, return their dot products."""
    v, ndim = _lead_vec(v)
    u = _follow_vec(v, u)
    n = v.shape[1]
    dst = np.empty(n)
    lib().sk_dots(addr(v), addr(u), n, addr(dst))
    return float(dst[0]) if ndim == 1 else dst


def T(M):
    """Swap the first two dimensions of an array."""
    m, ndim = _lead_mat(M)
    n = m.shape[2]
    dst = out_mat(n)
    lib().sk_T(addr(m), n, addr(dst))
    return dst[:, :, 0] if ndim == 2 else dst


def mxv(M, v):
    """Matrix times vector: multiply an NxN matrix by a vector."""
    v, vdim = _lead_vec(v)
    m, _ = _lead_mat(M)
    # Upstream is `einsum('ij...,j...->i...')`, which broadcasts the trailing
    # axes against each other rather than letting the first argument decide
    # `n`: a (3, 3) matrix times a (3, n) vector is an (3, n) product.
    n = max(m.shape[2], v.shape[1])
    m = np.ascontiguousarray(np.broadcast_to(m, (3, 3, n)))
    v = np.ascontiguousarray(np.broadcast_to(v, (3, n)))
    dst = out_vec(n)
    lib().sk_mxv(addr(m), addr(v), n, addr(dst))
    return dst[:, 0] if vdim == 1 and n == 1 else dst


def mxm(M1, M2):
    """Matrix times matrix: multiply two NxN matrices."""
    m1, ndim = _lead_mat(M1)
    m2 = _lead_mat(M2)[0]
    n = max(m1.shape[2], m2.shape[2])
    m1 = np.ascontiguousarray(np.broadcast_to(m1, (3, 3, n)))
    m2 = np.ascontiguousarray(np.broadcast_to(m2, (3, 3, n)))
    dst = out_mat(n)
    lib().sk_mxm(addr(m1), addr(m2), n, addr(dst))
    return dst[:, :, 0] if n == 1 else dst


def mxmxm(M1, M2, M3):
    """Matrix times matrix times matrix: multiply 3 NxN matrices together."""
    mats = [_lead_mat(a)[0] for a in (M1, M2, M3)]
    n = max(m.shape[2] for m in mats)
    mats = [np.ascontiguousarray(np.broadcast_to(m, (3, 3, n))) for m in mats]
    dst = out_mat(n)
    lib().sk_mxmxm(addr(mats[0]), addr(mats[1]), addr(mats[2]), n, addr(dst))
    return dst[:, :, 0] if n == 1 else dst


def length_of(xyz):
    """Given a 3-element array |xyz|, return its length."""
    v, ndim = _lead_vec(xyz)
    n = v.shape[1]
    dst = np.empty(n)
    lib().sk_length_of(addr(v), n, addr(dst))
    return float(dst[0]) if ndim == 1 else dst


def angle_between(u, v):
    """Given two vectors `v` and `u`, return the radian angle separating them."""
    u, ndim = _lead_vec(u)
    v = _follow_vec(u, v)
    n = u.shape[1]
    dst = np.empty(n)
    lib().sk_angle_between(addr(u), addr(v), n, addr(dst))
    return float(dst[0]) if ndim == 1 else dst


def to_spherical(xyz):
    """Convert |xyz| to spherical coordinates (r,theta,phi)."""
    v, ndim = _lead_vec(xyz)
    n = v.shape[1]
    r, theta, phi = np.empty(n), np.empty(n), np.empty(n)
    lib().sk_to_spherical(addr(v), n, addr(r), addr(theta), addr(phi))
    if ndim == 1:
        return float(r[0]), float(theta[0]), float(phi[0])
    return r, theta, phi


def from_spherical(r, theta, phi):
    """Convert (r,theta,phi) to Cartesian coordinates |xyz|."""
    raw = np.asarray(r, dtype=np.float64)
    scalar = raw.ndim == 0
    r = np.ascontiguousarray(raw.reshape(-1))
    n = r.size
    theta = np.ascontiguousarray(
        np.broadcast_to(np.asarray(theta, dtype=np.float64), (n,)))
    phi = np.ascontiguousarray(
        np.broadcast_to(np.asarray(phi, dtype=np.float64), (n,)))
    dst = out_vec(n)
    lib().sk_from_spherical(addr(r), addr(theta), addr(phi), n, addr(dst))
    return dst[:, 0] if scalar else dst


def rot_x(theta):
    return _rot("sk_rot_x", theta)


def rot_y(theta):
    return _rot("sk_rot_y", theta)


def rot_z(theta):
    return _rot("sk_rot_z", theta)


def R1(theta):
    """Left-handed rotation about x, per the Explanatory Supplement."""
    return _rot("sk_R1", theta)


def R2(theta):
    """Left-handed rotation about y, per the Explanatory Supplement."""
    return _rot("sk_R2", theta)


def R3(theta):
    """Left-handed rotation about z, per the Explanatory Supplement."""
    return _rot("sk_R3", theta)


def angular_velocity_matrix(angular_velocity_vector):
    v, ndim = _lead_vec(angular_velocity_vector)
    n = v.shape[1]
    dst = out_mat(n)
    lib().sk_angular_velocity_matrix(addr(v), n, addr(dst))
    return dst[:, :, 0] if ndim == 1 else dst


to_polar = to_spherical
from_polar = from_spherical


def _follow_flat(like, a, n):
    """`a` broadcast to a 1-D array of length `n`."""
    return np.ascontiguousarray(
        np.broadcast_to(np.asarray(a, dtype=np.float64), (n,)))
