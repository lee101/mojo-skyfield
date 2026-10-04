"""`skyfield.keplerlib` in Mojo.

Upstream's `propagate()` accepts a batch of orbits and an array of times
and returns positions of shape ``(3, norbits, n)``.  This port keeps a
single orbit, so it accepts one position/velocity pair and returns
``(3, n)``.  See the README's coverage section.
"""

import numpy as np

from ._lib import addr, lib, out_vec, scratch_i64, workers
from .functions import _flat, _follow_flat, _lead_vec


# dt, x, kfun, lower, upper, lcount, mostc, not_done, past, future, c0..c3.
_PROPAGATE_ARRAYS = 14

# Upstream's module-level `dpmax`, kept so `from mojo_skyfield.keplerlib import
# dpmax` works against code written for upstream. Nothing here uses it: the
# kernel carries its own `dpmax`, and this wrapper has no path that needs it.
dpmax = np.finfo(np.float64).max


def normpi(x):
    """Return `x` wrapped into the interval [-pi, pi)."""
    x, scalar = _flat(x)
    dst = np.empty_like(x)
    lib().sk_normpi(addr(x), x.size, addr(dst))
    return float(dst[0]) if scalar else dst


def eccentric_anomaly(e, M):
    """Iterate to solve Kepler's equation to find the eccentric anomaly."""
    e, scalar = _flat(e)
    n = e.size
    Mv = _follow_flat(e, M, n)
    dst = np.empty(n)
    lib().sk_eccentric_anomaly(addr(e), addr(Mv), n, addr(dst))
    return float(dst[0]) if scalar else dst


def true_anomaly_hyperbolic(e, E):
    """Calculates true anomaly from eccentricity and eccentric anomaly."""
    e, scalar = _flat(e)
    n = e.size
    E = _follow_flat(e, E, n)
    dst = np.empty(n)
    lib().sk_true_anomaly_hyperbolic(addr(e), addr(E), n, addr(dst))
    return float(dst[0]) if scalar else dst


def true_anomaly_closed(e, E):
    """Calculates true anomaly from eccentricity and eccentric anomaly."""
    e, scalar = _flat(e)
    n = e.size
    E = _follow_flat(e, E, n)
    dst = np.empty(n)
    lib().sk_true_anomaly_closed(addr(e), addr(E), n, addr(dst))
    return float(dst[0]) if scalar else dst


def true_anomaly_parabolic(p, gm, M):
    """Calculates true anomaly from semi-latus rectum, gm, and mean anomaly."""
    p, scalar = _flat(p)
    n = p.size
    Mv = _follow_flat(p, M, n)
    dst = np.empty(n)
    lib().sk_true_anomaly_parabolic(addr(p), float(gm), addr(Mv), n, addr(dst))
    return float(dst[0]) if scalar else dst


def ele_to_vec(p, e, i, Om, w, v, mu):
    """Calculates state vectors from orbital elements."""
    # Upstream broadcasts the six element vectors against *each other*, so
    # `ele_to_vec(7000.0, [0.1, 0.2], ...)` is legal and returns two states.
    # Take `n` as the longest of the six, which is what upstream's own
    # broadcasting gives, and repeat the shorter ones up to it: the guards
    # below compare the six against each other, and the kernel indexes all of
    # them as length-`n` runs, so a short one would read past its own end.
    # (`_follow_flat(a, b, n)` is not usable here -- its `like` argument would
    # be a one-element buffer, and it would rebuild `n` from that.)
    flat = [_flat(a) for a in (p, e, i, Om, w, v)]
    n = max(a.size for a, _ in flat)
    scalar = all(flag for _, flag in flat)
    p, e, inc, node, argp, nu = (
        a if a.size == n else np.ascontiguousarray(np.broadcast_to(a, (n,)))
        for a, _ in flat)

    # Upstream's guards, kept here because the kernel does not raise.  The
    # `arccos` is evaluated only where `e > 1`, as upstream does: taking it
    # over the whole array would divide by zero at `e == 0` and warn.
    hyperbolic = e > 1.0
    if np.any(hyperbolic & (nu > np.arccos(-1.0 / np.where(hyperbolic, e, 1.0)))):
        raise ValueError('If eccentricity is >1, abs(true anomaly) cannot be '
                         'more than arccos(-1/e)')
    if not np.all((inc >= 0.0) & (inc <= np.pi)):
        raise ValueError('Inclination outside the range [0, pi] radians')

    position, velocity = out_vec(n), out_vec(n)
    lib().sk_ele_to_vec(addr(p), addr(e), addr(inc), addr(node), addr(argp),
                        addr(nu), float(mu), n, addr(position), addr(velocity))
    if scalar:
        return position[:, 0], velocity[:, 0]
    return position, velocity


def stumpff(x):
    """Calculates Stumpff functions.

    Based on the function toolkit/src/spicelib/stmp03.f from the SPICE
    toolkit, which can be downloaded from
    naif.jpl.nasa.gov/naif/toolkit_FORTRAN.html
    """
    x, scalar = _flat(x)
    n = x.size
    c0, c1, c2, c3 = (np.empty(n) for _ in range(4))
    lib().sk_stumpff(addr(x), n, addr(c0), addr(c1), addr(c2), addr(c3))
    if scalar:
        return float(c0[0]), float(c1[0]), float(c2[0]), float(c3[0])
    return c0, c1, c2, c3


def propagate(position, velocity, t0, t1, gm):
    """Propagates a position and velocity vector with an array of times.

    Parameters
    ----------
    position : ndarray
        Position vector with shape (3,)
    velocity : ndarray
        Velocity vector with shape (3,)
    t0 : float
        Time corresponding to `position` and `velocity`
    t1 : ndarray
        Times to propagate to
    gm : float
        Gravitational parameter in units that match the other arguments
    """
    if gm <= 0:
        raise ValueError("'gm' should be positive")
    p, _ = _lead_vec(position)
    v, _ = _lead_vec(velocity)
    t1, _ = _flat(t1)
    n = t1.size
    if np.allclose(np.linalg.norm(v), 0.0):
        raise ValueError('Velocity vector has zero magnitude')
    if np.allclose(np.linalg.norm(p), 0.0):
        raise ValueError('Position vector has zero magnitude')

    new_p, new_v = out_vec(n), out_vec(n)
    # Fourteen length-n per-epoch arrays for the kernel to work in, plus one
    # integer per worker for the "is anything still going" reductions.
    scratch = np.empty(_PROPAGATE_ARRAYS * n)
    # One int per worker the kernel may use, plus one.  The kernel clamps its
    # own worker count to `n // 2048`, so sizing this by the `workers()` passed
    # to it is always sufficient; a hardcoded 32 under-allocates whenever
    # `set_workers` is given more than that.
    flags = scratch_i64(workers() + 1)
    status = lib().sk_propagate(addr(p), addr(v), float(t0), addr(t1),
                                float(gm), n, workers(), addr(scratch),
                                addr(flags), addr(new_p), addr(new_v))
    if status != 0:
        raise ValueError(
            'The input delta time (dt) has a value beyond the range of DT for '
            'which we can reliably propagate states, for this GM and initial '
            'state.')
    return new_p, new_v
