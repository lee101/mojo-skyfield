"""`skyfield.relativity` in Mojo."""

import numpy as np

from ._lib import addr, lib, out_vec
from .functions import _follow_vec, _lead_vec

deflectors = ['sun', 'jupiter', 'saturn', 'moon', 'venus', 'uranus', 'neptune']

rmasses = {
    199: 6023600.0,             # mercury
    299: 408523.71,             # venus
    399: 332946.050895,         # earth
    499: 3098708.0,             # mars
    599: 1047.3486,             # jupiter
    699: 3497.898,              # saturn
    799: 22902.98,              # uranus
    899: 19412.24,              # neptune
    999: 135200000.0,           # pluto
    10: 1.0,                    # sun
    301: 27068700.387534,       # moon

    # Names, for compatibility with code written against Skyfield 1.53+.
    'mercury': 6023600.0,
    'venus': 408523.71,
    'earth': 332946.050895,
    'mars': 3098708.0,
    'jupiter': 1047.3486,
    'saturn': 3497.898,
    'uranus': 22902.98,
    'neptune': 19412.24,
    'pluto': 135200000.0,
    'sun': 1.0,
    'moon': 27068700.387534,
}


def light_time_difference(position, deflector_position):
    """When did the light from ``position`` pass closest to a given deflector?"""
    p, ndim = _lead_vec(position)
    n = p.shape[1]
    d = _follow_vec(p, deflector_position)
    dst = np.empty(n)
    lib().sk_light_time_difference(addr(p), addr(d), n, addr(dst))
    return float(dst[0]) if ndim == 1 else dst


def _compute_deflection(position, pe, rmass):
    """Compute how much a mass will deflect a position."""
    p, ndim = _lead_vec(position)
    n = p.shape[1]
    e = _follow_vec(p, pe)
    dst = out_vec(n)
    lib().sk_compute_deflection(addr(p), addr(e), float(rmass), n, addr(dst))
    return dst[:, 0] if ndim == 1 else dst


# Upstream's name is private; expose it under both so callers can use
# either spelling.
compute_deflection = _compute_deflection


def add_aberration(position, velocity, light_time):
    """Correct a relative position vector for aberration of light, in place."""
    p, ndim = _lead_vec(position)
    n = p.shape[1]
    v = _follow_vec(p, velocity)
    lt = np.ascontiguousarray(
        np.broadcast_to(np.asarray(light_time, dtype=np.float64), (n,)))
    lib().sk_add_aberration(addr(p), addr(v), addr(lt), n)
    return p[:, 0] if ndim == 1 else p
