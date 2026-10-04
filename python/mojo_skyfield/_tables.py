"""The IAU 2000A coefficient tables that upstream bundles in `nutation.npz`.

Upstream loads these with `load_bundled_npy('nutation.npz')` at import time
and passes them to `iau2000a()` as ordinary arrays.  The port keeps the
same file and the same array names; the Mojo kernel receives their
addresses instead of importing them itself.
"""

import os

import numpy as np

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                     "nutation.npz")
_arrays = np.load(_PATH)

ke0_t = np.ascontiguousarray(_arrays["ke0_t"], dtype=np.int64)
ke1 = np.ascontiguousarray(_arrays["ke1"], dtype=np.int64)
lunisolar_longitude_coefficients = np.ascontiguousarray(
    _arrays["lunisolar_longitude_coefficients"], dtype=np.float64)
lunisolar_obliquity_coefficients = np.ascontiguousarray(
    _arrays["lunisolar_obliquity_coefficients"], dtype=np.float64)
nals_t = np.ascontiguousarray(_arrays["nals_t"], dtype=np.int64)
napl_t = np.ascontiguousarray(_arrays["napl_t"], dtype=np.int64)
nutation_coefficients_longitude = np.ascontiguousarray(
    _arrays["nutation_coefficients_longitude"], dtype=np.float64)
nutation_coefficients_obliquity = np.ascontiguousarray(
    _arrays["nutation_coefficients_obliquity"], dtype=np.float64)
se0_t_0 = np.ascontiguousarray(_arrays["se0_t_0"], dtype=np.float64)
se0_t_1 = np.ascontiguousarray(_arrays["se0_t_1"], dtype=np.float64)
