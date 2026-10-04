"""Loads (and if necessary builds) the compiled Mojo library."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")

# `MOJO_SKYFIELD_LIB` points at an already-built `capi.so`. Two cases need it:
# an ordinary (non-editable) `pip install`, where the package is copied into
# site-packages and `ROOT` no longer names the checkout that holds `src/` and
# `build/`; and any server that must never pay a `mojo build` inside a request
# and would rather ship the kernel next to the deploy. Set it and nothing is
# ever compiled at import or call time.
LIB = os.environ.get("MOJO_SKYFIELD_LIB") or os.path.join(ROOT, "build", "capi.so")

I = ctypes.c_int64
F = ctypes.c_double

# name -> (argtypes, restype)
_SIGNATURES = {
    "sk_sqrt_nan": ([I, I, I], None),
    "sk_dots": ([I, I, I, I], None),
    "sk_T": ([I, I, I], None),
    "sk_mxv": ([I, I, I, I], None),
    "sk_mxm": ([I, I, I, I], None),
    "sk_mxmxm": ([I, I, I, I, I], None),
    "sk_length_of": ([I, I, I], None),
    "sk_angle_between": ([I, I, I, I], None),
    "sk_to_spherical": ([I, I, I, I, I], None),
    "sk_from_spherical": ([I, I, I, I, I], None),
    "sk_rot_x": ([I, I, I], None),
    "sk_rot_y": ([I, I, I], None),
    "sk_rot_z": ([I, I, I], None),
    "sk_R1": ([I, I, I], None),
    "sk_R2": ([I, I, I], None),
    "sk_R3": ([I, I, I], None),
    "sk_angular_velocity_matrix": ([I, I, I], None),
    "sk_julian_day": ([I, I, I, F, I, I], None),
    "sk_julian_date": ([I, I, I, I, I, I, F, I, I], None),
    "sk_julian_date_of_besselian_epoch": ([I, I, I], None),
    "sk_compute_calendar_date": ([I, F, I, I, I, I], None),
    "sk_calendar_tuple": ([I, I, F, I, I, I, I, I, I, I], None),
    "sk_tdb_minus_tt": ([I, I, I, I], None),
    "sk_terra": ([I, I, I, I, I, I, I], None),
    "sk_reverse_terra": ([I, I, I, I, I, I, I], None),
    "sk_compute_limb_angle": ([I, I, I, I, I], None),
    "sk_earth_rotation_angle": ([I, I, I, I], None),
    "sk_sidereal_time": ([I, I, I, I, I], None),
    "sk_refraction": ([I, F, F, I, I], None),
    "sk_refract": ([I, F, F, I, I], None),
    "sk_compute_precession": ([I, I, I, I], None),
    "sk_fundamental_arguments": ([I, I, I, I, I], None),
    "sk_mean_obliquity": ([I, I, I], None),
    "sk_build_nutation_matrix": ([I, I, I, I, I], None),
    "sk_equation_of_the_equinoxes_complimentary_terms": (
        [I, I, I, I, I, I, I, I, I], None),
    "sk_iau2000a": ([I, I, I, I, I, I, I, I, I, I, I, I, I, I], None),
    "sk_iau2000b": ([I, I, I, I, I, I, I, I, I, I, I, I, I, I], None),
    "sk_build_matrix": ([I], None),
    "sk_build_ecliptic_matrix": ([I, I, I, I, I], None),
    "sk_tirs_rotation_at": ([I, I, I, I, I], None),
    "sk_itrs_rotation_at": ([I, I, I, I, I, I, I], None),
    "sk_light_time_difference": ([I, I, I, I], None),
    "sk_compute_deflection": ([I, I, F, I, I], None),
    "sk_add_aberration": ([I, I, I, I], None),
    "sk_cross": ([I, I, I, I], None),
    "sk_theta_GMST1982": ([I, I, I, I, I], None),
    "sk_TEME_to_ITRF": ([I, I, I, F, F, I, I, I, I, I], None),
    "sk_TEME_rotation_at": ([I, I, I, I, I, I], None),
    "sk_eccentric_anomaly": ([I, I, I, I], None),
    "sk_true_anomaly_hyperbolic": ([I, I, I, I], None),
    "sk_true_anomaly_closed": ([I, I, I, I], None),
    "sk_true_anomaly_parabolic": ([I, F, I, I, I], None),
    "sk_ele_to_vec": ([I, I, I, I, I, I, F, I, I, I], None),
    "sk_stumpff": ([I, I, I, I, I, I], None),
    "sk_normpi": ([I, I, I], None),
    "sk_propagate": ([I, I, F, I, F, I, I, I, I, I, I], I),
}


class BuildError(RuntimeError):
    pass


def mojo_command() -> list[str]:
    override = os.environ.get("MOJO_SKYFIELD_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    raise BuildError("mojo not found; set MOJO_SKYFIELD_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    """Compile `src/capi.mojo` into `build/capi.so` if it is missing or stale."""
    if os.environ.get("MOJO_SKYFIELD_LIB") and os.path.exists(LIB) and not force:
        # An explicitly supplied library is taken as given: the sources it
        # was built from need not be present at all, so there is nothing to
        # compare mtimes against.
        return LIB
    if not os.path.isdir(SRC):
        if os.path.exists(LIB):
            return LIB
        raise BuildError(
            f"no Mojo sources at {SRC} and no library at {LIB}; this looks like "
            "a copied install - point MOJO_SKYFIELD_LIB at a built capi.so"
        )
    sources = [
        os.path.join(dirpath, name)
        for dirpath, _, names in os.walk(SRC)
        for name in names
        if name.endswith(".mojo")
    ]
    if not force and os.path.exists(LIB):
        newest = max(os.path.getmtime(s) for s in sources)
        if os.path.getmtime(LIB) >= newest:
            return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    cmd = mojo_command() + [
        "build", "--emit", "shared-lib", "-I", SRC,
        os.path.join(SRC, "capi.mojo"), "-o", LIB,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except OSError as exc:
        # `MOJO_SKYFIELD_MOJO` can name something that is not executable; say
        # so in the package's own error type rather than leaking errno.
        raise BuildError(f"could not run {cmd[0]!r}: {exc}") from exc
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_lib = None

# How many cores the kernels may spread a per-epoch loop over.  Mojo has no
# mutable global, so this travels as an argument to each kernel that has a
# loop that divides cleanly; `set_workers(1)` forces the serial path.
_workers = min(os.cpu_count() or 1, 32)


def set_workers(count: int) -> None:
    """Cap how many cores the kernels spread their per-epoch loops over."""
    global _workers
    _workers = max(1, int(count))


def workers() -> int:
    return _workers




def lib() -> ctypes.CDLL:
    """Load the shared library on first use."""
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_lib, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _lib



def scratch_i64(count: int) -> np.ndarray:
    """A C-contiguous int64 scratch buffer of `count` elements."""
    return np.zeros(count, dtype=np.int64)


def main() -> int:
    """`python -m mojo_skyfield._lib` rebuilds the library."""
    print(build(force="--force" in sys.argv))
    return 0


# ----------------------------------------------------------------------------
# Array marshalling shared by every binding module
# ----------------------------------------------------------------------------
def addr(a) -> int:
    """The address of a C-contiguous buffer, without changing its dtype."""
    return np.ascontiguousarray(a).ctypes.data



def out_vec(n: int) -> np.ndarray:
    return np.empty((3, n))


def out_mat(n: int) -> np.ndarray:
    return np.empty((3, 3, n))


def squeeze(a: np.ndarray, scalar: bool) -> np.ndarray | float:
    """Return `float(a[0])` for a one-element result built from a scalar input."""
    if scalar:
        return float(a[0])
    return a


def sentinel(julian_before):
    """The kernel's `julian_before` argument: -1.0 stands for upstream's None."""
    return -1.0 if julian_before is None else float(julian_before)


if __name__ == "__main__":
    raise SystemExit(main())
