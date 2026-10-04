"""mojo-skyfield vs upstream Skyfield on the same inputs.

Run through `pixi run bench`, never directly: the pixi task holds a
machine-wide flock so that a concurrent factory job cannot distort the
numbers.
"""

import os
import platform
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "python"))

from mojo_skyfield import (keplerlib as ms_kep,  # noqa: E402
                           nutationlib as ms_nut, precessionlib as ms_pre,
                           timelib as ms_time, functions as ms_fn,
                           sgp4lib as ms_sgp)
from skyfield import (functions as up_fn,  # noqa: E402
                      keplerlib as up_kep,  # noqa: E402
                      nutationlib as up_nut, precessionlib as up_pre,
                      sgp4lib as up_sgp)

N = 100_000
GM = 398600.4418


def timeit(fn, budget=0.30, reps=1):
    """The fastest of `reps` runs, with enough repeats to cover `budget`.

    A shared 72-core box is noisy; taking the best of a time-budgeted
    number of runs keeps a single scheduling hiccup from deciding a row.
    """
    best = float("inf")
    out = None
    reps = 1
    while True:
        for _ in range(reps):
            t0 = time.perf_counter()
            out = fn()
            best = min(best, time.perf_counter() - t0)
        reps *= 2
        if best >= budget or reps >= 64:
            return best, out


def row(name, ours, theirs, note=""):
    speed = theirs / ours if ours > 0 else float("inf")
    print(f"| {name} | {ours * 1e3:.2f} ms | {theirs * 1e3:.2f} ms | "
          f"{speed:.2f}x | {note} |")


def machine():
    cpu = ""
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if line.startswith("Model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    return f"{cpu or platform.processor()} / {os.cpu_count()} threads / Linux"


def main():
    print(f"machine: {machine()}")
    print(f"python:  {platform.python_version()}  numpy: {np.__version__}")
    print()
    print("| kernel | mojo-skyfield | upstream skyfield | speedup | note |")
    print("| --- | --- | --- | --- | --- |")

    jd = 2451545.0 + np.random.default_rng(0).random(N) * 8000.0
    t_cent = (jd - 2451545.0) / 36525.0

    ours, _ = timeit(lambda: ms_nut.iau2000a(jd))
    theirs, _ = timeit(lambda: up_nut.iau2000a(jd))
    row(f"`iau2000a`, {N:,} epochs", ours, theirs,
        "678 luni-solar + 687 planetary terms, one sin and cos each; "
        "upstream materialises 678xN temporaries")

    ours, _ = timeit(lambda: ms_nut.iau2000b(jd))
    theirs, _ = timeit(lambda: up_nut.iau2000b(jd))
    row(f"`iau2000b`, {N:,} epochs", ours, theirs, "77-term series")

    ours, _ = timeit(lambda: ms_nut.equation_of_the_equinoxes_complimentary_terms(jd))
    theirs, _ = timeit(
        lambda: up_nut.equation_of_the_equinoxes_complimentary_terms(jd))
    row(f"eqeq complementary terms, {N:,}", ours, theirs)

    ours, _ = timeit(lambda: ms_nut.fundamental_arguments(t_cent))
    theirs, _ = timeit(lambda: up_nut.fundamental_arguments(t_cent))
    row(f"`fundamental_arguments`, {N:,}", ours, theirs)

    ours, _ = timeit(lambda: ms_pre.compute_precession(jd))
    theirs, _ = timeit(lambda: up_pre.compute_precession(jd))
    row(f"`compute_precession`, {N:,}", ours, theirs)

    ours, _ = timeit(lambda: ms_nut.mean_obliquity(jd))
    theirs, _ = timeit(lambda: up_nut.mean_obliquity(jd))
    row(f"`mean_obliquity`, {N:,}", ours, theirs, "memory-bound in both")

    position = np.array([7000.0, 0.0, 1000.0])
    velocity = np.array([0.0, 5.0, 4.0])
    t1 = np.linspace(0.0, 40000.0, N)

    def run_ours():
        return ms_kep.propagate(position, velocity, 0.0, t1, GM)

    def run_theirs():
        p, v = up_kep.propagate(position, velocity, 0.0, t1, GM)
        return p, v

    ours, _ = timeit(run_ours)
    theirs, _ = timeit(run_theirs)
    row(f"`propagate`, {N:,} epochs", ours, theirs,
        "universal-variable Kepler with Stumpff series")

    x = np.linspace(-4.0, 4.0, N)
    ours, _ = timeit(lambda: np.array(ms_kep.stumpff(x)))
    theirs, _ = timeit(lambda: np.array(up_kep.stumpff(x)))
    row(f"`stumpff`, {N:,}", ours, theirs)

    v = np.empty((3, N))
    v[0] = 1.0
    v[1] = 0.1 * np.sin(np.linspace(0, 6, N))
    v[2] = 0.1 * np.cos(np.linspace(0, 6, N))
    u = np.roll(v, 1, axis=0)
    ours, _ = timeit(lambda: ms_sgp.TEME_to_ITRF(jd, v, u))
    theirs, _ = timeit(lambda: up_sgp.TEME_to_ITRF(jd, v, u))
    row(f"`TEME_to_ITRF`, {N:,}", ours, theirs)

    m = np.repeat(np.eye(3)[:, :, None], N, axis=2)
    ours, _ = timeit(lambda: ms_fn.mxv(m, v))
    theirs, _ = timeit(lambda: up_fn.mxv(m, v))
    row(f"`mxv`, {N:,}", ours, theirs, "memory-bound in both")

    ours, _ = timeit(lambda: ms_fn.rot_z(np.linspace(0, 6, N)))
    theirs, _ = timeit(lambda: up_fn.rot_z(np.linspace(0, 6, N)))
    row(f"`rot_z`, {N:,}", ours, theirs)

    # End-to-end: sidereal time and the ICRS rotation matrix for a whole
    # timeline, which is what an ephemeris consumer actually asks for.
    ts = None
    try:
        from skyfield.api import load
        ts = load.timescale()
    except Exception:
        pass

    def ours_pipeline():
        # A fresh Time each rep, so nothing is served from a cache.
        t = ms_time.Time.from_julian_date(jd)
        return t.gast, t.M

    if ts is not None:
        def theirs_pipeline():
            # Upstream caches `gast` and `M` on the Time object, so it has
            # to be a fresh Time each rep too.
            t = ts.tt_jd(jd)
            return t.gast, t.M

        ours, _ = timeit(ours_pipeline)
        theirs, _ = timeit(theirs_pipeline)
        row(f"`Time.gast` + `Time.M`, {N:,}", ours, theirs,
            "full nutation + precession + sidereal time")
    print()


if __name__ == "__main__":
    main()
