# mojo-skyfield

[Skyfield](https://rhodesmill.org/skyfield/)'s ephemeris and
satellite-propagation kernels, rewritten in [Mojo](https://www.modular.com/mojo).

Precession, IAU 2000A nutation, sidereal time, the ICRS/ITRS/TIRS/ecliptic
reference frames, gravitational deflection and aberration, the geodetic
conversions for a ground station, the TEME-to-ITRF transform that SGP4
states are viewed through, and universal-variable Kepler propagation. All of
it runs in one compiled shared library reached through ctypes, and every
function keeps the upstream name, argument order and array shapes, so the
covered subset is a drop-in for `skyfield`.

```python
import numpy as np
from mojo_skyfield import timelib, precessionlib, nutationlib, wgs84

# A day of passes over a ground station, sampled every 30 s.
jd = (timelib.julian_date(2024, 4, 8, 0, 0, 0.0)
      + np.arange(0.0, 86400.0, 30.0) / 86400.0)
t = timelib.Time.from_julian_date(jd)

print(f'{len(jd)} epochs from {jd[0]:.5f} to {jd[-1]:.5f} TT')
print(f'  GAST  {t.gast[0]:.6f} .. {t.gast[-1]:.6f} hours')
print(f'  dpsi  {nutationlib.iau2000a(jd)[0][0]:.6e} (0.1 microarcsec units)')

topos = wgs84.latlon(37.3414, -121.6429, 30.0)
print(f'  station ITRS |xyz| km: {topos.itrs_xyz * 149597870700.0 / 1e3}')

P = precessionlib.compute_precession(jd)
print(f'  P[0,0] {P[0, 0, 0]:.9f}  P[1,0] {P[1, 0, 0]:.3e}')
```

```
2880 epochs from 2460408.50000 to 2460409.49965 TT
  GAST  13.116315 .. 13.173666 hours
  dpsi  -5.235005e+07 (0.1 microarcsec units)
  station ITRS |xyz| km: [-2663.57894809 -4322.33086359  3847.60263346]
  P[0,0] 0.999982497  P[1,0] 5.427e-03
```

## What is covered

Each module here mirrors the `skyfield` module of the same name, and within
each the functions appear in upstream's source order.

| module | ported |
| --- | --- |
| `functions` | `dots`, `T`, `mxv`, `mxm`, `mxmxm`, `length_of`, `angle_between`, `to_spherical`, `from_spherical`, `rot_x/y/z`, `R1/R2/R3`, `angular_velocity_matrix`, `sqrt_nan`, and the `to_polar`/`from_polar` aliases |
| `timelib` | `julian_day`, `julian_date`, `julian_date_of_besselian_epoch`, `compute_calendar_date` (aliased `calendar_date`), `calendar_tuple`, `tdb_minus_tt`, and `Time` with `tt`/`tdb`/`ut1`, `gmst`, `gast`, `M`, `MT`, `N`, `P`, `true_obliquity`, `nutation_matrix()`, `precession_matrix()`, `polar_motion_angles()`, `polar_motion_matrix()` |
| `nutationlib` | `iau2000a`, `iau2000b`, `iau2000a_radians`, `iau2000b_radians`, `fundamental_arguments`, `mean_obliquity`, `build_nutation_matrix`, `equation_of_the_equinoxes_complimentary_terms` |
| `precessionlib` | `compute_precession` |
| `earthlib` | `earth_rotation_angle`, `sidereal_time`, `refraction`, `refract`, `compute_limb_angle`, `terra`, `reverse_terra` |
| `framelib` | `build_matrix` / `ICRS_to_J2000`, `polar_motion_matrix`, `build_ecliptic_matrix`, `tirs.rotation_at`, `itrs.rotation_at`, `ecliptic_frame.rotation_at`, `ICRS`, `InertialFrame`, `mean_equinox_of_date`, `true_equinox_of_date` |
| `relativity` | `light_time_difference`, `_compute_deflection` (also `compute_deflection`), `add_aberration`, `rmasses`, `deflectors` |
| `toposlib` | `Geoid` (`latlon`, `latlon_of`, `height_of`, `geographic_position_of`, `subpoint_of`, `subpoint`), `GeographicPosition` (`lst_hours_at`, `rotation_at`, `refract`, `_dRdt_times_RT_at`), `wgs84`, `iers2010`, `Topos` |
| `sgp4lib` | `theta_GMST1982`, `_cross` (aliased `cross`), `TEME_to_ITRF`, `TEME.rotation_at` |
| `keplerlib` | `normpi`, `eccentric_anomaly`, `true_anomaly_closed/hyperbolic/parabolic`, `ele_to_vec`, `stumpff`, `propagate` |

Every function in that table has a test comparing it against the real upstream
`skyfield` package, element by element. Two things are worth knowing about
how that is arranged:

- **Every exported kernel is also called by name.** `tests/test_kernels.py`
  reaches each of the 56 `sk_*` symbols through `ctypes` itself and asserts
  on its output, so a kernel that was wired up and completely wrong could
  not ship green: nothing observes it unless it is called directly.
  `test_kernel_coverage` re-scans `src/*.mojo` for `@export` names and fails
  if one is not named there, which is what keeps that true as the port grows.
  Those tests exercise the compiled `dist/*.so`; there is no pure-Python
  fallback, and `tests/conftest.py` exits the run rather than skipping if the
  library cannot be loaded.
- **`nutationlib`'s tables bound the term counts.** Upstream's `iau2000a`
  defaults to `lunisolar_terms=687`, but the `nals_t` table it indexes has
  678 rows. The Python wrapper clamps to the table length, and the direct
  kernel test does the same, because a caller that passes 687 to the raw
  symbol reads past the end of the table.

One note on an edge:

- `toposlib.Topos` follows upstream's *source* (it builds on the `iers2010`
  geoid), but upstream's own `Topos` takes a different argument list from the
  rest of its class and does not round-trip its own inputs, so the test pins
  the geoid choice rather than comparing positions against it.

## What is not covered, and why

These are the only divergences. Each one is deliberate; nothing else in the
ported functions differs from upstream.

- **`Timescale` and the leap-second / Delta-T tables.** Upstream builds a
  `Time` from a UTC calendar date by looking up leap seconds and Delta-T in
  bundled tables. Those are table lookups, not compute, so they are not
  ported; `mojo_skyfield.timelib.Time` is constructed from Julian dates
  instead, and takes `ut1_fraction` (defaulting to `tt_fraction`, i.e.
  `dut1 = 0`) so that the caller can supply the same UT1 upstream would.
  Everything computed *from* a `Time` is ported.
- **`keplerlib.propagate` handles one orbit.** Upstream accepts a batch of
  `#orbits` states and returns `(3, norbits, n)`; this port takes a single
  `position`/`velocity` pair and returns `(3, n)`. The algorithm, the
  Stumpff series, the bracketing and the bisection are unchanged — only the
  batch axis is gone. `ele_to_vec` and the anomaly solvers are vectorised
  over time as upstream has them.
- **`sgp4lib.EarthSatellite` is not here.** Since Skyfield 1.49 the SGP4/SDP4
  orbit integration itself lives in the separate `sgp4` package; what is
  ported is everything in `sgp4lib` that is Skyfield's own: the TEME frame
  definition, `theta_GMST1982` and the TEME-to-ITRF transform.
- **`relativity.add_deflection` and `Astrometric.apparent`** need an
  ephemeris object and a `Time`-carrying position class, neither of which is
  part of this port. Their numeric cores — `_compute_deflection`,
  `light_time_difference` and `add_aberration` — are.
- **Range guards that upstream raises on.** `keplerlib.eccentric_anomaly`
  raises when its ten Newton iterations fail to converge; the port returns
  the last iterate instead. `keplerlib.stumpff` rejects arguments below
  upstream's `stumpff_bound` of `-504775.9`, where `cosh(sqrt(-x))` overflows
  a float64; the port does not, and returns `inf`/`nan`. The two other
  `ele_to_vec` guards, and `propagate`'s unpropagatable-dt guard, are
  reproduced exactly, from the Python wrapper.
- **Two numerical footnotes.** `keplerlib.true_anomaly_parabolic` takes a
  cube root, and this toolchain's `pow(x, y)` with a fractional `y` is
  `exp(y * log(x))`, which is only good to about `1e-10` relative rather
  than correctly rounded; the near-cancellation in `B - 1/B` would then put
  a `3e-10` error into the anomaly, two orders past float64. The kernel
  therefore uses `cbrt()`, which agrees with libm to the last bit and brings
  the result back in line with upstream. And the summation order inside
  `dots`, `mxv`, `mxm` and the nutation series is sequential, where NumPy
  may use pairwise or BLAS reductions, so those agree to a few units in the
  last place rather than bit-for-bit. One place where that bites visibly is
  `nutationlib.build_nutation_matrix` entry `[1][2]`,
  `cpsi*sobm*cobt - cobm*sobt`: its two ~0.365 terms cancel to ~3e-7, a
  factor of 1.1e6, so a one-ulp `sin`/`cos` difference shows up there as
  ~5e-11. That is a property of upstream's own formula, not of the port,
  and the test's tolerance for that kernel says so.

## FFI divergences

Three more, the first two forced by the C ABI rather than by the algorithms:

- **Signatures take raw pointers and an explicit length.** Mojo has no
  parametric `@export`, so a buffer crosses as an `Int` address and the
  pointer is rebuilt inside `src/capi.mojo`. The upstream-shaped arguments
  are reconstructed in the Python wrapper, and the kernel bodies in
  `src/ported.mojo` are upstream's arithmetic verbatim.
- **A few composed functions take a caller-supplied `scratch` buffer.**
  `framelib.tirs.rotation_at` and friends build `rot_z(angle)` and then
  multiply it by `t.M`, exactly as upstream does; the intermediate 3x3 has to
  live somewhere, and Mojo will not allocate inside an `abi("C")` function
  it must not leak from. `keplerlib.propagate` takes fourteen length-`n`
  work arrays and one integer per worker for the same reason: a parallel
  worker writes through them and cannot be handed a list it owns. Every
  other temporary is a scalar local. `sgp4lib.TEME_to_ITRF` used to need
  seven such buffers (two `(3, 3, n)` rotation stacks, three `(3, n)`
  vectors and the cross product); since every one of those intermediates is
  only three numbers long, they are held in registers instead and the kernel
  takes no scratch arguments at all.
- `nutationlib`'s coefficient tables still come from the same
  `nutation.npz` that Skyfield bundles, redistributed under `python/`; the
  kernel receives their addresses rather than importing them itself.

## Install

```bash
pixi install
pixi run build     # mojo build --emit shared-lib -> dist/libmojo-skyfield.so
pixi run test      # 358 tests against the real upstream skyfield
pixi run bench     # the table below
```

The kernel builds itself on first import, and rebuilds only when a `.mojo`
file under `src/` is newer than the library, so a bare
`import mojo_skyfield` is enough to get going. Two caveats, both verified
here:

- That on-demand build needs `mojo` on `PATH` **and** `MODULAR_HOME` pointing
  at the environment's `share/max`; without the latter every `std` import
  fails with `unable to locate module 'std'`. Inside `pixi run` both are
  already set. Outside it, a failed build raises `mojo_skyfield._lib.BuildError`
  rather than falling back to anything, because there is no fallback to fall
  back to. If `mojo` is not on `PATH`, either set it, or point
  `MOJO_SKYFIELD_LIB` at a library you built with `pixi run build` to skip the
  compile entirely.
- A plain (non-editable) `pip install` copies the package into site-packages,
  where `src/` no longer exists. Run `pixi run build` first, then set
  `MOJO_SKYFIELD_LIB` to that library.

`pixi.toml` pins `max` alongside `mojo`, because
`max.algorithm.parallelize` ships in the `max` package and not in `mojo`.
Without it every `max.*` import fails and the parallel loops do not build.

## Benchmarks

Intel Xeon E5-2697 v4 @ 2.30 GHz, 72 threads visible, Linux, Python 3.13.14,
NumPy 2.5.1. Five consecutive `pixi run bench` passes. The mojo column is
the median of the five; the upstream column is the best of the five:

| kernel | mojo-skyfield | upstream skyfield | speedup |
| --- | --- | --- | --- |
| `iau2000a`, 100,000 epochs | 366.21 ms | 19219.19 ms | 52.48x |
| `propagate`, 100,000 epochs | 21.51 ms | 643.88 ms | 29.93x |
| eqeq complementary terms, 100,000 | 17.15 ms | 449.60 ms | 26.22x |
| `iau2000b`, 100,000 epochs | 24.11 ms | 593.95 ms | 24.64x |
| `Time.gast` + `Time.M`, 100,000 | 901.67 ms | 14699.69 ms | 16.30x |
| `stumpff`, 100,000 | 3.46 ms | 18.77 ms | 5.42x |
| `mean_obliquity`, 100,000 | 0.23 ms | 0.82 ms | 3.57x |
| `compute_precession`, 100,000 | 5.67 ms | 13.28 ms | 2.34x |
| `TEME_to_ITRF`, 100,000 | 6.02 ms | 14.01 ms | 2.33x |
| `fundamental_arguments`, 100,000 | 4.95 ms | 10.12 ms | 2.04x |
| `mxv`, 100,000 | 0.55 ms | 0.90 ms | 1.64x |
| `rot_z`, 100,000 | 2.69 ms | 3.76 ms | 1.40x |

Every row is faster than upstream; nothing here regressed.

Within a pass, `bench/bench.py` already takes the fastest of a time-budgeted
set of runs, so the five passes differ only by how busy the box was. Taking
the median for this port and the best of the five for upstream is
deliberately conservative: the ratio it reports is the one least flattering
to this port. Treat the speedups as a lower bound, and re-run `pixi run
bench` for numbers that reflect your own machine.

The run-to-run spread is worth knowing before comparing any single figure,
because on this shared box the upstream column is the unstable one. The two
nutation-heavy rows are the extremes: upstream's `iau2000a` read 19219 to
44671 ms and its `Time.gast` + `Time.M` row 14700 to 44068 ms across the five
passes, 2.3x and 3.0x, while this port's own figures for those same rows moved
1.1x and 1.2x. Upstream materialises a 678xN temporary per pass, so those rows
are bound by allocation and memory traffic. No row in this port's column
varied by more than 1.3x across the five passes, and most by less than 1.2x.

### Why the big rows are big

- **`iau2000a` is transcendental-bound, not loop-bound.** The series is 678
  luni-solar plus 687 planetary terms, and every term costs a `sin` and a
  `cos` call. Upstream also materialises a 678xN temporary per pass; the port
  keeps the five fundamental arguments in registers. The speedup comes from
  spreading epochs across cores, and it is bounded by how fast this machine's
  `libm` is.
- **`propagate` runs one launch, not one per bisection round.** Bracketing and
  bisection are over a hundred inner passes, and `parallelize` starts its
  workers afresh on every call, so launching per pass costs more in thread
  startup than the cores return. Giving each worker the whole
  bracketing-and-bisection for one contiguous slice of epochs collapses that to
  a single launch. That is why `_propagate_slice` exists as its own shape.
- **The fused passes remove whole walks of the arrays.** Folding the IAU 2000B
  offsets into the summing pass, and folding the Stumpff bracket test into the
  pass that produces `kfun`, each drop a full traversal of the output or input
  array per round. `TEME_to_ITRF` went the same way: it used to walk its inputs
  seven times — build `R`, two `mxv`, a cross, an accumulate, a copy — every
  one of those intermediates being only three numbers long, so they now live in
  registers and the kernel reads each input exactly once.
- **`_stumpff_one` precomputes its factorial reciprocals.** Upstream multiplies
  `(2k+2)!` and `(2k+3)!` out from scratch, twice per term, on every
  evaluation; the port holds the reciprocals as constants.

### Parallelism

`max.algorithm.parallelize` is the live import on this toolchain; `std`
no longer has it. A kernel with a per-epoch loop takes a worker count
(default `min(os.cpu_count(), 32)`, set from Python and changeable with
`mojo_skyfield._lib.set_workers`) and splits its loop into
`(num_work_items, num_workers)` contiguous chunks. Below 2048 elements per
worker a launch costs more than the work saves, so the loop stays serial.

Every one of these kernels is bit-for-bit identical at one worker and at
32; `tests/test_parallel.py` checks that at sizes chosen to leave a ragged
tail for every worker count.

### The GPU

The blocker to a GPU path is narrow and worth stating precisely — a float64
`sin` or `cos` on the GPU target fails to instantiate. Re-checked on this
pass, not inherited:

```
$ mojo build --target-triple nvptx64-nvidia-cuda --emit asm sin_kernel.mojo
Mojo/stdlib/std/math/math.mojo:1785:9: note: constraint failed:
    DType.float64 is not supported for sin on NVIDIA GPU
```

`std.math` has no float64 `sin` or `cos` device implementation for NVIDIA
targets. That is the entire content of `iau2000a`, `iau2000b`, the equation
of the equinoxes and the Stumpff evaluation, so there is no GPU path for the
kernels this port is actually slow on. Approximating them on the device would
break the `1e-13` parity tolerance, which is not negotiable. No GPU code is
shipped, and the claim is not that the API is missing — it is that this one
data type has no device implementation for the one function these kernels are
made of. The row is a CPU-only answer on the merits, not a toolchain excuse.

## How it works

**FFI.** `src/ported.mojo` holds the kernels; `src/capi.mojo` is a thin
`@export` layer whose only job is turning `Int` addresses back into
`Pointer[Float64, AnyOrigin[mut=True]]`. Everything is one compilation unit,
because build cost is fixed and per-function cost is not. The library is
`dist/libmojo-skyfield.so`; `python/mojo_skyfield/_lib.py` compiles it on
first import when it is missing or older than `src/`, and every entry point
declares its `argtypes` once in `_SIGNATURES`.

**Memory layout.** Nothing is copied. Upstream is NumPy, so a `|xyz|` vector
of shape `(3, n)` is already C-contiguous, and Mojo indexes element `(i, j)`
at `i * n + j`; a `(3, 3, n)` rotation matrix element `(i, k, j)` is at
`(i * 3 + k) * n + j`. The Python wrapper guarantees contiguity with
`np.ascontiguousarray` and hands the kernel `array.ctypes.data`. Output
arrays are allocated with `np.empty` in the upstream shape and written in
place, which is why `add_aberration` and `relativity` can stay in-place
exactly as they are upstream. Where a composed function needs an
intermediate, the wrapper passes a scratch array, as described above.

**The one invariant to respect when adding a kernel.** Every kernel takes a
single length `n` and indexes all of its buffers as if they were exactly that
long. So the wrapper, not the kernel, is responsible for two things. First,
every input has to agree on `n` — a scalar or a shorter array passed straight
to a kernel whose `n` came from a longer one is an out-of-bounds read, and a
flags or scratch buffer sized from the wrong bound is an out-of-bounds write.
Second, every buffer has to be float64, because the kernel reads it as
`Pointer[Float64]` and nothing on the C ABI checks that: an `int32` array of
`n` elements handed over as-is is re-read 8 bytes per element, which is both
wrong and past the end of the buffer. The helpers in `functions.py`
(`_flat`, `_lead_vec`, `_lead_mat`, `_follow_vec`, `_follow_flat`) and the
explicit `np.broadcast_to` calls in `framelib.py` exist for these two things,
and every buffer that reaches a kernel goes through one of them.

**Reading the port.** `src/ported.mojo` is sectioned by upstream module in
upstream's order — `functions`, `timelib`, `earthlib`, `precessionlib`,
`nutationlib`, `framelib`, `relativity`, `sgp4lib`, `keplerlib` — and inside
each section the functions appear as upstream wrote them, with upstream's
branch structure and arithmetic. Read it next to `skyfield/functions.py`,
`skyfield/nutationlib.py`, `skyfield/keplerlib.py` and the rest.

What you will find that is not upstream is all of it deliberate and confined
to four places: the per-epoch loops, which are wrapped in a
`max.algorithm.parallelize` region; the few bodies reshaped so a worker can
run one epoch without a shared list (`_iau2000a_one`, `_propagate_slice`,
`_precession_at`, `_stumpff_one`); the passes upstream walks twice that are
fused here into one (the IAU 2000B offsets, the Stumpff bracket test, and
the whole of `TEME_to_ITRF`); and the precomputed Stumpff factorial
reciprocals. The arithmetic inside all of them is still upstream's, in
upstream's order.

**Tests.** `tests/test_parity.py` and `tests/test_time.py` compare every
ported function against the installed upstream `skyfield` on the same
inputs, element by element, and require agreement to within float64
rounding for the operation — not "it ran". `tests/test_parallel.py` covers
what the new code paths can get wrong: that splitting a loop across workers
leaves every element bit-for-bit identical, that a split with a ragged tail
still covers the whole input, and that the fused IAU 2000B and bracketing
passes still return upstream's answer (including the `ValueError` that
`propagate` raises for an unpropagatable delta time). The fused
`TEME_to_ITRF` gets the same treatment directly: a ragged-tail sweep over
sizes 1, 5, 4097 and 20011 at 1, 3, 8 and 32 workers, on both the
polar-motion and no-polar-motion branches, plus a serial-versus-split
bit-identity check. That branch is worth pinning hard, because it is the one
place where a wrong element in the hand-expanded `W` matrix still produces a
plausible-looking answer, and only the `xp != 0` cases exercise it.

MIT.