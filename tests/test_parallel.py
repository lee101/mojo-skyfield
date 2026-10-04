"""The parallel dispatch, the fused passes and the rewritten Stumpff series.

Everything in the port that has a per-epoch loop now splits that loop across
cores.  These tests pin down the three things that can go wrong there: that
splitting a loop changes no element, that a split covers the whole input
including a ragged tail, and that the passes which were folded together
(IAU 2000B's offsets, the Stumpff bracket test, the past/future bracketing
loops) still return upstream's answer.

Sizes are chosen to straddle the per-worker threshold and to leave a ragged
tail for every worker count used here, so no epoch is left unvisited.
"""

import numpy as np
import pytest

from skyfield import keplerlib as up_kep, nutationlib as up_nut
from skyfield import precessionlib as up_pre

from mojo_skyfield import _lib
from mojo_skyfield import keplerlib as ms_kep
from mojo_skyfield import nutationlib as ms_nut
from mojo_skyfield import precessionlib as ms_pre

GM = 398600.4418

# Below 2 * _PAR_MIN (2048) the kernel stays serial no matter how many workers
# are offered, so 4095 and 4096 straddle the threshold; the rest are prime to
# every worker count used here, which forces an uneven last chunk.
SIZES = [1, 3, 100, 4095, 4096, 4097, 5000, 20011]


@pytest.fixture(autouse=True)
def restore_workers():
    saved = _lib.workers()
    yield
    _lib.set_workers(saved)


def epochs(n):
    return 2451545.0 + np.random.default_rng(11).random(n) * 8000.0


def assert_identical(ours, theirs):
    ours = np.asarray(ours, dtype=np.float64)
    theirs = np.asarray(theirs, dtype=np.float64)
    assert ours.shape == theirs.shape
    # Epochs are independent, so a split loop must be bit-for-bit identical,
    # not merely close: no reduction crosses an epoch boundary here.
    assert np.array_equal(ours, theirs)


def both_ways(fn, *args, workers):
    """Return `(result at one worker, result at `workers)` for `fn`."""
    _lib.set_workers(1)
    serial = fn(*args)
    _lib.set_workers(workers)
    return serial, fn(*args)


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("workers", [1, 3, 8, 32])
def test_iau2000a_split_matches_serial(n, workers):
    jd = epochs(n)
    serial, split = both_ways(ms_nut.iau2000a, jd, workers=workers)
    assert_identical(split[0], serial[0])
    assert_identical(split[1], serial[1])


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("workers", [1, 8, 32])
def test_iau2000b_split_matches_serial(n, workers):
    jd = epochs(n)
    serial, split = both_ways(ms_nut.iau2000b, jd, workers=workers)
    assert_identical(split[0], serial[0])
    assert_identical(split[1], serial[1])


def test_iau2000b_offsets_are_fused_into_one_pass():
    """The two IAU 2000B constants are added in the summing pass, not after.

    A separate pass would have been free to drop, so check the fused result
    still differs from IAU 2000A by exactly those two constants, and still
    matches upstream.
    """
    jd = epochs(1000)
    dpsi_a, deps_a = ms_nut.iau2000a(jd, 2, 77, 0)
    dpsi_b, deps_b = ms_nut.iau2000b(jd)
    assert np.allclose(dpsi_b - dpsi_a, -0.000135e7, rtol=0, atol=1e-6)
    assert np.allclose(deps_b - deps_a, 0.000388e7, rtol=0, atol=1e-6)
    ours, theirs = ms_nut.iau2000b(jd), up_nut.iau2000b(jd)
    for axis in (0, 1):
        scale = max(np.max(np.abs(theirs[axis])), 1e-300)
        assert np.max(np.abs(ours[axis] - theirs[axis])) <= 1e-13 * scale


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("workers", [1, 8, 32])
def test_equation_of_equinoxes_split_matches_serial(n, workers):
    jd = epochs(n)
    serial, split = both_ways(
        ms_nut.equation_of_the_equinoxes_complimentary_terms, jd,
        workers=workers)
    assert_identical(split, serial)


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("workers", [1, 8, 32])
def test_fundamental_arguments_split_matches_serial(n, workers):
    t = np.linspace(-0.2, 0.2, n)
    serial, split = both_ways(ms_nut.fundamental_arguments, t, workers=workers)
    assert_identical(split, serial)


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("workers", [1, 3, 8, 32])
def test_compute_precession_split_matches_serial(n, workers):
    jd = epochs(n)
    serial, split = both_ways(ms_pre.compute_precession, jd, workers=workers)
    assert_identical(split, serial)
    theirs = up_pre.compute_precession(jd)
    scale = max(np.max(np.abs(theirs)), 1e-300)
    assert np.max(np.abs(split - theirs)) <= 1e-13 * scale


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("workers", [1, 3, 8, 32])
def test_stumpff_split_matches_serial(n, workers):
    # Straddle the |x| <= 1 branch on both sides of the boundary.
    x = np.concatenate([np.linspace(-8.0, 8.0, n), np.full(n, 1.0 - 1e-16),
                        np.full(n, -(1.0 - 1e-16))])
    serial, split = both_ways(lambda a: np.array(ms_kep.stumpff(a)), x,
                              workers=workers)
    assert_identical(split, serial)
    theirs = np.array(up_kep.stumpff(x))
    for axis in range(4):
        scale = max(np.max(np.abs(theirs[axis])), 1e-300)
        assert np.max(np.abs(serial[axis] - theirs[axis])) <= 1e-11 * scale


@pytest.mark.parametrize("n", [1, 3, 100, 4097, 20011])
@pytest.mark.parametrize("workers", [1, 3, 8, 32])
@pytest.mark.parametrize("position,velocity", [
    (np.array([7000.0, 0.0, 1000.0]), np.array([0.0, 5.0, 4.0])),
    (np.array([1.0e7, 0.0, 0.0]), np.array([0.0, 12.0, 0.0])),
])
def test_propagate_split_matches_serial(n, workers, position, velocity):
    # Times before, at and after t0, so the past and future bracketing
    # branches are both taken, and far enough out to force several doublings.
    t1 = np.concatenate([np.linspace(-1e6, 1e7, n), np.array([0.0, 0.5, -0.5])])
    serial, split = both_ways(
        ms_kep.propagate, position, velocity, 0.0, t1, GM, workers=workers)
    assert_identical(split[0], serial[0])
    assert_identical(split[1], serial[1])


@pytest.mark.parametrize("workers", [1, 8, 32])
def test_propagate_out_of_range_raises_like_upstream(workers):
    """The bracket-widening loop still reports "no reliable propagation".

    `propagate` returns 1 when a bracket stops moving, which the wrapper
    turns into upstream's ValueError.  The past and future loops were merged
    and their stall test moved into the Stumpff pass, so this path needs
    covering at every worker count.
    """
    position = np.array([7000.0, 0.0, 0.0])
    velocity = np.array([0.0, 2.0, 0.0])
    t1 = np.array([1e200, 1.0, -1e200])
    with pytest.raises(ValueError):
        up_kep.propagate(position, velocity, 0.0, t1, GM)
    _lib.set_workers(workers)
    with pytest.raises(ValueError):
        ms_kep.propagate(position, velocity, 0.0, t1, GM)


def test_workers_are_clamped_to_at_least_one():
    _lib.set_workers(0)
    assert _lib.workers() == 1
    _lib.set_workers(-5)
    assert _lib.workers() == 1
    _lib.set_workers(4)
    assert _lib.workers() == 4
