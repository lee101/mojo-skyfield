import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import pytest

from mojo_skyfield._lib import lib

# The tests below call `lib()` for every kernel they check, so if the compiled
# library were missing they would fail anyway -- but a skip is a much clearer
# signal than 60 collection errors, and it must never be silent.  Resolve the
# path here so a missing `mojo` toolchain is reported at collection time.
try:
    _LIB_PATH = lib()._name
except OSError as exc:  # pragma: no cover - only on a broken checkout
    pytest.exit(f"could not load the compiled Mojo library: {exc}", returncode=1)


def pytest_configure(config):
    config.addinivalue_line("markers", "kernel: calls one exported `sk_*` symbol directly")


@pytest.fixture(scope="session")
def kern():
    """The loaded shared library, for tests that name an `sk_*` symbol."""
    return lib()
