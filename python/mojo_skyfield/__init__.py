"""mojo-skyfield: Skyfield's ephemeris and satellite-propagation kernels in Mojo.

Each module here mirrors the upstream `skyfield` module of the same name,
keeps upstream's function names, argument order and array shapes, and runs
the arithmetic in a compiled Mojo kernel reached through ctypes.
"""

from . import constants
from . import framelib
from . import functions
from . import keplerlib
from . import nutationlib
from . import precessionlib
from . import relativity
from . import sgp4lib
from . import timelib
from . import toposlib
from ._lib import lib
from .constants import (AU_KM, AU_M, ASEC2RAD, B1950, C, DAY_S, DEG2RAD,
                        ERAD, GS, RAD2DEG, T0, pi, tau)
from .functions import (angle_between, dots, from_spherical, length_of, mxm,
                        mxmxm, mxv, rot_x, rot_y, rot_z, to_spherical)
from .keplerlib import propagate
from .nutationlib import iau2000a, iau2000b
from .precessionlib import compute_precession
from .relativity import add_aberration
from .timelib import Time, calendar_tuple, julian_date, tdb_minus_tt
from .toposlib import wgs84, iers2010

__version__ = "0.1.0"

__all__ = [
    'AU_KM', 'AU_M', 'ASEC2RAD', 'B1950', 'C', 'DAY_S', 'DEG2RAD', 'ERAD',
    'GS', 'RAD2DEG', 'T0', 'Time', 'add_aberration', 'angle_between',
    'calendar_tuple', 'compute_precession', 'constants', 'dots', 'framelib',
    'from_spherical', 'functions', 'iau2000a', 'iau2000b', 'iers2010',
    'julian_date', 'keplerlib', 'length_of', 'lib', 'mxm', 'mxmxm', 'mxv',
    'nutationlib', 'pi', 'precessionlib', 'propagate', 'relativity',
    'rot_x', 'rot_y', 'rot_z', 'sgp4lib', 'tau', 'tdb_minus_tt', 'timelib',
    'to_spherical', 'toposlib', 'wgs84',
]
