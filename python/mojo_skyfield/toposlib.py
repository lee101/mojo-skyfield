"""`skyfield.toposlib` in Mojo."""

import numpy as np

from .constants import ANGVEL, AU_M, DAY_S, T0, pi, tau
from .functions import angular_velocity_matrix, mxm, rot_y, rot_z, _lead_mat
from . import earthlib

_EARTH_ANGULAR_VELOCITY_VECTOR = np.array((0, 0, DAY_S * ANGVEL))


class ITRSPosition(object):
    """An |xyz| position in the Earth-centered Earth-fixed (ITRS) frame."""

    center = 399

    def __init__(self, itrs_xyz):
        self.itrs_xyz = itrs_xyz
        x, y, z = itrs_xyz
        self._velocity_au_per_d = ANGVEL * DAY_S * np.array((-y, x, 0.0 * z))

    @property
    def target(self):
        return self

    def _at(self, t):
        """Compute GCRS position and velocity at time `t`."""
        from . import framelib
        r = np.ascontiguousarray(self.itrs_xyz.reshape(3, 1))
        v = np.ascontiguousarray(self._velocity_au_per_d.reshape(3, 1))
        RT, _ = _lead_mat(framelib.T(framelib.itrs.rotation_at(t)))
        return mxm_v(RT, r), mxm_v(RT, v), None, None


def mxm_v(M, v):
    from .functions import mxv
    return mxv(M, v)


class GeographicPosition(ITRSPosition):
    """A latitude-longitude-elevation position on Earth.

    Build one with a reference geoid rather than directly::

        topos = wgs84.latlon(37.3414, -121.6429)
    """

    vector_name = 'Geodetic'

    def __init__(self, model, latitude, longitude, elevation, itrs_xyz):
        super(GeographicPosition, self).__init__(itrs_xyz)
        self.model = model
        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation
        self._R_lat = rot_y(np.radians(latitude))[::-1]
        self._R_latlon = mxm(self._R_lat, rot_z(-np.radians(longitude)))

    def __repr__(self):
        return (f'{self.model.name} latitude {self.latitude:+.4f} N '
                f'longitude {self.longitude:+.4f} E '
                f'elevation {self.elevation:.1f} m')

    def lst_hours_at(self, t):
        """Return the Local Apparent Sidereal Time, in hours, at time ``t``."""
        sprime = -47.0e-6 * (t.whole - T0 + t.tdb_fraction) / 36525.0
        return (t.gast + self.longitude / 15.0 + sprime / 54000.0) % 24.0

    def refract(self, altitude_degrees, temperature_C, pressure_mbar):
        """Predict how the atmosphere will refract a position."""
        if temperature_C == 'standard':
            temperature_C = 10.0
        if pressure_mbar == 'standard':
            pressure_mbar = 1010.0 * np.exp(-self.elevation / 9.1e3)
        return earthlib.refract(altitude_degrees, temperature_C, pressure_mbar)

    def rotation_at(self, t):
        """Compute rotation from GCRS to this location's altazimuth system."""
        from . import framelib
        return mxm(self._R_latlon, framelib.itrs.rotation_at(t))

    def _dRdt_times_RT_at(self, t):
        from .functions import mxv
        R = mxv(self._R_lat, _EARTH_ANGULAR_VELOCITY_VECTOR)
        return angular_velocity_matrix(R)


class Geoid(object):
    """An Earth ellipsoid: maps latitudes and longitudes to |xyz| positions."""

    def __init__(self, name, radius_m, inverse_flattening):
        self.name = name
        self.radius = radius_m / AU_M
        self.inverse_flattening = inverse_flattening
        omf = (inverse_flattening - 1.0) / inverse_flattening
        self._one_minus_flattening_squared = omf * omf
        f = 1.0 / inverse_flattening
        self._e2 = 2.0 * f - f * f

    def __repr__(self):
        return f'Geoid({self.name!r})'

    @property
    def polar_radius(self):
        """The Earth's polar radius, in au."""
        return self.radius * (1.0 - 1.0 / self.inverse_flattening)

    def latlon(self, latitude_degrees, longitude_degrees, elevation_m=0.0,
               cls=GeographicPosition):
        """Return a `GeographicPosition` for a given latitude and longitude."""
        lat = np.radians(latitude_degrees)
        lon = np.radians(longitude_degrees)
        radius_au = self.radius
        elevation_au = elevation_m / AU_M

        sinphi = np.sin(lat)
        cosphi = np.cos(lat)
        omf2 = self._one_minus_flattening_squared
        c = 1.0 / np.sqrt(cosphi * cosphi + sinphi * sinphi * omf2)
        s = omf2 * c

        radius_xy = radius_au * c
        xy = (radius_xy + elevation_au) * cosphi
        x = xy * np.cos(lon)
        y = xy * np.sin(lon)

        radius_z = radius_au * s
        z = (radius_z + elevation_au) * sinphi

        r = np.array((x, y, z))
        return cls(self, latitude_degrees, longitude_degrees, elevation_m, r)

    def latlon_of(self, position):
        """Return the latitude and longitude of a geocentric ``position``."""
        _, _, _, _, _, _, lat = self._compute_latitude(position)
        x, y = position[0], position[1]
        lon = (np.arctan2(y, x) - pi) % tau - pi
        return lat, lon

    def height_of(self, position):
        """Return the height above the Earth's ellipsoid of a ``position``."""
        _, _, _, R, aC, hyp, _ = self._compute_latitude(position)
        return np.sqrt(hyp * hyp + R * R) - aC

    def geographic_position_of(self, position):
        """Return the `GeographicPosition` of a geocentric ``position``."""
        xyz_au, x, y, R, aC, hyp, lat = self._compute_latitude(position)
        lon = (np.arctan2(y, x) - pi) % tau - pi
        height_au = np.sqrt(hyp * hyp + R * R) - aC
        return GeographicPosition(
            latitude=lat * (180.0 / np.pi),
            longitude=lon * (180.0 / np.pi),
            elevation=height_au * AU_M,
            itrs_xyz=xyz_au,
            model=self,
        )

    def subpoint_of(self, position):
        """Return the point on the ellipsoid directly below a ``position``."""
        _, x, y, R, aC, hyp, lat = self._compute_latitude(position)
        lon = (np.arctan2(y, x) - pi) % tau - pi
        return self.latlon(lat * (180.0 / np.pi), lon * (180.0 / np.pi))

    subpoint = geographic_position_of

    def _compute_latitude(self, position):
        xyz_au = np.asarray(position, dtype=np.float64)
        x, y, z = xyz_au
        a = self.radius
        e2 = self._e2
        R = np.sqrt(x * x + y * y)
        lat = np.arctan2(z, R)
        for iteration in 0, 1, 2:
            sin_lat = np.sin(lat)
            e2_sin_lat = e2 * sin_lat
            aC = a / np.sqrt(1.0 - e2_sin_lat * sin_lat)
            hyp = z + aC * e2_sin_lat
            lat = np.arctan2(hyp, R)
        return xyz_au, x, y, R, aC, hyp, lat


wgs84 = Geoid('WGS84', 6378137.0, 298.257223563)
iers2010 = Geoid('IERS2010', 6378136.6, 298.25642)


class Topos(GeographicPosition):
    """Deprecated: use ``wgs84.latlon()`` or ``iers2010.latlon()``."""

    def __init__(self, latitude, longitude, elevation_m=0.0, cls=None):
        # Upstream's `Topos` builds on the IERS2010 geoid, not WGS84.
        super(Topos, self).__init__(
            iers2010, latitude, longitude, elevation_m,
            iers2010.latlon(latitude, longitude, elevation_m).itrs_xyz)
