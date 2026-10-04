"""The C ABI the Python bindings call through.

Buffers cross as `Int` addresses: `@export` refuses parametric functions,
and a pointer with an inferred origin is parametric. Nothing here
allocates, and nothing here rearranges upstream's arithmetic -- each
wrapper only rebuilds the pointers that `ported.mojo` works on.
"""

from ported import (
    T,
    _compute_deflection,
    _cross,
    add_aberration,
    angle_between,
    build_ecliptic_matrix,
    build_matrix,
    build_nutation_matrix,
    calendar_tuple,
    compute_calendar_date,
    compute_limb_angle,
    compute_precession,
    dots,
    earth_rotation_angle,
    ele_to_vec,
    equation_of_the_equinoxes_complimentary_terms,
    eccentric_anomaly,
    from_spherical,
    fundamental_arguments,
    iau2000a,
    iau2000b,
    itrs_rotation_at,
    julian_date,
    julian_date_of_besselian_epoch,
    julian_day,
    length_of,
    light_time_difference,
    mean_obliquity,
    mxv,
    mxm,
    mxmxm,
    normpi,

    propagate,

    refract,
    refraction,
    reverse_terra,
    rot_x,
    rot_y,
    rot_z,
    sidereal_time,
    sqrt_nan,
    stumpff,

    tdb_minus_tt,
    TEME_rotation_at,
    TEME_to_ITRF,
    terra,
    theta_GMST1982,
    tirs_rotation_at,
    true_anomaly_closed,
    true_anomaly_hyperbolic,
    true_anomaly_parabolic,
    to_spherical,
    angular_velocity_matrix,
    R1,
    R2,
    R3,
)
from ported import IPtr, Ptr


# ----------------------------------------------------------------------------
# skyfield.functions
# ----------------------------------------------------------------------------

@export("sk_sqrt_nan")
def sk_sqrt_nan(x: Int, size: Int, dst: Int) abi("C") -> None:
    sqrt_nan(Ptr(unsafe_from_address=x), size, Ptr(unsafe_from_address=dst))


@export("sk_dots")
def sk_dots(v: Int, u: Int, n: Int, dst: Int) abi("C") -> None:
    dots(Ptr(unsafe_from_address=v), Ptr(unsafe_from_address=u), n,
         Ptr(unsafe_from_address=dst))


@export("sk_T")
def sk_T(M: Int, n: Int, dst: Int) abi("C") -> None:
    T(Ptr(unsafe_from_address=M), n, Ptr(unsafe_from_address=dst))


@export("sk_mxv")
def sk_mxv(M: Int, v: Int, n: Int, dst: Int) abi("C") -> None:
    mxv(Ptr(unsafe_from_address=M), Ptr(unsafe_from_address=v), n,
        Ptr(unsafe_from_address=dst))


@export("sk_mxm")
def sk_mxm(M1: Int, M2: Int, n: Int, dst: Int) abi("C") -> None:
    mxm(Ptr(unsafe_from_address=M1), Ptr(unsafe_from_address=M2), n,
        Ptr(unsafe_from_address=dst))


@export("sk_mxmxm")
def sk_mxmxm(M1: Int, M2: Int, M3: Int, n: Int, dst: Int) abi("C") -> None:
    mxmxm(Ptr(unsafe_from_address=M1), Ptr(unsafe_from_address=M2),
          Ptr(unsafe_from_address=M3), n, Ptr(unsafe_from_address=dst))


@export("sk_length_of")
def sk_length_of(xyz: Int, n: Int, dst: Int) abi("C") -> None:
    length_of(Ptr(unsafe_from_address=xyz), n, Ptr(unsafe_from_address=dst))


@export("sk_angle_between")
def sk_angle_between(u: Int, v: Int, n: Int, dst: Int) abi("C") -> None:
    angle_between(Ptr(unsafe_from_address=u), Ptr(unsafe_from_address=v), n,
                  Ptr(unsafe_from_address=dst))


@export("sk_to_spherical")
def sk_to_spherical(xyz: Int, n: Int, d_r: Int, d_theta: Int,
                    d_phi: Int) abi("C") -> None:
    to_spherical(Ptr(unsafe_from_address=xyz), n, Ptr(unsafe_from_address=d_r),
                 Ptr(unsafe_from_address=d_theta), Ptr(unsafe_from_address=d_phi))


@export("sk_from_spherical")
def sk_from_spherical(r: Int, theta: Int, phi: Int, n: Int,
                      dst: Int) abi("C") -> None:
    from_spherical(Ptr(unsafe_from_address=r), Ptr(unsafe_from_address=theta),
                   Ptr(unsafe_from_address=phi), n, Ptr(unsafe_from_address=dst))


@export("sk_rot_x")
def sk_rot_x(theta: Int, n: Int, dst: Int) abi("C") -> None:
    rot_x(Ptr(unsafe_from_address=theta), n, Ptr(unsafe_from_address=dst))


@export("sk_rot_y")
def sk_rot_y(theta: Int, n: Int, dst: Int) abi("C") -> None:
    rot_y(Ptr(unsafe_from_address=theta), n, Ptr(unsafe_from_address=dst))


@export("sk_rot_z")
def sk_rot_z(theta: Int, n: Int, dst: Int) abi("C") -> None:
    rot_z(Ptr(unsafe_from_address=theta), n, Ptr(unsafe_from_address=dst))


@export("sk_R1")
def sk_R1(theta: Int, n: Int, dst: Int) abi("C") -> None:
    R1(Ptr(unsafe_from_address=theta), n, Ptr(unsafe_from_address=dst))


@export("sk_R2")
def sk_R2(theta: Int, n: Int, dst: Int) abi("C") -> None:
    R2(Ptr(unsafe_from_address=theta), n, Ptr(unsafe_from_address=dst))


@export("sk_R3")
def sk_R3(theta: Int, n: Int, dst: Int) abi("C") -> None:
    R3(Ptr(unsafe_from_address=theta), n, Ptr(unsafe_from_address=dst))


@export("sk_angular_velocity_matrix")
def sk_angular_velocity_matrix(v: Int, n: Int, dst: Int) abi("C") -> None:
    angular_velocity_matrix(Ptr(unsafe_from_address=v), n,
                            Ptr(unsafe_from_address=dst))


# ----------------------------------------------------------------------------
# skyfield.timelib
# ----------------------------------------------------------------------------

@export("sk_julian_day")
def sk_julian_day(year: Int, month: Int, day: Int, julian_before: Float64,
                  n: Int, dst: Int) abi("C") -> None:
    julian_day(Ptr(unsafe_from_address=year), Ptr(unsafe_from_address=month),
               Ptr(unsafe_from_address=day), julian_before, n,
               Ptr(unsafe_from_address=dst))


@export("sk_julian_date")
def sk_julian_date(year: Int, month: Int, day: Int, hour: Int, minute: Int,
                   second: Int, julian_before: Float64, n: Int,
                   dst: Int) abi("C") -> None:
    julian_date(Ptr(unsafe_from_address=year), Ptr(unsafe_from_address=month),
                Ptr(unsafe_from_address=day), Ptr(unsafe_from_address=hour),
                Ptr(unsafe_from_address=minute), Ptr(unsafe_from_address=second),
                julian_before, n, Ptr(unsafe_from_address=dst))


@export("sk_julian_date_of_besselian_epoch")
def sk_julian_date_of_besselian_epoch(b: Int, n: Int, dst: Int) abi("C") -> None:
    julian_date_of_besselian_epoch(Ptr(unsafe_from_address=b), n,
                                   Ptr(unsafe_from_address=dst))


@export("sk_compute_calendar_date")
def sk_compute_calendar_date(jd: Int, julian_before: Float64, n: Int, d_year: Int,
                             d_month: Int, d_day: Int) abi("C") -> None:
    compute_calendar_date(Ptr(unsafe_from_address=jd), julian_before, n,
                          Ptr(unsafe_from_address=d_year),
                          Ptr(unsafe_from_address=d_month),
                          Ptr(unsafe_from_address=d_day))


@export("sk_calendar_tuple")
def sk_calendar_tuple(jd: Int, fraction: Int, julian_before: Float64, n: Int,
                      d_year: Int, d_month: Int, d_day: Int, d_hour: Int,
                      d_minute: Int, d_second: Int) abi("C") -> None:
    calendar_tuple(Ptr(unsafe_from_address=jd), Ptr(unsafe_from_address=fraction),
                   julian_before, n, Ptr(unsafe_from_address=d_year),
                   Ptr(unsafe_from_address=d_month), Ptr(unsafe_from_address=d_day),
                   Ptr(unsafe_from_address=d_hour),
                   Ptr(unsafe_from_address=d_minute),
                   Ptr(unsafe_from_address=d_second))


@export("sk_tdb_minus_tt")
def sk_tdb_minus_tt(jd_tdb: Int, fraction_tdb: Int, n: Int,
                    dst: Int) abi("C") -> None:
    tdb_minus_tt(Ptr(unsafe_from_address=jd_tdb),
                 Ptr(unsafe_from_address=fraction_tdb), n,
                 Ptr(unsafe_from_address=dst))


# ----------------------------------------------------------------------------
# skyfield.earthlib
# ----------------------------------------------------------------------------

@export("sk_terra")
def sk_terra(latitude: Int, longitude: Int, elevation: Int, gast: Int, n: Int,
             d_pos: Int, d_vel: Int) abi("C") -> None:
    terra(Ptr(unsafe_from_address=latitude), Ptr(unsafe_from_address=longitude),
          Ptr(unsafe_from_address=elevation), Ptr(unsafe_from_address=gast), n,
          Ptr(unsafe_from_address=d_pos), Ptr(unsafe_from_address=d_vel))


@export("sk_reverse_terra")
def sk_reverse_terra(xyz_au: Int, gast: Int, iterations: Int, n: Int, d_lat: Int,
                     d_lon: Int, d_elevation_m: Int) abi("C") -> None:
    reverse_terra(Ptr(unsafe_from_address=xyz_au), Ptr(unsafe_from_address=gast),
                  iterations, n, Ptr(unsafe_from_address=d_lat),
                  Ptr(unsafe_from_address=d_lon),
                  Ptr(unsafe_from_address=d_elevation_m))


@export("sk_compute_limb_angle")
def sk_compute_limb_angle(position_au: Int, observer_au: Int, n: Int,
                          d_limb: Int, d_nadir: Int) abi("C") -> None:
    compute_limb_angle(Ptr(unsafe_from_address=position_au),
                       Ptr(unsafe_from_address=observer_au), n,
                       Ptr(unsafe_from_address=d_limb),
                       Ptr(unsafe_from_address=d_nadir))


@export("sk_earth_rotation_angle")
def sk_earth_rotation_angle(jd_ut1: Int, fraction_ut1: Int, n: Int,
                            dst: Int) abi("C") -> None:
    earth_rotation_angle(Ptr(unsafe_from_address=jd_ut1),
                         Ptr(unsafe_from_address=fraction_ut1), n,
                         Ptr(unsafe_from_address=dst))


@export("sk_sidereal_time")
def sk_sidereal_time(jd_ut1: Int, fraction_ut1: Int, tdb_fraction: Int, n: Int,
                     dst: Int) abi("C") -> None:
    sidereal_time(Ptr(unsafe_from_address=jd_ut1),
                  Ptr(unsafe_from_address=fraction_ut1),
                  Ptr(unsafe_from_address=tdb_fraction), n,
                  Ptr(unsafe_from_address=dst))


@export("sk_refraction")
def sk_refraction(alt_degrees: Int, temperature_C: Float64, pressure_mbar: Float64,
                  n: Int, dst: Int) abi("C") -> None:
    refraction(Ptr(unsafe_from_address=alt_degrees), temperature_C, pressure_mbar,
               n, Ptr(unsafe_from_address=dst))


@export("sk_refract")
def sk_refract(alt_degrees: Int, temperature_C: Float64, pressure_mbar: Float64,
               n: Int, dst: Int) abi("C") -> None:
    refract(Ptr(unsafe_from_address=alt_degrees), temperature_C, pressure_mbar, n,
            Ptr(unsafe_from_address=dst))


# ----------------------------------------------------------------------------
# skyfield.precessionlib
# ----------------------------------------------------------------------------

@export("sk_compute_precession")
def sk_compute_precession(jd_tdb: Int, n: Int, workers: Int,
                          dst: Int) abi("C") -> None:
    compute_precession(Ptr(unsafe_from_address=jd_tdb), n, workers,
                       Ptr(unsafe_from_address=dst))


# ----------------------------------------------------------------------------
# skyfield.nutationlib
# ----------------------------------------------------------------------------

@export("sk_fundamental_arguments")
def sk_fundamental_arguments(t: Int, terms: Int, n: Int, workers: Int,
                             dst: Int) abi("C") -> None:
    fundamental_arguments(Ptr(unsafe_from_address=t), terms, n, workers,
                          Ptr(unsafe_from_address=dst))


@export("sk_mean_obliquity")
def sk_mean_obliquity(jd_tdb: Int, n: Int, dst: Int) abi("C") -> None:
    mean_obliquity(Ptr(unsafe_from_address=jd_tdb), n, Ptr(unsafe_from_address=dst))


@export("sk_build_nutation_matrix")
def sk_build_nutation_matrix(mean_obliquity: Int, true_obliquity: Int, psi: Int,
                             n: Int, dst: Int) abi("C") -> None:
    build_nutation_matrix(Ptr(unsafe_from_address=mean_obliquity),
                          Ptr(unsafe_from_address=true_obliquity),
                          Ptr(unsafe_from_address=psi), n,
                          Ptr(unsafe_from_address=dst))


@export("sk_equation_of_the_equinoxes_complimentary_terms")
def sk_equation_of_the_equinoxes_complimentary_terms(
        jd_tt: Int, ke0_t: Int, ke1: Int, se0_t_0: Int, se0_t_1: Int, nke: Int,
        n: Int, workers: Int, dst: Int) abi("C") -> None:
    equation_of_the_equinoxes_complimentary_terms(
        Ptr(unsafe_from_address=jd_tt), IPtr(unsafe_from_address=ke0_t),
        IPtr(unsafe_from_address=ke1), Ptr(unsafe_from_address=se0_t_0),
        Ptr(unsafe_from_address=se0_t_1), nke, n, workers,
        Ptr(unsafe_from_address=dst))


@export("sk_iau2000a")
def sk_iau2000a(jd_tt: Int, fundamental_argument_terms: Int, lunisolar_terms: Int,
                planetary_terms: Int, nals_t: Int, ls_long: Int, ls_obl: Int,
                napl_t: Int, nut_long: Int, nut_obl: Int, n: Int, workers: Int,
                d_psi: Int, d_eps: Int) abi("C") -> None:
    iau2000a(Ptr(unsafe_from_address=jd_tt), fundamental_argument_terms,
             lunisolar_terms, planetary_terms, IPtr(unsafe_from_address=nals_t),
             Ptr(unsafe_from_address=ls_long), Ptr(unsafe_from_address=ls_obl),
             IPtr(unsafe_from_address=napl_t), Ptr(unsafe_from_address=nut_long),
             Ptr(unsafe_from_address=nut_obl), n, workers,
             Ptr(unsafe_from_address=d_psi), Ptr(unsafe_from_address=d_eps))


@export("sk_iau2000b")
def sk_iau2000b(jd_tt: Int, fundamental_argument_terms: Int, lunisolar_terms: Int,
                planetary_terms: Int, nals_t: Int, ls_long: Int, ls_obl: Int,
                napl_t: Int, nut_long: Int, nut_obl: Int, n: Int, workers: Int,
                d_psi: Int, d_eps: Int) abi("C") -> None:
    iau2000b(Ptr(unsafe_from_address=jd_tt), fundamental_argument_terms,
             lunisolar_terms, planetary_terms, IPtr(unsafe_from_address=nals_t),
             Ptr(unsafe_from_address=ls_long), Ptr(unsafe_from_address=ls_obl),
             IPtr(unsafe_from_address=napl_t), Ptr(unsafe_from_address=nut_long),
             Ptr(unsafe_from_address=nut_obl), n, workers,
             Ptr(unsafe_from_address=d_psi), Ptr(unsafe_from_address=d_eps))


# ----------------------------------------------------------------------------
# skyfield.framelib
# ----------------------------------------------------------------------------

@export("sk_build_matrix")
def sk_build_matrix(dst: Int) abi("C") -> None:
    build_matrix(Ptr(unsafe_from_address=dst))


@export("sk_build_ecliptic_matrix")
def sk_build_ecliptic_matrix(M: Int, true_obliquity: Int, n: Int, dst: Int,
                             scratch: Int) abi("C") -> None:
    build_ecliptic_matrix(Ptr(unsafe_from_address=M),
                          Ptr(unsafe_from_address=true_obliquity), n,
                          Ptr(unsafe_from_address=dst),
                          Ptr(unsafe_from_address=scratch))


@export("sk_tirs_rotation_at")
def sk_tirs_rotation_at(M: Int, gast: Int, n: Int, dst: Int,
                        scratch: Int) abi("C") -> None:
    tirs_rotation_at(Ptr(unsafe_from_address=M), Ptr(unsafe_from_address=gast), n,
                     Ptr(unsafe_from_address=dst),
                     Ptr(unsafe_from_address=scratch))


@export("sk_itrs_rotation_at")
def sk_itrs_rotation_at(M: Int, gast: Int, polar_motion: Int, n: Int, dst: Int,
                        scratch: Int, scratch2: Int) abi("C") -> None:
    itrs_rotation_at(Ptr(unsafe_from_address=M), Ptr(unsafe_from_address=gast),
                     Ptr(unsafe_from_address=polar_motion), n,
                     Ptr(unsafe_from_address=dst),
                     Ptr(unsafe_from_address=scratch),
                     Ptr(unsafe_from_address=scratch2))


# ----------------------------------------------------------------------------
# skyfield.relativity
# ----------------------------------------------------------------------------

@export("sk_light_time_difference")
def sk_light_time_difference(position: Int, deflector_position: Int, n: Int,
                             dst: Int) abi("C") -> None:
    light_time_difference(Ptr(unsafe_from_address=position),
                          Ptr(unsafe_from_address=deflector_position), n,
                          Ptr(unsafe_from_address=dst))


@export("sk_compute_deflection")
def sk_compute_deflection(position: Int, pe: Int, rmass: Float64, n: Int,
                          dst: Int) abi("C") -> None:
    _compute_deflection(Ptr(unsafe_from_address=position),
                        Ptr(unsafe_from_address=pe), rmass, n,
                        Ptr(unsafe_from_address=dst))


@export("sk_add_aberration")
def sk_add_aberration(position: Int, velocity: Int, light_time: Int,
                      n: Int) abi("C") -> None:
    add_aberration(Ptr(unsafe_from_address=position),
                   Ptr(unsafe_from_address=velocity),
                   Ptr(unsafe_from_address=light_time), n)


# ----------------------------------------------------------------------------
# skyfield.sgp4lib
# ----------------------------------------------------------------------------

@export("sk_cross")
def sk_cross(a: Int, b: Int, n: Int, dst: Int) abi("C") -> None:
    _cross(Ptr(unsafe_from_address=a), Ptr(unsafe_from_address=b), n,
           Ptr(unsafe_from_address=dst))


@export("sk_theta_GMST1982")
def sk_theta_GMST1982(jd_ut1: Int, fraction_ut1: Int, n: Int, d_theta: Int,
                      d_theta_dot: Int) abi("C") -> None:
    theta_GMST1982(Ptr(unsafe_from_address=jd_ut1),
                   Ptr(unsafe_from_address=fraction_ut1), n,
                   Ptr(unsafe_from_address=d_theta),
                   Ptr(unsafe_from_address=d_theta_dot))


@export("sk_TEME_to_ITRF")
def sk_TEME_to_ITRF(jd_ut1: Int, rTEME: Int, vTEME: Int, xp: Float64,
                    yp: Float64, fraction_ut1: Int, n: Int, max_workers: Int,
                    d_rITRF: Int, d_vITRF: Int) abi("C") -> None:
    TEME_to_ITRF(Ptr(unsafe_from_address=jd_ut1), Ptr(unsafe_from_address=rTEME),
                 Ptr(unsafe_from_address=vTEME), xp, yp,
                 Ptr(unsafe_from_address=fraction_ut1), n, max_workers,
                 Ptr(unsafe_from_address=d_rITRF), Ptr(unsafe_from_address=d_vITRF))


@export("sk_TEME_rotation_at")
def sk_TEME_rotation_at(theta: Int, gast: Int, M: Int, n: Int, dst: Int,
                        scratch: Int) abi("C") -> None:
    TEME_rotation_at(Ptr(unsafe_from_address=theta), Ptr(unsafe_from_address=gast),
                     Ptr(unsafe_from_address=M), n, Ptr(unsafe_from_address=dst),
                     Ptr(unsafe_from_address=scratch))


# ----------------------------------------------------------------------------
# skyfield.keplerlib
# ----------------------------------------------------------------------------

@export("sk_eccentric_anomaly")
def sk_eccentric_anomaly(e: Int, M: Int, n: Int, dst: Int) abi("C") -> None:
    eccentric_anomaly(Ptr(unsafe_from_address=e), Ptr(unsafe_from_address=M), n,
                      Ptr(unsafe_from_address=dst))


@export("sk_true_anomaly_hyperbolic")
def sk_true_anomaly_hyperbolic(e: Int, E: Int, n: Int, dst: Int) abi("C") -> None:
    true_anomaly_hyperbolic(Ptr(unsafe_from_address=e), Ptr(unsafe_from_address=E),
                            n, Ptr(unsafe_from_address=dst))


@export("sk_true_anomaly_closed")
def sk_true_anomaly_closed(e: Int, E: Int, n: Int, dst: Int) abi("C") -> None:
    true_anomaly_closed(Ptr(unsafe_from_address=e), Ptr(unsafe_from_address=E), n,
                        Ptr(unsafe_from_address=dst))


@export("sk_true_anomaly_parabolic")
def sk_true_anomaly_parabolic(p: Int, gm: Float64, M: Int, n: Int,
                              dst: Int) abi("C") -> None:
    true_anomaly_parabolic(Ptr(unsafe_from_address=p), gm,
                           Ptr(unsafe_from_address=M), n,
                           Ptr(unsafe_from_address=dst))


@export("sk_ele_to_vec")
def sk_ele_to_vec(p: Int, e: Int, i: Int, Om: Int, w: Int, v: Int, mu: Float64,
                  n: Int, d_position: Int, d_velocity: Int) abi("C") -> None:
    ele_to_vec(Ptr(unsafe_from_address=p), Ptr(unsafe_from_address=e),
               Ptr(unsafe_from_address=i), Ptr(unsafe_from_address=Om),
               Ptr(unsafe_from_address=w), Ptr(unsafe_from_address=v), mu, n,
               Ptr(unsafe_from_address=d_position),
               Ptr(unsafe_from_address=d_velocity))


@export("sk_stumpff")
def sk_stumpff(x: Int, n: Int, d_c0: Int, d_c1: Int, d_c2: Int,
               d_c3: Int) abi("C") -> None:
    stumpff(Ptr(unsafe_from_address=x), n, Ptr(unsafe_from_address=d_c0),
                  Ptr(unsafe_from_address=d_c1), Ptr(unsafe_from_address=d_c2),
                  Ptr(unsafe_from_address=d_c3))


@export("sk_normpi")
def sk_normpi(x: Int, n: Int, dst: Int) abi("C") -> None:
    normpi(Ptr(unsafe_from_address=x), n, Ptr(unsafe_from_address=dst))


@export("sk_propagate")
def sk_propagate(position: Int, velocity: Int, t0: Float64, t1: Int, gm: Float64,
                 n: Int, workers: Int, scratch: Int, flags: Int, d_position: Int,
                 d_velocity: Int) abi("C") -> Int:
    return propagate(Ptr(unsafe_from_address=position),
                     Ptr(unsafe_from_address=velocity), t0,
                     Ptr(unsafe_from_address=t1), gm, n, workers,
                     Ptr(unsafe_from_address=scratch),
                     IPtr(unsafe_from_address=flags),
                     Ptr(unsafe_from_address=d_position),
                     Ptr(unsafe_from_address=d_velocity))
