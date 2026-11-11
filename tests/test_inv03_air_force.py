"""Invariant 3 (spec sections 12, 17.1, 17.3): MuJoCo's density and viscosity are 0 and opt.wind is unused; the air
force is zero when drone and air move together; odd in relative velocity; computed in the body frame at the current
attitude and rotor speed, so flat at hover it equals section 17.3's table for each air model at every azimuth; the
linear term's moment equals its force times the 5 mm rotor-plane offset.
"""
import numpy as np
import pytest

import oracles as O
import spec_numbers as S
from drone_study.air import (LINEAR_ONLY, PRIMARY, QUAD_COP_10MM, QUAD_DOUBLED, air_wrench_body, air_wrench_world)
from drone_study.condition import Condition
from drone_study.physics import Physics

MODELS = {"primary": PRIMARY, "linear-only": LINEAR_ONLY, "quad-doubled": QUAD_DOUBLED, "quad-cop-10mm": QUAD_COP_10MM}
HOVER = np.full(4, S.OMEGA_HOVER)
AZIMUTHS = np.arange(0.0, 360.0, 22.5)
RNG = np.random.default_rng(20261105)


def rand_rotation(rng, max_tilt_deg=60.0):
    axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
    ang = np.radians(rng.uniform(0, max_tilt_deg))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def test_inv3_mujocos_own_fluid_model_is_off_and_opt_wind_is_unused():
    p = Physics(Condition(case="fig8-faster", az=0.0, el=0.0, speed=8.0, phase=0.0), 1)
    assert p.model.opt.density == 0.0
    assert p.model.opt.viscosity == 0.0
    assert np.all(p.model.opt.wind == 0.0)


def test_inv3_hover_rotor_speed_sum_is_the_one_in_the_table():
    assert S.SUM_OMEGA_HOVER == pytest.approx(6064, abs=1)           # section 17.3: "about 6,064 rad/s"
    f, _ = air_wrench_body(np.array([1.0, 0.0, 0.0]), HOVER, LINEAR_ONLY)
    assert f[0] == pytest.approx(5.57e-3, abs=5e-6)                  # "5.57e-3 N/(m/s)" at hover


@pytest.mark.parametrize("yaw_deg", (0.0, 37.0))
@pytest.mark.parametrize("az", AZIMUTHS)
def test_inv3_flat_at_hover_it_equals_the_table_at_every_azimuth(az, yaw_deg):
    R = O.rot_z(np.radians(yaw_deg))
    for i, v in enumerate(S.TABLE_SPEEDS):
        direction = np.array([np.cos(np.radians(az)), np.sin(np.radians(az)), 0.0])
        v_air = v * direction
        f_pri, _ = air_wrench_world(v_air, np.zeros(3), R, HOVER, PRIMARY)
        f_lin, _ = air_wrench_world(v_air, np.zeros(3), R, HOVER, LINEAR_ONLY)
        # the table, to half a unit of its last printed digit
        assert np.linalg.norm(f_lin) == pytest.approx(S.TABLE_LINEAR_N[i], abs=S.TABLE_TOL_N)
        assert np.linalg.norm(f_pri) == pytest.approx(S.TABLE_PRIMARY_N[i], abs=S.TABLE_TOL_N)
        assert np.linalg.norm(f_pri) / S.MASS == pytest.approx(S.TABLE_PRIMARY_MS2[i], abs=S.TABLE_TOL_MS2 + 0.01)
        assert 100 * np.linalg.norm(f_pri) / S.LATERAL_LIMIT_N == pytest.approx(S.TABLE_PRIMARY_SHARE_PCT[i], abs=S.TABLE_TOL_PCT)
        # and the formula behind it, which does not round
        want, _ = O.air_wrench_body(R.T @ v_air, S.SUM_OMEGA_HOVER)
        np.testing.assert_allclose(R.T @ f_pri, want, rtol=S.PARAM_RTOL, atol=1e-9)
        # the force points along the relative air velocity, whatever the azimuth
        cosine = f_pri @ v_air / (np.linalg.norm(f_pri) * v)
        assert cosine == pytest.approx(1.0, abs=1e-9)


def test_inv3_the_quadratic_term_is_the_same_all_round_in_plan():
    """Section 17.3: a diagonal 12 m/s gust is 0.302 N, not the 0.233 N a per-axis form would give."""
    along = air_wrench_world(np.array([12.0, 0.0, 0.0]), np.zeros(3), np.eye(3), HOVER, PRIMARY)[0]
    diagonal = air_wrench_world(12.0 * np.array([1.0, 1.0, 0.0]) / np.sqrt(2), np.zeros(3), np.eye(3), HOVER, PRIMARY)[0]
    assert np.linalg.norm(along) == pytest.approx(0.302, abs=S.TABLE_TOL_N)
    assert np.linalg.norm(diagonal) == pytest.approx(np.linalg.norm(along), rel=1e-12)


@pytest.mark.parametrize("name", list(MODELS))
def test_inv3_each_air_model_matches_its_definition(name):
    kw = O.AIR_MODELS[name]
    rng = np.random.default_rng(3)
    for _ in range(40):
        v = rng.normal(0, 6, 3)
        omega = HOVER * rng.uniform(0.3, 1.5)
        f, m = air_wrench_body(v, omega, MODELS[name])
        wf, wm = O.air_wrench_body(v, 4 * omega[0], **kw)
        np.testing.assert_allclose(f, wf, rtol=S.PARAM_RTOL, atol=1e-9)
        np.testing.assert_allclose(m, wm, rtol=S.PARAM_RTOL, atol=1e-10)


def test_inv3_the_s1_models_differ_from_the_primary_only_where_they_say():
    v = np.array([7.0, -3.0, 2.0])
    f0, m0 = air_wrench_body(v, HOVER, PRIMARY)
    f_lin, m_lin = air_wrench_body(v, HOVER, LINEAR_ONLY)
    f_dbl, m_dbl = air_wrench_body(v, HOVER, QUAD_DOUBLED)
    f_cop, m_cop = air_wrench_body(v, HOVER, QUAD_COP_10MM)
    f_quad = f0 - f_lin
    np.testing.assert_allclose(f_dbl, f_lin + 2 * f_quad, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(m_dbl, m_lin, rtol=1e-9, atol=1e-15)             # the quadratic term has no moment
    np.testing.assert_allclose(f_cop, f0, rtol=1e-12, atol=1e-15)               # the offset moves the moment, not the force
    np.testing.assert_allclose(m_cop - m0, np.cross([0, 0, 0.010], f_quad), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(m0, m_lin, rtol=1e-9, atol=1e-15)                # primary: quadratic acts at the centre of mass
    assert np.linalg.norm(f_quad) > 0.01


@pytest.mark.parametrize("name", list(MODELS))
def test_inv3_it_is_zero_when_drone_and_air_move_together(name):
    rng = np.random.default_rng(5)
    for _ in range(20):
        v = rng.normal(0, 5, 3)
        R = rand_rotation(rng)
        f, m = air_wrench_world(v, v, R, HOVER * rng.uniform(0.5, 1.5), MODELS[name])
        assert np.all(f == 0.0) and np.all(m == 0.0)
    f, m = air_wrench_world(np.array([9.0, 0, 0]), np.array([0.0, 0, 0]), np.eye(3), HOVER, MODELS[name])
    assert np.linalg.norm(f) > 0.05                      # the control: it is not zero when they differ


@pytest.mark.parametrize("name", list(MODELS))
def test_inv3_it_is_odd_in_relative_velocity(name):
    rng = np.random.default_rng(6)
    for _ in range(30):
        v = rng.normal(0, 6, 3)
        omega = HOVER * rng.uniform(0.4, 1.6)
        f1, m1 = air_wrench_body(v, omega, MODELS[name])
        f2, m2 = air_wrench_body(-v, omega, MODELS[name])
        np.testing.assert_allclose(f2, -f1, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(m2, -m1, rtol=1e-12, atol=1e-18)
        assert np.linalg.norm(f1) > 0


def test_inv3_it_is_computed_in_the_body_frame_at_the_current_attitude():
    rng = np.random.default_rng(7)
    differs = 0
    for _ in range(40):
        R = rand_rotation(rng)
        v_air = rng.normal(0, 6, 3)
        v_drone = rng.normal(0, 1, 3)
        omega = HOVER * rng.uniform(0.6, 1.4)
        f, m = air_wrench_world(v_air, v_drone, R, omega, PRIMARY)
        wf, wm = O.air_wrench_body(R.T @ (v_air - v_drone), 4 * omega[0])
        np.testing.assert_allclose(R.T @ f, wf, rtol=S.PARAM_RTOL, atol=1e-9)
        np.testing.assert_allclose(R.T @ m, wm, rtol=S.PARAM_RTOL, atol=1e-10)
        flat, _ = air_wrench_world(v_air, v_drone, np.eye(3), omega, PRIMARY)
        differs += np.linalg.norm(f - flat) > 0.05 * np.linalg.norm(flat)
    assert differs > 20                                  # attitude matters: a tilted drone shows more top face


def test_inv3_a_tilted_drone_catches_more_air_than_a_flat_one():
    """Section 17.3: the top face (92 x 92 mm) is bigger than the side (92 x 29 mm)."""
    th = np.radians(45.0)
    R = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    v = np.array([10.0, 0.0, 0.0])
    f_flat = np.linalg.norm(air_wrench_world(v, np.zeros(3), np.eye(3), HOVER, PRIMARY)[0])
    f_tilt = np.linalg.norm(air_wrench_world(v, np.zeros(3), R, HOVER, PRIMARY)[0])
    assert f_tilt > 1.15 * f_flat


def test_inv3_the_linear_term_follows_the_current_rotor_speed():
    v = np.array([6.0, -2.0, 1.0])
    f1, m1 = air_wrench_body(v, HOVER, LINEAR_ONLY)
    f2, m2 = air_wrench_body(v, 2 * HOVER, LINEAR_ONLY)
    np.testing.assert_allclose(f2, 2 * f1, rtol=1e-12)
    np.testing.assert_allclose(m2, 2 * m1, rtol=1e-12)
    f0, m0 = air_wrench_body(v, np.zeros(4), PRIMARY)               # rotors stopped: the quadratic term alone
    want, _ = O.air_wrench_body(v, 0.0)
    np.testing.assert_allclose(f0, want, rtol=S.PARAM_RTOL, atol=1e-9)
    assert np.linalg.norm(f0) > 0.01
    assert np.all(m0 == 0.0)
    # the rotor speeds enter as a sum: the same total spread differently gives the same force
    f3, _ = air_wrench_body(v, np.array([1.2, 0.8, 1.1, 0.9]) * S.OMEGA_HOVER, LINEAR_ONLY)
    np.testing.assert_allclose(f3, f1, rtol=1e-12)


@pytest.mark.parametrize("name", list(MODELS))
def test_inv3_the_linear_moment_is_its_force_times_the_five_millimetre_offset(name):
    rng = np.random.default_rng(8)
    lever = np.array([0.0, 0.0, S.ROTOR_PLANE_OFFSET_M])
    for _ in range(20):
        v = rng.normal(0, 6, 3)
        f_lin, m_lin = air_wrench_body(v, HOVER, LINEAR_ONLY)
        assert np.linalg.norm(f_lin) > 1e-3
        np.testing.assert_allclose(m_lin, np.cross(lever, f_lin), rtol=1e-9, atol=1e-15)
        assert m_lin[2] == 0.0
        f, m = air_wrench_body(v, HOVER, MODELS[name])
        quad = f - f_lin                                           # whatever else this model adds is the quadratic term
        cop = 0.010 if name == "quad-cop-10mm" else 0.0
        np.testing.assert_allclose(m, np.cross(lever, f_lin) + np.cross([0, 0, cop], quad), rtol=1e-9, atol=1e-13)
    f, m = air_wrench_body(np.array([12.0, 0.0, 0.0]), HOVER, LINEAR_ONLY)
    assert abs(m[1]) == pytest.approx(S.ROTOR_PLANE_OFFSET_M * S.K_LIN_XY * S.SUM_OMEGA_HOVER * 12.0, rel=1e-3)
    assert abs(m[1]) == pytest.approx(3.3e-4, rel=0.02)            # section 17.3: "3.3e-4 N m" at 12 m/s, 24 rad/s^2 on roll
