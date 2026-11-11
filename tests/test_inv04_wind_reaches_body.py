"""Invariant 4 (spec sections 12, 17.1, 0): wind reaches the body. A held drone's net force changes by exactly the air
force. (The test the shipped wrapper would fail: its force is overwritten before MuJoCo steps.)

Method: two Physics objects start from the same state at rest at time t, one in a condition with a gust and one
without; each takes one substep with the same rotor thrust. A constant world-frame force on a rigid body changes its
velocity by F dt / m exactly, so the difference of the two velocities, times m / dt, is the air force MuJoCo applied.
"""
import numpy as np
import pytest

import oracles as O
import spec_numbers as S
from drone_study.air import PRIMARY, air_wrench_world
from drone_study.condition import Condition
from drone_study.physics import Physics

SHAPE, TIER = "fig8", "faster"
PEAK = O.gust_peak(SHAPE, TIER, 0.0)
_, TRAVEL, _ = O.case_state(SHAPE, TIER, PEAK)
HOVER_THRUST = np.full(4, S.WEIGHT / 4)
HOVER_OMEGA = np.full(4, S.OMEGA_HOVER)
I_BODY = np.diag([S.IXX, S.IYY, S.IZZ])


def quat_from_rotation(R):
    w = np.sqrt(max(1 + np.trace(R), 0)) / 2
    return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)])


def rotation(roll_deg, pitch_deg):
    r, p = np.radians(roll_deg), np.radians(pitch_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    return Ry @ Rx


def one_substep(condition, t, R):
    p = Physics(condition, 1)
    p.set_time(t)
    p.set_state(np.array([0.0, 0.0, 3.05]), quat_from_rotation(R), np.zeros(3), np.zeros(3))
    thrust = HOVER_THRUST
    rec = p.substep(thrust, HOVER_OMEGA)
    return p, rec


def gust_condition(az, speed, el=0.0):
    return Condition(case=f"{SHAPE}-{TIER}", az=az, el=el, speed=speed, phase=0.0, turb=0.0)


CALM = Condition(case=f"{SHAPE}-{TIER}", turb=0.0)
TIMES = (PEAK - 0.35, PEAK - 0.25, PEAK, PEAK + 0.1, PEAK + 0.3)          # rising, falling, at the crest
ATTITUDES = ((0.0, 0.0), (25.0, 0.0), (0.0, -30.0), (20.0, 35.0))


@pytest.mark.parametrize("roll,pitch", ATTITUDES)
@pytest.mark.parametrize("t", TIMES)
@pytest.mark.parametrize("az,speed,el", ((0.0, 8.0, 0.0), (135.0, 12.0, 0.0), (270.0, 5.0, 35.0)))
def test_inv4_a_held_drones_net_force_changes_by_exactly_the_air_force(az, speed, el, t, roll, pitch):
    R = rotation(roll, pitch)
    wind, rec = one_substep(gust_condition(az, speed, el), t, R)
    calm, _ = one_substep(CALM, t, R)
    dv = wind.data.qvel[0:3] - calm.data.qvel[0:3]
    net_change = S.MASS * dv / S.DT_PHYS                       # what MuJoCo integrated, in newtons
    # the air velocity at the substep's start time, from the gust alone (the turbulence tape is zero here)
    v_air = O.gust_magnitude(t, speed, PEAK) * O.gust_direction(TRAVEL, az, el)
    f_stack, _ = air_wrench_world(v_air, np.zeros(3), R, HOVER_OMEGA, PRIMARY)
    f_oracle_body, _ = O.air_wrench_body(R.T @ v_air, S.SUM_OMEGA_HOVER)
    assert np.linalg.norm(f_oracle_body) > 0.005               # the gust is really blowing at this time
    np.testing.assert_allclose(net_change, f_stack, rtol=1e-9, atol=1e-9)         # exactly the air force the stack states
    np.testing.assert_allclose(net_change, R @ f_oracle_body, rtol=S.PARAM_RTOL, atol=1e-6)   # and that is section 17.3's
    np.testing.assert_allclose(rec.air_force_world, net_change, rtol=1e-9, atol=1e-9)         # the record says what was applied
    assert rec.time == pytest.approx(t, abs=1e-12)
    np.testing.assert_allclose(rec.air_velocity, v_air, atol=1e-9)


@pytest.mark.parametrize("roll,pitch", ATTITUDES)
def test_inv4_the_moment_reaches_the_body_too(roll, pitch):
    R = rotation(roll, pitch)
    t = PEAK
    wind, rec = one_substep(gust_condition(0.0, 12.0), t, R)
    calm, _ = one_substep(CALM, t, R)
    d_omega_body = wind.data.qvel[3:6] - calm.data.qvel[3:6]    # free joint: angular velocity in the body frame
    torque_body = I_BODY @ d_omega_body / S.DT_PHYS
    v_air = O.gust_magnitude(t, 12.0, PEAK) * O.gust_direction(TRAVEL, 0.0, 0.0)
    _, m_body = O.air_wrench_body(R.T @ v_air, S.SUM_OMEGA_HOVER)
    assert np.linalg.norm(m_body) > 1e-4                       # about 3.3e-4 N m at 12 m/s (section 17.3)
    np.testing.assert_allclose(torque_body, m_body, rtol=S.PARAM_RTOL + 1e-3, atol=1e-7)   # the body turns a little in the step
    np.testing.assert_allclose(R @ torque_body, rec.air_moment_world, rtol=1e-3, atol=1e-7)


def test_inv4_no_gust_means_no_change():
    """The control: with no gust the two runs are the same run, so the test above is not seeing something else."""
    a, _ = one_substep(CALM, PEAK, np.eye(3))
    b, _ = one_substep(Condition(case=f"{SHAPE}-{TIER}", az=0.0, el=0.0, speed=0.0, phase=0.0, turb=0.0), PEAK, np.eye(3))
    np.testing.assert_array_equal(a.data.qvel, b.data.qvel)
    assert np.linalg.norm(a.data.qvel[0:3]) < 1e-9             # thrust equals weight, drone at rest: it stays at rest
