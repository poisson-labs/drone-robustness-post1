"""Invariant 6 (spec sections 12, 17.4): a step in rate command settles with the firmware gains' structure; the loop
reads the on-board gyro and nothing from the outer stream, so changing the delay axis leaves it unchanged given the
same rate command; its D term runs on the measured rate; its integrals clamp at section 17.4's limits; the mixer's cap
takes the overshoot off all four motors, without negative thrust.

Units: the firmware's rate PID works in degrees per second (3A-4). Firmware facts are from crazyflie-firmware
e828ed2: pid.c (D on the measured value, integral clamped before ki, no output limit in the PID itself) and
attitude_pid_controller.c (saturateSignedInt16 is +-INT16_MAX).
"""
import math

import numpy as np
import pytest

import oracles as O
import spec_numbers as S
from drone_study.actuation import Actuation, mix
from drone_study.condition import Condition
from drone_study.physics import Physics
from drone_study.rate_loop import RateLoop
from drone_study.sensing import OnboardGyro, SensingConfig
from drone_study.stack import ScriptedController, fly
from drone_study.streams import build_tapes

DEG = math.degrees(1.0)
AXES = ("roll", "pitch", "yaw")


def zeros3():
    return np.zeros(3)


# ---- the PID's structure ------------------------------------------------------------------------------------------------
def test_inv6_p_term_is_the_firmware_gain_per_degree_per_second():
    loop = RateLoop()
    out = loop.step(np.array([1.0, 1.0, 1.0]), zeros3())
    for i, a in enumerate(AXES):
        kp, ki, kd = S.RATE_GAINS[a]
        assert loop.terms["p"][i] == pytest.approx(kp * DEG, rel=1e-9)        # 1 rad/s is 57.2958 deg/s into the gain
        assert loop.terms["d"][i] == 0.0
        assert loop.terms["i"][i] == pytest.approx(ki * DEG * S.DT_PHYS, rel=1e-9)
        assert out[i] == pytest.approx(loop.terms["p"][i] + loop.terms["i"][i], rel=1e-9)


def test_inv6_the_whole_loop_matches_a_reference_written_from_the_firmware_source():
    rng = np.random.default_rng(1)
    loop, ref = RateLoop(), O.RefRateLoop()
    meas = np.concatenate([np.zeros((5, 3)), np.cumsum(rng.normal(0, 0.02, (495, 3)), axis=0)])
    cmd = np.concatenate([np.zeros((5, 3)), np.repeat(rng.normal(0, 1.0, (55, 3)), 9, axis=0)])
    for k in range(len(meas)):
        out = loop.step(cmd[k], meas[k])
        want, terms = ref.step(cmd[k], meas[k])
        np.testing.assert_allclose(out, want, rtol=1e-9, atol=1e-6)
        for key in "pid":
            np.testing.assert_allclose(loop.terms[key], terms[key], rtol=1e-9, atol=1e-6)


def test_inv6_the_d_term_runs_on_the_measured_rate_and_does_not_kick_on_a_command_step():
    loop = RateLoop()
    for k in range(20):
        loop.step(np.array([0.0 if k < 10 else 3.0] * 3), zeros3())          # a step in command, measurement still
        assert np.all(loop.terms["d"] == 0.0)
    loop = RateLoop()
    slope = 0.001                                                             # rad/s per tick
    for k in range(1, 30):
        loop.step(zeros3(), np.full(3, slope * k))
        for i, a in enumerate(AXES):
            kd = S.RATE_GAINS[a][2]
            assert loop.terms["d"][i] == pytest.approx(-kd * math.degrees(slope) / S.DT_PHYS, rel=1e-9)
    assert loop.terms["d"][0] < 0                                             # rate rising: D opposes it


@pytest.mark.parametrize("sign", (+1.0, -1.0))
def test_inv6_integrals_clamp_at_the_firmware_limits(sign):
    loop = RateLoop()
    for _ in range(4000):                                                     # 8 s of a 1 rad/s error: far past any clamp
        loop.step(sign * np.ones(3), zeros3())
    for i, a in enumerate(AXES):
        ki = S.RATE_GAINS[a][1]
        assert loop.terms["i"][i] == pytest.approx(sign * ki * S.RATE_I_LIMIT[a], rel=1e-9)   # clamp on the state, then * ki
    # the clamp bounds the state, so the integral leaves it at once when the error reverses
    loop.step(-sign * np.ones(3), zeros3())
    for i, a in enumerate(AXES):
        assert abs(loop.terms["i"][i]) < S.RATE_GAINS[a][1] * S.RATE_I_LIMIT[a]


@pytest.mark.parametrize("sign", (+1.0, -1.0))
def test_inv6_the_output_saturates_to_a_signed_16_bit_value(sign):
    loop = RateLoop()
    out = loop.step(sign * np.full(3, 100.0), zeros3())
    np.testing.assert_array_equal(out, sign * np.full(3, S.INT16_MAX))
    out = RateLoop().step(sign * np.array([0.5, 0.5, 0.5]), zeros3())
    assert np.all(np.abs(out) < S.INT16_MAX)                                  # the control: below the limit it passes


# ---- the loop in the stack ------------------------------------------------------------------------------------------------
def gimbal_run(axis, step, seconds=3.0, t_on=0.1):
    """The rate loop on the real plant with the drone's position held, so only its rotation is under test: the on-board
    gyro, the rate PID, the firmware mixer, the 72 ms motors and the rigid body, composed from the public pieces. (A drone
    free to translate adds a drag moment of its own, which has nothing to do with the loop.)"""
    cond = Condition.calm("circle-medium")
    tapes = build_tapes(cond, 3)
    phys = Physics(cond, 3)
    gyro, loop, act = OnboardGyro(tapes.sensor, scale=0.0), RateLoop(), Actuation(S.DT_PHYS)
    act.motors.omega[:] = S.OMEGA_HOVER
    phys.set_state(np.array([0.0, 0.0, S.HEIGHT_M]), np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), np.zeros(3))
    hover_units = S.UNITS_MAX / 1.5
    n = int(round(seconds / S.DT_PHYS))
    t, rate = np.arange(n) * S.DT_PHYS, np.zeros((n, 3))
    for j in range(n):
        cmd = np.zeros(3)
        if t[j] >= t_on:
            cmd[axis] = step
        measured = gyro.measure(np.array(phys.data.qvel[3:6]))
        roll, pitch, yaw = loop.step(cmd, measured)
        thrust = act.step(hover_units, roll, pitch, yaw)
        phys.substep(thrust, act.motors.omega)
        phys.data.qpos[0:3] = [0.0, 0.0, S.HEIGHT_M]
        phys.data.qvel[0:3] = 0.0
        rate[j] = phys.data.qvel[3:6]
    return t, rate


@pytest.mark.parametrize("axis", (0, 1, 2), ids=AXES)
def test_inv6_a_step_in_rate_command_settles_with_the_firmware_gains(axis):
    """3A-1: 'settles' is within 1% of the step (borrowed from invariant 5) by 1.5 s after the step, staying inside to
    the end of the run. A single-axis model of this plant and these gains (checks/rate_axis_check.py) reaches 1% at
    0.57 s on roll and pitch and 0.87 s on yaw, with overshoot of 39% and 64%."""
    step, t_on = 0.2, 0.1
    t, rate = gimbal_run(axis, step, t_on=t_on)
    late = t >= t_on + S.SETTLE_BY_S
    assert late.sum() > 500
    assert np.abs(rate[late, axis] - step).max() <= S.SETTLE_TOL * step
    assert rate[late, axis].mean() == pytest.approx(step, rel=S.SETTLE_TOL)             # it settles to the commanded rate
    others = [a for a in range(3) if a != axis]
    assert np.abs(rate[:, others]).max() <= S.SETTLE_TOL * step                          # and the step moves one axis only
    assert np.abs(rate[t < t_on]).max() < 1e-9                                           # nothing moves before the command


def _open_loop_script(view):
    k = int(round(view.t * 100))
    return [S.WEIGHT * (1.0 + 0.1 * math.sin(0.2 * k)), 0.5 * math.sin(0.13 * k), 0.4 * math.cos(0.09 * k), 0.3 * math.sin(0.05 * k)]


def test_inv6_it_reads_the_gyro_and_nothing_from_the_outer_stream():
    base = Condition(case="circle-fast", turb=0.3)
    delayed = Condition(case="circle-fast", turb=0.3, delay=60)
    runs = {
        "clean": fly(base, 5, ScriptedController(_open_loop_script), max_steps=300),
        "delay 60 ms": fly(delayed, 5, ScriptedController(_open_loop_script), max_steps=300),
        "outer noise x100": fly(base, 5, ScriptedController(_open_loop_script), sensing=SensingConfig(outer_scale=100.0), max_steps=300),
        "no outer noise": fly(base, 5, ScriptedController(_open_loop_script), sensing=SensingConfig(outer_scale=0.0), max_steps=300),
    }
    ref = runs["clean"]
    assert np.abs(ref.rotor_omega).max() > 0 and np.abs(ref.mixer_output).max() > 0       # it flew
    for name, rec in runs.items():
        for field in ("position", "quat", "body_rate", "rotor_omega", "mixer_output", "applied_action"):
            np.testing.assert_array_equal(getattr(rec, field), getattr(ref, field), err_msg=f"{name}: {field}")
    # the control: the gyro IS what the loop reads, so removing the gyro's noise changes the flight
    quiet = fly(base, 5, ScriptedController(_open_loop_script), sensing=SensingConfig(gyro_scale=0.0), max_steps=300)
    assert not np.array_equal(quiet.rotor_omega, ref.rotor_omega)
    loud = fly(base, 5, ScriptedController(_open_loop_script), sensing=SensingConfig(gyro_scale=50.0), max_steps=300)
    assert not np.array_equal(loud.mixer_output, ref.mixer_output)


# ---- the mixer and its cap -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("request_", [
    (60000.0, 20000.0, 0.0, 0.0), (60000.0, 0.0, -24000.0, 0.0), (62000.0, 6000.0, 6000.0, 9000.0),
    (65535.0, 0.0, 0.0, 0.0), (43690.0, 1000.0, -1000.0, 500.0), (5000.0, 20000.0, 0.0, 0.0), (3000.0, -9000.0, 4000.0, -8000.0),
])
def test_inv6_the_cap_takes_the_overshoot_off_all_four_motors_and_never_goes_negative(request_):
    got = mix(*request_)
    want = O.mix_capped(*request_)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-9)
    assert np.all(got >= 0.0) and np.all(got <= S.UNITS_MAX + 1e-9)
    raw = O.mix_uncapped(*request_)
    over = max(raw.max() - S.UNITS_MAX, 0.0)
    if over > 0 and raw.min() - over >= 0:
        np.testing.assert_allclose(got, raw - over, atol=1e-9)                # the same amount off every motor
        assert got.max() == pytest.approx(S.UNITS_MAX, abs=1e-9)
        np.testing.assert_allclose(got - got.mean(), raw - raw.mean(), atol=1e-9)    # torques are untouched; thrust gave way


def test_inv6_thrust_gives_way_first_when_the_cap_bites():
    request = (60000.0, 20000.0, 0.0, 4000.0)
    got = mix(*request)
    t, r, p, y = O.unmix(got)
    assert t < request[0]
    assert r == pytest.approx(request[1], abs=1e-9) and p == pytest.approx(request[2], abs=1e-9)
    assert y == pytest.approx(request[3], abs=1e-9)


def test_inv6_a_random_request_never_gives_negative_thrust_or_exceeds_the_motor_maximum():
    rng = np.random.default_rng(9)
    for _ in range(500):
        req = (rng.uniform(0, 65535), rng.uniform(-40000, 40000), rng.uniform(-40000, 40000), rng.uniform(-20000, 20000))
        got = mix(*req)
        np.testing.assert_allclose(got, O.mix_capped(*req), atol=1e-9)
        assert np.all(got >= 0.0) and np.all(got <= S.UNITS_MAX + 1e-9)


@pytest.mark.parametrize("which", (1, 2, 3), ids=("roll", "pitch", "yaw"))
def test_inv6_the_mixers_signs_give_the_commanded_body_torque(which):
    base = S.UNITS_MAX / 1.5
    req = [base, 0.0, 0.0, 0.0]
    req[which] = 2000.0
    got = mix(*req)
    f = S.T_MAX / 4 * (got / S.UNITS_MAX) ** 2                                 # thrust goes as units squared (3A-3)
    tau = O.body_torques_from_thrusts(f)
    assert tau[which - 1] > 0
    for other in range(3):
        if other != which - 1:
            assert abs(tau[other]) < 1e-6 * abs(tau[which - 1]) + 1e-12
