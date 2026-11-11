"""Invariant 7 (spec sections 12, 17.4, 17.9): delay shifts the outer stream by the stated steps; noise statistics and
the autocorrelation at section 17.4's time constant match the table; truth never reaches the outer stream.

Statistical checks use the convention in 3A-6: an estimate is accepted within 5 standard errors.
"""
import dataclasses
import functools
import math

import numpy as np
import pytest
from scipy import signal

import oracles as O
import spec_numbers as S
from drone_study.condition import Condition
from drone_study.sensing import OnboardGyro, OuterReading, OuterStream, SensingConfig, TrueState
from drone_study.stack import ControllerView, ScriptedController, StackEnv, fly
from drone_study.streams import build_tapes

COND = Condition(case="circle-medium", turb=0.3)


def truth_at(k):
    return TrueState(np.array([0.01 * k, -0.02 * k, 3.05]), np.array([0.5 + 0.001 * k, 0.0, -0.001 * k]), np.eye(3),
                     np.array([0.001 * k, 0.0, 0.002 * k]))


def rodrigues(v):
    th = np.linalg.norm(v)
    if th < 1e-15:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


# ---- delay -----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("delay_ms,steps", sorted(S.DELAY_STEPS.items()))
def test_inv7_the_delay_axis_shifts_the_outer_stream_by_the_stated_steps(delay_ms, steps):
    tape = build_tapes(COND, 1).sensor
    stream = OuterStream(tape, delay_steps=steps, scale=0.0)
    for k in range(40):
        stream.push(truth_at(k))
        r = stream.read()
        if k >= S.SENSOR_LATENCY_STEPS + steps:
            want = truth_at(k - S.SENSOR_LATENCY_STEPS - steps)
            np.testing.assert_allclose(r.position, want.position, atol=1e-12)
            np.testing.assert_allclose(r.velocity, want.velocity, atol=1e-12)
            np.testing.assert_allclose(r.body_rate, want.body_rate, atol=1e-12)


def test_inv7_nominal_sensor_latency_is_one_control_step_and_the_delay_axis_adds_to_it():
    tape = build_tapes(COND, 1).sensor
    base, late = OuterStream(tape, delay_steps=0, scale=0.0), OuterStream(tape, delay_steps=6, scale=0.0)
    seen_base, seen_late = [], []
    for k in range(30):
        base.push(truth_at(k)); late.push(truth_at(k))
        seen_base.append(base.read().position[0]); seen_late.append(late.read().position[0])
    assert seen_base[10] == pytest.approx(0.01 * 9, abs=1e-12)                 # 10 ms: one step
    assert seen_late[16] == pytest.approx(seen_base[10], abs=1e-12)            # 60 ms more: six steps later


@pytest.mark.parametrize("delay_ms,steps", sorted(S.DELAY_STEPS.items()))
def test_inv7_in_the_stack_the_condition_delay_is_the_same_shift(delay_ms, steps):
    cond = Condition(case="circle-medium", turb=0.3, delay=delay_ms)
    env = StackEnv(cond, 2, sensing=SensingConfig(outer_scale=0.0))
    reading = env.reset()
    truths, readings = [env.truth], [reading]
    for k in range(60):
        result = env.step(np.array([S.WEIGHT, 0.0, 0.0, 0.0]))
        truths.append(env.truth); readings.append(result.reading)
    moved = np.linalg.norm(truths[-1].position - truths[0].position)
    assert moved > 1e-4                                                         # the drone has moved, so a shift can be seen
    for k in range(S.SENSOR_LATENCY_STEPS + steps + 1, 60):
        np.testing.assert_allclose(readings[k].position, truths[k - S.SENSOR_LATENCY_STEPS - steps].position, atol=1e-9)


def test_inv7_an_action_takes_effect_one_control_step_later():
    """Section 17.4 table: action latency, outer stream, one control step (10 ms)."""
    hover = np.array([S.WEIGHT, 0.0, 0.0, 0.0])
    actions = [np.array([S.WEIGHT * (1 + 0.001 * k), 0.01 * k, -0.01 * k, 0.005 * k]) for k in range(1, 60)]
    chosen = []

    def script(view):
        a = actions[len(chosen)]
        chosen.append(a)
        return a

    rec = fly(COND, 4, ScriptedController(script), max_steps=40)
    np.testing.assert_allclose(rec.applied_action[0], hover, atol=1e-12)                 # nothing chosen yet: the hover hold
    for k in range(1, 40):
        np.testing.assert_allclose(rec.applied_action[k], actions[k - 1], atol=1e-12)    # step k flies step k-1's choice


def test_inv7_in_flight_the_response_to_an_action_starts_one_control_step_later():
    """The record above says what was applied; this checks it in the air. A rate step chosen at step k0 flies during step
    k0 + 1, so the body rate moves at the start of step k0 + 2 and not before."""
    k0 = 20

    def script(view):
        k = int(round(view.t * 100))
        return [S.WEIGHT, 0.5 if k >= k0 else 0.0, 0.0, 0.0]

    rec = fly(Condition.calm("circle-medium"), 4, ScriptedController(script), sensing=SensingConfig(gyro_scale=0.0, outer_scale=0.0), max_steps=40)
    roll = rec.body_rate[:, 0]
    assert abs(roll[k0 + 1] - roll[k0]) < 0.005                  # the step after the choice: nothing yet
    assert roll[k0 + 2] - roll[k0 + 1] > 0.03                    # one step later: it has begun
    assert roll[k0 + 4] > 0.2


# ---- the noise ---------------------------------------------------------------------------------------------------------------
def test_inv7_the_noise_added_to_the_outer_stream_is_the_tape():
    tapes = build_tapes(COND, 3)
    stream = OuterStream(tapes.sensor, delay_steps=0, scale=1.0)
    truth = truth_at(0)
    for k in range(60):
        stream.push(truth)
        r = stream.read()
        np.testing.assert_allclose(r.position - truth.position, tapes.sensor.pos_offset + tapes.sensor.pos_ar[k], atol=1e-12)
        np.testing.assert_allclose(r.velocity - truth.velocity, tapes.sensor.vel_ar[k], atol=1e-12)
        np.testing.assert_allclose(r.body_rate - truth.body_rate, tapes.sensor.rate_white[k], atol=1e-12)
        np.testing.assert_allclose(r.rotation, rodrigues(tapes.sensor.att_ar[k]) @ truth.rotation, atol=1e-9)
    assert np.abs(tapes.sensor.pos_ar).max() > 0


@functools.lru_cache(maxsize=None)
def _ensemble(attr, seeds=300):
    x = np.stack([getattr(build_tapes(COND, s).sensor, attr) for s in range(seeds)])
    x.setflags(write=False)
    return x


def _std_ok(x, sigma):
    n = x.size
    return abs(x.std(ddof=0) - sigma) <= S.SIGMA_K * sigma / math.sqrt(2 * n)


@pytest.mark.parametrize("attr,sigma", [("pos_ar", S.POS_AR_SIGMA), ("vel_ar", S.VEL_AR_SIGMA), ("att_ar", S.ATT_AR_SIGMA),
                                        ("rate_white", S.RATE_WHITE_SIGMA)])
def test_inv7_marginal_spreads_match_the_table(attr, sigma):
    x = _ensemble(attr)                                    # (seeds, 1000, 3)
    for step in (200, 500, 800):                           # far apart, so the three are independent
        assert _std_ok(x[:, step, :], sigma), (attr, step, x[:, step, :].std(), sigma)
    assert abs(x.mean()) <= S.SIGMA_K * sigma / math.sqrt(x[:, ::50, :].size)


def test_inv7_position_has_a_per_episode_offset_on_top_of_the_ar_noise():
    offs = np.stack([build_tapes(COND, s).sensor.pos_offset for s in range(600)])
    assert _std_ok(offs, S.POS_OFFSET_SIGMA)
    ar = _ensemble("pos_ar", 300)
    both = offs[:300, None, :] + ar
    total = math.hypot(S.POS_OFFSET_SIGMA, S.POS_AR_SIGMA)
    assert _std_ok(both[:, 500, :], total)


def test_inv7_the_gyro_tape_is_white_noise_at_the_sensors_density():
    x = np.stack([build_tapes(COND, s).sensor.gyro_white for s in range(20)])
    assert _std_ok(x, S.GYRO_SIGMA)
    lag1 = np.mean(x[:, 1:, :] * x[:, :-1, :]) / x.var()
    assert abs(lag1) <= S.SIGMA_K / math.sqrt(x[:, 1:, :].size)


def _lag_corr(x, step_list, lag):
    a = np.concatenate([x[:, s, :].ravel() for s in step_list])
    b = np.concatenate([x[:, s + lag, :].ravel() for s in step_list])
    n = a.size
    rho = float(np.mean(a * b) / math.sqrt(np.mean(a * a) * np.mean(b * b)))
    return rho, n


@pytest.mark.parametrize("attr", ("pos_ar", "vel_ar", "att_ar"))
@pytest.mark.parametrize("lag_steps", (1, 10, 30))
def test_inv7_the_autocorrelation_is_exp_minus_lag_over_tau(attr, lag_steps):
    x = _ensemble(attr, 400)
    rho, n = _lag_corr(x, (200, 500, 800), lag_steps)
    want = math.exp(-lag_steps * S.DT_CTRL / S.AR_TAU_S)                    # tau = 0.1 s
    se = (1 - want ** 2) / math.sqrt(n)
    assert abs(rho - want) <= S.SIGMA_K * se, (attr, lag_steps, rho, want)


def test_inv7_body_rate_noise_on_the_outer_stream_is_white():
    x = _ensemble("rate_white", 400)
    rho, n = _lag_corr(x, (200, 500, 800), 1)
    assert abs(rho) <= S.SIGMA_K / math.sqrt(n)
    assert x.std() == pytest.approx(S.RATE_WHITE_SIGMA, rel=0.05)


# ---- the on-board gyro -----------------------------------------------------------------------------------------------------
B80, A80 = signal.butter(2, S.GYRO_LPF_HZ / (S.PHYS_HZ / 2))


def test_inv7_the_gyro_is_the_true_rate_plus_white_noise_through_the_80_hz_two_pole_filter_with_no_added_delay():
    tape = build_tapes(COND, 6).sensor
    rng = np.random.default_rng(2)
    true = np.concatenate([np.zeros((10, 3)), np.cumsum(rng.normal(0, 0.05, (990, 3)), axis=0)])
    for scale in (0.0, 1.0):
        gyro = OnboardGyro(tape, scale=scale)
        out = np.stack([gyro.measure(true[j]) for j in range(1000)])
        want = signal.lfilter(B80, A80, true + scale * tape.gyro_white[:1000], axis=0)
        np.testing.assert_allclose(out, want, rtol=0, atol=1e-12)
    assert np.abs(out).max() > 0


def test_inv7_the_gyro_filter_has_unit_dc_gain_and_minus_3_db_at_80_hz():
    tape = build_tapes(COND, 6).sensor
    gyro = OnboardGyro(tape, scale=0.0)
    for _ in range(400):
        settled = gyro.measure(np.array([1.0, -2.0, 0.5]))
    np.testing.assert_allclose(settled, [1.0, -2.0, 0.5], rtol=1e-9)
    gyro = OnboardGyro(tape, scale=0.0)
    t = np.arange(2000) / S.PHYS_HZ
    out = np.array([gyro.measure(np.full(3, math.sin(2 * math.pi * S.GYRO_LPF_HZ * tk)))[0] for tk in t])
    assert np.abs(out[1000:]).max() == pytest.approx(1 / math.sqrt(2), rel=1e-2)


def test_inv7_filtered_gyro_noise_has_the_spread_the_filter_gives():
    h = signal.lfilter(B80, A80, np.r_[1.0, np.zeros(2000)])
    sigma_out = S.GYRO_SIGMA * math.sqrt(np.sum(h ** 2))
    vals = []
    for s in range(8):
        tape = build_tapes(COND, s).sensor
        g = OnboardGyro(tape, scale=1.0)
        vals.append(np.stack([g.measure(np.zeros(3)) for _ in range(5000)])[200:])
    x = np.concatenate(vals)
    rms, se = O.rms_with_se(x[:, 0])
    assert abs(rms - sigma_out) <= S.SIGMA_K * max(se, sigma_out * 0.01)


# ---- truth never reaches the policy ---------------------------------------------------------------------------------------
def test_inv7_truth_never_reaches_the_outer_stream():
    views = []

    def spy(view):
        views.append(view)
        return [S.WEIGHT, 0.0, 0.0, 0.0]

    env = StackEnv(COND, 8)
    reading = env.reset()
    controller = ScriptedController(spy)
    differ = {shift: 0 for shift in range(-2, 9)}
    truths = [env.truth]
    n = 60
    for k in range(n):
        view = ControllerView(env.time, reading, np.zeros((5, 3)), env.privileged(controller.privileges))
        result = env.step(controller.act(view))
        reading = result.reading
        truths.append(env.truth)
    assert all(v.privileged is None for v in views)
    assert {f.name for f in dataclasses.fields(ControllerView)} == {"t", "reading", "reference", "privileged"}
    assert {f.name for f in dataclasses.fields(OuterReading)} == {"position", "velocity", "rotation", "body_rate"}
    for k in range(10, n - 10):
        for shift in differ:
            if not np.allclose(views[k].reading.position, truths[k + shift].position, atol=1e-6):
                differ[shift] += 1
    for shift, count in differ.items():
        assert count >= 0.99 * (n - 20), (shift, count)          # at no shift does the reading equal the true position
