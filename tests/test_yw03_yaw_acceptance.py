"""The yaw addenda of 2026-11-07 and 2026-11-08 (session 4D-A), rule 3: session 4C's heading checks as frozen acceptance functions,
with the addenda's thresholds. For one policy over the bar's 1,200 calm episodes (12 cases x 100 seeds, salt 76e7b463820f72ad):
  - the median yaw-rate RMS (the true body z rate) is at most 12 deg/s; the heading rate (the Euler yaw rate) is reported beside it
    and not gated (4DR-4 asked for the rate line on the heading rate; the re-fly read the geometric controller at 13.5 deg/s of it,
    so the line stays on the body z rate: ruling 4DA-10);
  - the median |heading drift| (least-squares slope of the unwrapped true heading) is at most 1 deg/s;
  - at most 12 episodes have a largest |heading| past 45 deg, and at most 3 past 90 deg;
  - the median over the episodes of the largest |heading| is at most 15 deg (the fifth line, 4DR-4: a steady crab is not a held heading).
Every line inclusive. The heading numbers ride in the bar rows' `meta` beside `rmse`, so the acceptance is read from the stored
bar sweep on the same seeds (rulings 4DA-4, 4DA-8). The altitude offset is reported, never gated (4DA-7).

Closed-form cases on constructed records; session 4C's episodes re-flown read-only for the heading rate (tests/data/4c-yaw-episodes.json:
the 16 pi0 of 4B and the two controllers), which the thresholds must separate; a real flight's meta; a Prove store.
"""
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from poisson_prove.executor import derive_seed
from poisson_prove.store import config_hash

import training_numbers as N
import yaw_oracles as Y
import yaw_prove_targets as T
from drone_study.condition import Condition, condition_to_cfg
from drone_study.controllers import GeometricController
from drone_study.rollout import rollout_for
from drone_study.stack import fly
from drone_study.training import behaviour as B
from drone_study.training.bar import bar_sweep_spec
from drone_study.training.behaviour import (EpisodeAltitude, EpisodeYaw, YawMismatch, YawReading, YawRow, altitude_metrics, heading_deg,
                                            heading_rate_deg_s, yaw_acceptance, yaw_metrics, yaw_rows_from_store)

ROOT = Path(__file__).resolve().parents[1]
DEG = math.pi / 180.0
CALM = [Condition.calm(name) for name in N.CASE_NAMES]
KEYS = [(c.case, derive_seed(N.BAR_SALT, config_hash(condition_to_cfg(c)), k)) for c in CALM for k in range(N.BAR_REPLICATES)]
FIXTURE_PATH = Path(__file__).parent / "data" / "4c-yaw-episodes.json"
FIXTURE_SHA256 = "6f667e88bfeda2cb145731c6119072f9ff60e0a77ccf29dfedb7cb9fd83d646a"
FIXTURE = json.loads(FIXTURE_PATH.read_text())
COL = {name: i for i, name in enumerate(FIXTURE["fields"])}


def test_yw03_the_thresholds_are_the_addendums():
    assert (B.YAW_RATE_RMS_MAX_DEG_S, B.YAW_DRIFT_MAX_DEG_S) == (N.YAW_RATE_RMS_MAX_DEG_S, N.YAW_DRIFT_MAX_DEG_S) == (12.0, 1.0)
    assert (B.PAST_45_MAX, B.PAST_90_MAX) == (N.PAST_45_MAX, N.PAST_90_MAX) == (12, 3)
    assert (B.HEADING_45_DEG, B.HEADING_90_DEG) == (45.0, 90.0)
    assert B.MAX_HEADING_MEDIAN_MAX_DEG == N.MAX_HEADING_MEDIAN_MAX_DEG == 15.0
    assert tuple(B.YAW_META_FIELDS) == N.YAW_META_FIELDS == ("yaw_rate_rms_deg_s", "heading_rate_rms_deg_s", "yaw_drift_deg_s", "max_abs_yaw_deg")
    # the lines sit between the controllers' readings and every pi0's (session 4C, table 3; the 2026-11-08 re-fly)
    assert max(N.CONTROLLER_4C_MAX_HEADING_MEDIAN.values()) < N.MAX_HEADING_MEDIAN_MAX_DEG                 # 6.1 under the 15 line
    assert N.PI0_4C_MAX_HEADING_MEDIAN_RANGE[0] < N.MAX_HEADING_MEDIAN_MAX_DEG < 16.3                      # pi0 6 (13.9) passes it, pi0 9 (16.35) and the rest fail
    assert max(N.CONTROLLER_4C_YAW_RATE_RMS.values()) < N.YAW_RATE_RMS_MAX_DEG_S < N.PI0_4C_YAW_RATE_RMS_RANGE[0]
    assert not (max(N.CONTROLLER_4C_HEADING_RATE_RMS.values()) < 12.0)                     # why the line is not on the heading rate (4DA-10)


# ---- the heading and the heading rate of a record ---------------------------------------------------------------------------------
def test_yw03_heading_deg_is_the_unwrapped_zyx_yaw_and_a_tilt_does_not_change_it():
    psi = np.linspace(0.0, 3.0 * math.pi, 400)                                             # one and a half turns
    rec = Y.constructed_record(psi, pitch_rad=0.3, roll_rad=-0.2)
    got = heading_deg(rec.quat)
    np.testing.assert_allclose(got, np.degrees(psi), atol=1e-6)                             # unwrapped: no jump at 180 deg
    assert got.max() > 500.0
    level = Y.constructed_record(psi)
    np.testing.assert_allclose(heading_deg(level.quat), got, atol=1e-6)
    for q, p in zip(rec.quat[::50], psi[::50]):
        assert Y.heading_of_quat(q) == pytest.approx(Y.wrap(p), abs=1e-9)                   # session 4C's formula, wrapped


def test_yw03_heading_rate_is_the_euler_yaw_rate_and_a_banked_turn_at_a_held_heading_has_none():
    """The heading rate the reward damps and the gate reports: a tilt direction rotating at a held heading (a lap of a circle
    facing one way) carries a body z rate and no heading rate; a real turn reads its rate whatever the tilt."""
    n, dt, alpha, omega = 600, 0.01, 0.45, 2.2
    t = np.arange(n) * dt
    Rs = np.array([Y.rot_zyx(0.0, alpha * math.cos(omega * tt), alpha * math.sin(omega * tt)) for tt in t])
    w = Y.body_rates_of(Rs, dt)
    rec = Y.constructed_record(np.zeros(n), pitch_rad=alpha * np.cos(omega * t), roll_rad=alpha * np.sin(omega * t), body_rate=w)
    rate = heading_rate_deg_s(rec.quat, rec.body_rate)
    assert rate.shape == (n,) and np.max(np.abs(rate[2:-2])) < 0.2                            # under 0.2 deg/s of heading rate
    m = yaw_metrics(rec)
    assert m.heading_rate_rms_deg_s < 0.2 and m.yaw_rate_rms_deg_s > 10.0                   # and 10 deg/s and more of body z rate: the two differ
    assert m.yaw_rate_rms_deg_s == pytest.approx(np.degrees(math.sqrt(np.mean(w[:, 2] ** 2))), rel=1e-9)
    turning = np.array([Y.rot_zyx(0.8 * tt, alpha * math.cos(omega * tt), alpha * math.sin(omega * tt)) for tt in t])
    w2 = Y.body_rates_of(turning, dt)
    rec2 = Y.constructed_record(0.8 * t, pitch_rad=alpha * np.cos(omega * t), roll_rad=alpha * np.sin(omega * t), body_rate=w2)
    rate2 = heading_rate_deg_s(rec2.quat, rec2.body_rate)
    np.testing.assert_allclose(rate2[2:-2], math.degrees(0.8), atol=0.2)                    # the control: a real turn reads 45.8 deg/s
    assert yaw_metrics(rec2).heading_rate_rms_deg_s == pytest.approx(math.degrees(0.8), abs=0.2)
    assert yaw_metrics(rec2).heading_rate_rms_deg_s == pytest.approx(Y.heading_rate_rms_of(rec2), rel=1e-9)


@pytest.mark.parametrize("omega_deg_s", [6.05, -11.01, 30.0, 0.5])
def test_yw03_a_steady_spin_reads_its_rate_as_rms_and_drift_and_its_turn_as_the_largest_heading(omega_deg_s):
    n = 1001
    t = np.arange(n) * 0.01
    psi = np.radians(omega_deg_s) * t
    rec = Y.constructed_record(psi, r_rad_s=np.full(n, np.radians(omega_deg_s)))
    m = yaw_metrics(rec)
    assert isinstance(m, EpisodeYaw)
    assert m.heading_rate_rms_deg_s == pytest.approx(abs(omega_deg_s), abs=1e-9)             # level: the heading rate is the body z rate
    assert m.yaw_rate_rms_deg_s == pytest.approx(abs(omega_deg_s), abs=1e-9)
    assert m.drift_deg_s == pytest.approx(omega_deg_s, abs=1e-9)
    assert m.max_abs_yaw_deg == pytest.approx(abs(omega_deg_s) * 10.0, abs=1e-9)
    assert m.final_yaw_deg == pytest.approx(omega_deg_s * 10.0, abs=1e-9)


def test_yw03_a_held_heading_with_a_tilt_reads_no_rate_no_drift_and_the_offset_as_its_largest():
    n = 1001
    rec = Y.constructed_record(np.full(n, 30 * DEG), r_rad_s=np.zeros(n), pitch_rad=0.4, roll_rad=0.3)
    m = yaw_metrics(rec)
    assert (m.heading_rate_rms_deg_s, m.yaw_rate_rms_deg_s, m.drift_deg_s) == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)
    assert m.max_abs_yaw_deg == pytest.approx(30.0, abs=1e-6) and m.final_yaw_deg == pytest.approx(30.0, abs=1e-6)


def test_yw03_a_wander_reads_session_4cs_numbers_and_the_heading_rate():
    n = 1001
    t = np.arange(n) * 0.01
    psi = 0.4 * np.sin(1.3 * t) + 0.05 * t                                                   # wander with a slow drift
    r = 0.4 * 1.3 * np.cos(1.3 * t) + 0.05 + 0.02 * np.sin(7.0 * t)                          # a body rate of its own
    rec = Y.constructed_record(psi, r_rad_s=r, pitch_rad=0.2)
    rms, slope, biggest = Y.yaw_numbers(t, np.degrees(psi), r)
    m = yaw_metrics(rec)
    assert m.yaw_rate_rms_deg_s == pytest.approx(rms, rel=1e-9)
    assert m.heading_rate_rms_deg_s == pytest.approx(Y.heading_rate_rms_of(rec), rel=1e-9)
    assert m.heading_rate_rms_deg_s == pytest.approx(rms / math.cos(0.2), rel=1e-9)         # a body z rate at 0.2 rad of pitch and no roll
    assert m.drift_deg_s == pytest.approx(slope, rel=1e-9)
    assert m.max_abs_yaw_deg == pytest.approx(biggest, rel=1e-9)
    assert m.final_yaw_deg == pytest.approx(math.degrees(psi[-1]), abs=1e-6)


def test_yw03_a_short_episode_reads_over_the_rows_it_flew():
    n = 230                                                                                 # a training crash at 2.3 s
    t = np.arange(n) * 0.01
    psi = 0.9 * t
    rec = Y.constructed_record(psi, r_rad_s=np.full(n, 0.9))
    rec.end_reason = "radius"
    m = yaw_metrics(rec)
    assert m.heading_rate_rms_deg_s == pytest.approx(math.degrees(0.9), abs=1e-9)
    assert m.max_abs_yaw_deg == pytest.approx(math.degrees(0.9 * t[-1]), abs=1e-6)


# ---- the altitude offset (reported, not gated) ------------------------------------------------------------------------------------
def test_yw03_the_altitude_offset_is_the_mean_height_above_the_reference_over_the_episode_and_its_last_second():
    n = 1001
    t = np.arange(n) * 0.01
    rec = Y.constructed_record(np.zeros(n), r_rad_s=np.zeros(n), dz_m=np.full(n, 0.02))
    a = altitude_metrics(rec)
    assert isinstance(a, EpisodeAltitude)
    assert (a.bias_cm, a.last_second_bias_cm, a.rms_cm, a.lowest_cm) == pytest.approx((2.0, 2.0, 2.0, 2.0), abs=1e-9)
    ramp = Y.constructed_record(np.zeros(n), r_rad_s=np.zeros(n), dz_m=0.04 * (t / 10.0) - 0.01)   # -1 cm to +3 cm
    a = altitude_metrics(ramp)
    assert a.bias_cm == pytest.approx(100 * np.mean(0.04 * (t / 10.0) - 0.01), abs=1e-9)
    assert a.last_second_bias_cm == pytest.approx(100 * np.mean((0.04 * (t / 10.0) - 0.01)[t >= t[-1] - 1.0 + 1e-9]), abs=1e-9)   # 4C's last second: the 100 rows after 9.0 s
    assert a.lowest_cm == pytest.approx(-1.0, abs=1e-9) and a.rms_cm == pytest.approx(100 * math.sqrt(np.mean((0.04 * (t / 10.0) - 0.01) ** 2)), abs=1e-9)
    assert a.bias_cm > 0.0


# ---- the acceptance on the bar's 1,200 rows -------------------------------------------------------------------------------------
def rows(rate=4.0, drift=0.05, biggest=10.0, heading_rate=6.0, keys=KEYS):
    """The bar's 1,200 YawRows with one value per column, or one per row. `rate` is the body z rate the rate line reads."""
    n = len(keys)
    rate = np.broadcast_to(np.asarray(rate, dtype=float), (n,))
    drift = np.broadcast_to(np.asarray(drift, dtype=float), (n,))
    biggest = np.broadcast_to(np.asarray(biggest, dtype=float), (n,))
    heading_rate = np.broadcast_to(np.asarray(heading_rate, dtype=float), (n,))
    return [YawRow(case, seed, EpisodeYaw(float(rate[i]), float(heading_rate[i]), float(drift[i]), float(biggest[i]), float(biggest[i])))
            for i, (case, seed) in enumerate(keys)]


def split(low, high, n_low):
    return [low] * n_low + [high] * (N.BAR_EPISODES - n_low)


def test_yw03_a_heading_holder_passes_all_five_lines_and_the_reading_carries_its_numbers():
    r = yaw_acceptance(rows())
    assert isinstance(r, YawReading) and r.passed is True
    assert (r.passes_rate, r.passes_drift, r.passes_45, r.passes_90, r.passes_heading) == (True, True, True, True, True)
    assert r.episodes == 1200 and r.yaw_rate_rms_median == pytest.approx(4.0) and r.drift_abs_median == pytest.approx(0.05)
    assert r.heading_rate_rms_median == pytest.approx(6.0)                                     # reported beside the body z rate
    assert (r.past_45, r.past_90) == (0, 0) and r.max_heading_median == pytest.approx(10.0)
    d = r.to_dict()
    assert d["passed"] is True and d["past_45"] == 0 and d["yaw_rate_rms_median_deg_s"] == pytest.approx(4.0)
    assert d["heading_rate_rms_median_deg_s"] == pytest.approx(6.0)
    assert d["max_heading_median_deg"] == pytest.approx(10.0) and d["passes_heading"] is True


@pytest.mark.parametrize("values,median,ok", [(split(3.0, 40.0, 601), 3.0, True), (split(3.0, 40.0, 600), 21.5, False),
                                              (12.0, 12.0, True), (12.0 + 1e-9, 12.0 + 1e-9, False), (split(1.0, 12.0, 300), 12.0, True)])
def test_yw03_the_median_yaw_rate_rms_is_at_most_twelve_degrees_a_second(values, median, ok):
    r = yaw_acceptance(rows(rate=values))
    assert r.yaw_rate_rms_median == pytest.approx(median, rel=1e-12)
    assert r.passes_rate is ok and r.passed is ok
    assert (r.passes_drift, r.passes_45, r.passes_90, r.passes_heading) == (True, True, True, True)


def test_yw03_the_rate_line_reads_the_body_z_rate_and_the_heading_rate_is_reported_beside_it():
    """Ruling 4DA-10: the geometric controller's heading wobbles at 13.5 deg/s while it holds heading within 6 degrees, so the line
    stays on the body z rate (4DA-4) and the heading rate is carried in the reading and not gated."""
    r = yaw_acceptance(rows(rate=5.8, heading_rate=13.5))                                   # the geometric controller's medians
    assert r.passes_rate is True and r.passed is True
    assert r.yaw_rate_rms_median == pytest.approx(5.8) and r.heading_rate_rms_median == pytest.approx(13.5)
    r = yaw_acceptance(rows(rate=24.0, heading_rate=4.0))
    assert r.passes_rate is False and r.passed is False and r.yaw_rate_rms_median == pytest.approx(24.0)
    r = yaw_acceptance(rows(rate=4.0, heading_rate=40.0))                                   # the heading rate alone gates nothing
    assert r.passes_rate is True and r.passed is True and r.heading_rate_rms_median == pytest.approx(40.0)


@pytest.mark.parametrize("values,median,ok", [(0.3, 0.3, True), (-0.9, 0.9, True), (1.0, 1.0, True), (-1.0, 1.0, True),
                                              (1.0 + 1e-9, 1.0 + 1e-9, False), (split(-6.05, 0.1, 700), 6.05, False),
                                              (split(11.01, 0.1, 600), 5.555, False), (split(11.01, 0.1, 599), 0.1, True)])
def test_yw03_the_median_absolute_drift_is_at_most_one_degree_a_second(values, median, ok):
    r = yaw_acceptance(rows(drift=values))
    assert r.drift_abs_median == pytest.approx(median, rel=1e-9)
    assert r.passes_drift is ok and r.passed is ok
    assert (r.passes_rate, r.passes_45, r.passes_90, r.passes_heading) == (True, True, True, True)


@pytest.mark.parametrize("n45,n90,ok45,ok90", [(0, 0, True, True), (12, 0, True, True), (13, 0, False, True), (12, 3, True, True),
                                               (12, 4, True, False), (3, 3, True, True), (1, 1, True, True)])
def test_yw03_at_most_twelve_episodes_past_45_and_three_past_90(n45, n90, ok45, ok90):
    assert n90 <= n45
    biggest = [100.0] * n90 + [60.0] * (n45 - n90) + [10.0] * (N.BAR_EPISODES - n45)        # a 90 crossing is a 45 crossing too
    r = yaw_acceptance(rows(biggest=biggest))
    assert (r.past_45, r.past_90) == (n45, n90)
    assert (r.passes_45, r.passes_90) == (ok45, ok90)
    assert r.passed is (ok45 and ok90)
    assert (r.passes_rate, r.passes_drift, r.passes_heading) == (True, True, True)             # the median largest heading is still 10


def test_yw03_exactly_45_or_90_is_not_past():
    r = yaw_acceptance(rows(biggest=split(45.0, 10.0, 500)))
    assert (r.past_45, r.past_90) == (0, 0) and r.passed is True
    r = yaw_acceptance(rows(biggest=split(90.0, 10.0, 500)))
    assert (r.past_45, r.past_90) == (500, 0) and r.passes_45 is False and r.passes_90 is True
    r = yaw_acceptance(rows(biggest=split(45.0 + 1e-9, 10.0, 13)))
    assert r.past_45 == 13 and r.passes_45 is False


@pytest.mark.parametrize("values,median,ok", [(10.0, 10.0, True), (15.0, 15.0, True), (15.0 + 1e-9, 15.0 + 1e-9, False),
                                              (split(30.0, 5.0, 600), 17.5, False), (split(30.0, 5.0, 599), 5.0, True),
                                              (split(2.0, 44.0, 601), 2.0, True), (split(2.0, 44.0, 600), 23.0, False)])
def test_yw03_the_median_largest_heading_is_at_most_fifteen_degrees(values, median, ok):
    """The fifth line (4DR-4): a policy crabbed at a steady 25 to 35 degrees passes the first four lines and fails this one."""
    r = yaw_acceptance(rows(biggest=values))
    assert r.max_heading_median == pytest.approx(median, rel=1e-12)
    assert r.passes_heading is ok and r.passed is ok
    assert (r.passes_rate, r.passes_drift, r.passes_45, r.passes_90) == (True, True, True, True)   # under 45, so the count lines pass


def test_yw03_a_steady_crab_fails_only_the_fifth_line():
    r = yaw_acceptance(rows(rate=3.0, drift=0.02, biggest=30.0))                               # held at 30 degrees, no rate, no drift
    assert (r.passes_rate, r.passes_drift, r.passes_45, r.passes_90) == (True, True, True, True)
    assert r.passes_heading is False and r.passed is False and r.max_heading_median == pytest.approx(30.0)


def test_yw03_all_five_lines_are_needed():
    assert yaw_acceptance(rows()).passed is True
    assert yaw_acceptance(rows(rate=30.0)).passed is False
    assert yaw_acceptance(rows(drift=2.0)).passed is False
    assert yaw_acceptance(rows(biggest=split(50.0, 10.0, 13))).passed is False
    assert yaw_acceptance(rows(biggest=split(95.0, 10.0, 4))).passed is False
    assert yaw_acceptance(rows(biggest=20.0)).passed is False
    bad = yaw_acceptance(rows(rate=30.0, drift=2.0, biggest=95.0))
    assert (bad.passes_rate, bad.passes_drift, bad.passes_45, bad.passes_90, bad.passes_heading) == (False, False, False, False, False)


def test_yw03_rows_that_are_not_the_bars_1200_are_refused():
    with pytest.raises(YawMismatch):
        yaw_acceptance(rows()[:1199])
    with pytest.raises(YawMismatch):
        yaw_acceptance(rows() + rows()[:1])                                                   # a row twice
    other = [(c, s + 1) for c, s in KEYS]
    with pytest.raises(YawMismatch):
        yaw_acceptance(rows(keys=other))                                                      # not the bar's seeds
    zone_d = [(c.case, derive_seed(N.salt("D"), config_hash(condition_to_cfg(c)), k)) for c in CALM for k in range(100)]
    with pytest.raises(YawMismatch):
        yaw_acceptance(rows(keys=zone_d))
    r = rows()
    r[5] = YawRow(r[5].case, r[5].seed, EpisodeYaw(), error=True)
    with pytest.raises(YawMismatch):
        yaw_acceptance(r)                                                                     # an error row stops the read
    r = rows()
    r[7] = YawRow(r[7].case, r[7].seed, EpisodeYaw(float("nan"), 1.0, 0.0, 1.0, 1.0))
    with pytest.raises(YawMismatch):
        yaw_acceptance(r)
    r = rows()
    r[8] = YawRow(r[8].case, r[8].seed, EpisodeYaw(1.0, float("nan"), 0.0, 1.0, 1.0))         # a non-finite heading rate too
    with pytest.raises(YawMismatch):
        yaw_acceptance(r)
    assert yaw_acceptance(rows()).passed is True                                              # the control


# ---- session 4C's episodes: the thresholds must separate the controllers from every pi0 of 4B ----------------------------------------
def fixture_rows(flyer):
    arr = FIXTURE["flyers"][flyer]
    assert len(arr) == 1200 and FIXTURE["fields"][:4] == list(N.YAW_META_FIELDS)
    return [YawRow(case, seed, EpisodeYaw(a[COL["yaw_rate_rms_deg_s"]], a[COL["heading_rate_rms_deg_s"]], a[COL["yaw_drift_deg_s"]],
                                          a[COL["max_abs_yaw_deg"]], float("nan"))) for (case, seed), a in zip(KEYS, arr)]


def test_yw03_the_two_controllers_of_session_4c_pass_every_line():
    for flyer in ("pid", "geometric"):
        r = yaw_acceptance(fixture_rows(flyer))
        assert r.passed is True, flyer
        assert (r.passes_rate, r.passes_drift, r.passes_45, r.passes_90, r.passes_heading) == (True, True, True, True, True)
        assert r.yaw_rate_rms_median == pytest.approx(N.CONTROLLER_4C_YAW_RATE_RMS[flyer], abs=0.06)
        assert r.heading_rate_rms_median == pytest.approx(N.CONTROLLER_4C_HEADING_RATE_RMS[flyer], abs=0.06)   # reported: 13.5 would fail a 12 line
        assert r.max_heading_median == pytest.approx(N.CONTROLLER_4C_MAX_HEADING_MEDIAN[flyer], abs=0.06)
        assert r.drift_abs_median < 0.2 and (r.past_45, r.past_90) == (0, 0)


@pytest.mark.parametrize("seed", range(1, 17))
def test_yw03_every_pi0_of_4b_fails_on_session_4cs_numbers(seed):
    r = yaw_acceptance(fixture_rows(f"pi0-{seed}"))
    assert r.passed is False
    assert r.passes_rate is False and N.PI0_4C_YAW_RATE_RMS_RANGE[0] - 0.1 <= r.yaw_rate_rms_median <= N.PI0_4C_YAW_RATE_RMS_RANGE[1] + 0.1
    assert N.PI0_4C_HEADING_RATE_RMS_RANGE[0] - 0.1 <= r.heading_rate_rms_median <= N.PI0_4C_HEADING_RATE_RMS_RANGE[1] + 0.1
    assert r.max_heading_median >= N.PI0_4C_MAX_HEADING_MEDIAN_RANGE[0] - 0.1
    assert r.passes_heading is (seed == 6)                                                     # pi0 6 holds within 13.9 degrees and fails the rate line alone
    if seed in (5, 16):                                                                        # the two spinners fail every line
        assert (r.passes_drift, r.passes_45, r.passes_90) == (False, False, False) and r.past_90 == 900
    else:
        assert r.passes_drift is True and r.drift_abs_median < 0.6
    if seed in (4, 5, 8, 14, 15, 16):
        assert r.passes_45 is False and r.past_45 >= 15                                        # the wanderers (21, 421, 15, 708) and the spinners (900)
    else:
        assert r.passes_45 is True and r.past_45 <= 6                                          # pi0 11 has 6, pi0 10 has 1, the rest 0


def test_yw03_the_fixture_is_session_4cs_episodes_re_flown_for_the_heading_rate():
    assert hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest() == FIXTURE_SHA256
    assert set(FIXTURE["flyers"]) == {f"pi0-{i}" for i in range(1, 17)} | {"pid", "geometric"}
    assert FIXTURE["fields"] == ["yaw_rate_rms_deg_s", "heading_rate_rms_deg_s", "yaw_drift_deg_s", "max_abs_yaw_deg", "alt_bias_cm"]
    # the altitude column says what 4C said: every pi0 above the reference, the controllers not
    for f in FIXTURE["flyers"]:
        bias = float(np.median([a[COL["alt_bias_cm"]] for a in FIXTURE["flyers"][f]]))
        assert (1.0 < bias < 4.5) if f.startswith("pi0") else (abs(bias) < 0.5), (f, bias)
    # and the body z rate column is 4C's table 3: the controllers 3.3 and 5.8, the pi0 23.2 to 62.4
    body_z = {f: float(np.median([a[COL["yaw_rate_rms_deg_s"]] for a in FIXTURE["flyers"][f]])) for f in FIXTURE["flyers"]}
    assert body_z["pid"] == pytest.approx(3.3, abs=0.06) and body_z["geometric"] == pytest.approx(5.8, abs=0.06)
    assert min(body_z[f"pi0-{i}"] for i in range(1, 17)) == pytest.approx(23.2, abs=0.06)
    assert max(body_z[f"pi0-{i}"] for i in range(1, 17)) == pytest.approx(62.4, abs=0.06)
    # and the heading rate column is the re-fly's: the PID 4.2, the geometric controller 13.5, the pi0 24.4 to 62.8
    hr = {f: float(np.median([a[COL["heading_rate_rms_deg_s"]] for a in FIXTURE["flyers"][f]])) for f in FIXTURE["flyers"]}
    assert hr["pid"] == pytest.approx(4.2, abs=0.06) and hr["geometric"] == pytest.approx(13.5, abs=0.06)
    assert min(hr[f"pi0-{i}"] for i in range(1, 17)) == pytest.approx(24.4, abs=0.06) and max(hr[f"pi0-{i}"] for i in range(1, 17)) == pytest.approx(62.8, abs=0.06)
    r = yaw_acceptance(fixture_rows("pid"))
    assert r.episodes == 1200                                                                   # the stub's zero would not do


# ---- a real flight's heading numbers ride in the bar rows' meta -------------------------------------------------------------------
def test_yw03_the_rollouts_meta_carries_the_four_heading_numbers_of_the_flight():
    condition, seed = Condition.calm("oval-medium"), 20261108
    ep = rollout_for(GeometricController())(condition_to_cfg(condition), seed)
    rec = fly(condition, seed, GeometricController())
    want = yaw_metrics(rec)
    assert set(N.YAW_META_FIELDS) <= set(ep.meta) and "rmse" in ep.meta
    assert ep.meta["yaw_rate_rms_deg_s"] == pytest.approx(want.yaw_rate_rms_deg_s, rel=1e-9)
    assert ep.meta["heading_rate_rms_deg_s"] == pytest.approx(want.heading_rate_rms_deg_s, rel=1e-9)
    assert ep.meta["yaw_drift_deg_s"] == pytest.approx(want.drift_deg_s, rel=1e-9)
    assert ep.meta["max_abs_yaw_deg"] == pytest.approx(want.max_abs_yaw_deg, rel=1e-9)
    assert 0.0 < want.yaw_rate_rms_deg_s < 12.0 and want.max_abs_yaw_deg < 15.0               # the geometric controller holds heading
    assert want.heading_rate_rms_deg_s > want.yaw_rate_rms_deg_s                              # and its heading wobble reads faster than its body z rate
    oracle_rms, _, oracle_max = Y.yaw_numbers(rec.t, np.degrees(np.unwrap([Y.heading_of_quat(q) for q in rec.quat])), rec.body_rate[:, 2])
    assert want.yaw_rate_rms_deg_s == pytest.approx(oracle_rms, rel=1e-9) and want.max_abs_yaw_deg == pytest.approx(oracle_max, rel=1e-9)
    assert want.heading_rate_rms_deg_s == pytest.approx(Y.heading_rate_rms_of(rec), rel=1e-9)


# ---- through Prove, as the bar is stored -----------------------------------------------------------------------------------------
def prove(store, *args):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / "tests"), os.environ.get("PYTHONPATH", "")]))
    return subprocess.run([sys.executable, "-m", "poisson_prove", "--store", str(store), *args, "--plain"], env=env, cwd=ROOT,
                          capture_output=True, text=True)


_STORE = {}


@pytest.fixture(scope="module")
def tmp_root(tmp_path_factory):
    return tmp_path_factory.mktemp("yaw-bar")


def yaw_store(root):
    if "store" in _STORE:
        return _STORE["store"]
    text = bar_sweep_spec(target="yaw_prove_targets:bar_yaw_policy", target_id="drone-robustness-4d")
    assert text.strip(), "the study wrote no sweep spec for the bar"
    spec = root / "bar.toml"
    spec.write_text(text)
    st = root / "store"
    for name, target in (("with-yaw", "bar_yaw_policy"), ("without-yaw", "bar_no_yaw"), ("body-z-only", "bar_body_z_only")):
        r = prove(st, "sweep", str(spec), "--target", f"yaw_prove_targets:{target}", "--salt", N.BAR_SALT, "--yes", "--max-mde", "1",
                  "--name", name)
        assert r.returncode == 0, f"sweep {name} failed:\n{r.stdout}\n{r.stderr}"
    _STORE["store"] = st
    return st


def test_yw03_a_stored_bar_run_reads_back_its_1200_heading_rows(tmp_root):
    store = yaw_store(tmp_root)
    got = yaw_rows_from_store(store, "with-yaw")
    assert len(got) == 1200 and all(isinstance(r, YawRow) for r in got)
    assert sorted((r.case, r.seed) for r in got) == sorted(KEYS)
    for r in got:
        rate, heading_rate, drift, biggest = T.yaw_outcome(r.seed)
        assert not r.error
        assert (r.yaw.yaw_rate_rms_deg_s, r.yaw.heading_rate_rms_deg_s, r.yaw.drift_deg_s, r.yaw.max_abs_yaw_deg) == pytest.approx((rate, heading_rate, drift, biggest), rel=1e-12)
    reading = yaw_acceptance(got)
    want_rate = float(np.median([T.yaw_outcome(s)[0] for _, s in KEYS]))
    assert reading.heading_rate_rms_median == pytest.approx(float(np.median([T.yaw_outcome(s)[1] for _, s in KEYS])), rel=1e-12)
    want_drift = float(np.median([abs(T.yaw_outcome(s)[2]) for _, s in KEYS]))
    want_heading = float(np.median([T.yaw_outcome(s)[3] for _, s in KEYS]))
    assert reading.yaw_rate_rms_median == pytest.approx(want_rate, rel=1e-12) and reading.drift_abs_median == pytest.approx(want_drift, rel=1e-12)
    assert reading.max_heading_median == pytest.approx(want_heading, rel=1e-12)
    assert reading.past_45 == sum(T.yaw_outcome(s)[3] > 45.0 for _, s in KEYS) and reading.past_90 == 0
    assert reading.passed is (want_rate <= 12.0 and want_drift <= 1.0 and reading.past_45 <= 12 and want_heading <= 15.0)


def test_yw03_a_run_whose_rows_carry_no_heading_numbers_is_not_read_as_a_pass(tmp_root):
    store = yaw_store(tmp_root)
    assert len(yaw_rows_from_store(store, "with-yaw")) == 1200                                # the control
    rows_ = yaw_rows_from_store(store, "without-yaw")
    assert len(rows_) == 1200 and all(r.error for r in rows_)
    with pytest.raises(YawMismatch):
        yaw_acceptance(rows_)
    body_z_only = yaw_rows_from_store(store, "body-z-only")                                     # the 2026-11-07 three fields, no heading rate
    assert len(body_z_only) == 1200 and all(r.error for r in body_z_only)
    with pytest.raises(YawMismatch):
        yaw_acceptance(body_z_only)
