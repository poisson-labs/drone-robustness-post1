"""Ruling 4DB-1 (Taylor, 2026-11-09), added by session 4D-B as an additive change: a third lineage line beside the bar and the yaw
gate, read from the stored bar rows like the other two. Over the bar's 1,200 calm episodes (12 cases x 100 seeds, salt
76e7b463820f72ad), the share of control steps on which any motor's command is at its limit (0 or 65,535 units) is at most 0.5%,
inclusive. The reading is the record's mixer output at the end of each control step (row k of `mixer_output` is the last 500 Hz
substep of step k - 1, so row 0, the reset's zeros, is not a reading). The rollout writes the episode's share into each row's
`meta`; the line pools the episodes' steps. A failed seed is reported with the gate "motor" and the next seed is tried; there is
no stop; the pick is untouched (it reads the bar and the yaw gate).

Why: pilot 9001 at the picked weight passed both gates while pegging motors 1 and 3 on 18% to 25% of calm steps, which leaves
it no margin for a gust. Nothing in 3A's, 4A's, 4D-A's, 5A's or 8A's tests changes.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from poisson_prove.executor import derive_seed
from poisson_prove.store import config_hash

import motor_prove_targets as MT
import training_numbers as N
from drone_study.condition import Condition, condition_to_cfg
from drone_study.controllers import GeometricController
from drone_study.rollout import rollout_for
from drone_study.stack import ScriptedController, fly
from drone_study.training import behaviour as B
from drone_study.training.bar import BarReading, bar_sweep_spec
from drone_study.training.behaviour import (MotorMismatch, MotorReading, MotorRow, YawReading, motor_acceptance, motor_limit_share,
                                            motor_rows_from_store)
from drone_study.training.yaw_weight import LINEAGES, failed_gate, select_lineages_with_yaw
from drone_study.vehicle import T_MAX, UNITS_MAX

ROOT = Path(__file__).resolve().parents[1]
CALM = [Condition.calm(name) for name in N.CASE_NAMES]
KEYS = [(c.case, derive_seed(N.BAR_SALT, config_hash(condition_to_cfg(c)), k)) for c in CALM for k in range(N.BAR_REPLICATES)]


def oracle_share(mixer_output) -> float:
    """Written from the ruling, independently of the package: the ends of the control steps are rows 1 .. n - 1; a step counts
    when any of the four commands is <= 0 or >= 65,535."""
    rows = [list(r) for r in np.asarray(mixer_output, dtype=float)][1:]
    if not rows:
        return 0.0
    hit = 0
    for r in rows:
        if any(u <= 0.0 or u >= 65535.0 for u in r):
            hit += 1
    return hit / len(rows)


class Rec:
    def __init__(self, mixer_output):
        self.mixer_output = np.asarray(mixer_output, dtype=float)


# ---- the constants ------------------------------------------------------------------------------------------------------------------
def test_yw04_the_line_is_half_a_percent_of_steps_and_the_limit_is_the_firmware_full_scale():
    assert B.MOTOR_LIMIT_SHARE_MAX == 0.005
    assert B.UNITS_LIMIT == UNITS_MAX == 65535.0
    assert B.MOTOR_META_FIELD == "motor_limit_share"
    assert B.MOTOR_META_FIELD not in N.YAW_META_FIELDS                                          # the yaw gate's four fields are untouched


# ---- one episode's share, on constructed records ------------------------------------------------------------------------------------
def test_yw04_the_share_reads_the_ends_of_the_steps_and_not_the_resets_zeros():
    mid = np.full((11, 4), 30000.0)
    mid[0] = 0.0                                                                                # the reset's row: no step flown yet
    assert motor_limit_share(Rec(mid)) == 0.0 == oracle_share(mid)
    mid[3, 2] = 65535.0                                                                         # one motor at the top on one of ten steps
    assert motor_limit_share(Rec(mid)) == pytest.approx(0.1, abs=1e-15) == oracle_share(mid)
    mid[7, 0] = 0.0                                                                             # one at the floor on another
    assert motor_limit_share(Rec(mid)) == pytest.approx(0.2, abs=1e-15)
    mid[7, 1] = 65535.0                                                                         # two motors on the same step count once
    assert motor_limit_share(Rec(mid)) == pytest.approx(0.2, abs=1e-15)


def test_yw04_a_command_just_inside_either_limit_is_not_at_it():
    m = np.full((5, 4), 30000.0)
    m[1:, 0] = 65535.0 - 1e-3
    m[1:, 1] = 1e-3
    assert motor_limit_share(Rec(m)) == 0.0
    m[2, 3] = 65535.0
    assert motor_limit_share(Rec(m)) == pytest.approx(0.25, abs=1e-15) == oracle_share(m)


def test_yw04_every_one_of_the_four_motors_is_read():
    for k in range(4):
        m = np.full((3, 4), 20000.0)
        m[2, k] = 0.0
        assert motor_limit_share(Rec(m)) == pytest.approx(0.5, abs=1e-15), k


def test_yw04_an_episode_with_no_step_read_has_share_zero():
    assert motor_limit_share(Rec(np.zeros((1, 4)))) == 0.0


# ---- a real flight: the rollout's meta carries the share ----------------------------------------------------------------------------
def test_yw04_the_rollouts_meta_carries_the_flights_motor_limit_share():
    condition, seed = Condition.calm("oval-medium"), 20261109
    ep = rollout_for(GeometricController())(condition_to_cfg(condition), seed)
    rec = fly(condition, seed, GeometricController())
    assert B.MOTOR_META_FIELD in ep.meta and set(N.YAW_META_FIELDS) <= set(ep.meta) and "rmse" in ep.meta
    assert ep.meta["motor_limit_share"] == oracle_share(rec.mixer_output) == motor_limit_share(rec)
    assert ep.meta["motor_limit_share"] <= 0.005                                                # a controller in calm air is inside the line
    assert ep.meta["steps"] == rec.n_steps


def test_yw04_full_throttle_reads_a_motor_at_the_top_on_every_step_after_the_hold():
    """A scripted flyer at full thrust: the first step flies the hold action (the action lag), every later step ends with a motor
    at 65,535; the share is (n - 2) / (n - 1)."""
    condition, seed = Condition.calm("oval-medium"), 7
    full = ScriptedController(lambda view: np.array([T_MAX, 0.0, 0.0, 0.0]))
    rec = fly(condition, seed, full)
    n = rec.n_steps
    assert n > 10
    ep = rollout_for(ScriptedController(lambda view: np.array([T_MAX, 0.0, 0.0, 0.0])))(condition_to_cfg(condition), seed)
    assert ep.meta["motor_limit_share"] == pytest.approx((n - 2) / (n - 1), abs=1e-12) == pytest.approx(oracle_share(rec.mixer_output), abs=1e-15)
    assert ep.meta["motor_limit_share"] > 0.005


# ---- the line on the bar's 1,200 rows -----------------------------------------------------------------------------------------------
def rows(shares, steps=1001, keys=KEYS):
    shares = list(shares) if not np.isscalar(shares) else [float(shares)] * len(keys)
    return [MotorRow(c, int(s), float(x), int(steps)) for (c, s), x in zip(keys, shares)]


def test_yw04_the_line_pools_the_steps_and_is_inclusive():
    r = motor_acceptance(rows(0.0))
    assert isinstance(r, MotorReading) and r.episodes == 1200 and r.share == 0.0 and r.passed
    r = motor_acceptance(rows(0.005))                                                           # exactly at the line passes
    assert r.share == pytest.approx(0.005, abs=1e-15) and r.passed
    r = motor_acceptance(rows(0.0051))
    assert not r.passed
    # six episodes pegged on every step and the rest clean: 6/1200 = 0.5% of the steps, a pass at the line
    shares = [1.0] * 6 + [0.0] * 1194
    r = motor_acceptance(rows(shares))
    assert r.share == pytest.approx(0.005, abs=1e-15) and r.passed and r.episodes_over == 6 and r.episode_share_max == 1.0
    assert r.episode_share_median == 0.0                                                        # the median would miss them; the line pools
    r = motor_acceptance(rows([1.0] * 7 + [0.0] * 1193))
    assert not r.passed and r.episodes_over == 7


def test_yw04_the_pool_weights_each_episode_by_its_steps_read():
    keys = KEYS
    shares = [0.5] + [0.0] * 1199
    short = [MotorRow(keys[0][0], keys[0][1], 0.5, 11)] + rows([0.0] * 1199, keys=keys[1:])     # ten steps read, five pegged
    r = motor_acceptance(short)
    assert r.share == pytest.approx(5 / (10 + 1199 * 1000), rel=1e-12)
    r = motor_acceptance(rows(shares))
    assert r.share == pytest.approx(500 / (1200 * 1000), rel=1e-12)


def test_yw04_pilot_9001s_buzz_fails_the_line_and_the_controllers_pass_it():
    assert not motor_acceptance(rows(0.179)).passed                                             # 17.9% of steps at 5M (fail-fast.md)
    assert not motor_acceptance(rows(0.251)).passed                                             # 25.1% at 10M
    assert motor_acceptance(rows(0.00005)).passed                                               # the geometric controller's 0.005%
    assert motor_acceptance(rows(0.0042)).passed                                                # 4B's worst pi0, 0.42%


def test_yw04_rows_that_are_not_the_bars_scored_episodes_are_refused():
    with pytest.raises(MotorMismatch):
        motor_acceptance(rows(0.0)[:-1])                                                        # one short
    with pytest.raises(MotorMismatch):
        motor_acceptance(rows(0.0)[:-1] + [rows(0.0)[0]])                                      # a row twice
    other = [(c, s + 1) for c, s in KEYS]
    with pytest.raises(MotorMismatch):
        motor_acceptance(rows(0.0, keys=other))                                                 # not the bar's seeds
    bad = rows(0.0)
    for broken in (MotorRow(bad[5].case, bad[5].seed, error=True), MotorRow(bad[5].case, bad[5].seed, float("nan"), 1001),
                   MotorRow(bad[5].case, bad[5].seed, 1.5, 1001), MotorRow(bad[5].case, bad[5].seed, -0.1, 1001),
                   MotorRow(bad[5].case, bad[5].seed, 0.0, 0)):
        with pytest.raises(MotorMismatch):
            motor_acceptance(bad[:5] + [broken] + bad[6:])


def test_yw04_the_reading_says_what_it_read():
    d = motor_acceptance(rows([1.0] * 3 + [0.0] * 1197)).to_dict()
    assert d["episodes"] == 1200 and d["limit"] == 0.005 and d["passed"] is True and d["episodes_over"] == 3
    assert d["motor_limit_share"] == pytest.approx(3 / 1200, rel=1e-12)


# ---- through Prove, as the bar is stored ---------------------------------------------------------------------------------------------
def prove(store, *args):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / "tests"), os.environ.get("PYTHONPATH", "")]))
    return subprocess.run([sys.executable, "-m", "poisson_prove", "--store", str(store), *args, "--plain"], env=env, cwd=ROOT,
                          capture_output=True, text=True)


_STORE = {}


@pytest.fixture(scope="module")
def motor_store(tmp_path_factory):
    if "store" in _STORE:
        return _STORE["store"]
    root = tmp_path_factory.mktemp("motor-bar")
    spec = root / "bar.toml"
    spec.write_text(bar_sweep_spec(target="motor_prove_targets:bar_motor_policy", target_id="drone-robustness-4d"))
    st = root / "store"
    for name, target in (("with-motor", "bar_motor_policy"), ("without-motor", "bar_no_motor"), ("buzz", "bar_motor_buzz")):
        r = prove(st, "sweep", str(spec), "--target", f"motor_prove_targets:{target}", "--salt", N.BAR_SALT, "--yes", "--max-mde", "1",
                  "--name", name)
        assert r.returncode == 0, f"sweep {name} failed:\n{r.stdout}\n{r.stderr}"
    _STORE["store"] = st
    return st


def test_yw04_a_stored_bar_run_reads_back_its_1200_motor_rows(motor_store):
    got = motor_rows_from_store(motor_store, "with-motor")
    assert len(got) == 1200 and all(isinstance(r, MotorRow) and not r.error for r in got)
    assert sorted((r.case, r.seed) for r in got) == sorted(KEYS)
    for r in got:
        assert r.share == MT.motor_share(r.seed) and r.steps == MT.STEPS
    reading = motor_acceptance(got)
    want = sum(MT.motor_share(s) for _, s in KEYS) / 1200
    assert reading.share == pytest.approx(want, rel=1e-12) and want > 0.0
    assert reading.passed is (want <= 0.005)
    assert not motor_acceptance(motor_rows_from_store(motor_store, "buzz")).passed


def test_yw04_a_run_swept_before_the_motor_line_is_not_read_as_a_pass(motor_store):
    rows_ = motor_rows_from_store(motor_store, "without-motor")
    assert len(rows_) == 1200 and all(r.error for r in rows_)
    with pytest.raises(MotorMismatch):
        motor_acceptance(rows_)
    with pytest.raises(MotorMismatch):
        motor_rows_from_store(motor_store, "no-such-run")


# ---- the selection's third line ------------------------------------------------------------------------------------------------------
def _yaw(passed):
    return YawReading(passed=passed, passes_rate=passed, passes_drift=passed, passes_45=passed, passes_90=passed, passes_heading=passed)


def readings(pattern):
    """One letter per nominal seed: P passes all three; B bar, Y yaw, F bar and yaw, M motor alone; b, y, f the same with the
    motor line failed too."""
    kind = {"P": (1, 1, 1), "B": (0, 1, 1), "Y": (1, 0, 1), "F": (0, 0, 1), "M": (1, 1, 0), "b": (0, 1, 0), "y": (1, 0, 0), "f": (0, 0, 0)}
    return [(i + 1, BarReading(passed=bool(kind[ch][0])), _yaw(bool(kind[ch][1])), MotorReading(passed=bool(kind[ch][2])))
            for i, ch in enumerate(pattern)]


def test_yw04_a_seed_that_passes_the_bar_and_the_yaw_gate_but_pegs_a_motor_is_no_lineage():
    sel = select_lineages_with_yaw(readings("M" + "P" * 16))
    assert sel.failures == (1,) and dict(sel.failed_on) == {1: "motor"} and sel.lineages == tuple(range(2, 18)) and sel.tried == 17


def test_yw04_each_failure_names_every_gate_it_failed():
    sel = select_lineages_with_yaw(readings("PBYFMbyf" + "P" * 15))
    assert sel.failures == tuple(range(2, 9))
    assert dict(sel.failed_on) == {2: "bar", 3: "yaw", 4: "both", 5: "motor", 6: "bar+motor", 7: "yaw+motor", 8: "both+motor"}
    assert sel.lineages == (1,) + tuple(range(9, 24)) and len(sel.lineages) == LINEAGES
    assert failed_gate(True, True, True) is None and failed_gate(True, True) is None


def test_yw04_motor_failures_do_not_stop_the_selection():
    sel = select_lineages_with_yaw(readings("M" * 40 + "P" * 16))
    assert len(sel.failures) == 40 and sel.lineages == tuple(range(41, 57)) and sel.tried == 56
    assert set(sel.failed_on.values()) == {"motor"}
    sel = select_lineages_with_yaw(readings("P" * 16 + "MMMM"))
    assert sel.lineages == tuple(range(1, 17)) and sel.tried == 16                              # nothing read after the sixteenth pass


def test_yw04_the_three_reading_form_is_4da_5s_selection_unchanged():
    three = [(s, b, y) for s, b, y, _ in readings("PYBFP" + "P" * 14)]
    sel = select_lineages_with_yaw(three)
    assert dict(sel.failed_on) == {2: "yaw", 3: "bar", 4: "both"} and sel.tried == 19
    with pytest.raises(ValueError):
        select_lineages_with_yaw([(1, BarReading(passed=True))])
