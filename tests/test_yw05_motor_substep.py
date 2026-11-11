"""Ruling 4DB-2 (Taylor, 2026-11-09), added by session 4D-B as an additive change: the motor line is re-based on the sub-step count.
A control step counts when any of the four motors' commands is at its limit (0 or 65,535 units) on ANY of the step's five 500 Hz
substeps, the reading 4C's numbers came from when they set the 0.5% line. The stack's record carries it (one entry per control step
flown), the rollout writes the episode's share and the steps flown into each bar row's `meta`, and the selection reads it. The
line stays at 0.5% of the bar's control steps, inclusive, pooled over the 1,200 calm episodes. The end-of-step reading (4DB-1)
stays in the record and in `meta` and is reported beside it.

Why: near the line the two readings differ by 1.5x to 9x; pilots 9003 and 9004 at 40M read 0.22% and 0.34% at the ends of the
steps and 1.20% and 1.91% on 4C's reading, and post 2's fine-tunes train past 5M, where the two disagree. Nothing in 3A's, 4A's,
4D-A's, 4DB-1's, 5A's or 8A's tests changes.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from poisson_prove.executor import derive_seed
from poisson_prove.store import config_hash

import motor_substep_prove_targets as ST
import training_numbers as N
from drone_study.condition import Condition, condition_to_cfg
from drone_study.controllers import GeometricController
from drone_study.rollout import rollout_for
from drone_study.stack import N_STEPS, ControllerView, ScriptedController, StackEnv, fly
from drone_study.training import behaviour as B
from drone_study.training.bar import BarReading, bar_sweep_spec
from drone_study.training.behaviour import (MotorMismatch, MotorReading, MotorRow, YawReading, motor_acceptance, motor_limit_share,
                                            motor_limit_share_substep, motor_rows_from_store, motor_substep_rows_from_store)
from drone_study.training.yaw_weight import select_lineages_with_yaw
from drone_study.vehicle import T_MAX, WEIGHT

ROOT = Path(__file__).resolve().parents[1]
CALM = [Condition.calm(name) for name in N.CASE_NAMES]
KEYS = [(c.case, derive_seed(N.BAR_SALT, config_hash(condition_to_cfg(c)), k)) for c in CALM for k in range(N.BAR_REPLICATES)]
SUB = B.READING_SUBSTEP


def bang():
    """A flyer that slams its yaw rate back and forth: some substeps of some steps at a limit, and it lands before 10 s."""
    return ScriptedController(lambda view: np.array([WEIGHT * 1.05, 0.0, 0.0, 3.0 * np.sign(np.sin(view.t * 15))]))


def instrumented(condition, seed, controller):
    """Written from the ruling, independently of the package, the way 4C's instrument read it: every substep's mixer output is
    captured by wrapping the actuation, and a step counts when any of its five rows has a command <= 0 or >= 65,535."""
    env = StackEnv(condition, seed, mixer=controller.mixer(condition.motor_scales()))
    act, inner, cap = env.actuation, env.actuation.step, []

    def step(*a):
        out = inner(*a)
        cap.append(np.array(act.mixer_output, dtype=float).copy())
        return out

    act.step = step
    controller.reset()
    reading = env.reset()
    for _ in range(N_STEPS):
        result = env.step(controller.act(ControllerView(env.time, reading, env.reference_ahead(), env.privileged(controller.privileges))))
        reading = result.reading
        if result.done:
            break
    per = np.array(cap).reshape(-1, 5, 4)
    flags = [any(u <= 0.0 or u >= 65535.0 for sub in stp for u in sub) for stp in per]
    return env.record, np.array(flags, dtype=bool)


class Rec:
    def __init__(self, flags):
        self.motor_limit_substep = None if flags is None else np.asarray(flags, dtype=bool)


# ---- the constants ------------------------------------------------------------------------------------------------------------------
def test_yw05_the_line_reads_any_substep_and_stays_at_half_a_percent():
    assert B.MOTOR_LINE_READING == B.READING_SUBSTEP == "any-substep" and B.READING_END_OF_STEP == "end-of-step"
    assert B.MOTOR_LIMIT_SHARE_MAX == 0.005
    assert B.MOTOR_SUBSTEP_META_FIELD == "motor_limit_share_substep" and B.MOTOR_STEPS_META_FIELD == "motor_steps_flown"
    assert B.MOTOR_META_FIELD == "motor_limit_share"                                            # the end-of-step field is kept
    assert not {B.MOTOR_SUBSTEP_META_FIELD, B.MOTOR_STEPS_META_FIELD} & set(N.YAW_META_FIELDS)


# ---- the stack's record --------------------------------------------------------------------------------------------------------------
def test_yw05_the_record_has_one_entry_per_step_flown_on_a_timed_out_episode():
    rec = fly(Condition.calm("oval-medium"), 7, GeometricController())
    assert rec.end_reason == "time" and rec.n_steps == 1000
    assert rec.motor_limit_substep.dtype == bool and len(rec.motor_limit_substep) == 1000


def test_yw05_the_record_has_one_entry_per_step_flown_on_a_landed_episode():
    rec = fly(Condition.calm("oval-medium"), 7, bang())
    assert rec.end_reason == "ground"
    assert len(rec.motor_limit_substep) == rec.n_steps - 1                                      # the last row is the state it ended in


def test_yw05_the_records_reading_is_4cs_reading_step_for_step():
    for seed in (7, 11):
        rec, flags = instrumented(Condition.calm("oval-medium"), seed, bang())
        assert np.array_equal(rec.motor_limit_substep, flags), seed
        assert 0 < flags.sum() < len(flags)                                                     # some steps, not all


def test_yw05_reading_the_substeps_changes_nothing_in_the_flight():
    c = Condition.calm("slalom-fast")
    a = fly(c, 3, GeometricController())
    b, _ = instrumented(c, 3, GeometricController())
    for f in ("position", "velocity", "quat", "body_rate", "rotor_omega", "mixer_output", "applied_action"):
        assert np.array_equal(getattr(a, f), getattr(b, f)), f
    assert np.array_equal(a.motor_limit_substep, b.motor_limit_substep)


def test_yw05_every_step_whose_end_is_at_a_limit_is_counted_on_its_substeps():
    rec = fly(Condition.calm("oval-medium"), 7, bang())
    ends = ((rec.mixer_output <= 0.0) | (rec.mixer_output >= 65535.0)).any(axis=1)[1:]           # row k + 1 ends step k
    sub = rec.motor_limit_substep[:len(ends)]
    assert np.all(sub[ends])                                                                    # the end is one of the five
    assert sub.sum() > ends.sum()                                                               # and the substeps see more
    assert motor_limit_share_substep(rec) > motor_limit_share(rec)


def test_yw05_full_throttle_is_at_the_top_on_every_step_but_the_hold():
    rec = fly(Condition.calm("oval-medium"), 7, ScriptedController(lambda view: np.array([T_MAX, 0.0, 0.0, 0.0])))
    assert rec.n_steps == 1000 and not rec.motor_limit_substep[0] and rec.motor_limit_substep[1:].all()
    assert motor_limit_share_substep(rec) == pytest.approx(999 / 1000, abs=1e-15)


def test_yw05_zero_thrust_is_at_the_floor_on_every_step_but_the_hold():
    rec, flags = instrumented(Condition.calm("oval-medium"), 7, ScriptedController(lambda view: np.zeros(4)))
    assert rec.end_reason == "ground" and not rec.motor_limit_substep[0] and rec.motor_limit_substep[1:].all()
    assert np.array_equal(rec.motor_limit_substep, flags)
    assert (rec.mixer_output[2:] <= 0.0).any(axis=1).all() and (rec.mixer_output[2:] < 65535.0).all()   # the floor, never the top


def test_yw05_a_new_episode_starts_with_no_step_counted():
    c = Condition.calm("oval-medium")
    env = StackEnv(c, 7)
    env.reset()
    for _ in range(50):
        env.step(np.array([T_MAX, 0.0, 0.0, 0.0]))
    assert env.record.motor_limit_substep.sum() == 49
    env.reset()
    assert len(env.record.motor_limit_substep) == 0
    env.step(np.array([WEIGHT, 0.0, 0.0, 0.0]))
    assert len(env.record.motor_limit_substep) == 1 and not env.record.motor_limit_substep[0]


# ---- one episode's share --------------------------------------------------------------------------------------------------------------
def test_yw05_the_share_is_the_flagged_steps_over_the_steps_flown():
    assert motor_limit_share_substep(Rec([False] * 9 + [True])) == pytest.approx(0.1, abs=1e-15)
    assert motor_limit_share_substep(Rec([True, True, False, False])) == 0.5
    assert motor_limit_share_substep(Rec([])) == 0.0


def test_yw05_a_record_without_the_reading_is_refused_not_read_as_zero():
    with pytest.raises(MotorMismatch):
        motor_limit_share_substep(Rec(None))


def test_yw05_the_rollouts_meta_carries_both_readings_and_the_steps_flown():
    condition, seed = Condition.calm("oval-medium"), 7
    ep = rollout_for(bang())(condition_to_cfg(condition), seed)
    rec = fly(condition, seed, bang())
    assert rec.end_reason == "ground" and ep.meta["motor_steps_flown"] == rec.n_steps - 1                # flown, not rows
    assert ep.meta["motor_limit_share_substep"] == motor_limit_share_substep(rec)
    assert ep.meta["motor_steps_flown"] == len(rec.motor_limit_substep)
    assert ep.meta["motor_limit_share"] == motor_limit_share(rec)                               # 4DB-1's reading, beside it
    assert ep.meta["motor_limit_share_substep"] > ep.meta["motor_limit_share"]


# ---- the line on the bar's 1,200 rows --------------------------------------------------------------------------------------------------
def rows(shares, steps=1000, keys=KEYS, reading=SUB):
    shares = list(shares) if not np.isscalar(shares) else [float(shares)] * len(keys)
    return [MotorRow(c, int(s), float(x), int(steps), reading=reading) for (c, s), x in zip(keys, shares)]


def test_yw05_the_line_on_substeps_pools_the_steps_flown_and_is_inclusive():
    assert motor_acceptance(rows(0.005)).passed and not motor_acceptance(rows(0.0051)).passed
    r = motor_acceptance(rows(0.0))
    assert r.passed and r.reading == SUB and r.to_dict()["reading"] == SUB
    short = rows(0.0)
    short[0] = MotorRow(short[0].case, short[0].seed, 1.0, 1, reading=SUB)                      # one step flown, at a limit
    assert motor_acceptance(short).share == pytest.approx(1 / (1199 * 1000 + 1), rel=1e-12)  # weighted by steps flown, not - 1


def test_yw05_the_pilots_at_40m_pass_on_the_ends_of_the_steps_and_fail_on_the_substeps():
    for end, sub in ((0.00219, 0.01199), (0.00343, 0.01912)):                                   # 9003 and 9004 at 40M
        assert motor_acceptance(rows(end, steps=1001, reading=B.READING_END_OF_STEP)).passed
        assert not motor_acceptance(rows(sub)).passed


def test_yw05_rows_of_both_readings_are_never_pooled():
    mixed = rows(0.0)[:600] + rows(0.0, reading=B.READING_END_OF_STEP)[600:]
    with pytest.raises(MotorMismatch):
        motor_acceptance(mixed)
    odd = rows(0.0, reading="end-of-something")
    with pytest.raises(MotorMismatch):
        motor_acceptance(odd)


def test_yw05_an_end_of_step_row_still_reads_as_4db_1_read_it():
    assert MotorRow("x", 1).reading == B.READING_END_OF_STEP and MotorReading().reading == B.READING_END_OF_STEP


# ---- through Prove, as the bar is stored ---------------------------------------------------------------------------------------------
def prove(store, *args):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / "tests"), os.environ.get("PYTHONPATH", "")]))
    return subprocess.run([sys.executable, "-m", "poisson_prove", "--store", str(store), *args, "--plain"], env=env, cwd=ROOT,
                          capture_output=True, text=True)


_STORE = {}


@pytest.fixture(scope="module")
def substep_store(tmp_path_factory):
    if "store" in _STORE:
        return _STORE["store"]
    root = tmp_path_factory.mktemp("motor-substep-bar")
    spec = root / "bar.toml"
    spec.write_text(bar_sweep_spec(target="motor_substep_prove_targets:bar_substep_policy", target_id="drone-robustness-4d"))
    st = root / "store"
    for name, target in (("substep", "motor_substep_prove_targets:bar_substep_policy"),
                         ("hidden", "motor_substep_prove_targets:bar_substep_hidden"),
                         ("end-only", "motor_prove_targets:bar_motor_policy")):
        r = prove(st, "sweep", str(spec), "--target", target, "--salt", N.BAR_SALT, "--yes", "--max-mde", "1", "--name", name)
        assert r.returncode == 0, f"sweep {name} failed:\n{r.stdout}\n{r.stderr}"
    _STORE["store"] = st
    return st


def test_yw05_a_stored_bar_run_reads_back_its_1200_substep_rows(substep_store):
    got = motor_substep_rows_from_store(substep_store, "substep")
    assert len(got) == 1200 and all(not r.error and r.reading == SUB for r in got)
    assert sorted((r.case, r.seed) for r in got) == sorted(KEYS)
    for r in got:
        assert r.share == ST.substep_share(r.seed) and r.steps == ST.FLOWN
    reading = motor_acceptance(got)
    want = sum(ST.substep_share(s) for _, s in KEYS) / 1200
    assert reading.share == pytest.approx(want, rel=1e-12) and want > 0.0 and reading.reading == SUB


def test_yw05_a_flyer_inside_the_line_at_the_ends_of_its_steps_fails_it_on_its_substeps(substep_store):
    end = motor_acceptance(motor_rows_from_store(substep_store, "hidden"))
    sub = motor_acceptance(motor_substep_rows_from_store(substep_store, "hidden"))
    assert end.passed and end.reading == B.READING_END_OF_STEP
    assert not sub.passed and sub.share == pytest.approx(0.012, rel=1e-12)


def test_yw05_a_run_swept_before_4db_2_is_not_read_as_a_pass_on_the_line(substep_store):
    rows_ = motor_substep_rows_from_store(substep_store, "end-only")
    assert len(rows_) == 1200 and all(r.error for r in rows_)
    with pytest.raises(MotorMismatch):
        motor_acceptance(rows_)
    assert motor_acceptance(motor_rows_from_store(substep_store, "end-only")).episodes == 1200      # 4DB-1's reading still reads
    with pytest.raises(MotorMismatch):
        motor_substep_rows_from_store(substep_store, "no-such-run")


# ---- the selection --------------------------------------------------------------------------------------------------------------------
def test_yw05_the_selection_fails_a_seed_on_the_substep_reading_with_gate_motor():
    def yaw():
        return YawReading(passed=True, passes_rate=True, passes_drift=True, passes_45=True, passes_90=True, passes_heading=True)

    sub_fail = motor_acceptance(rows(0.012))
    sub_pass = motor_acceptance(rows(0.0))
    seq = [(1, BarReading(passed=True), yaw(), sub_fail)] + [(s, BarReading(passed=True), yaw(), sub_pass) for s in range(2, 18)]
    sel = select_lineages_with_yaw(seq)
    assert dict(sel.failed_on) == {1: "motor"} and sel.lineages == tuple(range(2, 18)) and sel.tried == 17
