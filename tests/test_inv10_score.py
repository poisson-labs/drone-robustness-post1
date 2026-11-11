"""Invariant 10 (spec section 4; 12; S6 in section 10): each class fires on a constructed trajectory at its threshold
and not just under it; the held-tilt timer resets; a non-finite state is an error row; in evaluation the drone flies on
past the tilt and radius thresholds to section 4's stops, and S6's limits re-score from what's stored, equal to a
direct calculation.

A hold of D seconds is a run of consecutive samples above the threshold from t_a to a sample at t >= t_a + D; the
crash time is that sample's time (3A-5). Samples here are at 100 Hz.
"""
import json
import math

import numpy as np
import pytest

import spec_numbers as S
from drone_study.condition import Condition
from drone_study.score import Limits, NonFiniteState, PRIMARY_LIMITS, Trajectory, rescore, score
from drone_study.sensing import SensingConfig
from drone_study.stack import ScriptedController, StackEnv, fly

DT = 0.01
T = np.round(np.arange(0, 1001) * DT, 10)
N = len(T)


def tilt_quat(deg):
    h = math.radians(deg) / 2
    return np.array([math.cos(h), math.sin(h), 0.0, 0.0])             # rotation about x: tilt equals the angle


def traj(height=3.0, tilt_deg=None, offset=None, contact=None):
    """A drone hovering on a fixed reference; arguments are functions of the sample index or arrays."""
    pos = np.tile([1.0, 0.0, 3.0], (N, 1)).astype(float)
    quat = np.tile(tilt_quat(0.0), (N, 1))
    floor = np.zeros(N, dtype=bool)
    if height is not None and not np.isscalar(height):
        pos[:, 2] = height
    ref = pos.copy()                                                   # the reference goes where the drone goes, unless offset
    if tilt_deg is not None:
        for i, d in enumerate(tilt_deg):
            quat[i] = tilt_quat(d)
    if offset is not None:
        pos = pos + np.asarray(offset)
    if contact is not None:
        floor = np.asarray(contact, dtype=bool)
    return Trajectory(T.copy(), pos, quat, ref, floor)


def tilt_profile(*segments):
    """segments: (start_s, end_s_inclusive, degrees) on a 100 Hz grid."""
    d = np.zeros(N)
    for a, b, deg in segments:
        d[int(round(a / DT)): int(round(b / DT)) + 1] = deg
    return d


# ---- ground -------------------------------------------------------------------------------------------------------------
def test_inv10_ground_fires_at_0_05_m_and_not_just_over_it():
    for h, crashes in ((0.0501, False), (0.05, True), (0.0499, True), (0.0, True)):
        z = np.full(N, 3.0); z[500:] = h
        out = score(traj(height=z))
        assert out.passed is (not crashes), h
        if crashes:
            assert out.crash_class == "ground" and out.crash_time == pytest.approx(5.0, abs=1e-9)


def test_inv10_contact_with_the_floor_is_a_ground_crash_whatever_the_height_reads():
    c = np.zeros(N, dtype=bool); c[300:] = True
    out = score(traj(contact=c))
    assert (out.crash_class, out.passed) == ("ground", False)
    assert out.crash_time == pytest.approx(3.0, abs=1e-9)


# ---- tilt ---------------------------------------------------------------------------------------------------------------
TILT60 = [(60.5, 1.00, True), (60.5, 0.99, False), (60.01, 1.00, True), (59.99, 5.0, False)]      # a hundredth of a degree
TILT101 = [(101.6, 0.10, True), (101.6, 0.09, False), (101.51, 0.10, True), (101.49, 0.50, False)]  # either side: tilt comes from a quaternion


def _tilt_pairs(cases):
    for deg, held_s, crashes in cases:
        out = score(traj(tilt_deg=tilt_profile((2.0, 2.0 + held_s, deg))))
        assert out.passed is (not crashes), (deg, held_s)
        if crashes:
            assert out.crash_class == "tilt" and out.crash_time == pytest.approx(2.0 + held_s, abs=1e-9), (deg, held_s)


def test_inv10_tilt_over_60_degrees_held_one_second_fires_at_its_threshold_and_not_just_under():
    _tilt_pairs(TILT60)


def test_inv10_tilt_over_101_5_degrees_held_a_tenth_of_a_second_fires_at_its_threshold_and_not_just_under():
    _tilt_pairs(TILT101)


def test_inv10_the_held_tilt_timer_resets_when_the_tilt_drops_back():
    gap = tilt_profile((2.0, 2.6, 70.0), (2.61, 2.61, 10.0), (2.62, 3.22, 70.0))            # two 0.6 s holds with a gap
    assert score(traj(tilt_deg=gap)).passed is True
    straight = tilt_profile((2.0, 3.2, 70.0))
    assert score(traj(tilt_deg=straight)).passed is False
    gap2 = tilt_profile((2.0, 2.06, 110.0), (2.07, 2.07, 80.0), (2.08, 2.14, 110.0))        # the 101.5 rule: 0.06 s twice
    assert score(traj(tilt_deg=gap2)).passed is True                                          # (80 deg for one sample: under both rules' hold)
    assert score(traj(tilt_deg=tilt_profile((2.0, 2.12, 110.0)))).passed is False


def test_inv10_the_first_class_to_fire_is_the_one_reported_and_later_ones_are_still_stored():
    z = np.full(N, 3.0); z[700:] = 0.04
    out = score(traj(height=z, tilt_deg=tilt_profile((2.0, 3.0, 65.0))))
    assert (out.crash_class, out.crash_time) == ("tilt", pytest.approx(3.0, abs=1e-9))
    assert any(v == pytest.approx(7.0, abs=1e-9) for v in out.crossings.values() if v is not None)   # the ground crossing is kept


# ---- radius ---------------------------------------------------------------------------------------------------------------
RADIUS = [((0.5, 0.0, 0.0), False), ((0.5001, 0.0, 0.0), True), ((0.4999, 0.0, 0.0), False),
          ((0.0, 0.0, 0.5), False), ((0.0, 0.0, 0.5001), True), ((0.0, -0.5001, 0.0), True),
          ((0.3 * 1.0001, 0.4 * 1.0001, 0.0), True), ((0.3 * 0.9999, 0.4 * 0.9999, 0.0), False)]


def test_inv10_radius_fires_over_half_a_metre_from_the_reference_and_not_just_under():
    for offset, crashes in RADIUS:
        off = np.zeros((N, 3)); off[400:] = offset
        out = score(traj(offset=off))
        assert out.passed is (not crashes), offset
        if crashes:
            assert (out.crash_class, out.crash_time) == ("radius", pytest.approx(4.0, abs=1e-9)), offset


def test_inv10_radius_is_measured_from_where_the_reference_is_at_that_moment():
    t = traj()
    moving_ref = t.reference.copy(); moving_ref[:, 0] = np.linspace(1.0, 6.0, N)
    follows = Trajectory(t.t, moving_ref.copy(), t.quat, moving_ref, t.floor_contact)
    assert score(follows).passed is True
    stays = Trajectory(t.t, t.position, t.quat, moving_ref, t.floor_contact)
    out = score(stays)
    assert out.crash_class == "radius"
    assert out.crash_time == pytest.approx(1.01, abs=1e-9)                             # 0.5 m behind a 0.5 m/s reference at 1.0 s: over it one sample later


# ---- non-finite -------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("field,value", [("position", np.nan), ("position", np.inf), ("quat", np.nan), ("reference", -np.inf), ("t", np.nan)])
def test_inv10_a_non_finite_state_is_an_error_row_never_a_crash(field, value):
    good = traj()
    assert score(good).passed is True                                   # the control: the same trajectory, finite, is accepted
    arrays = {f: getattr(good, f).copy() for f in ("t", "position", "quat", "reference", "floor_contact")}
    arrays[field] = arrays[field].astype(float)
    arrays[field].flat[arrays[field].size // 2] = value
    with pytest.raises(NonFiniteState):
        score(Trajectory(**arrays))


# ---- S6: stored first crossings re-score exactly ----------------------------------------------------------------------------
S6_LIMITS = [PRIMARY_LIMITS] + [Limits(tilt_rules=((a, 0.0),)) for a in S.S6_TILT_INSTANT] + \
            [Limits(radius=r) for r in S.S6_RADIUS] + [Limits(radius=None)]


def random_trajectory(rng):
    t = T.copy()
    walk = np.cumsum(rng.normal(0, 0.012, (N, 3)), axis=0) * rng.choice([0.2, 1.0, 3.0])
    ref = np.stack([np.cos(t * 0.6), np.sin(t * 0.6), np.full(N, 3.05)], axis=1)
    pos = ref + walk
    if rng.random() < 0.3:
        pos[:, 2] = np.maximum(pos[:, 2] - np.linspace(0, rng.uniform(0, 3.3), N), 0.0)
    tilt = np.abs(np.cumsum(rng.normal(0, rng.choice([0.5, 2.0, 5.0]), N)))
    tilt = np.where(rng.random(N) < 0.02, tilt + rng.uniform(0, 60), tilt)
    quat = np.stack([tilt_quat(d) for d in np.clip(tilt, 0, 179)])
    floor = np.zeros(N, dtype=bool)
    if rng.random() < 0.2:
        floor[rng.integers(100, N):] = True
    return Trajectory(t, pos, quat, ref, floor)


def test_inv10_s6_limits_re_score_from_what_is_stored_equal_to_a_direct_calculation():
    rng = np.random.default_rng(10)
    classes = set()
    for _ in range(150):
        tr = random_trajectory(rng)
        stored = score(tr).crossings
        json.dumps(stored)                                               # what goes in the sidecar is plain data
        for limits in S6_LIMITS:
            direct = score(tr, limits)
            again = rescore(json.loads(json.dumps(stored)), limits)
            assert (again.passed, again.crash_class) == (direct.passed, direct.crash_class), limits
            assert again.crash_time == direct.crash_time, limits
            classes.add(direct.crash_class)
    assert {"ground", "tilt", "radius", None} <= classes                 # the fuzz reached every outcome


def test_inv10_s6_ground_and_tilt_only_ignores_the_radius():
    off = np.zeros((N, 3)); off[400:] = (2.0, 0.0, 0.0)
    out = score(traj(offset=off), Limits(radius=None))
    assert out.passed is True
    assert score(traj(offset=off), PRIMARY_LIMITS).crash_class == "radius"


def test_inv10_s6_instantaneous_tilt_limits_fire_on_a_single_sample():
    one = np.zeros(N); one[250] = 75.5
    assert score(traj(tilt_deg=one), Limits(tilt_rules=((75.0, 0.0),))).crash_class == "tilt"
    assert score(traj(tilt_deg=one), Limits(tilt_rules=((90.0, 0.0),))).passed is True
    assert score(traj(tilt_deg=one)).passed is True                      # and the primary rule, which needs a hold, does not


# ---- the episode flies on in evaluation and stops at section 4's stops ------------------------------------------------------
def _wild(view):
    if view.t < 0.5:
        return [1.4 * S.WEIGHT, 2.0, 0.0, 0.0]                           # roll hard, then hold the attitude and keep the thrust
    return [S.T_MAX, 0.0, 0.0, 0.0]


def test_inv10_in_evaluation_the_episode_flies_on_past_the_radius_and_tilt_thresholds():
    cond = Condition(case="circle-fast", turb=0.3)
    ev = fly(cond, 6, ScriptedController(_wild), mode="evaluation")
    out = score(ev.trajectory())
    assert out.crash_class in ("tilt", "radius") and out.crash_time is not None
    assert out.crash_time < 2.0
    assert ev.t[-1] >= out.crash_time + 0.5                              # it did not stop at the first crossing
    assert ev.end_reason in ("time", "ground")
    crossings = {k: v for k, v in out.crossings.items() if v is not None}
    assert len(crossings) >= 2                                           # a second threshold was crossed after the first


def test_inv10_in_training_the_episode_ends_at_the_first_crash_class_the_same_one_score_finds():
    cond = Condition(case="circle-fast", turb=0.3)
    ev = fly(cond, 6, ScriptedController(_wild), mode="evaluation")
    tr = fly(cond, 6, ScriptedController(_wild), mode="training")
    first = score(ev.trajectory())
    assert tr.end_reason.startswith("training-crash")
    assert tr.t[-1] == pytest.approx(first.crash_time, abs=DT + 1e-9)
    assert tr.n_steps < ev.n_steps


def test_inv10_evaluation_stops_at_floor_contact_and_runs_to_ten_seconds_otherwise():
    cond = Condition.calm("circle-fast")
    fall = fly(cond, 2, ScriptedController(lambda v: [0.0, 0.0, 0.0, 0.0]))
    assert fall.end_reason == "ground"
    out = score(fall.trajectory())
    assert out.crash_class == "radius"                                    # it leaves the 0.5 m sphere (about 0.3 s) before it lands
    assert score(fall.trajectory(), Limits(radius=None)).crash_class == "ground"
    ground_at = out.crossings["ground"]
    assert ground_at is not None
    assert fall.t[-1] >= ground_at - 1e-9 and fall.t[-1] < 2.0            # free fall from 3.05 m takes about 0.8 s
    assert fall.floor_contact[-1] or fall.position[-1, 2] <= S.GROUND_M
    # Ruling 3B-1 (A, Taylor, 2026-11-05): an episode starts at the attitude the reference's acceleration needs (spec section 2),
    # about 19 degrees on circle-fast, so the hold carries the tilt: thrust whose vertical part is the weight. (As first written
    # the hold was [S.WEIGHT, 0, 0, 0], which lands from that start at 4.8 s.) The assertions are unchanged.
    hover = fly(Condition(case="circle-fast", turb=0.3), 2, ScriptedController(lambda v: [S.WEIGHT / v.reading.rotation[2, 2], 0.0, 0.0, 0.0]))
    assert hover.end_reason == "time" and hover.n_steps == S.N_STEPS
    assert hover.t[-1] == pytest.approx(S.EPISODE_S - S.DT_CTRL, abs=1e-9)
