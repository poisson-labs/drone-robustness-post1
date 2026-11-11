"""The yaw addenda of 2026-11-07 and 2026-11-08 (session 4D-A), rule 2: the weight is chosen by a rule written before any pilot
trains. Four pilots per weight to 40M, read at 5M, 10M, 20M and 40M against the frozen bar (unchanged) and the yaw acceptance;
a weight passes at a rung when all four pass both. The pick (4DR-1): B0 is the smallest rung at which some weight passes, and
the weight is the smallest passing at that rung. If no base weight (0.004, 0.008, 0.016) passes at any rung the ladder extends
to 0.032, then 0.064 (4DA-3 as amended); the study never stops on this. Each weight carries its own crash penalty (4DR-3),
recorded in the pick. A lineage needs the bar and the yaw gate; failed seeds are reported with the gate they failed and new
seeds are trained until 16 pass, with no stop (4DA-5 as amended).

Every input is synthetic: pass/fail flags per pilot per rung, and bar and yaw readings with their `passed` flag set.
"""
import hashlib
import json
import random
from pathlib import Path

import pytest

import training_numbers as N
from drone_study.training import reward as R
from drone_study.training import yaw_weight as YW
from drone_study.training.bar import BarReading
from drone_study.training.behaviour import YawReading
from drone_study.training.yaw_weight import (LineageSelection, NoWeightPasses, PilotReading, WeightMismatch, YawPick, load_yaw_weight,
                                             next_weight, pick_yaw_weight, select_lineages_with_yaw, write_yaw_weight)

BASE, EXT, LADDER, RUNGS = N.YAW_WEIGHT_LADDER_BASE, N.YAW_WEIGHT_LADDER_EXTENSION, N.YAW_WEIGHT_LADDER, N.RUNGS
ALL = (True, True)
R5, R10, R20, R40 = RUNGS


def table(passes=None, default=ALL, weights=BASE):
    """{weight: {rung: [PilotReading x 4]}} for the weights tried; `passes` maps (weight, rung) to a list of four (bar, yaw) pairs,
    or one pair for all four."""
    passes = passes or {}
    out = {}
    for w in weights:
        out[w] = {}
        for rung in RUNGS:
            spec = passes.get((w, rung), default)
            four = spec if isinstance(spec, list) else [spec] * 4
            out[w][rung] = [PilotReading(bar_passed=b, yaw_passed=y) for b, y in four]
    return out


def fails(*weights, rungs=RUNGS, how=(True, False)):
    """Every pilot of these weights fails at these rungs (by default the yaw gate)."""
    return {(w, r): how for w in weights for r in rungs}


def test_yw02_the_ladder_the_extension_the_rungs_and_the_files_are_the_addendums():
    assert tuple(R.YAW_WEIGHT_LADDER_BASE) == BASE == (0.004, 0.008, 0.016)
    assert tuple(R.YAW_WEIGHT_LADDER_EXTENSION) == EXT == (0.032, 0.064)
    assert tuple(R.YAW_WEIGHT_LADDER) == LADDER == BASE + EXT
    assert list(LADDER) == sorted(LADDER) and len(set(LADDER)) == 5
    assert all(b == 2 * a for a, b in zip(LADDER, LADDER[1:]))                                  # each rung of the ladder doubles
    assert YW.YAW_WEIGHT_FILE == Path("data") / "yaw-weight.json" and YW.LINEAGES_FILE == Path("data") / "4d-lineages.json"
    assert N.RUNGS == (5_000_000, 10_000_000, 20_000_000, 40_000_000) and N.N_PILOTS == 4
    assert {k: R.crash_penalty_for(k) for k in LADDER} == N.CRASH_PENALTIES                     # the penalty travels with the weight


def test_yw02_a_pilot_passes_only_when_the_bar_and_the_yaw_gate_both_pass():
    assert PilotReading(True, True).passed is True
    assert PilotReading(True, False).passed is False and PilotReading(False, True).passed is False and PilotReading(False, False).passed is False


@pytest.mark.parametrize("passes,weight,b0", [
    ({}, 0.004, R5),                                                                               # everything passes: the smallest of both
    (fails(0.004), 0.008, R5),                                                                     # 0.004 holds no heading at any rung
    ({**fails(0.004), **fails(0.008, rungs=RUNGS[:2])}, 0.016, R5),                                   # 0.016 passes at 5M: the rung decides before the weight
    ({**fails(0.004), **fails(0.008, rungs=RUNGS[:2]), **fails(0.016, rungs=RUNGS[:2])}, 0.008, R20),
    (fails(0.004, rungs=RUNGS[:3], how=(False, True)), 0.008, R5),                                 # 0.004 passes at 40M only: the rung is read first
    ({**fails(0.004, how=(False, False)), **fails(0.008)}, 0.016, R5),
    ({(0.004, R5): [ALL, ALL, ALL, (True, False)]}, 0.008, R5),                                    # three of four is not a pass; 0.008 passes at 5M
    ({(0.004, R5): [ALL, ALL, ALL, (False, True)], **fails(0.008, rungs=(R5,)), **fails(0.016, rungs=(R5,))}, 0.004, R10),
    ({**fails(0.004, rungs=(R5, R10)), **fails(0.008, rungs=(R5,)), **fails(0.016, rungs=(R5,))}, 0.008, R10),   # at 10M, 0.008 is the smallest passing
    ({**fails(0.004, rungs=(R5, R10, R20)), **fails(0.008, rungs=(R5, R10)), **fails(0.016, rungs=(R5, R10, R20))}, 0.008, R20),
])
def test_yw02_the_pick_is_the_smallest_rung_at_which_some_weight_passes_then_the_smallest_weight_there(passes, weight, b0):
    pick = pick_yaw_weight(table(passes))
    assert isinstance(pick, YawPick)
    assert (pick.weight, pick.b0) == (weight, b0)
    assert pick.b0 in RUNGS and pick.weight in LADDER
    assert pick.crash_penalty == N.CRASH_PENALTIES[weight]


def test_yw02_a_larger_weight_that_passes_at_a_smaller_rung_beats_a_smaller_weight_that_passes_only_later():
    """4DR-1 reverses the 2026-11-07 order: 0.004 passing at 40M only and 0.008 everywhere picks (5M, 0.008), not (40M, 0.004)."""
    passes = fails(0.004, rungs=RUNGS[:3])
    pick = pick_yaw_weight(table(passes))
    assert (pick.weight, pick.b0) == (0.008, R5)
    passes = {**fails(0.004, rungs=RUNGS[:3]), **fails(0.008, rungs=RUNGS[:2]), **fails(0.016, rungs=(R5,))}
    pick = pick_yaw_weight(table(passes))
    assert (pick.weight, pick.b0) == (0.016, R10)                                                 # the smallest rung is 10M, where only 0.016 passes


# ---- the extension -----------------------------------------------------------------------------------------------------------------
def test_yw02_no_base_weight_passing_is_not_a_stop_the_next_weight_is_the_extensions():
    base_fails = table(default=(True, False))                                                      # the bar passes everywhere, heading never held
    with pytest.raises(NoWeightPasses):
        pick_yaw_weight(base_fails)
    assert next_weight(base_fails) == 0.032
    with_032 = table(default=(True, False), weights=BASE + (0.032,))
    with pytest.raises(NoWeightPasses):
        pick_yaw_weight(with_032)
    assert next_weight(with_032) == 0.064
    with_064 = table(default=(True, False), weights=LADDER)
    with pytest.raises(NoWeightPasses):
        pick_yaw_weight(with_064)
    assert next_weight(with_064) is None                                                           # nothing beyond 0.064 is written: a ruling
    assert issubclass(NoWeightPasses, ValueError)
    assert next_weight(table()) is None                                                            # a table that passes needs no next weight
    assert pick_yaw_weight(table()).weight == 0.004                                                # the control


@pytest.mark.parametrize("passes,weights,weight,b0", [
    ({**fails(*BASE)}, BASE + (0.032,), 0.032, R5),                                                # the base fails everywhere; 0.032 passes at 5M
    ({**fails(*BASE), **fails(0.032, rungs=(R5, R10))}, BASE + (0.032,), 0.032, R20),
    ({**fails(*BASE), **fails(0.032)}, LADDER, 0.064, R5),                                         # and 0.032 everywhere: 0.064
    ({**fails(*BASE), **fails(0.032), **fails(0.064, rungs=(R5, R10, R20))}, LADDER, 0.064, R40),  # the worst case: 40M at 0.064
])
def test_yw02_the_extension_weights_are_picked_by_the_same_rule_with_their_own_crash_penalty(passes, weights, weight, b0):
    pick = pick_yaw_weight(table(passes, weights=weights))
    assert (pick.weight, pick.b0) == (weight, b0)
    assert pick.crash_penalty == N.CRASH_PENALTIES[weight] and pick.crash_penalty > N.CRASH_PENALTY   # 22 at 0.032, 26 at 0.064


def test_yw02_an_extension_weight_in_the_table_while_a_base_weight_passes_is_refused():
    """The extension is tried only when the base passes nowhere (4DA-3 as amended), so a table with 0.032 beside a passing base
    weight is not one the rule produces."""
    with pytest.raises(ValueError):
        pick_yaw_weight(table({}, weights=BASE + (0.032,)))
    with pytest.raises(ValueError):
        pick_yaw_weight(table({**fails(0.004), **fails(0.008)}, weights=LADDER))
    with pytest.raises(ValueError):
        pick_yaw_weight(table({**fails(*BASE)}, weights=BASE + (0.064,)))                           # 0.064 before 0.032
    assert pick_yaw_weight(table({**fails(*BASE)}, weights=BASE + (0.032,))).weight == 0.032         # the control


def test_yw02_a_table_missing_a_weight_a_rung_or_a_pilot_is_refused_but_a_prefix_of_the_ladder_is_the_climb():
    t = table()
    del t[0.008]                                                                                   # a hole in the ladder
    with pytest.raises(ValueError):
        pick_yaw_weight(t)
    t = table()
    del t[0.016]                                                                                   # the climb stopped after 0.008: a prefix, accepted
    assert (pick_yaw_weight(t).weight, pick_yaw_weight(t).b0) == (0.004, R5)
    assert pick_yaw_weight(table(weights=(0.004,))).weight == 0.004                                # or after 0.004
    t = table()
    del t[0.004][R20]
    with pytest.raises(ValueError):
        pick_yaw_weight(t)
    t = table()
    t[0.008][R5] = t[0.008][R5][:3]
    with pytest.raises(ValueError):
        pick_yaw_weight(t)
    t = table()
    t[0.05] = t[0.004]                                                                             # a weight off the ladder
    with pytest.raises(ValueError):
        pick_yaw_weight(t)


def test_yw02_the_pick_does_not_depend_on_the_tables_order():
    passes = {**fails(0.004), (0.008, R5): (True, False), (0.016, R5): (True, False)}
    want = pick_yaw_weight(table(passes))
    assert (want.weight, want.b0) == (0.008, R10)
    rng = random.Random(3)
    for _ in range(5):
        t = table(passes)
        weights = list(t)
        rng.shuffle(weights)
        shuffled = {}
        for w in weights:
            rungs = list(t[w])
            rng.shuffle(rungs)
            shuffled[w] = {r: t[w][r] for r in rungs}
        got = pick_yaw_weight(shuffled)
        assert (got.weight, got.b0) == (want.weight, want.b0)


def test_yw02_the_climb_in_ladder_order_gives_the_full_tables_pick():
    """4DA-9 rewritten: climbing the base ladder in order, 4D-B may stop once a weight passes at 5M (no later weight can enter the
    pick); otherwise it holds the best (rung, weight) and trains the next; the extension only when the base passes nowhere. The
    climb's pick equals the full table's in every case here."""
    def climb(full):
        tried = {}
        for w in BASE:
            tried[w] = full[w]
            best = None
            try:
                best = pick_yaw_weight(tried)
            except NoWeightPasses:
                continue
            if best.b0 == R5:
                return best, tuple(tried)                                                          # nothing later can beat 5M
        while True:
            try:
                return pick_yaw_weight(tried), tuple(tried)
            except NoWeightPasses:
                nxt = next_weight(tried)
                if nxt is None:
                    raise
                tried[nxt] = full[nxt]

    cases = [({}, (0.004,)), (fails(0.004), (0.004, 0.008)), (fails(0.004, rungs=RUNGS[:3]), (0.004, 0.008)),
             ({**fails(0.004, rungs=(R5, R10)), **fails(0.008, rungs=(R5,)), **fails(0.016, rungs=(R5,))}, BASE),
             ({**fails(*BASE)}, BASE + (0.032,)), ({**fails(*BASE), **fails(0.032)}, LADDER)]
    for passes, tried_want in cases:
        full = table(passes, weights=LADDER)
        want = pick_yaw_weight({w: full[w] for w in tried_want})
        got, tried = climb(full)
        assert (got.weight, got.b0, got.crash_penalty) == (want.weight, want.b0, want.crash_penalty)
        assert tried == tried_want                                                                  # and no weight beyond what the rule needs


# ---- the file ------------------------------------------------------------------------------------------------------------------------
def test_yw02_a_pick_writes_reads_back_and_a_file_the_rule_would_not_give_is_refused(tmp_path):
    passes = fails(0.004)
    pick = pick_yaw_weight(table(passes))
    path = tmp_path / "yaw-weight.json"
    digest = write_yaw_weight(path, pick)
    assert path.exists() and digest == hashlib.sha256(path.read_bytes()).hexdigest()
    back = load_yaw_weight(path)
    assert isinstance(back, YawPick) and (back.weight, back.b0, back.crash_penalty) == (0.008, R5, 20.0) == (pick.weight, pick.b0, pick.crash_penalty)
    rt = YawPick.from_dict(pick.to_dict())
    assert (rt.weight, rt.b0, rt.crash_penalty) == (pick.weight, pick.b0, pick.crash_penalty)
    d = json.loads(path.read_text())
    assert d["weight"] == 0.008 and d["b0"] == R5 and d["crash_penalty"] == 20.0                  # plain JSON, readable by 5B's tests
    for tamper in ({"weight": 0.05}, {"b0": 7_000_000}, {"weight": 0.004}, {"b0": R10}, {"crash_penalty": 22.0}, {"weight": 0.032}):
        bad = dict(d)
        bad.update(tamper)
        (tmp_path / "bad.json").write_text(json.dumps(bad))
        with pytest.raises(WeightMismatch):
            load_yaw_weight(tmp_path / "bad.json")                                                  # off the ladder, not a rung, not the rule's penalty, or not what the readings give
    with pytest.raises((WeightMismatch, FileNotFoundError, OSError)):
        load_yaw_weight(tmp_path / "missing.json")


def test_yw02_an_extension_pick_writes_its_raised_penalty_and_reads_back(tmp_path):
    pick = pick_yaw_weight(table({**fails(*BASE), **fails(0.032)}, weights=LADDER))
    assert (pick.weight, pick.b0, pick.crash_penalty) == (0.064, R5, 26.0)
    path = tmp_path / "yaw-weight.json"
    write_yaw_weight(path, pick)
    back = load_yaw_weight(path)
    assert (back.weight, back.b0, back.crash_penalty) == (0.064, R5, 26.0)
    d = json.loads(path.read_text())
    d["crash_penalty"] = 20.0
    (tmp_path / "bad.json").write_text(json.dumps(d))
    with pytest.raises(WeightMismatch):
        load_yaw_weight(tmp_path / "bad.json")                                                      # 4A-1's 20 is not this weight's penalty


# ---- lineages ------------------------------------------------------------------------------------------------------------------------
def _bar(passed):
    return BarReading(passed=passed, passes_crashes=passed, passes_rmse=passed, passes_ratio=passed)


def _yaw(passed):
    return YawReading(passed=passed, passes_rate=passed, passes_drift=passed, passes_45=passed, passes_90=passed, passes_heading=passed)


def readings(pattern):
    """One letter per nominal seed in order: P passes both, B fails the bar only, Y fails the yaw gate only, F fails both."""
    kind = {"P": (True, True), "B": (False, True), "Y": (True, False), "F": (False, False)}
    return [(i + 1, _bar(kind[ch][0]), _yaw(kind[ch][1])) for i, ch in enumerate(pattern)]


def test_yw02_lineages_need_the_bar_and_the_yaw_gate_and_each_failure_names_its_gate():
    sel = select_lineages_with_yaw(readings("P" * 16))
    assert isinstance(sel, LineageSelection) and sel.lineages == tuple(range(1, 17)) and sel.failures == () and sel.tried == 16
    assert dict(sel.failed_on) == {}
    sel = select_lineages_with_yaw(readings("PYBFP" + "P" * 14))
    assert sel.failures == (2, 3, 4)                                                                # a spinner fails like a slow one
    assert dict(sel.failed_on) == {2: "yaw", 3: "bar", 4: "both"}
    assert sel.lineages == (1, 5) + tuple(range(6, 20)) and sel.tried == 19
    sel = select_lineages_with_yaw(readings("P" * 16 + "YYYY"))
    assert sel.lineages == tuple(range(1, 17)) and sel.tried == 16                                  # nothing read after the sixteenth pass
    sel = select_lineages_with_yaw(readings("PPYPP"))
    assert sel.lineages == (1, 2, 4, 5) and sel.failures == (3,) and sel.tried == 5 and len(sel.lineages) < 16


def test_yw02_failed_seeds_are_reported_and_new_seeds_tried_until_sixteen_pass_with_no_stop():
    """4DA-5 as amended: no count of failures stops the selection. Nine, twenty and forty failures of either kind all read on."""
    sel = select_lineages_with_yaw(readings("YBYBYBYB" + "P" * 16))
    assert sel.failures == tuple(range(1, 9)) and sel.lineages == tuple(range(9, 25))
    sel = select_lineages_with_yaw(readings("Y" * 9 + "P" * 16))                                     # the ninth, which 4A's bar rule stopped on
    assert sel.failures == tuple(range(1, 10)) and sel.lineages == tuple(range(10, 26)) and sel.tried == 25
    sel = select_lineages_with_yaw(readings("PYPBPYPBPYPBPYPBPF" + "P" * 16))
    assert len(sel.failures) == 9 and len(sel.lineages) == 16 and sel.tried == 25
    sel = select_lineages_with_yaw(readings("Y" * 20 + "P" * 16))
    assert len(sel.failures) == 20 and sel.lineages == tuple(range(21, 37)) and sel.tried == 36
    sel = select_lineages_with_yaw(readings("F" * 40 + "P" * 16))
    assert len(sel.failures) == 40 and sel.lineages == tuple(range(41, 57)) and sel.tried == 56
    assert all(v == "both" for v in sel.failed_on.values()) and len(sel.failed_on) == 40
