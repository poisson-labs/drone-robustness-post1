"""Training rule 13 (spec section 2, 'The calm-air bar'; N9 of section 20.1; section 20.2's row for N9; section 20.3
point 10): "All three, by code:
  - At most 5 crashes in 1,200 episodes (12 cases x 100), which keeps the Wilson 95% upper end under 1%.
  - Median per-episode tracking RMSE at most 10 cm.
  - That RMSE at most 1.25 times the cascaded PID's on the same seeds."
The bar has its own salt, `76e7b463820f72ad`, so its 1,200 episodes are the 100 seeds Prove derives for each of the 12
calm cells under it, for the policy and for the PID alike.

Readings (ruling 4A-11): the median is over all 1,200 per-episode RMSEs, crashed episodes included; "at most" includes
the line; the bar is read only on 1,200 scored episodes, so an error row stops the read instead of being dropped.
"""
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest
from poisson_prove.executor import derive_seed
from poisson_prove.stats import wilson as prove_wilson
from poisson_prove.store import config_hash

import training_numbers as N
import training_oracles as TO
import training_prove_targets as T
from drone_study.condition import Condition, cell_name, condition_to_cfg
from drone_study.training import bar as BAR
from drone_study.training.bar import BarMismatch, BarReading, BarRow, bar_conditions, bar_seeds, bar_sweep_spec, read_bar, rows_from_store

ROOT = Path(__file__).resolve().parents[1]
CALM = [Condition.calm(name) for name in N.CASE_NAMES]
KEYS = [(c.case, derive_seed(N.BAR_SALT, config_hash(condition_to_cfg(c)), k)) for c in CALM for k in range(N.BAR_REPLICATES)]


def rows(rmse=0.05, crashes=0, keys=KEYS):
    """The bar's 1,200 rows with one RMSE (or one per row) and the first `crashes` of them crashed."""
    values = np.broadcast_to(np.asarray(rmse, dtype=float), (len(keys),))
    return [BarRow(case, seed, i < crashes, float(values[i])) for i, (case, seed) in enumerate(keys)]


def split(low, high, n_low):
    return [low] * n_low + [high] * (N.BAR_EPISODES - n_low)


def test_tr13_the_bars_numbers_are_section_2s():
    assert (BAR.BAR_SALT, BAR.BAR_REPLICATES) == (N.BAR_SALT, 100) and N.BAR_SALT == N.salt("bar") == "76e7b463820f72ad"
    assert (BAR.MAX_CRASHES, BAR.MAX_RMSE_M, BAR.MAX_RMSE_RATIO) == (5, 0.10, 1.25)
    assert (BAR.LINEAGES, BAR.MAX_FAILURES) == (16, 8)


def test_tr13_the_bar_flies_the_twelve_cases_in_calm_air():
    cells = bar_conditions()
    assert cells == CALM and len(cells) == 12
    assert all(c.turb == 0.0 and not c.has_gust and c.payload == 0.0 and c.motor_index is None and c.delay == 0 for c in cells)


def test_tr13_its_seeds_are_the_hundred_prove_derives_for_each_cell_under_the_bars_salt():
    assert len(CALM) == 12
    got = [(c.case, s) for c in CALM for s in bar_seeds(c)]
    assert got == KEYS and len(got) == N.BAR_EPISODES == 1200
    assert len(set(got)) == 1200
    zone_d = derive_seed(N.salt("D"), config_hash(condition_to_cfg(CALM[0])), 0)
    assert bar_seeds(CALM[0])[:1] != (zone_d,)                                # N9's episodes are not Zone D's (section 20.3)


def test_tr13_the_sweep_is_twelve_cells_by_a_hundred_replicates():
    text = bar_sweep_spec(target="training_prove_targets:bar_policy", target_id="pi0-lineage-1")
    assert text.strip()
    spec = tomllib.loads(text)
    assert spec["replicates"] == 100 and spec["target"] == "training_prove_targets:bar_policy" and spec["target_id"] == "pi0-lineage-1"
    assert spec["axes"]["cell"]["values"] == [cell_name(c) for c in CALM]
    assert spec["declare"]["failures"] == ["crash"]


@pytest.mark.parametrize("crashes,ok", [(0, True), (5, True), (6, False), (40, False)])
def test_tr13_at_most_five_crashes_in_1200(crashes, ok):
    reading = read_bar(rows(crashes=crashes), rows(0.05))
    assert isinstance(reading, BarReading)
    assert (reading.crashes, reading.episodes) == (crashes, 1200)
    assert reading.passes_crashes is ok and reading.passed is ok
    assert reading.passes_rmse is True and reading.passes_ratio is True


@pytest.mark.parametrize("crashes", [0, 5, 6, 40])
def test_tr13_the_crash_rate_carries_its_wilson_95_interval(crashes):
    reading = read_bar(rows(crashes=crashes), rows(0.05))
    lo, hi = TO.wilson(crashes, 1200)
    assert reading.wilson == pytest.approx((lo, hi), rel=1e-9, abs=1e-15)
    w = prove_wilson(crashes, 1200)
    assert reading.wilson == pytest.approx((w.lo, w.hi), rel=1e-9, abs=1e-15)
    assert (reading.wilson[1] < N.BAR_WILSON_UPPER) == (crashes <= 5)         # why the line is at 5: the upper end stays under 1%


@pytest.mark.parametrize("values,median,ok", [
    (0.08, 0.08, True), (0.10, 0.10, True), (0.1001, 0.1001, False),          # at most 10 cm: the line is in
    (split(0.08, 0.30, 601), 0.08, True),                                     # a median, not a mean (the mean is 19 cm)
    (split(0.02, 0.12, 599), 0.12, False),                                    # ... in both directions (the mean is 7 cm)
    (split(0.09, 0.11, 600), 0.10, True), (split(0.09, 0.1102, 600), 0.1001, False),   # 1,200 is even: the middle two, averaged
])
def test_tr13_the_median_per_episode_rmse_is_at_most_ten_centimetres(values, median, ok):
    reading = read_bar(rows(values), rows(0.2))                               # a PID at 20 cm, so only the 10 cm line can bind
    assert reading.median_rmse == pytest.approx(median, rel=1e-12)
    assert reading.passes_rmse is ok and reading.passed is ok
    assert reading.passes_crashes is True and reading.passes_ratio is True


def test_tr13_a_crashed_episodes_rmse_is_in_the_median():
    values = split(0.02, 0.12, 599)                                           # the 601 at 12 cm carry the median
    crashed_low = read_bar(rows(values, crashes=5), rows(0.2))                # the five crashed ones are among the 2 cm rows
    assert crashed_low.median_rmse == pytest.approx(0.12) and crashed_low.crashes == 5
    lifted = [0.12] * 601 + [0.02] * 599                                      # the first 5 of 601 high rows crashed: still counted
    assert read_bar(rows(lifted, crashes=5), rows(0.2)).median_rmse == pytest.approx(0.12)


@pytest.mark.parametrize("policy,pid,ok", [
    (0.078125, 0.0625, True),                                                 # exactly 1.25 times (5/64 over 1/16): at most
    (0.0782, 0.0625, False), (0.05, 0.0625, True), (0.09, 0.03, False),
])
def test_tr13_the_rmse_is_at_most_one_and_a_quarter_times_the_pids_on_the_same_seeds(policy, pid, ok):
    reading = read_bar(rows(policy), rows(pid))
    assert reading.pid_median_rmse == pytest.approx(pid, rel=1e-12) and reading.ratio == pytest.approx(policy / pid, rel=1e-12)
    assert reading.passes_ratio is ok and reading.passed is ok
    assert reading.passes_crashes is True and reading.passes_rmse is True     # under 10 cm, so only the PID's line can bind


def test_tr13_the_pids_own_crashes_are_reported_and_its_median_is_over_all_its_rows():
    reading = read_bar(rows(0.05), rows(split(0.04, 0.5, 601), crashes=3))
    assert reading.pid_crashes == 3 and reading.pid_median_rmse == pytest.approx(0.04)
    assert reading.passed is True and reading.crashes == 0


def test_tr13_all_three_are_needed():
    good = read_bar(rows(0.05), rows(0.05))
    assert good.passed is True and (good.passes_crashes, good.passes_rmse, good.passes_ratio) == (True, True, True)
    for policy, pid, flags in ((rows(0.05, crashes=6), rows(0.05), (False, True, True)),
                               (rows(0.11), rows(0.2), (True, False, True)),
                               (rows(0.09), rows(0.05), (True, True, False)),
                               (rows(0.3, crashes=9), rows(0.05), (False, False, False))):
        reading = read_bar(policy, pid)
        assert (reading.passes_crashes, reading.passes_rmse, reading.passes_ratio) == flags and reading.passed is False


def test_tr13_rows_that_are_not_the_bars_1200_on_the_same_seeds_are_refused():
    assert read_bar(rows(0.05), rows(0.05)).episodes == 1200                 # the control
    other = [(c.case, derive_seed(N.salt("D"), config_hash(condition_to_cfg(c)), k)) for c in CALM for k in range(100)]
    bad = {
        "a row short": (rows(0.05)[:-1], rows(0.05)),
        "the PID a row short": (rows(0.05), rows(0.05)[1:]),
        "a row twice": (rows(0.05)[:-1] + rows(0.05)[:1], rows(0.05)),
        "the policy on Zone D's seeds": (rows(0.05, keys=other), rows(0.05)),
        "the PID on other seeds": (rows(0.05), rows(0.05, keys=other)),
        "one seed moved": (rows(0.05)[:-1] + [BarRow(KEYS[-1][0], KEYS[-1][1] + 1, False, 0.05)], rows(0.05)),
        "a case that is not one of the twelve": (rows(0.05)[:-1] + [BarRow("hover", KEYS[-1][1], False, 0.05)], rows(0.05)),
    }
    for name, (policy, pid) in bad.items():
        with pytest.raises(BarMismatch):
            read_bar(policy, pid)


def test_tr13_an_error_row_stops_the_read_and_is_never_a_crash_or_a_pass():
    errored = rows(0.05)
    errored[17] = BarRow(errored[17].case, errored[17].seed, False, float("nan"), error=True)
    with pytest.raises(BarMismatch):
        read_bar(errored, rows(0.05))
    with pytest.raises(BarMismatch):
        read_bar(rows(0.05), errored)
    assert read_bar(rows(0.05), rows(0.05)).passed is True                    # the control: the same rows with none errored


# ---- through Prove, as the bar is flown -----------------------------------------------------------------------------------
def prove(store, *args):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / "tests"), os.environ.get("PYTHONPATH", "")]))
    return subprocess.run([sys.executable, "-m", "poisson_prove", "--store", str(store), *args, "--plain"], env=env, cwd=ROOT,
                          capture_output=True, text=True)


_STORE = {}


@pytest.fixture(scope="module")
def tmp_root(tmp_path_factory):
    return tmp_path_factory.mktemp("bar")


def bar_store(root):
    """Five sweeps, made once and reused: the stand-in policy and PID under the bar's salt, the policy under Zone D's salt,
    a flyer that faults on some seeds, and a sweep of 4 replicates."""
    if "store" in _STORE:
        return _STORE["store"]
    text = bar_sweep_spec(target="training_prove_targets:bar_policy", target_id="drone-robustness-4a")
    assert text.strip(), "the study wrote no sweep spec for the bar"
    spec = root / "bar.toml"
    spec.write_text(text)
    short = root / "short.toml"
    short_text, swapped = re.subn(r"replicates\s*=\s*100", "replicates = 4", text)
    assert swapped == 1
    short.write_text(short_text)
    st = root / "store"
    for name, target, salt, file in (("pi0", "bar_policy", N.BAR_SALT, spec), ("pid", "bar_pid", N.BAR_SALT, spec),
                                     ("pi0-zone-d", "bar_policy", N.salt("D"), spec), ("faulty", "bar_faulty", N.BAR_SALT, spec),
                                     ("short", "bar_policy", N.BAR_SALT, short)):
        r = prove(st, "sweep", str(file), "--target", f"training_prove_targets:{target}", "--salt", salt, "--yes", "--max-mde", "1",
                  "--name", name)
        assert r.returncode == 0, f"sweep {name} failed:\n{r.stdout}\n{r.stderr}"
    _STORE["store"] = st
    return st


def test_tr13_a_stored_bar_run_reads_back_as_its_1200_rows(tmp_root):
    store = bar_store(tmp_root)
    got = rows_from_store(store, "pi0")
    assert len(got) == 1200 and all(isinstance(r, BarRow) for r in got)
    assert sorted((r.case, r.seed) for r in got) == sorted(KEYS)
    for r in got:
        crashed, rmse = T.policy_outcome(r.seed)
        assert (r.crashed, r.error) == (crashed, False) and r.rmse == pytest.approx(rmse, rel=1e-12)
    pid = rows_from_store(store, "pid")
    assert sorted((r.case, r.seed) for r in pid) == sorted(KEYS)              # the PID on the same seeds
    reading = read_bar(got, pid)
    want_crashes = sum(T.policy_outcome(seed)[0] for _, seed in KEYS)
    assert reading.crashes == want_crashes and reading.episodes == 1200
    assert reading.median_rmse == pytest.approx(float(np.median([T.policy_outcome(s)[1] for _, s in KEYS])), rel=1e-12)
    assert reading.pid_median_rmse == pytest.approx(float(np.median([T.pid_outcome(s)[1] for _, s in KEYS])), rel=1e-12)
    assert reading.passed is (want_crashes <= 5 and reading.median_rmse <= 0.10 and reading.median_rmse <= 1.25 * reading.pid_median_rmse)


def test_tr13_a_run_swept_under_another_salt_is_not_the_bar(tmp_root):
    store = bar_store(tmp_root)
    assert len(rows_from_store(store, "pi0")) == 1200                         # the control: the bar's salt
    with pytest.raises(BarMismatch):
        read_bar(rows_from_store(store, "pi0-zone-d"), rows_from_store(store, "pid"))


def test_tr13_a_run_with_fewer_replicates_is_not_the_bar(tmp_root):
    store = bar_store(tmp_root)
    assert len(rows_from_store(store, "pid")) == 1200
    with pytest.raises(BarMismatch):
        read_bar(rows_from_store(store, "short"), rows_from_store(store, "pid"))


def test_tr13_a_stored_error_row_reads_back_as_an_error_and_stops_the_bar(tmp_root):
    store = bar_store(tmp_root)
    assert len(rows_from_store(store, "pi0")) == 1200
    want_errors = sum(seed % 300 == 11 for _, seed in KEYS)
    faulty = rows_from_store(store, "faulty")
    assert want_errors > 0 and len(faulty) == 1200
    assert sum(r.error for r in faulty) == want_errors and not any(r.crashed for r in faulty if r.error)   # an error, never a crash
    with pytest.raises(BarMismatch):
        read_bar(faulty, rows_from_store(store, "pid"))
