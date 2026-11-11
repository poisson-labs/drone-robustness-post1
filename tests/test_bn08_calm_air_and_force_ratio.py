"""Rule 6c (spec 20.1, 17.3, 20.2; the yaw addendum of 2026-11-08, 4DR-5): N9, and the flat force ratio printed with every speed.

N9  Calm air, from the bar of section 2 and not from Zone D. For each pi0: crashes out of 1,200 with the Wilson 95% interval, the
    median RMSE, and that RMSE over the cascaded PID's on the same seeds. The PID's own median RMSE and crash count on those seeds. The
    geometric controller's calm RMSE from its gain selection, labelled a tuning figure. B0, and how many seeds were tried and how many
    failed the bar.  "N9 comes from the bar's code, frozen in 4A": the writer reads the stored readings and does not recompute them.
    The 2026-11-08 addendum adds (4DR-5): k_yaw, the crash penalty, the yaw gate's reading per lineage, and which gate each failed seed
    failed. The stored file is 4D-B's lineage file (`data/4d-lineages.json`, 4B's `bars.json` shape with `k_yaw`, `crash_penalty`,
    `yaw_readings` and `selection.failed_on`); the fixture here is a file in that shape built on 4B's bar readings.
"Every speed is printed with its flat force ratio under section 17.3's primary air model.\""""
import copy
import json
from pathlib import Path

import pytest

import boundary_numbers as B
import boundary_oracles as O
import spec_numbers as S
from drone_study.boundary import results as N
from drone_study.training import bar as bar_module

DATA = Path(__file__).resolve().parent / "data"
BARS = json.loads((DATA / "4d-lineages-shape.json").read_text())   # 4B's bars.json readings in 4D-B's lineage-file shape (scripts/make_4d_lineages_shape.py)
TUNING = json.loads((DATA / "4b-tuning-pick.json").read_text())  # 4B's raw/tuning-pick.json without its 75-point grid (314a6067... whole)


@pytest.fixture(autouse=True)
def no_recompute(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("N9 recomputed the bar from rows")
    monkeypatch.setattr(bar_module, "read_bar", refuse)
    monkeypatch.setattr(bar_module, "rows_from_store", refuse)
    monkeypatch.setattr(bar_module, "select_lineages", refuse)
    from drone_study.training import behaviour as behaviour_module
    from drone_study.training import yaw_weight as yaw_module
    monkeypatch.setattr(behaviour_module, "yaw_acceptance", refuse)
    monkeypatch.setattr(behaviour_module, "yaw_rows_from_store", refuse)
    monkeypatch.setattr(yaw_module, "select_lineages_with_yaw", refuse)


# ---- N9 on 4B's own stored readings ----------------------------------------------------------------------------------------
def test_bn08_n9_reads_the_stored_bar_for_the_16_lineages_verbatim():
    got = N.n9_calm(BARS, TUNING)
    assert got["b0"] == 5_000_000 and got["tried"] == 16 and list(got["failures"]) == []
    assert sorted(got["lineages"]) == list(range(1, 17))
    for seed in range(1, 17):
        assert got["lineages"][seed] == BARS["readings"][f"pi0-seed-{seed}"]
        assert got["lineages"][seed]["episodes"] == 1200 and got["lineages"][seed]["crashes"] == 0


def test_bn08_n9_carries_the_weight_the_penalty_and_the_yaw_gates_reading_per_lineage_verbatim():
    """Re-frozen by 4D-A (2026-11-08, 4DR-5). Before: N9 had no yaw and no weight."""
    got = N.n9_calm(BARS, TUNING)
    assert got["k_yaw"] == 0.008 == BARS["k_yaw"] and got["crash_penalty"] == 20.0 == BARS["crash_penalty"]
    assert sorted(got["yaw"]) == list(range(1, 17)) and dict(got["failed_on"]) == {}
    for seed in range(1, 17):
        assert got["yaw"][seed] == BARS["yaw_readings"][f"pi0-seed-{seed}"]
        assert got["yaw"][seed]["passed"] is True and got["yaw"][seed]["episodes"] == 1200
        assert {"yaw_rate_rms_median_deg_s", "drift_abs_median_deg_s", "past_45", "past_90", "max_heading_median_deg", "passed"} <= set(got["yaw"][seed])
    assert not [k for k in got["yaw"] if not isinstance(k, int)]


def test_bn08_n9_has_the_pids_own_readings_and_the_geometric_controllers_tuning_figure():
    got = N.n9_calm(BARS, TUNING)
    assert got["pid"] == {"median_rmse_m": 0.06227632689501371, "crashes": 0}
    assert got["geometric_tuning_figure_m"] == 0.04755207605147789
    assert got["geometric_label"] == "tuning figure"


def test_bn08_n9_does_not_carry_the_pilots_which_are_not_lineages():
    got = N.n9_calm(BARS, TUNING)
    assert not [k for k in got["lineages"] if not isinstance(k, int)] and len(got["lineages"]) == 16


# ---- never recomputed -------------------------------------------------------------------------------------------------------
def test_bn08_n9_prints_what_is_stored_even_when_it_is_not_what_the_rows_would_say():
    bars = copy.deepcopy(BARS)
    r = bars["readings"]["pi0-seed-3"]
    r.update(crashes=4, median_rmse_m=0.123456789, ratio=9.87654321, wilson_hi=0.5)           # internally inconsistent on purpose
    got = N.n9_calm(bars, TUNING)
    assert 3 in got["lineages"]
    assert got["lineages"][3] == r and got["lineages"][3]["ratio"] == 9.87654321 and got["lineages"][3]["wilson_hi"] == 0.5


def test_bn08_n9_reports_the_stored_selection_when_seeds_failed_the_bar_or_the_yaw_gate():
    """Re-frozen by 4D-A (2026-11-08, 4DR-5): a seed may fail the yaw gate, and N9 says which gate each failed seed failed."""
    bars = copy.deepcopy(BARS)
    for new in (17, 18):
        bars["readings"][f"pi0-seed-{new}"] = {**bars["readings"]["pi0-seed-16"], "median_rmse_m": 0.09}
        bars["yaw_readings"][f"pi0-seed-{new}"] = {**bars["yaw_readings"]["pi0-seed-16"], "yaw_rate_rms_median_deg_s": 7.5}
    bars["readings"]["pi0-seed-4"].update(crashes=9, passes_crashes=False, passed=False)
    bars["yaw_readings"]["pi0-seed-9"].update(yaw_rate_rms_median_deg_s=31.0, passes_rate=False, passed=False)
    bars["selection"] = {"lineages": [1, 2, 3] + list(range(5, 9)) + list(range(10, 19)), "failures": [4, 9], "tried": 18,
                         "failed_on": {"4": "bar", "9": "yaw"}}
    got = N.n9_calm(bars, TUNING)
    assert got["tried"] == 18 and list(got["failures"]) == [4, 9]
    assert dict(got["failed_on"]) == {4: "bar", 9: "yaw"}
    assert sorted(got["lineages"]) == [1, 2, 3] + list(range(5, 9)) + list(range(10, 19)) and len(got["lineages"]) == 16
    assert sorted(got["yaw"]) == sorted(got["lineages"])
    assert got["lineages"][17]["median_rmse_m"] == 0.09 and got["yaw"][18]["yaw_rate_rms_median_deg_s"] == 7.5
    assert 4 not in got["lineages"] and 9 not in got["yaw"]                                   # the failed seeds are not lineages


# ---- the flat force ratio ---------------------------------------------------------------------------------------------------
def test_bn08_the_flat_force_ratio_is_the_primary_air_models_force_over_the_lateral_limit_at_every_speed():
    for v in list(B.SPEEDS) + [16.0, 2.5, 9.0, 11.0]:
        assert N.flat_force_ratio(v) == pytest.approx(O.flat_force_ratio(v), rel=1e-9), v


def test_bn08_it_reproduces_17_3s_table_to_its_printed_digits():
    for v, newtons, share in zip(S.TABLE_SPEEDS, S.TABLE_PRIMARY_N, S.TABLE_PRIMARY_SHARE_PCT):
        got = N.flat_force_ratio(float(v))
        assert abs(got * S.LATERAL_LIMIT_N - newtons) <= S.TABLE_TOL_N, v
        assert abs(got * 100 - share) <= S.TABLE_TOL_PCT, v


def test_bn08_it_is_zero_at_rest_increasing_and_the_extensions_16_ms_is_95_percent():
    assert N.flat_force_ratio(0.0) == 0.0
    vals = [N.flat_force_ratio(v) for v in B.SPEEDS_EXTENDED]
    assert all(b > a for a, b in zip(vals, vals[1:]))
    assert vals[-1] == pytest.approx(0.95, abs=0.005)                          # section 3: "its flat force ratio is 0.95"
    assert N.flat_force_ratio(12.0) == pytest.approx(0.565, abs=0.001)
