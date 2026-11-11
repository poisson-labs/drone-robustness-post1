"""Rule 6b (spec 20.1, 20.2): N6, N7 and N8.

N6  Paired counts, cell by cell, for each pi0 with each controller on their shared seeds: both crash, only pi0 crashes, only the
    controller crashes, neither. Counts only: no flag, word or p-value from Prove's `diff`.
N7  The anchor slice, for each flyer: crashes out of 16 in each of its 16 cells and the mean over them, beside the same cells' rates
    in N1. For pi0, the mean of the 16 lineages and all 16 values.
N8  S6's re-read of N2 and N3 under each of its crash limits, for each flyer. No flight."""
import subprocess

import numpy as np
import pandas as pd
import pytest

import boundary_numbers as B
import boundary_oracles as O
import boundary_rows as R
from drone_study.analysis import estimator as E
from drone_study.analysis import sensitivities as S
from drone_study.boundary import results as N


# ---- N6 -------------------------------------------------------------------------------------------------------------------
def paired(pi0_counts, ctl_counts):
    rows = R.frame(R.zone_b_rows("pi0-seed-1", pi0_counts) + R.zone_b_rows("pid", ctl_counts))
    return rows, N.n6_paired_counts(rows, "pi0-seed-1", "pid")


def test_bn07_n6_counts_both_only_one_only_the_other_and_neither_in_every_cell():
    rng = np.random.default_rng(6)
    a, b = rng.integers(0, 25, size=R.SHAPE), rng.integers(0, 25, size=R.SHAPE)
    _, got = paired(a, b)
    assert len(got) == 320
    order = [(az, el, v, ph) for az, el in B.DIRECTIONS for v in B.SPEEDS for ph in B.PHASES]
    assert [(g["az"], g["el"], g["speed"], g["phase"]) for g in got] == order
    for i, g in enumerate(got):
        d, s, t = i // 20, (i // 2) % 10, i % 2
        both = min(a[d, s, t], b[d, s, t])                                         # crashes are on each flyer's first k replicates: nested
        assert g["both"] == both
        assert g["only_pi0"] == a[d, s, t] - both and g["only_controller"] == b[d, s, t] - both
        assert g["neither"] == 24 - a[d, s, t] - b[d, s, t] + both
        assert g["both"] + g["only_pi0"] + g["only_controller"] + g["neither"] == 24


def test_bn07_n6_is_counts_only_nothing_of_provs_diff_is_in_it(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("N6 ran a subprocess")
    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setattr(subprocess, "Popen", refuse)
    _, got = paired(np.full(R.SHAPE, 5), np.full(R.SHAPE, 9))
    assert set(got[0]) == {"az", "el", "speed", "phase", "both", "only_pi0", "only_controller", "neither"}


def test_bn07_n6_pairs_on_the_seeds_not_on_the_replicate_order_and_drops_a_pair_with_an_error_on_either_side():
    a, b = np.full(R.SHAPE, 8), np.full(R.SHAPE, 4)
    rows, _ = paired(a, b)
    first = (0.0, 0.0, 2.0, 0.0)
    keep = rows["cell"] == first
    pi0_rows = rows[keep & (rows["flyer"] == "pi0-seed-1")].sort_values("seed")
    ctl_rows = rows[keep & (rows["flyer"] == "pid")].sort_values("seed")
    bad_pi0, bad_ctl = pi0_rows["seed"].iloc[0], ctl_rows["seed"].iloc[20]                      # pi0 replicate 0 (a crash); pid replicate 20 (ok)
    rows = rows.copy()
    rows.loc[(rows["flyer"] == "pi0-seed-1") & (rows["cell"] == first) & (rows["seed"] == bad_pi0), ["outcome", "error", "crash_class"]] = ["error", True, None]
    rows.loc[(rows["flyer"] == "pid") & (rows["cell"] == first) & (rows["seed"] == bad_ctl), ["outcome", "error", "crash_class"]] = ["error", True, None]
    rows = rows.sample(frac=1, random_state=0).reset_index(drop=True)                          # order in the table means nothing
    every = N.n6_paired_counts(rows, "pi0-seed-1", "pid")
    assert len(every) == 320
    got = every[0]
    # 22 pairs left. pi0 crashes on replicates 0-7 (replicate 0 gone), pid on 0-3: both = 3 (replicates 1-3), only pi0 = 4 (4-7), none else.
    assert (got["both"], got["only_pi0"], got["only_controller"], got["neither"]) == (3, 4, 0, 15)


def test_bn07_n6_refuses_two_flyers_whose_seeds_differ_on_a_cell():
    a = R.zone_b_rows("pi0-seed-1", np.full(R.SHAPE, 5))
    b = [(f, z, c, s + 1, *rest) for (f, z, c, s, *rest) in R.zone_b_rows("pid", np.full(R.SHAPE, 5))]
    with pytest.raises(N.PairingError):
        N.n6_paired_counts(R.frame(a + b), "pi0-seed-1", "pid")


# ---- N7 -------------------------------------------------------------------------------------------------------------------
ANCHOR_SPEED = [B.ANCHOR[c] for c in R.anchor_cells()]


def zone_b_horizon_rows(flyer, rate_fn):
    c = np.zeros((8, 10, 2), int)
    for d in range(8):
        for s in range(10):
            for t in range(2):
                c[d, s, t] = rate_fn(d, s, t)
    return R.zone_b_rows(flyer, c, directions=B.HORIZON)


def anchor_fixture():
    flyers = [f"pi0-seed-{i}" for i in range(1, 17)] + ["pid"]
    rows = []
    for i, f in enumerate(flyers):
        rows += R.anchor_rows(f, [(3 * i + j) % 17 for j in range(16)])
        rows += zone_b_horizon_rows(f, lambda d, s, t, i=i: (i + d + 2 * s + t) % 25)
    return flyers, R.frame(rows)


def test_bn07_n7_gives_each_flyers_16_cells_out_of_16_and_their_mean_beside_the_same_cells_on_the_grid():
    flyers, rows = anchor_fixture()
    got = N.n7_anchor(rows, flyers, flyers[:16])
    assert list(got["flyers"]) == flyers
    for i, f in enumerate(flyers):
        cells = got["flyers"][f]["cells"]
        assert len(cells) == 16
        rates = []
        for j, ((az, ph), cell) in enumerate(zip(R.anchor_cells(), cells)):
            c = (3 * i + j) % 17
            speed = B.ANCHOR[(az, ph)]
            d, s, t = B.HORIZON_AZ.index(az), B.SPEEDS.index(speed), B.PHASES.index(ph)
            gc = (i + d + 2 * s + t) % 25
            assert (cell["az"], cell["phase"], cell["speed"]) == (az, ph, speed)
            assert (cell["crashes"], cell["n"]) == (c, 16) and cell["rate"] == pytest.approx(c / 16)
            assert (cell["grid_crashes"], cell["grid_n"]) == (gc, 24) and cell["grid_rate"] == pytest.approx(gc / 24)
            rates.append(c / 16)
        assert got["flyers"][f]["mean"] == pytest.approx(np.mean(rates), abs=1e-12)


def test_bn07_n7_for_the_pi0_is_the_mean_of_the_16_lineages_and_all_16_values():
    flyers, rows = anchor_fixture()
    summary = N.n7_anchor(rows, flyers, flyers[:16])
    assert "pi0" in summary
    got = summary["pi0"]
    per_lineage = np.array([[((3 * i + j) % 17) / 16 for j in range(16)] for i in range(16)])      # (lineage, cell)
    assert np.asarray(got["lineage_means"]) == pytest.approx(per_lineage.mean(1), abs=1e-12)
    assert np.asarray(got["cell_means"]) == pytest.approx(per_lineage.mean(0), abs=1e-12)
    assert got["mean"] == pytest.approx(per_lineage.mean(), abs=1e-12)
    assert len(got["lineage_means"]) == 16 and len(got["cell_means"]) == 16


def test_bn07_n7_scores_an_anchor_cell_on_scored_rows_only_and_stops_on_a_missing_cell():
    rows = R.frame(R.anchor_rows("pid", [4] * 16, n=14, errors=[2] * 16) + zone_b_horizon_rows("pid", lambda d, s, t: 6))
    cells = N.n7_anchor(rows, ["pid"])["flyers"]["pid"]["cells"]
    assert len(cells) == 16
    got = cells[0]
    assert (got["crashes"], got["n"]) == (4, 14) and got["rate"] == pytest.approx(4 / 14)
    assert "pi0" not in N.n7_anchor(rows, ["pid"])
    gone = rows[~((rows["zone"] == "anchor") & (rows["cell"] == (45.0, 0.0)))]
    with pytest.raises(N.MissingCell):
        N.n7_anchor(gone, ["pid"])


# ---- N8 -------------------------------------------------------------------------------------------------------------------
NAMES = list(B.S6_NAMES)


class FakeLimits(dict):
    pass


def fake_rescore(rows, name):
    """Stand-in for S6's re-score: a crash stays a crash under limit `name` when `name` is in the row's stored `keep` list."""
    out = rows.copy()
    mask = (out["outcome"] != "error")
    keep = [(x is not None and name in x["keep"]) for x in out["crossings"]]
    crash = [bool(m and k and o == "crash") for m, k, o in zip(mask, keep, out["outcome"])]
    out["outcome"] = np.where(~mask, "error", np.where(crash, "crash", "ok"))
    return out


def limits_rows(flyer, base, speeds=B.SPEEDS):
    """Zone B rows whose k-th crash in a cell is kept under the first (k + d + s) % 7 of the six limits' names."""
    def crossings(d, s, t, k):
        return {"keep": NAMES[: (k + d + s) % 7]}
    return R.frame(R.zone_b_rows(flyer, base, speeds=speeds, crossings=crossings))


def test_bn07_n8_re_reads_n2_and_n3_under_each_of_the_six_limits_from_the_rescored_rows(monkeypatch):
    monkeypatch.setattr(S, "S6_LIMITS", {n: object() for n in NAMES})
    monkeypatch.setattr(S, "s6_rescore", fake_rescore)
    rng = np.random.default_rng(8)
    base = np.sort(rng.integers(0, 25, size=(16, 10)), axis=1)
    c = np.stack([base, base], axis=-1)
    rows = limits_rows("f", c)
    got = N.n8_limits(rows, "f")
    assert list(got) == NAMES
    for name in NAMES:
        re = fake_rescore(rows, name)
        g = N.flyer_grid(re, "f")
        edges = N.n2_edges(g)
        for d in range(16):
            pc, pn = g.crashes[d].sum(-1), g.n[d].sum(-1)
            for lvl, e, by in ((0.5, got[name]["v50"][d], got[name]["v50_by_timing"][d]), (0.1, got[name]["v10"][d], got[name]["v10_by_timing"][d])):
                v, m = O.edge_exact(pc, pn, B.SPEEDS, lvl)
                assert (e.value, e.censored) == (pytest.approx(v, abs=1e-12), m), (name, d, lvl)
                for t in range(2):
                    vt, mt = O.edge_exact(g.crashes[d, :, t], g.n[d, :, t], B.SPEEDS, lvl)
                    assert (by[t].value, by[t].censored) == (pytest.approx(vt, abs=1e-12), mt)
            assert got[name]["integrated"][d] == pytest.approx(O.integrated(pc, pn, B.SPEEDS), abs=1e-12)
        assert got[name]["v50"] == [e.v50 for e in edges]


def test_bn07_the_limits_change_the_answer_so_n8_is_not_the_primary_read_six_times(monkeypatch):
    monkeypatch.setattr(S, "S6_LIMITS", {n: object() for n in NAMES})
    monkeypatch.setattr(S, "s6_rescore", fake_rescore)
    c = np.stack([R.per_speed([0, 0, 0, 24, 24, 24, 24, 24, 24, 24])[:, :, 0]] * 2, axis=-1)[:16]
    rows = limits_rows("f", c)
    got = N.n8_limits(rows, "f")
    assert list(got) == NAMES
    primary = N.n2_edges(N.flyer_grid(rows, "f"))
    assert any(got[n]["v50"][0].value != primary[0].v50.value or got[n]["v50"][0].censored != primary[0].v50.censored for n in NAMES)
    assert got[NAMES[0]]["v50"][0] != got[NAMES[-1]]["v50"][0] or got[NAMES[0]]["integrated"][0] != got[NAMES[-1]]["integrated"][0]


def test_bn07_n8_takes_a_16_top(monkeypatch):
    monkeypatch.setattr(S, "S6_LIMITS", {n: object() for n in NAMES})
    monkeypatch.setattr(S, "s6_rescore", fake_rescore)
    c = np.zeros((16, 11, 2), int)
    c[:, 8:, :] = 24
    rows = limits_rows("f", c, speeds=B.SPEEDS_EXTENDED)
    got = N.n8_limits(rows, "f", speeds=B.SPEEDS_EXTENDED)
    assert list(got) == NAMES
    assert all(len(got[n]["integrated"]) == 16 for n in NAMES)
    assert all(e.value <= 16.0 for n in NAMES for e in got[n]["v50"])
