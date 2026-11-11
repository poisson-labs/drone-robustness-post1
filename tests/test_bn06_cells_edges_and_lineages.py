"""Rule 6a (spec 20.1, 20.2): N1, N2, N4 and N5 as functions from stored rows.

N1  The crash rate in every Zone B cell, for each flyer: crashes out of 24, split by crash class, with Prove's own interval
    for the cell. 320 cells.
N2  v50 and v10 per direction, for each flyer, by section 9's estimator, pooled over the two timings; the same fit on each timing
    alone; censored directions marked.
N4  Across the 16 pi0, per direction: the mean of N2 and of N3, all 16 values, and how many are censored.
N5  pi0 beside each controller, per direction: each lineage's v50 minus the controller's, the mean of those 16, the lowest and the
    highest, with pairs where either is censored marked. The same for the integrated rate. All 16 differences share one
    controller's estimate, and the post says so.
Error rows are in no count (8A-10, ruling 5A-6); a cell with no scored row stops the read."""
import numpy as np
import pandas as pd
import pytest
from poisson_prove import stats as prove_stats

import boundary_numbers as B
import boundary_oracles as O
import boundary_rows as R
from drone_study.analysis import boundary as analysis_boundary
from drone_study.analysis import estimator as E
from drone_study.boundary import results as N

N5_NOTE = "All 16 differences share one controller's estimate, so its error moves them together."


def grid_of(flyer, crashes, n=24, **kw):
    return N.flyer_grid(R.frame(R.zone_b_rows(flyer, crashes, n, **kw)), flyer)


# ---- the flyer's grid -----------------------------------------------------------------------------------------------------
def test_bn06_a_flyers_grid_holds_the_counts_by_direction_speed_and_timing_and_scored_rows_only():
    rng = np.random.default_rng(11)
    c = rng.integers(0, 25, size=R.SHAPE)
    errors = (rng.integers(0, 3, size=R.SHAPE))
    rows = R.frame(R.zone_b_rows("f", c, 24, errors=errors))
    g = N.flyer_grid(rows, "f")
    assert g.flyer == "f" and g.speeds == B.SPEEDS and tuple(g.directions) == B.DIRECTIONS
    assert (g.crashes == c).all() and (g.n == 24).all()                         # error rows are in neither count
    for k, cls in enumerate(B.CRASH_CLASSES):
        want = np.array([[[R.class_counts(c[d, s, t], d)[k] for t in range(2)] for s in range(10)] for d in range(16)])
        assert (np.asarray(g.classes[cls]) == want).all()


def test_bn06_the_grid_reads_only_its_own_flyer_and_only_zone_b_rows():
    c = np.full(R.SHAPE, 3)
    other = R.zone_b_rows("g", np.full(R.SHAPE, 20))
    anchor = R.anchor_rows("f", [5] * 16)
    rows = R.frame(R.zone_b_rows("f", c) + other + anchor)
    g = N.flyer_grid(rows, "f")
    assert (g.crashes == 3).all() and (g.n == 24).all()


def test_bn06_a_cell_with_no_scored_row_stops_the_read():
    c = np.full(R.SHAPE, 3)
    n = np.full(R.SHAPE, 24)
    n[2, 4, 1] = 0
    errors = np.zeros(R.SHAPE, int)
    errors[2, 4, 1] = 5
    with pytest.raises(N.MissingCell):
        N.flyer_grid(R.frame(R.zone_b_rows("f", c, n, errors=errors)), "f")
    missing = R.zone_b_rows("f", c)
    missing = [r for r in missing if r[2] != (90.0, 0.0, 6.0, 0.125)]
    with pytest.raises(N.MissingCell):
        N.flyer_grid(R.frame(missing), "f")


# ---- N1 -------------------------------------------------------------------------------------------------------------------
def n1_for(flyers, crashes_by_flyer, **kw):
    rows = R.frame(sum((R.zone_b_rows(f, crashes_by_flyer[f], **kw) for f in flyers), []))
    return N.n1_cells(rows, flyers)


def test_bn06_n1_is_one_line_per_flyer_and_cell_320_each_in_the_order_given():
    rng = np.random.default_rng(1)
    cs = {"b": rng.integers(0, 25, size=R.SHAPE), "a": rng.integers(0, 25, size=R.SHAPE)}
    got = n1_for(["b", "a"], cs)
    assert len(got) == 640
    assert [g["flyer"] for g in got] == ["b"] * 320 + ["a"] * 320
    order = [(az, el, v, ph) for az, el in B.DIRECTIONS for v in B.SPEEDS for ph in B.PHASES]
    assert [(g["az"], g["el"], g["speed"], g["phase"]) for g in got[:320]] == order
    assert [(g["az"], g["el"], g["speed"], g["phase"]) for g in got[320:]] == order


def test_bn06_n1_gives_crashes_out_of_n_split_by_class_and_the_flat_force_ratio():
    rng = np.random.default_rng(2)
    c = rng.integers(0, 25, size=R.SHAPE)
    got = n1_for(["f"], {"f": c})
    assert len(got) == 320
    for i, g in enumerate(got):
        d, s, t = i // 20, (i // 2) % 10, i % 2
        assert (g["crashes"], g["n"]) == (c[d, s, t], 24)
        assert (g["ground"], g["tilt"], g["radius"]) == R.class_counts(c[d, s, t], d)
        assert g["ground"] + g["tilt"] + g["radius"] == g["crashes"]
        assert g["value"] == pytest.approx(c[d, s, t] / 24, abs=1e-15)
        assert g["force_ratio"] == pytest.approx(O.flat_force_ratio(B.SPEEDS[s]), rel=1e-9)


def test_bn06_n1s_interval_is_provs_own_for_the_cell_wilson_95():
    c = np.zeros(R.SHAPE, int)
    c[0, 0, 0], c[0, 1, 0], c[0, 2, 0], c[0, 3, 0] = 0, 1, 12, 24
    got = n1_for(["f"], {"f": c})
    assert len(got) == 320
    for g in got[:8:2]:
        want = prove_stats.share(int(g["crashes"]), 24)
        assert (g["lo"], g["hi"]) == pytest.approx((want["lo"], want["hi"]), abs=1e-15)
        assert g["level"] == want["level"] == 0.95 and g["method"] == want["method"] == "wilson"
    assert got[0]["lo"] == pytest.approx(0.0, abs=1e-12) and got[6]["hi"] == pytest.approx(1.0, abs=1e-12)


def test_bn06_n1_counts_scored_rows_so_an_error_row_changes_the_denominator_not_the_crashes():
    c = np.full(R.SHAPE, 6)
    errors = np.zeros(R.SHAPE, int)
    errors[0, 0, 0] = 2
    n = np.full(R.SHAPE, 22)
    got = n1_for(["f"], {"f": c}, n=n, errors=errors)
    assert len(got) == 320
    assert (got[0]["crashes"], got[0]["n"]) == (6, 22) and got[0]["value"] == pytest.approx(6 / 22)
    assert got[0]["lo"] == pytest.approx(prove_stats.share(6, 22)["lo"])


def test_bn06_n1_stops_on_a_missing_cell():
    rows = [r for r in R.zone_b_rows("f", np.zeros(R.SHAPE, int)) if r[2] != (0.0, 0.0, 14.0, 0.0)]
    with pytest.raises(N.MissingCell):
        N.n1_cells(R.frame(rows), ["f"])


# ---- N2 -------------------------------------------------------------------------------------------------------------------
THRESHOLDS = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 5, 5, 5, 5, 5])      # a step at speed index t in each direction


def test_bn06_n2_on_step_curves_is_the_midpoint_between_the_speeds_either_side_of_the_step():
    g = grid_of("f", R.steps(THRESHOLDS))
    got = N.n2_edges(g)
    assert len(got) == 16
    s = B.SPEEDS
    for d, t in enumerate(THRESHOLDS):
        e = got[d]
        if t == 0:
            assert (e.v50.value, e.v50.censored, e.v10.value, e.v10.censored) == (2.0, "bottom", 2.0, "bottom")
        elif t == 10:
            assert (e.v50.value, e.v50.censored, e.v10.value, e.v10.censored) == (14.0, "top", 14.0, "top")
        else:
            assert e.v50.value == pytest.approx((s[t - 1] + s[t]) / 2) and e.v50.censored is None
            assert e.v10.value == pytest.approx(s[t - 1] + 0.1 * (s[t] - s[t - 1])) and e.v10.censored is None
            for k in range(2):
                assert e.v50_by_timing[k].value == pytest.approx((s[t - 1] + s[t]) / 2)


def test_bn06_n2_is_section_9s_estimator_on_the_pooled_counts_and_on_each_timing_alone():
    rng = np.random.default_rng(5)
    base = np.sort(rng.integers(0, 25, size=(16, 10)), axis=1)
    c = np.stack([base, np.clip(base + rng.integers(-3, 4, size=(16, 10)), 0, 24)], axis=-1)
    got = N.n2_edges(grid_of("f", c))
    for d in range(16):
        for level, pooled, by_timing in ((0.5, got[d].v50, got[d].v50_by_timing), (0.1, got[d].v10, got[d].v10_by_timing)):
            v, mark = O.edge_exact(c[d].sum(-1), np.full(10, 48), B.SPEEDS, level)
            assert (pooled.value, pooled.censored) == (pytest.approx(v, abs=1e-12), mark)
            assert pooled.n_per_speed == (48,) * 10
            for t in range(2):
                vt, mt = O.edge_exact(c[d, :, t], np.full(10, 24), B.SPEEDS, level)
                assert (by_timing[t].value, by_timing[t].censored) == (pytest.approx(vt, abs=1e-12), mt)
                assert by_timing[t].n_per_speed == (24,) * 10


def test_bn06_n2_reads_its_edges_through_the_one_estimator(monkeypatch):
    seen = []
    sentinel = E.DirectionEdges(E.Edge(1.5, None), E.Edge(0.5, "top"), (E.Edge(2.5, None),) * 2, (E.Edge(3.5, None),) * 2)

    def fake(crashes, n, speeds):
        seen.append((np.asarray(crashes).shape, tuple(speeds)))
        return sentinel

    monkeypatch.setattr(E, "direction_edges", fake)
    got = N.n2_edges(grid_of("f", R.steps(THRESHOLDS)))
    assert got == [sentinel] * 16 and seen == [((10, 2), B.SPEEDS)] * 16


# ---- N4 -------------------------------------------------------------------------------------------------------------------
def pi0_grids(thresholds_of_lineage):
    return [grid_of(f"pi0-seed-{i + 1}", R.steps(thresholds_of_lineage(i))) for i in range(16)]


def test_bn06_n4_is_the_mean_of_the_16_values_all_16_values_and_the_censored_counts():
    def thr(i):
        t = THRESHOLDS.copy()
        t[0], t[1] = (0 if i < 5 else 3), (10 if i < 3 else 4)                      # direction 0: 5 censored at the bottom; direction 1: 3 at the top
        t[2:] = np.clip(THRESHOLDS[2:] + (i % 3) - 1, 1, 9)
        return t
    grids = pi0_grids(thr)
    got = N.n4_across_lineages(grids)
    assert tuple(got["directions"]) == B.DIRECTIONS
    v50 = np.asarray(got["v50"]["values"])
    assert v50.shape == (16, 16)
    want = np.array([[E.v50(g.crashes[d].sum(-1), g.n[d].sum(-1), B.SPEEDS).value for g in grids] for d in range(16)])
    assert v50 == pytest.approx(want, abs=1e-12)
    assert np.asarray(got["v50"]["mean"]) == pytest.approx(want.mean(1), abs=1e-12)                 # censored values stand as their stand-ins
    assert tuple(got["v50"]["censored_bottom"][:2]) == (5, 0) and tuple(got["v50"]["censored_top"][:2]) == (0, 3)
    assert sum(got["v50"]["censored_top"][2:]) == 0 and sum(got["v50"]["censored_bottom"][2:]) == 0
    v10 = np.asarray(got["v10"]["values"])
    want10 = np.array([[E.v10(g.crashes[d].sum(-1), g.n[d].sum(-1), B.SPEEDS).value for g in grids] for d in range(16)])
    assert v10 == pytest.approx(want10, abs=1e-12) and np.asarray(got["v10"]["mean"]) == pytest.approx(want10.mean(1), abs=1e-12)
    assert tuple(got["v10"]["censored_bottom"][:2]) == (5, 0) and tuple(got["v10"]["censored_top"][:2]) == (0, 3)


def test_bn06_n4_carries_the_integrated_rate_per_lineage_and_its_mean():
    grids = pi0_grids(lambda i: np.clip(THRESHOLDS + (i % 4) - 1, 0, 10))
    got = N.n4_across_lineages(grids)
    vals = np.asarray(got["integrated"]["values"])
    want = np.array([[O.integrated(g.crashes[d].sum(-1), g.n[d].sum(-1), B.SPEEDS) for g in grids] for d in range(16)])
    assert vals.shape == (16, 16) and vals == pytest.approx(want, abs=1e-12)
    assert np.asarray(got["integrated"]["mean"]) == pytest.approx(want.mean(1), abs=1e-12)
    assert "censored_top" not in got["integrated"]                                  # nothing censors it


# ---- N5 -------------------------------------------------------------------------------------------------------------------
def test_bn06_n5_each_lineages_value_minus_the_controllers_with_the_mean_lowest_highest_and_a_mask():
    def thr(i):
        t = np.clip(THRESHOLDS + (i % 3) - 1, 1, 9)
        t[0] = 0 if i % 2 == 0 else 4                                                # direction 0: π0 censored at the bottom on even lineages
        return t
    grids = pi0_grids(thr)
    ctl_thr = np.full(16, 5)
    ctl_thr[3] = 10                                                                  # direction 3: the controller never crashes: censored at the top
    controller = grid_of("pid", R.steps(ctl_thr))
    got = N.n5_against_controller(grids, controller)
    diff = np.asarray(got["v50"]["differences"])
    assert diff.shape == (16, 16)
    cv = [E.v50(controller.crashes[d].sum(-1), controller.n[d].sum(-1), B.SPEEDS) for d in range(16)]
    want = np.array([[E.v50(g.crashes[d].sum(-1), g.n[d].sum(-1), B.SPEEDS).value - cv[d].value for g in grids] for d in range(16)])
    assert diff == pytest.approx(want, abs=1e-12)
    assert np.asarray(got["v50"]["mean"]) == pytest.approx(want.mean(1), abs=1e-12)
    assert np.asarray(got["v50"]["lowest"]) == pytest.approx(want.min(1), abs=1e-12)
    assert np.asarray(got["v50"]["highest"]) == pytest.approx(want.max(1), abs=1e-12)
    mask = np.asarray(got["v50"]["censored_pair"])
    assert mask.shape == (16, 16) and mask.dtype == bool
    pi0_cens = np.array([[E.v50(g.crashes[d].sum(-1), g.n[d].sum(-1), B.SPEEDS).censored is not None for g in grids] for d in range(16)])
    want_mask = pi0_cens | np.array([[c.censored is not None for _ in grids] for c in cv])
    assert (mask == want_mask).all()
    assert mask[3].all() and mask[0, ::2].all() and not mask[0, 1::2].any()


def test_bn06_n5_all_16_differences_share_the_controllers_one_estimate_and_the_note_says_so():
    grids = pi0_grids(lambda i: np.clip(THRESHOLDS + (i % 3) - 1, 1, 9))
    controller = grid_of("geometric", R.steps(np.full(16, 6)))
    got = N.n5_against_controller(grids, controller)
    pi0_values = np.asarray(N.n4_across_lineages(grids)["v50"]["values"])
    implied = pi0_values - np.asarray(got["v50"]["differences"])                    # the π0 value minus the difference is the controller's value
    assert np.allclose(implied, implied[:, :1], atol=1e-12)                          # one number per direction, the same for all 16
    assert implied[:, 0] == pytest.approx([E.v50(controller.crashes[d].sum(-1), controller.n[d].sum(-1), B.SPEEDS).value for d in range(16)])
    assert got["note"] == N5_NOTE and N.N5_NOTE == N5_NOTE


def test_bn06_n5_does_the_same_for_the_integrated_rate_without_a_mask():
    grids = pi0_grids(lambda i: np.clip(THRESHOLDS + (i % 3) - 1, 1, 9))
    controller = grid_of("oracle", R.steps(np.full(16, 7)))
    got = N.n5_against_controller(grids, controller)["integrated"]
    cv = [O.integrated(controller.crashes[d].sum(-1), controller.n[d].sum(-1), B.SPEEDS) for d in range(16)]
    want = np.array([[O.integrated(g.crashes[d].sum(-1), g.n[d].sum(-1), B.SPEEDS) - cv[d] for g in grids] for d in range(16)])
    assert np.asarray(got["differences"]) == pytest.approx(want, abs=1e-12)
    assert np.asarray(got["mean"]) == pytest.approx(want.mean(1), abs=1e-12)
    assert np.asarray(got["lowest"]) == pytest.approx(want.min(1), abs=1e-12) and np.asarray(got["highest"]) == pytest.approx(want.max(1), abs=1e-12)
    assert "censored_pair" not in got


def test_bn06_n5_reads_the_estimator_and_the_integrated_rate_through_their_one_function(monkeypatch):
    monkeypatch.setattr(analysis_boundary, "integrated_rate", lambda c, n, s: 0.25)
    monkeypatch.setattr(E, "direction_edges", lambda c, n, s: E.DirectionEdges(E.Edge(7.0, None), E.Edge(5.0, None)))
    grids = pi0_grids(lambda i: THRESHOLDS)
    controller = grid_of("pid", R.steps(np.full(16, 5)))
    got = N.n5_against_controller(grids, controller)
    assert np.asarray(got["v50"]["differences"]) == pytest.approx(np.zeros((16, 16)))
    assert np.asarray(got["integrated"]["differences"]) == pytest.approx(np.zeros((16, 16)))
