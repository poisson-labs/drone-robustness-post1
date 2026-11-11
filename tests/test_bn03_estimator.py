"""Rule 3 (spec section 9, 20.3 point 1): the estimator for one policy and one direction, against closed-form curves.

"Pool crash counts across the two timings at each of the 10 speeds. Fit a monotone curve by pooled adjacent violators.
v50 is where it crosses 0.5, by straight-line interpolation. Same for v10, the safe-side edge. ... If the curve never
reaches 0.5, v50 is the grid's top (14 m/s, or 16 m/s after an extension), marked as censored. If it starts above 0.5, v50
is 2 m/s, marked likewise." N2 asks for "the same fit on each timing alone, at 24 episodes a speed". Section 20.3, point 1:
"pin whatever section 9 leaves to code for one policy, such as a fitted curve that sits at exactly 0.5 over a stretch."

The rules picked where section 9 is silent (OWED 5A-1 to 5A-3): a stretch at exactly the level crosses at its lowest speed and
is not censored; "reaches" and "above" are read on the counts in exact arithmetic; a speed with no scored episodes is left out."""
import numpy as np
import pytest

import boundary_numbers as B
import boundary_oracles as O
from drone_study.analysis import estimator as E

SPEEDS = B.SPEEDS
EXT = B.SPEEDS_EXTENDED
N48 = [48] * 10


# ---- pooling over the two timings ------------------------------------------------------------------------------------------
def test_bn03_pooling_sums_counts_over_the_timings_and_never_averages_rates():
    c = np.array([[3, 5], [10, 2], [0, 24]])           # (speeds, timings)
    n = np.array([[24, 24], [24, 12], [10, 24]])
    pc, pn = E.pool_timings(c, n)
    assert pc.tolist() == [8, 12, 24] and pn.tolist() == [48, 36, 34]
    assert pc[1] / pn[1] == pytest.approx(12 / 36)
    assert np.mean(c[1] / n[1]) == pytest.approx((10 / 24 + 2 / 12) / 2)       # a mean of the two rates is another number


def test_bn03_pooling_works_on_any_leading_shape_and_the_timings_axis_is_the_last():
    rng = np.random.default_rng(1)
    c = rng.integers(0, 24, size=(16, 10, 2))
    pc, pn = E.pool_timings(c, np.full(c.shape, 24))
    assert pc.shape == (16, 10) and (pc == c.sum(-1)).all() and (pn == 48).all()
    pc0, _ = E.pool_timings(np.moveaxis(c, -1, 0), np.full((2, 16, 10), 24), axis=0)
    assert (pc0 == c.sum(-1)).all()


# ---- pooled adjacent violators ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("crashes,n,want", [
    ([5, 2], [10, 10], [0.35, 0.35]),
    ([15, 2], [30, 10], [17 / 40, 17 / 40]),                                  # weights are the episodes: (15 + 2) / 40
    ([0, 3, 6, 9], [10, 10, 10, 10], [0, 0.3, 0.6, 0.9]),                    # already monotone: unchanged
    ([8, 6, 2, 9], [10, 10, 10, 10], [16 / 30] * 3 + [0.9]),                  # 0.8, 0.6, 0.2 pool to their mean, then 0.9 stands
    ([5, 1, 9, 2, 3], [10, 10, 10, 10, 10], [0.3, 0.3, 0.466666666666667, 0.466666666666667, 0.466666666666667]),
    ([10, 0], [10, 10], [0.5, 0.5]),
    ([4, 4, 4], [8, 8, 8], [0.5, 0.5, 0.5]),
])
def test_bn03_the_monotone_fit_is_pooled_adjacent_violators_with_episode_weights(crashes, n, want):
    got = E.monotone_fit(crashes, n)
    assert np.allclose(got, want, atol=1e-12)
    assert (np.diff(got) >= -1e-15).all()


def test_bn03_the_fit_conserves_crashes_in_every_pooled_block_and_matches_the_exact_fit():
    rng = np.random.default_rng(20261104)
    for _ in range(200):
        n = rng.choice([12, 24, 36, 48], size=10)
        c = np.array([rng.integers(0, m + 1) for m in n])
        got = E.monotone_fit(c, n)
        want = np.array([float(f) for f in O.pav_exact(c, n)])
        assert np.allclose(got, want, atol=1e-12)
        assert float((got * n).sum()) == pytest.approx(float(c.sum()), abs=1e-9)


# ---- v50 and v10 by straight-line interpolation ---------------------------------------------------------------------------
def test_bn03_a_ramp_that_hits_one_half_on_a_grid_speed_has_v50_there_and_v10_by_interpolation():
    """Rate (v - 4) / 8 clipped to [0, 1] on the grid: 0, 0, 0, 1/8, 1/4, 3/8, 1/2, 3/4, 1, 1."""
    crashes = [0, 0, 0, 6, 12, 18, 24, 36, 48, 48]
    got = E.v50(crashes, N48, SPEEDS)
    assert got.value == pytest.approx(8.0, abs=1e-12) and got.censored is None
    low = E.v10(crashes, N48, SPEEDS)
    assert low.value == pytest.approx(4.8, abs=1e-12) and low.censored is None          # 4 + (0.1 - 0) / (0.125 - 0) * 1


def test_bn03_v50_is_straight_line_interpolation_between_adjacent_grid_speeds():
    n24 = [24] * 10
    assert E.v50([0, 0, 0, 0, 6, 18, 24, 24, 24, 24], n24, SPEEDS).value == pytest.approx(6.5, abs=1e-12)     # 0.25 at 6, 0.75 at 7


def test_bn03_the_interpolation_uses_the_grids_uneven_spacing():
    between_8_and_10 = E.v50([0, 0, 0, 0, 0, 0, 19, 29, 48, 48], N48, SPEEDS)                  # 19/48 at 8 m/s, 29/48 at 10 m/s
    assert between_8_and_10.value == pytest.approx(8 + (24 - 19) / (29 - 19) * 2, abs=1e-12)
    assert between_8_and_10.value == pytest.approx(9.0, abs=1e-12)
    between_12_and_14 = E.v50([0, 0, 0, 0, 0, 0, 0, 0, 22, 26], N48, SPEEDS)                    # 22/48 at 12, 26/48 at 14: straddle 1/2
    assert between_12_and_14.value == pytest.approx(13.0, abs=1e-12) and between_12_and_14.censored is None


def test_bn03_v10_is_the_same_rule_at_one_tenth_and_sits_on_the_safe_side_of_v50():
    crashes = [0, 0, 1, 3, 9, 24, 40, 47, 48, 48]
    lo, hi = E.v10(crashes, N48, SPEEDS), E.v50(crashes, N48, SPEEDS)
    assert lo.value == pytest.approx(O.edge_exact(crashes, N48, SPEEDS, 0.1)[0], abs=1e-12) and lo.censored is None
    assert hi.value == pytest.approx(O.edge_exact(crashes, N48, SPEEDS, 0.5)[0], abs=1e-12) and hi.censored is None
    assert lo.value == pytest.approx(5 + (0.1 - 3 / 48) / (9 / 48 - 3 / 48), abs=1e-12)           # 3/48 at 5 m/s, 9/48 at 6 m/s
    assert lo.value < hi.value
    assert E.edge(crashes, N48, SPEEDS, 0.1) == lo and E.edge(crashes, N48, SPEEDS, 0.5) == hi


def test_bn03_the_fit_not_the_raw_curve_is_read_so_a_dip_cannot_make_an_early_crossing():
    """Raw rates 0, 0.21, 0.60, 0.40, 0.40, 0.83, 1: the 4 m/s point is above one half but the next two pull the block under
    it (67/144 = 0.465), so the fit crosses between 6 and 7 m/s, where a read of the raw curve would say 3.x."""
    crashes = [0, 10, 29, 19, 19, 40, 48, 48, 48, 48]
    fit = E.monotone_fit(crashes, N48)
    assert fit[2] == pytest.approx(67 / 144) and fit[3] == pytest.approx(67 / 144) and fit[4] == pytest.approx(67 / 144)
    got = E.v50(crashes, N48, SPEEDS)
    assert got.censored is None
    assert got.value == pytest.approx(6 + (0.5 - 67 / 144) / (40 / 48 - 67 / 144), abs=1e-12)
    assert got.value == pytest.approx(O.edge_exact(crashes, N48, SPEEDS)[0], abs=1e-12)


# ---- censoring -------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("speeds", [SPEEDS, EXT], ids=["14-top", "16-top"])
def test_bn03_a_curve_that_never_reaches_one_half_is_the_grids_top_marked_censored(speeds):
    s = len(speeds)
    got = E.v50([0] * (s - 1) + [23], [48] * s, speeds)                     # 23/48 at the top: under one half
    assert got.value == speeds[-1] and got.censored == "top"
    got0 = E.v50([0] * s, [48] * s, speeds)
    assert got0.value == speeds[-1] and got0.censored == "top"


@pytest.mark.parametrize("speeds", [SPEEDS, EXT], ids=["14-top", "16-top"])
def test_bn03_a_curve_that_starts_above_one_half_is_2_ms_marked_censored(speeds):
    s = len(speeds)
    got = E.v50([25] + [48] * (s - 1), [48] * s, speeds)
    assert got.value == 2.0 and got.censored == "bottom"
    assert E.v50([48] * s, [48] * s, speeds).censored == "bottom"


def test_bn03_the_16_ms_top_moves_where_a_never_reaching_curve_is_marked():
    c14 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 20]
    at14 = E.v50(c14, N48, SPEEDS)
    assert (at14.value, at14.censored) == (14.0, "top")
    got = E.v50(c14 + [30], N48 + [48], EXT)                              # the 16 m/s row crosses: no longer censored
    assert got.censored is None and got.value == pytest.approx(14 + (24 - 20) / (30 - 20) * 2, abs=1e-12)


def test_bn03_censoring_is_read_on_the_fit_so_a_start_that_pools_down_under_one_half_is_not_censored():
    """Raw rates 0.71, 0.21, 0.42 at the bottom pool to 64/144 = 0.444, under one half, so the curve does not start above it."""
    crashes = [34, 10, 20, 30, 48, 48, 48, 48, 48, 48]
    fit = E.monotone_fit(crashes, N48)
    assert fit[0] == pytest.approx(64 / 144) and fit[2] == pytest.approx(64 / 144)
    got = E.v50(crashes, N48, SPEEDS)
    assert got.censored is None and got.value == pytest.approx(4 + (0.5 - 64 / 144) / (30 / 48 - 64 / 144), abs=1e-12)
    assert got.value == pytest.approx(O.edge_exact(crashes, N48, SPEEDS)[0], abs=1e-12)


# ---- a fitted curve sitting at exactly one half (OWED 5A-1, 5A-2) -----------------------------------------------------------------
def test_bn03_a_curve_at_exactly_one_half_over_a_stretch_crosses_at_the_stretchs_lowest_speed():
    crashes = [0, 0, 0, 24, 24, 24, 48, 48, 48, 48]                          # 0.5 at 5, 6 and 7 m/s
    got = E.v50(crashes, N48, SPEEDS)
    assert got.value == pytest.approx(5.0, abs=1e-12) and got.censored is None
    assert O.edge_exact(crashes, N48, SPEEDS)[0] == 5.0


def test_bn03_a_stretch_at_one_half_that_never_rises_above_it_is_still_reached_and_not_censored():
    crashes = [0, 0, 6, 24, 24, 24, 24, 24, 24, 24]
    got = E.v50(crashes, N48, SPEEDS)
    assert got.censored is None and got.value == pytest.approx(5.0, abs=1e-12)


def test_bn03_a_stretch_at_one_half_at_the_very_bottom_is_2_ms_and_is_not_marked_because_it_does_not_start_above_one_half():
    crashes = [24, 24, 24, 30, 48, 48, 48, 48, 48, 48]
    got = E.v50(crashes, N48, SPEEDS)
    assert got.value == 2.0 and got.censored is None
    assert E.v50([24] * 10, N48, SPEEDS).value == 2.0 and E.v50([24] * 10, N48, SPEEDS).censored is None


def test_bn03_a_curve_that_reaches_one_half_only_at_the_top_is_the_top_and_is_not_marked():
    crashes = [0] * 9 + [24]
    got = E.v50(crashes, N48, SPEEDS)
    assert got.value == pytest.approx(14.0, abs=1e-12) and got.censored is None


def test_bn03_the_plateau_rule_holds_at_one_tenth_too():
    crashes = [0, 0, 0, 12, 12, 12, 30, 48, 48, 48]                          # 0.25: no; use n = 120 below for exactly 0.1
    n = [120] * 10
    crashes = [0, 0, 0, 12, 12, 12, 60, 120, 120, 120]                       # 12/120 = 0.1 at 5, 6, 7 m/s
    got = E.v10(crashes, n, SPEEDS)
    assert got.value == pytest.approx(5.0, abs=1e-12) and got.censored is None


@pytest.mark.parametrize("c,m", [([35, 14, 3], [40, 48, 16]), ([32, 13, 9], [48, 30, 30]), ([14, 9, 6], [24, 18, 16]), ([11, 12, 6], [18, 24, 16])])
def test_bn03_a_pooled_block_that_is_exactly_one_half_is_one_half_whatever_floating_point_says(c, m):
    """Three violators, for example 35/40, 14/48 and 3/16, pool to 52/104 = 1/2 exactly. In floating point the pooled mean
    comes out 0.49999999999999994 or 0.5000000000000001. The edge is read on the counts, so the block is at the level and it
    crosses at its lowest speed, 5 m/s here, with the curve rising to 1 above it."""
    from fractions import Fraction
    crashes = [0, 0, 0] + c + [24, 24, 24, 24]
    n = [24, 24, 24] + m + [24, 24, 24, 24]
    assert O.pav_exact(crashes, n)[3] == Fraction(1, 2)
    got = E.v50(crashes, n, SPEEDS)
    assert got.censored is None and got.value == pytest.approx(5.0, abs=1e-12)
    assert got.value == pytest.approx(O.edge_exact(crashes, n, SPEEDS)[0], abs=1e-12)


# ---- a speed with no scored episodes ------------------------------------------------------------------------------------------
def test_bn03_a_speed_with_no_scored_episodes_is_left_out_of_the_fit_and_a_fit_with_none_left_is_an_error():
    crashes = [0, 0, 0, 0, 0, 0, 0, 24, 48, 48]
    n = [48, 48, 48, 48, 0, 48, 48, 48, 48, 48]
    got = E.v50(crashes, n, SPEEDS)
    assert got.value == pytest.approx(O.edge_exact(crashes, n, SPEEDS)[0], abs=1e-12)
    assert got.value == pytest.approx(10.0, abs=1e-12)
    assert got.n_per_speed == (48,) * 9                                    # the 6 m/s speed is not in the fit
    with pytest.raises(ValueError):
        E.v50([0] * 10, [0] * 10, SPEEDS)


# ---- the per-timing fit at 24 an speed (N2) -----------------------------------------------------------------------------------
def test_bn03_each_timing_is_fitted_alone_at_24_episodes_a_speed_and_the_pooled_fit_is_the_two_together():
    """Timing 0 steps from 0 to 24 of 24 at 6 m/s; timing 0.125 steps at 8 m/s. Alone: 5.5 and 7.5 m/s. Pooled they sit at
    exactly one half over 6 and 7 m/s (24 of 48), so the pooled v50 is the plateau's lowest speed, 6 m/s."""
    c = np.zeros((10, 2), int)
    c[4:, 0] = 24
    c[6:, 1] = 24
    n = np.full((10, 2), 24)
    got = E.direction_edges(c, n, SPEEDS)
    assert [e.value for e in got.v50_by_timing] == [pytest.approx(5.5), pytest.approx(7.5)]
    assert [e.censored for e in got.v50_by_timing] == [None, None]
    assert all(e.n_per_speed == (24,) * 10 for e in got.v50_by_timing)
    assert got.v50.value == pytest.approx(6.0) and got.v50.censored is None
    assert got.v50.n_per_speed == (48,) * 10


def test_bn03_the_per_timing_fit_at_24_hits_one_half_exactly_often_and_the_rule_reads_it_the_same_way():
    """12 of 24 is one half exactly, so a stretch of 12s at 24 episodes is an ordinary case and not a rare one."""
    c = np.array([[0, 0], [0, 3], [2, 6], [9, 12], [12, 12], [12, 12], [20, 24], [24, 24], [24, 24], [24, 24]])
    n = np.full((10, 2), 24)
    got = E.direction_edges(c, n, SPEEDS)
    for t in range(2):
        want_50, _ = O.edge_exact(c[:, t], n[:, t], SPEEDS, 0.5)
        want_10, _ = O.edge_exact(c[:, t], n[:, t], SPEEDS, 0.1)
        assert got.v50_by_timing[t].value == pytest.approx(want_50) and got.v10_by_timing[t].value == pytest.approx(want_10)
    assert got.v50_by_timing[0].value == pytest.approx(6.0) and got.v50_by_timing[1].value == pytest.approx(5.0)   # the first speed at 12 of 24


def test_bn03_a_timing_can_be_censored_while_the_other_is_not_and_the_marks_follow_each_fit():
    c = np.zeros((10, 2), int)
    c[:, 1] = 24                                           # timing 0.125 crashes everywhere; timing 0 never crashes
    n = np.full((10, 2), 24)
    got = E.direction_edges(c, n, SPEEDS)
    assert (got.v50_by_timing[0].value, got.v50_by_timing[0].censored) == (14.0, "top")
    assert (got.v50_by_timing[1].value, got.v50_by_timing[1].censored) == (2.0, "bottom")
    assert got.v50.censored is None and got.v50.value == pytest.approx(O.edge_exact(c.sum(1), n.sum(1), SPEEDS)[0])
    assert (got.v10_by_timing[0].censored, got.v10_by_timing[1].censored) == ("top", "bottom")


def test_bn03_direction_edges_takes_a_16_top_and_marks_censoring_at_16():
    c = np.zeros((11, 2), int)
    n = np.full((11, 2), 24)
    got = E.direction_edges(c, n, EXT)
    assert got.v50.value == 16.0 and got.v50.censored == "top"
    assert all(e.value == 16.0 and e.censored == "top" for e in got.v50_by_timing)
