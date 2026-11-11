"""Rule 4 (spec section 9, N3): the integrated crash rate.

"For each direction, the area under the raw crash curve, pooled over timings, from 2 m/s to the grid's top by the trapezoid
rule, divided by that range. Nothing censors it." Session 8A froze the function (`analysis.boundary.integrated_rate`, 11 cases in
test_an11). This file adds only what is missing: N3 takes it per direction from stored rows, pooled on counts over the two
timings, and does so through that one function."""
import numpy as np
import pytest

import boundary_numbers as B
import boundary_oracles as O
import boundary_rows as R
from drone_study.analysis import boundary as analysis_boundary
from drone_study.boundary import results as N


def grid(crashes, n=24, speeds=B.SPEEDS, errors=None):
    rows = R.frame(R.zone_b_rows("f", crashes, n, speeds=speeds, errors=errors))
    return N.flyer_grid(rows, "f", speeds=speeds)


def test_bn04_n3_is_the_trapezoid_over_the_raw_pooled_counts_in_every_direction():
    rng = np.random.default_rng(3)
    c = rng.integers(0, 25, size=R.SHAPE)
    got = N.n3_integrated(grid(c))
    assert got.shape == (16,)
    for d in range(16):
        want = O.integrated(c[d].sum(-1), np.full(10, 48), B.SPEEDS)
        assert got[d] == pytest.approx(want, abs=1e-12), d


def test_bn04_a_flat_direction_integrates_to_its_rate_and_a_step_by_hand():
    c = R.per_speed([6] * 10)                                                    # 12 of 48 at every speed
    got = N.n3_integrated(grid(c))
    assert got == pytest.approx(np.full(16, 0.25), abs=1e-12)
    step = R.steps(np.full(16, 5))                                               # 0 up to 6 m/s... 1 from 7 m/s: 7.5 / 12
    assert N.n3_integrated(grid(step)) == pytest.approx(np.full(16, 7.5 / 12), abs=1e-12)


def test_bn04_timings_are_pooled_on_counts_so_a_cell_with_fewer_scored_episodes_weighs_less():
    """Timing 0 has 24 scored at every speed, timing 0.125 has 12 (12 error rows follow). The pooled rate at a speed is the
    sum of crashes over the sum of episodes, not the mean of the two rates."""
    c = np.zeros(R.SHAPE, int)
    c[:, :, 0] = 24                                                              # timing 0 always crashes
    n = np.full(R.SHAPE, 24)
    n[:, :, 1] = 12
    g = grid(c, n, errors=np.where(np.arange(2) == 1, 12, 0)[None, None, :] * np.ones(R.SHAPE, int))
    assert g.n[0, 0, 1] == 12
    pooled = 24 / 36                                                             # 24 of 24 and 0 of 12
    assert N.n3_integrated(g) == pytest.approx(np.full(16, pooled), abs=1e-12)
    assert pooled != pytest.approx((1.0 + 0.0) / 2)


def test_bn04_it_is_the_raw_curve_never_the_fitted_one():
    """Pooled raw counts over 48 at each speed: 0, 12, 6, 24, 18, 36, 30, 48, 40, 48. The dips are not smoothed."""
    per_timing = [0, 6, 3, 12, 9, 18, 15, 24, 20, 24]
    c = R.per_speed(per_timing)
    pooled = np.array(per_timing) * 2
    raw = O.integrated(pooled, np.full(10, 48), B.SPEEDS)
    fitted = np.array([float(x) for x in O.pav_exact(pooled, np.full(10, 48))]) * 48
    assert raw != pytest.approx(O.integrated(fitted, np.full(10, 48), B.SPEEDS), abs=1e-6)
    assert N.n3_integrated(grid(c))[0] == pytest.approx(raw, abs=1e-12)


def test_bn04_with_a_16_top_the_range_is_2_to_16():
    c = np.zeros((16, 11, 2), int)
    c[:, 5:, :] = 24
    got = N.n3_integrated(grid(c, speeds=B.SPEEDS_EXTENDED))
    assert got == pytest.approx(np.full(16, 9.5 / 14), abs=1e-12)               # 0 to 6 m/s, 1 from 7: 0.5 + 1 + 2 + 2 + 2 + 2 = 9.5 over 14


def test_bn04_n3_reads_every_direction_through_the_one_integrated_rate_function(monkeypatch):
    """One function per quantity: whatever analysis.boundary.integrated_rate returns is what N3 reports."""
    calls = []

    def sentinel(crashes, n, speeds):
        calls.append((tuple(np.asarray(crashes).tolist()), tuple(np.asarray(n).tolist()), tuple(speeds)))
        return 0.5 + len(calls) / 1000

    monkeypatch.setattr(analysis_boundary, "integrated_rate", sentinel)
    c = np.zeros(R.SHAPE, int)
    c[3, 4, :] = [5, 9]
    got = N.n3_integrated(grid(c))
    assert len(calls) == 16 and got == pytest.approx([0.5 + (i + 1) / 1000 for i in range(16)])
    assert calls[3][0][4] == 14 and calls[3][1][4] == 48 and calls[3][2] == B.SPEEDS


def test_bn04_8as_integrated_rate_is_the_function_n3_must_agree_with():
    """Where 8A's function is real, N3 equals it on the pooled counts; 8A pins the function's values (test_an11)."""
    c = R.per_speed([0, 6, 3, 12, 9, 18, 15, 24, 20, 24])
    got = N.n3_integrated(grid(c))
    assert got[0] == pytest.approx(analysis_boundary.integrated_rate(c[0].sum(-1), np.full(10, 48), B.SPEEDS), abs=1e-12)
