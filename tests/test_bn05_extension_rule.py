"""Rule 5 (spec section 3 "Grid extension", N10, 20.1, 20.2): the extension rule's first reading.

"In session 5B, once Zone B's rows at 12 m/s are in, and before any retrain exists: if, on any of the 16 directions, the mean pi0
crash rate at 12 m/s is under 80%, add 16 m/s to Zone B for every policy and controller." N10: "the mean pi0 crash rate at 12 m/s on
each of the 16 directions, and whether 16 m/s was added." And every grid-dependent function takes a 16 m/s top."""
import numpy as np
import pandas as pd
import pytest

import boundary_numbers as B
import boundary_oracles as O
import boundary_rows as R
from drone_study.analysis import estimator as E
from drone_study.boundary import results as N
from drone_study.boundary import sweeps as W

S12 = B.SPEEDS.index(12.0)


def pi0_grids(rate_at_12, n=24, speeds=B.SPEEDS, shape=R.SHAPE):
    """16 π0 grids whose crash counts at 12 m/s are rate_at_12[lineage][direction] (per timing); everywhere else 0."""
    out = []
    for lineage in range(16):
        c = np.zeros(shape, int)
        c[:, speeds.index(12.0), :] = np.asarray(rate_at_12[lineage])[:, None]
        rows = R.frame(R.zone_b_rows(f"pi0-seed-{lineage + 1}", c, n, speeds=speeds))
        out.append(N.flyer_grid(rows, f"pi0-seed-{lineage + 1}", speeds=speeds))
    return out


def uniform(count):
    return [np.full(16, count) for _ in range(16)]


def test_bn05_with_every_direction_at_or_over_80_percent_nothing_is_added():
    got = N.n10_extension(pi0_grids(uniform(20)))                            # 20 of 24 = 83%
    assert got["rate_at_12"] == pytest.approx(np.full(16, 20 / 24), abs=1e-12)
    assert got["fires"] is False and tuple(got["speeds"]) == B.SPEEDS
    assert tuple(got["directions"]) == B.DIRECTIONS


def test_bn05_one_direction_under_80_percent_adds_16_ms_for_everyone():
    counts = uniform(24)
    for lineage in range(16):
        counts[lineage] = counts[lineage].copy()
        counts[lineage][9] = 18                                              # direction 9: 18 of 24 = 75%
    got = N.n10_extension(pi0_grids(counts))
    assert got["fires"] is True and tuple(got["speeds"]) == B.SPEEDS_EXTENDED
    assert got["rate_at_12"][9] == pytest.approx(0.75, abs=1e-12) and (np.delete(got["rate_at_12"], 9) == 1.0).all()


def test_bn05_exactly_80_percent_is_not_under_and_does_not_fire_and_a_hair_under_does():
    """5 replicates a cell so 80% is exact: 4 of 5 in every cell. Under means strictly under, read on the counts."""
    at = N.n10_extension(pi0_grids(uniform(4), n=5))
    assert at["rate_at_12"] == pytest.approx(np.full(16, 0.8)) and at["fires"] is False
    counts = uniform(4)
    counts[0] = counts[0].copy()
    counts[0][2] = 3                                                          # one lineage 3 of 5 on direction 2: mean 0.7875
    under = N.n10_extension(pi0_grids(counts, n=5))
    assert under["rate_at_12"][2] == pytest.approx(0.7875) and under["fires"] is True


def test_bn05_the_rate_is_the_mean_over_lineages_of_the_mean_of_the_two_timings_cells_rates():
    """As 8A-10: a lineage's rate is the plain mean of its cells' rates. With unequal scored counts that is not the pooled rate.
    Lineage 0, direction 0: 24 of 24 at timing 0 and 0 of 6 at timing 0.125. Every other cell: 24 of 24."""
    c = np.zeros(R.SHAPE, int)
    c[:, S12, :] = 24
    grids = []
    for lineage in range(16):
        cc, nn = c.copy(), np.full(R.SHAPE, 24)
        if lineage == 0:
            cc[0, S12, 1], nn[0, S12, 1] = 0, 6
        rows = R.frame(R.zone_b_rows(f"pi0-seed-{lineage + 1}", cc, nn))
        grids.append(N.flyer_grid(rows, f"pi0-seed-{lineage + 1}"))
    got = N.n10_extension(grids)
    assert got["rate_at_12"][0] == pytest.approx((0.5 + 15 * 1.0) / 16, abs=1e-12)           # 0.96875
    assert got["rate_at_12"][0] != pytest.approx((24 + 0 + 15 * 48) / (24 + 6 + 15 * 48))     # the pooled rate is 0.992
    assert (got["rate_at_12"][1:] == 1.0).all()


def test_bn05_the_reading_needs_the_12_ms_row():
    speeds = B.SPEEDS[:8]                                                      # 2 to 10 m/s
    rows = R.frame(R.zone_b_rows("pi0-seed-1", np.zeros((16, 8, 2), int), speeds=speeds))
    with pytest.raises(ValueError):
        N.n10_extension([N.flyer_grid(rows, "pi0-seed-1", speeds=speeds)])


def test_bn05_the_reading_is_of_the_pi0_only_and_takes_the_16_lineages_in_the_order_given():
    got = N.n10_extension(pi0_grids(uniform(12)))
    assert got["fires"] is True and got["rate_at_12"] == pytest.approx(np.full(16, 0.5))


# ---- every grid-dependent function also takes a 16 m/s top --------------------------------------------------------------------
EXT = B.SPEEDS_EXTENDED


def ext_counts(top_count):
    c = np.zeros((16, 11, 2), int)
    c[:, 4:10, :] = np.array([3, 6, 9, 12, 18, 24])[None, :, None]
    c[:, 10, :] = top_count
    return c


def ext_rows(flyer, top_count):
    return R.frame(R.zone_b_rows(flyer, ext_counts(top_count), speeds=EXT))


def test_bn05_the_sweep_files_take_a_16_top():
    assert len(W.zone_b_conditions(EXT)) == 352 and len(W.extension_conditions()) == 32
    assert W.zone_b_sweep_spec(EXT).count("v=16.0") == 32


def test_bn05_the_estimator_takes_a_16_top_and_marks_censoring_there():
    never = E.v50([0] * 11, [48] * 11, EXT)
    assert (never.value, never.censored) == (16.0, "top")
    assert E.direction_edges(np.zeros((11, 2), int), np.full((11, 2), 24), EXT).v50.censored == "top"


def test_bn05_the_results_functions_take_a_16_top():
    rows = ext_rows("f", top_count=24)
    g = N.flyer_grid(rows, "f", speeds=EXT)
    assert g.crashes.shape == (16, 11, 2) and g.speeds == EXT
    cells = N.n1_cells(rows, ["f"], speeds=EXT)
    assert len(cells) == 352 and {c["speed"] for c in cells} == set(EXT)
    sixteen = [c for c in cells if c["speed"] == 16.0]
    assert len(sixteen) == 32 and all(c["force_ratio"] == pytest.approx(O.flat_force_ratio(16.0)) for c in sixteen)
    assert O.flat_force_ratio(16.0) == pytest.approx(0.95, abs=0.005)
    edges = N.n2_edges(g)
    assert len(edges) == 16 and all(e.v50.censored is None for e in edges)           # the curve reaches one half at 16
    assert N.n3_integrated(g).shape == (16,)
    never = N.flyer_grid(R.frame(R.zone_b_rows("g", np.zeros((16, 11, 2), int), speeds=EXT)), "g", speeds=EXT)
    assert all((e.v50.value, e.v50.censored) == (16.0, "top") for e in N.n2_edges(never))


def test_bn05_n4_n5_n6_n8_and_n10_take_a_16_top():
    rows = ext_rows("pi0-seed-1", 24)
    ctl = R.frame(R.zone_b_rows("pid", np.zeros((16, 11, 2), int), speeds=EXT))             # never crashes
    both = pd.concat([rows, ctl], ignore_index=True)
    grids = [N.flyer_grid(rows, "pi0-seed-1", speeds=EXT)] * 16
    controller = N.flyer_grid(ctl, "pid", speeds=EXT)
    n4 = N.n4_across_lineages(grids)
    assert np.asarray(n4["v50"]["values"]).shape == (16, 16)
    n5 = N.n5_against_controller(grids, controller)
    assert np.asarray(n5["v50"]["differences"]).shape == (16, 16)
    assert np.asarray(n5["v50"]["censored_pair"]).all()                          # the controller's v50 is censored at 16 everywhere
    n6 = N.n6_paired_counts(both, "pi0-seed-1", "pid", speeds=EXT)
    assert len(n6) == 352 and {c["speed"] for c in n6} == set(EXT)
    at16 = [c for c in n6 if c["speed"] == 16.0]
    assert all((c["both"], c["only_pi0"], c["only_controller"], c["neither"]) == (0, 24, 0, 0) for c in at16)
    ten = N.n10_extension(grids)
    assert ten["fires"] is True and tuple(ten["speeds"]) == EXT                   # 18 of 24 at 12 m/s fires; 16 is already in the grid, so it is not added twice
