"""Training rule 16 (spec section 17.2): "The geometric controller's Kp, Kv and Ki are picked from a fixed grid session 4A
names, by smallest median RMSE on the 12 cases in calm air, before any pi0 meets wind, and frozen." And N9 of section
20.1: "The geometric controller's calm RMSE from its gain selection, labelled a tuning figure."

The grid, the episodes behind "median RMSE" and the tie rule are ruling 4A-16 (Taylor, 2026-11-05, pick B): Kp 6, 9,
13.5, 20, 30; Kv 3.5, 5, 7, 10, 14; Ki 1, 2, 4 (75 points, 3B's provisional pick among them; Ki is never 0, so section
17.2's integral, the one that leaves a steady wind no standing offset, is always on); each point flies the 12 calm
cases on 8 seeds from a `tuning` salt (96 episodes); a point with any crash is out; ties go to the earlier point, Kp
first. The grid runs to Kp 30 and Kv 14 because calm-air tracking error keeps falling as the gains rise: a grid capped
lower would hand the classical baseline a worse tune than it can have.
"""
from itertools import product

import numpy as np
import pytest
from poisson_prove.executor import derive_seed
from poisson_prove.store import config_hash

import training_numbers as N
from drone_study.condition import Condition, condition_to_cfg
from drone_study.controllers import GeometricGains
from drone_study.training import tuning as TU
from drone_study.training.tuning import pick_gains, tuning_conditions, tuning_figure, tuning_seeds

GRID = [GeometricGains(kp, kv, ki) for kp, kv, ki in product(N.GRID_KP, N.GRID_KV, N.GRID_KI)]
EPISODES = 12 * N.TUNING_REPLICATES


def results(rmse_of, crashed_at=()):
    """{gains: 96 (rmse, crashed) pairs}; rmse_of(gains) gives the point's RMSE (one value or 96)."""
    out = {}
    for g in GRID:
        values = np.broadcast_to(np.asarray(rmse_of(g), dtype=float), (EPISODES,))
        out[g] = [(float(v), (g in crashed_at and i == 5)) for i, v in enumerate(values)]
    return out


def bowl(g, centre=GeometricGains(9.0, 5.0, 1.0)):
    return 0.05 + 0.01 * (abs(g.kp - centre.kp) + abs(g.kv - centre.kv) + abs(g.ki - centre.ki))


def test_tr16_the_grid_is_the_one_session_4a_names():
    assert list(TU.GAIN_GRID) == GRID and len(GRID) == 75 and len(set(GRID)) == 75
    assert GeometricGains(6.0, 3.5, 2.0) in GRID                              # session 3B's provisional pick is a candidate
    assert GRID[0] == GeometricGains(6.0, 3.5, 1.0) and GRID[-1] == GeometricGains(30.0, 14.0, 4.0)   # ruled B: up to Kp 30, Kv 14
    assert all(g.kp > 0 and g.kv > 0 and g.ki > 0 for g in GRID)                # the integral is never switched off


def test_tr16_each_point_flies_the_twelve_calm_cases_on_eight_seeds_of_the_tuning_salt():
    assert (TU.TUNING_SALT, TU.TUNING_REPLICATES) == (N.TUNING_SALT, 8) and N.TUNING_SALT == "5c2a0186f8a9244f"
    cells = tuning_conditions()
    assert cells == [Condition.calm(name) for name in N.CASE_NAMES]
    for c in cells:
        assert tuning_seeds(c) == tuple(derive_seed(N.TUNING_SALT, config_hash(condition_to_cfg(c)), k) for k in range(8))
    bar = derive_seed(N.BAR_SALT, config_hash(condition_to_cfg(cells[0])), 0)
    assert tuning_seeds(cells[0])[0] != bar                                   # not the bar's episodes


@pytest.mark.parametrize("centre", [GeometricGains(9.0, 5.0, 1.0), GeometricGains(6.0, 3.5, 1.0), GeometricGains(30.0, 14.0, 4.0),
                                    GeometricGains(20.0, 10.0, 4.0), GeometricGains(30.0, 7.0, 2.0), GeometricGains(13.5, 14.0, 1.0),
                                    GeometricGains(6.0, 3.5, 2.0)])
def test_tr16_the_pick_is_the_point_with_the_smallest_median_rmse(centre):
    assert pick_gains(results(lambda g: bowl(g, centre))) == centre


def test_tr16_it_is_a_median_over_the_points_96_episodes_not_a_mean():
    a, b = GRID[10], GRID[40]

    def rmse(g):
        if g == a:
            return [0.03] * 49 + [0.90] * 47                                  # median 3 cm, mean 46 cm
        if g == b:
            return [0.04] * 96
        return 0.2

    assert pick_gains(results(rmse)) == a
    swapped = results(lambda g: [0.03] * 47 + [0.90] * 49 if g == a else (0.04 if g == b else 0.2))   # median 90 cm
    assert pick_gains(swapped) == b


def test_tr16_a_point_that_crashed_in_calm_air_is_out_however_low_its_rmse():
    best, second = GeometricGains(9.0, 5.0, 1.0), GeometricGains(9.0, 5.0, 2.0)
    assert pick_gains(results(bowl)) == best                                  # the control
    assert pick_gains(results(bowl, crashed_at=(best,))) == second
    with pytest.raises(ValueError):
        pick_gains(results(bowl, crashed_at=tuple(GRID)))                     # nothing left to pick


def test_tr16_a_tie_goes_to_the_earlier_point_of_the_grid():
    flat = results(lambda g: 0.07)
    assert pick_gains(flat) == GRID[0] == GeometricGains(6.0, 3.5, 1.0)
    pair = results(lambda g: 0.05 if g in (GRID[20], GRID[45]) else 0.07)
    assert pick_gains(pair) == GRID[20]


def test_tr16_results_that_do_not_cover_the_grid_or_the_96_episodes_are_refused():
    full = results(bowl)
    assert pick_gains(full) == GeometricGains(9.0, 5.0, 1.0)                  # the control
    missing = dict(full); del missing[GRID[3]]
    with pytest.raises(ValueError):
        pick_gains(missing)
    extra = dict(full); extra[GeometricGains(1.0, 1.0, 1.0)] = full[GRID[0]]
    with pytest.raises(ValueError):
        pick_gains(extra)
    short = dict(full); short[GRID[3]] = full[GRID[3]][:-1]
    with pytest.raises(ValueError):
        pick_gains(short)


def test_tr16_the_tuning_figure_is_the_picked_points_median_rmse():
    """N9 prints the geometric controller's calm RMSE "from its gain selection, labelled a tuning figure": the smallest
    median on the grid, so it flatters, and it is labelled as that."""
    assert tuning_figure(results(bowl)) == pytest.approx(0.05, rel=1e-12)
    a = GRID[10]
    lopsided = results(lambda g: [0.03] * 49 + [0.90] * 47 if g == a else 0.2)        # the pick's median is 3 cm, its mean 46 cm
    assert pick_gains(lopsided) == a and tuning_figure(lopsided) == pytest.approx(0.03, rel=1e-12)
    crashed_best = results(bowl, crashed_at=(GeometricGains(9.0, 5.0, 1.0),))
    assert tuning_figure(crashed_best) == pytest.approx(0.06, rel=1e-12)             # the figure follows the pick, not the lowest RMSE
