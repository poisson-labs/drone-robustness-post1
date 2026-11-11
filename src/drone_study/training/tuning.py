"""The geometric controller's gain grid and how a point is picked (spec section 17.2; ruling 4A-16, pick B).

"The geometric controller's Kp, Kv and Ki are picked from a fixed grid session 4A names, by smallest median RMSE on the
12 cases in calm air, before any pi0 meets wind, and frozen." The grid: Kp 6, 9, 13.5, 20, 30 x Kv 3.5, 5, 7, 10, 14 x
Ki 1, 2, 4 (75 points, Kp first). Each point flies the 12 calm cases on 8 seeds of the `tuning` salt (96 episodes); the
pick is the smallest median RMSE among points with no crash; a tie goes to the earlier point. Ki is never 0.

Interface (4A-19):
  GAIN_GRID, TUNING_SALT, TUNING_REPLICATES
  tuning_conditions()         the 12 calm cells
  tuning_seeds(condition)     the 8 seeds Prove derives for that cell under the tuning salt
  pick_gains(results)         results: {GeometricGains: [(rmse, crashed), ...]} for every grid point, 96 pairs each
  tuning_figure(results)      the picked point's median RMSE in metres (N9's "tuning figure")
"""
from __future__ import annotations

import hashlib
from itertools import product
from typing import Mapping, Sequence

import numpy as np
from poisson_prove.executor import derive_seed
from poisson_prove.store import config_hash

from drone_study.cases import CASE_NAMES
from drone_study.condition import Condition, condition_to_cfg
from drone_study.controllers import GeometricGains

GRID_KP = (6.0, 9.0, 13.5, 20.0, 30.0)
GRID_KV = (3.5, 5.0, 7.0, 10.0, 14.0)
GRID_KI = (1.0, 2.0, 4.0)
GAIN_GRID: tuple = tuple(GeometricGains(kp, kv, ki) for kp, kv, ki in product(GRID_KP, GRID_KV, GRID_KI))
TUNING_SALT: str = hashlib.sha256(b"drone-robustness-2026-11-04:tuning").hexdigest()[:16]     # 5c2a0186f8a9244f
TUNING_REPLICATES: int = 8
EPISODES_PER_POINT = 12 * TUNING_REPLICATES


def tuning_conditions() -> list:
    return [Condition.calm(name) for name in CASE_NAMES]


def tuning_seeds(condition: Condition) -> tuple:
    h = config_hash(condition_to_cfg(condition))
    return tuple(derive_seed(TUNING_SALT, h, k) for k in range(TUNING_REPLICATES))


def _medians(results: Mapping[GeometricGains, Sequence]) -> dict:
    if set(results) != set(GAIN_GRID):
        raise ValueError("the tuning results must cover the grid's 75 points exactly")
    out = {}
    for g in GAIN_GRID:
        rows = list(results[g])
        if len(rows) != EPISODES_PER_POINT:
            raise ValueError(f"grid point {g} has {len(rows)} episodes, the grid flies {EPISODES_PER_POINT}")
        crashed = any(bool(c) for _, c in rows)
        out[g] = (float(np.median([float(r) for r, _ in rows])), crashed)
    return out


def pick_gains(results: Mapping[GeometricGains, Sequence]) -> GeometricGains:
    medians = _medians(results)
    best, best_median = None, None
    for g in GAIN_GRID:                                        # grid order: a tie goes to the earlier point
        median, crashed = medians[g]
        if crashed:
            continue
        if best is None or median < best_median:
            best, best_median = g, median
    if best is None:
        raise ValueError("every grid point crashed in calm air: nothing to pick")
    return best


def tuning_figure(results: Mapping[GeometricGains, Sequence]) -> float:
    return _medians(results)[pick_gains(results)][0]
