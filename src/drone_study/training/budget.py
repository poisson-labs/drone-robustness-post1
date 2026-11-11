"""Training budgets (spec section 2; session 7's row in section 12).

"Four pilot seeds ... train to 40M control steps with checkpoints at 5M, 10M, 20M and 40M. The budget B0 is the
smallest rung at which all four pass the bar." Readings (4A-2, 4A-15): a run's steps come in whole rollouts of 31,250;
no rung with all four pilots at the bar picks nothing.

This public slice carries the nominal arm's budget alone; the fine-tune budgets of the study's other arms (its second post)
are removed from this module and nothing else in it is changed.

Interface (4A-19), as the slice carries it:
  RUNGS, N_PILOTS, ROLLOUT_STEPS
  b0_from_pilots(passed)      passed: {rung: four booleans}; the smallest rung where all four pass; NoRungPasses if none
  budget(arm, b0)             the run's control steps (the nominal arm: B0)
  curve_marks(budget)         the steps at 25, 50, 75 and 100% of it
"""
from __future__ import annotations

from typing import Mapping, Sequence

RUNGS: tuple = (5_000_000, 10_000_000, 20_000_000, 40_000_000)
N_PILOTS: int = 4
ROLLOUT_STEPS: int = 25 * 1250
NOMINAL_ARM = "nominal"


class NoRungPasses(ValueError):
    """No rung has all four pilots at the bar: the study waits on a ruling (4A-15)."""


def b0_from_pilots(passed: Mapping[int, Sequence[bool]]) -> int:
    for rung in RUNGS:
        flags = list(passed.get(rung, ()))
        if len(flags) != N_PILOTS:
            raise ValueError(f"rung {rung} needs {N_PILOTS} pilot readings, got {len(flags)}")
        if all(bool(f) for f in flags):
            return rung
    raise NoRungPasses("no rung has all four pilots at the bar; no B0 is picked and the study waits on a ruling (4A-15)")


def _check_rung(b0: int) -> int:
    if b0 not in RUNGS:
        raise ValueError(f"B0 must be one of the rungs {RUNGS}, got {b0!r}")
    return int(b0)


def budget(arm: str, b0: int) -> int:
    b0 = _check_rung(b0)
    if arm == NOMINAL_ARM:
        return b0
    raise ValueError(f"unknown arm {arm!r}")


def curve_marks(budget_steps: int) -> tuple:
    b = int(budget_steps)
    return (b // 4, b // 2, 3 * b // 4, b)
