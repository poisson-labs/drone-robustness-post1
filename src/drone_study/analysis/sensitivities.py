"""S6 of the sensitivities named in advance (spec section 10): the crash rules re-read under other limits, from the crossings
each flight stored, with nothing flown. It is printed beside the primary and cannot replace it.

This public slice carries S6's limits and its re-score, the two functions of ruling 5A-5 that post 1's N8 reads. The study's
other sensitivities (section 10) read retrained arms and belong to its second post; they are not in this repository and
are removed from this module."""
from __future__ import annotations

import pandas as pd

import drone_study.score as _score
from drone_study.score import Limits

# S6 (section 10): tilt 75 and 90 degrees instantaneous, radius 0.3, 0.75 and 1.00 m, and `ground` and `tilt` only. The
# ground line stays at 0.05 m: it is the one physical line and the only one stored (score.rescore refuses another).
_PRIMARY_TILT = ((60.0, 1.0), (101.5, 0.1))
S6_LIMITS: dict = {
    "tilt-75": Limits(ground_height=0.05, tilt_rules=((75.0, 0.0),), radius=0.5),
    "tilt-90": Limits(ground_height=0.05, tilt_rules=((90.0, 0.0),), radius=0.5),
    "radius-0.3": Limits(ground_height=0.05, tilt_rules=_PRIMARY_TILT, radius=0.3),
    "radius-0.75": Limits(ground_height=0.05, tilt_rules=_PRIMARY_TILT, radius=0.75),
    "radius-1.0": Limits(ground_height=0.05, tilt_rules=_PRIMARY_TILT, radius=1.0),
    "ground-and-tilt": Limits(ground_height=0.05, tilt_rules=_PRIMARY_TILT, radius=None),
}


def s6_rescore(rows, limits_name: str):
    """The rows with `outcome` re-scored from each row's stored crossings by drone_study.score.rescore under
    S6_LIMITS[limits_name]. Error rows stay error rows. Every read then runs on the returned rows unchanged.
    Nothing is flown: each scored row is re-scored from the crossings its flight stored. The table given is not changed;
    a `crash_class` column, where the rows carry one, is re-read with the outcome (the class under these limits, None for a pass)."""
    if limits_name not in S6_LIMITS:
        raise KeyError(f"{limits_name!r} is not one of S6's limit sets: {sorted(S6_LIMITS)}")
    limits = S6_LIMITS[limits_name]
    out = rows.copy()
    outcomes, classes = [], []
    for err, crossings in zip(out["error"].tolist(), out["crossings"].tolist()):
        if err:
            outcomes.append("error")
            classes.append(None)
            continue
        if crossings is None:
            raise ValueError("a scored row stored no crossings, so it cannot be re-scored (S6 flies nothing)")
        result = _score.rescore(crossings, limits)
        outcomes.append("ok" if result.passed else "crash")
        classes.append(result.crash_class)
    out["outcome"] = pd.Series(outcomes, index=out.index, dtype=object)
    if "crash_class" in out.columns:
        out["crash_class"] = pd.Series(classes, index=out.index, dtype=object)
    return out
