"""The heading term's weight, picked by a rule written before any pilot trains (the yaw addenda of 2026-11-07 and 2026-11-08
to spec section 2; sections 2 and 5 for B0 and the lineages; rulings 4DA-3 as amended, 4DA-5 as amended, 4DA-9, 4DR-1, 4DR-3).

The rule. For each weight on the base ladder (`reward.YAW_WEIGHT_LADDER_BASE`, smallest first), four pilots (seeds 9001 to
9004, as 4A-13) train to 40M control steps with checkpoints at the rungs (`budget.RUNGS`), as 4B's did. At each rung each
pilot is read against the frozen calm-air bar (`bar.read_bar`, unchanged) and against the yaw acceptance (`behaviour.yaw_acceptance`)
on the same 1,200 episodes. A weight passes at a rung when all four pilots pass both there. The pick (4DR-1) is read rung
first: B0 is the smallest rung at which some weight passes, and the weight is the smallest that passes at that rung. If no
base weight passes at any rung, the ladder extends to 0.032, and if that passes nowhere, to 0.064, each with its own four
pilots to 40M, and the pick is read over the whole table the same way; the study never stops on this (4DA-3 as amended). A
table in which no tried weight passes anywhere raises NoWeightPasses, which 4D-B answers by training the next extension
weight; with the extension exhausted it is a ruling, since nothing beyond 0.064 is written. Each weight carries its crash
penalty (`reward.crash_penalty_for`, 4DR-3), which the pick records.

The pick is written to `data/yaw-weight.json` (weight, b0, crash_penalty, the table) and read back by the nominal retrain,
the bar's manifests and the boundary tests; a file whose weight is off the ladder, whose B0 is not a rung, whose crash
penalty is not the rule's for its weight, or that does not regenerate from its own readings is refused.

The climb (4DA-9 rewritten for the new order): 4D-B trains the base weights in order and may stop as soon as a weight passes
at the smallest rung (5M), since no later weight can then enter the pick; otherwise it trains the next weight and holds the
best (rung, weight) seen; it extends only when the base three pass nowhere. The pick is the same as from the full table.

Lineage selection with the pick (4DA-5 as amended; 4DB-1): a nominal seed is a lineage when it passes the bar (section 2's
three lines), the yaw acceptance and the motor line (`behaviour.motor_acceptance`, at most 0.5% of the bar's control steps with a
motor at its limit) on the bar's episodes; seeds are tried in order until 16 pass; every failure is reported with
which gate it failed; there is no stop. `bar.select_lineages` (4A's, with its stop at nine) is 4B's record and is not used
by 4D-B.

Interface (4D-B builds these; stubs here):
  PilotReading(bar_passed, yaw_passed)        one pilot at one rung; .passed when both
  pick_yaw_weight(table) -> YawPick           table: {weight: {rung: [PilotReading x 4]}}, the weights tried (a prefix of the
                                              ladder in order, every rung and four pilots each); NoWeightPasses when none passes
  YawPick(weight, b0, crash_penalty, table)   .to_dict() / YawPick.from_dict()
  write_yaw_weight(path, pick) -> sha256;  load_yaw_weight(path) -> YawPick (WeightMismatch when refused)
  next_weight(table) -> float | None          the extension weight to train next when the table passes nowhere; None past 0.064
  LineageSelection(lineages, failures, tried, failed_on)   failed_on: {seed: "bar" | "yaw" | "both"}, and with the motor line
                                              (4DB-1) "motor", "bar+motor", "yaw+motor" or "both+motor"
  select_lineages_with_yaw(readings) -> LineageSelection   readings: (seed, BarReading, YawReading[, MotorReading]) in seed order
  failed_gate(bar_ok, yaw_ok, motor_ok=True) -> str | None
  LINEAGES_FILE                               `data/4d-lineages.json`, 4D-B's bars file with the yaw readings and the selection
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

from drone_study.training.budget import N_PILOTS, RUNGS
from drone_study.training.reward import YAW_WEIGHT_LADDER, YAW_WEIGHT_LADDER_BASE, crash_penalty_for

YAW_WEIGHT_FILE = Path("data") / "yaw-weight.json"
LINEAGES_FILE = Path("data") / "4d-lineages.json"
LINEAGES: int = 16


class NoWeightPasses(ValueError):
    """No tried weight has a rung at which all four pilots pass the bar and the yaw acceptance."""


class WeightMismatch(ValueError):
    """The stored pick is not one the rule gives: a weight off the ladder, a B0 that is not a rung, a crash penalty that is not
    the rule's, or readings that do not give it."""


@dataclass(frozen=True)
class PilotReading:
    bar_passed: bool = False
    yaw_passed: bool = False

    @property
    def passed(self) -> bool:
        return bool(self.bar_passed) and bool(self.yaw_passed)


def _ladder_weight(w) -> float:
    for lw in YAW_WEIGHT_LADDER:
        if float(w) == lw:
            return lw
    raise ValueError(f"weight {w!r} is not on the ladder {YAW_WEIGHT_LADDER}")


def _checked_table(table: Mapping) -> dict:
    """The table as {ladder weight: {rung: [PilotReading x 4]}}, a prefix of the ladder with every rung and four pilots each."""
    out = {}
    for w, rungs in table.items():
        lw = _ladder_weight(w)
        if lw in out:
            raise ValueError(f"weight {lw} appears twice")
        by_rung = {}
        for rung, readings in dict(rungs).items():
            rung = int(rung)
            if rung not in RUNGS:
                raise ValueError(f"rung {rung} is not one of {RUNGS}")
            readings = list(readings)
            if len(readings) != N_PILOTS:
                raise ValueError(f"weight {lw} rung {rung}: {len(readings)} pilot readings, the rule needs {N_PILOTS}")
            by_rung[rung] = [r if isinstance(r, PilotReading) else PilotReading(bool(r[0]), bool(r[1])) for r in readings]
        if set(by_rung) != set(RUNGS):
            raise ValueError(f"weight {lw}: rungs {sorted(by_rung)} are not the four {RUNGS}")
        out[lw] = {rung: by_rung[rung] for rung in RUNGS}
    weights = sorted(out)
    if not weights:
        raise ValueError("an empty table")
    if tuple(weights) != tuple(YAW_WEIGHT_LADDER[:len(weights)]):
        raise ValueError(f"the weights tried {weights} are not a prefix of the ladder {YAW_WEIGHT_LADDER}")
    return {w: out[w] for w in weights}


def _passes_at(table: dict, w: float, rung: int) -> bool:
    return all(r.passed for r in table[w][rung])


def _passes_anywhere(table: dict, w: float) -> bool:
    return any(_passes_at(table, w, rung) for rung in RUNGS)


def _check_extension(table: dict) -> None:
    """An extension weight is tried only when every weight before it passes nowhere (4DA-3 as amended)."""
    for w in table:
        if w in YAW_WEIGHT_LADDER_BASE:
            continue
        before = [v for v in table if v < w]
        passing = [v for v in before if _passes_anywhere(table, v)]
        if passing:
            raise ValueError(f"extension weight {w} is in the table while {passing} pass: the rule would not have tried it")


@dataclass(frozen=True)
class YawPick:
    weight: float = 0.0
    b0: int = 0
    crash_penalty: float = 0.0
    table: Mapping = None

    def to_dict(self) -> dict:
        table = {}
        for w, rungs in (self.table or {}).items():
            table[repr(float(w))] = {str(int(rung)): [[bool(r.bar_passed), bool(r.yaw_passed)] for r in readings]
                                     for rung, readings in rungs.items()}
        return {"weight": float(self.weight), "b0": int(self.b0), "crash_penalty": float(self.crash_penalty), "table": table}

    @classmethod
    def from_dict(cls, d: Mapping) -> "YawPick":
        table = {}
        for w, rungs in dict(d.get("table") or {}).items():
            table[float(w)] = {int(rung): [PilotReading(bool(r[0]), bool(r[1])) for r in readings] for rung, readings in dict(rungs).items()}
        return cls(float(d["weight"]), int(d["b0"]), float(d["crash_penalty"]), table)


def pick_yaw_weight(table: Mapping) -> YawPick:
    """table: {weight: {rung: [PilotReading x 4]}} for the weights tried, every rung and four pilots present. B0 is the smallest
    rung at which some weight passes, the weight the smallest passing at that rung. NoWeightPasses when none does."""
    t = _checked_table(table)
    _check_extension(t)
    for rung in RUNGS:
        passing = [w for w in t if _passes_at(t, w, rung)]
        if passing:
            w = min(passing)
            return YawPick(w, int(rung), crash_penalty_for(w), t)
    raise NoWeightPasses(f"no weight of {sorted(t)} has a rung at which all four pilots pass the bar and the yaw gate")


def next_weight(table: Mapping) -> Optional[float]:
    """When the table passes nowhere: the next ladder weight to train (the first not in the table), or None past the ladder's end."""
    try:
        pick_yaw_weight(table)
        return None
    except NoWeightPasses:
        pass
    tried = set(_checked_table(table))
    for w in YAW_WEIGHT_LADDER:
        if w not in tried:
            return w
    return None


def write_yaw_weight(path, pick: YawPick) -> str:
    """Write the pick as JSON; returns the file's SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pick.to_dict(), indent=1, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_yaw_weight(path) -> YawPick:
    """Read a pick back, refusing one the rule would not give (WeightMismatch)."""
    d = json.loads(Path(path).read_text())
    try:
        pick = YawPick.from_dict(d)
    except (KeyError, TypeError, ValueError) as exc:
        raise WeightMismatch(f"{path}: not a pick file ({exc})") from exc
    if pick.weight not in YAW_WEIGHT_LADDER:
        raise WeightMismatch(f"{path}: weight {pick.weight} is not on the ladder {YAW_WEIGHT_LADDER}")
    if pick.b0 not in RUNGS:
        raise WeightMismatch(f"{path}: B0 {pick.b0} is not a rung {RUNGS}")
    if pick.crash_penalty != crash_penalty_for(pick.weight):
        raise WeightMismatch(f"{path}: crash penalty {pick.crash_penalty} is not the rule's {crash_penalty_for(pick.weight)} for {pick.weight}")
    try:
        again = pick_yaw_weight(pick.table)
    except ValueError as exc:
        raise WeightMismatch(f"{path}: the stored readings give no pick ({exc})") from exc
    if (again.weight, again.b0) != (pick.weight, pick.b0):
        raise WeightMismatch(f"{path}: the stored readings give ({again.weight}, {again.b0}), the file says ({pick.weight}, {pick.b0})")
    return YawPick(pick.weight, pick.b0, pick.crash_penalty, again.table)


@dataclass(frozen=True)
class LineageSelection:
    lineages: tuple = ()
    failures: tuple = ()
    tried: int = 0
    failed_on: Mapping = field(default_factory=dict)


def failed_gate(bar_ok: bool, yaw_ok: bool, motor_ok: bool = True) -> Optional[str]:
    """The label a failed seed is reported with: "bar", "yaw" or "both" (bar and yaw) as 4DA-16 names them; "motor" for the motor
    line alone (4DB-1); a motor failure beside the others is joined with "+": "bar+motor", "yaw+motor", "both+motor". None when
    every line passes."""
    if bar_ok and yaw_ok:
        first = None
    else:
        first = "both" if not bar_ok and not yaw_ok else ("bar" if not bar_ok else "yaw")
    if motor_ok:
        return first
    return "motor" if first is None else f"{first}+motor"


def select_lineages_with_yaw(readings: Sequence) -> LineageSelection:
    """readings: (nominal seed, BarReading, YawReading) or (nominal seed, BarReading, YawReading, MotorReading) in seed order. A
    seed passes when every line given passes (the motor line, ruling 4DB-1, when a fourth reading is given); seeds are tried in
    order until 16 pass; every failure is reported with the gate it failed (`failed_gate`); no count of failures stops the
    selection."""
    lineages, failures, failed_on, tried = [], [], {}, 0
    for item in readings:
        if len(item) not in (3, 4):
            raise ValueError(f"a reading is (seed, bar, yaw) or (seed, bar, yaw, motor), not {len(item)} items")
        seed, bar, yaw = item[0], item[1], item[2]
        motor = item[3] if len(item) > 3 else None
        if len(lineages) >= LINEAGES:
            break
        tried += 1
        label = failed_gate(bool(bar.passed), bool(yaw.passed), True if motor is None else bool(motor.passed))
        if label is None:
            lineages.append(int(seed))
        else:
            failures.append(int(seed))
            failed_on[int(seed)] = label
    return LineageSelection(tuple(lineages), tuple(failures), tried, failed_on)
