"""The calm-air bar (spec section 2; N9 of section 20.1; the salt of section 20.3 point 10).

"All three, by code: at most 5 crashes in 1,200 episodes (12 cases x 100) ...; median per-episode tracking RMSE at most
10 cm; that RMSE at most 1.25 times the cascaded PID's on the same seeds." "Seeds are tried in order until 16 pass,
every failure is reported, and more than 8 failures stop the study for a ruling."

The bar's 1,200 episodes are the 100 seeds Prove derives for each of the 12 calm cells under the bar's salt
(76e7b463820f72ad), for the policy and for the PID alike. Readings (4A-11): the median is over all 1,200 per-episode
RMSEs, crashed episodes included; "at most" includes the line; the PID's crashes are reported and do not gate; an error
row stops the read instead of being dropped or counted.

Interface (4A-19):
  BAR_SALT, BAR_REPLICATES, MAX_CRASHES, MAX_RMSE_M, MAX_RMSE_RATIO, LINEAGES, MAX_FAILURES
  bar_conditions(), bar_seeds(condition), bar_sweep_spec(target, target_id)
  BarRow(case, seed, crashed, rmse, error=False);  rows_from_store(store_path, run)
  read_bar(policy_rows, pid_rows) -> BarReading;  select_lineages(readings) -> Selection;  BarMismatch, StudyStop
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from poisson_prove.executor import derive_seed
from poisson_prove.stats import wilson
from poisson_prove.store import Store, config_hash

from drone_study.cases import CASE_NAMES
from drone_study.condition import Condition, condition_from_cfg, condition_to_cfg, sweep_spec_toml

BAR_SALT: str = hashlib.sha256(b"drone-robustness-2026-11-04:bar").hexdigest()[:16]        # 76e7b463820f72ad
BAR_REPLICATES: int = 100
MAX_CRASHES: int = 5
MAX_RMSE_M: float = 0.10
MAX_RMSE_RATIO: float = 1.25
LINEAGES: int = 16
MAX_FAILURES: int = 8
BAR_EPISODES = 12 * BAR_REPLICATES


class BarMismatch(ValueError):
    """The rows are not the bar's sweep: wrong cells, wrong seeds, a missing or an error row, or two flyers on different seeds."""


class StudyStop(RuntimeError):
    """More than 8 nominal seeds failed the bar: the study waits on a ruling."""


@dataclass(frozen=True)
class BarRow:
    case: str
    seed: int
    crashed: bool
    rmse: float
    error: bool = False


@dataclass(frozen=True)
class BarReading:
    crashes: int = 0
    episodes: int = 0
    wilson: tuple = (0.0, 0.0)
    median_rmse: float = 0.0
    pid_median_rmse: float = 0.0
    pid_crashes: int = 0
    ratio: float = 0.0
    passes_crashes: bool = False
    passes_rmse: bool = False
    passes_ratio: bool = False
    passed: bool = False

    def to_dict(self) -> dict:
        return {"crashes": self.crashes, "episodes": self.episodes, "wilson_lo": self.wilson[0], "wilson_hi": self.wilson[1],
                "median_rmse_m": self.median_rmse, "pid_median_rmse_m": self.pid_median_rmse, "pid_crashes": self.pid_crashes,
                "ratio": self.ratio, "passes_crashes": self.passes_crashes, "passes_rmse": self.passes_rmse,
                "passes_ratio": self.passes_ratio, "passed": self.passed}


@dataclass(frozen=True)
class Selection:
    lineages: tuple = ()
    failures: tuple = ()
    tried: int = 0


def bar_conditions() -> list:
    return [Condition.calm(name) for name in CASE_NAMES]


def bar_seeds(condition: Condition) -> tuple:
    h = config_hash(condition_to_cfg(condition))
    return tuple(derive_seed(BAR_SALT, h, k) for k in range(BAR_REPLICATES))


def bar_sweep_spec(*, target: str, target_id: str) -> str:
    return sweep_spec_toml(bar_conditions(), target=target, target_id=target_id, replicates=BAR_REPLICATES)


def _expected_keys() -> list:
    return [(c.case, s) for c in bar_conditions() for s in bar_seeds(c)]


def rows_from_store(store_path, run: str) -> list:
    """A stored Prove run's rows as BarRows: the run named `run` in the store at `store_path`. An errored row reads back
    as an error (crashed False, RMSE NaN). Whether the rows are the bar's is `read_bar`'s question."""
    store = Store(Path(store_path))
    ids = [r for r in store.runs() if store.manifest(r).name == run or r == run]
    if len(ids) != 1:
        raise BarMismatch(f"store {store_path} has {len(ids)} runs named {run!r}")
    df = store.read(ids[0], meta=True)
    out = []
    for cfg, seed, outcome, error, meta in zip(df["config"], df["seed"], df["outcome"], df["error"], df["meta"]):
        case = condition_from_cfg(cfg).case
        if error is not None or outcome is None:
            out.append(BarRow(case, int(seed), False, float("nan"), error=True))
        else:
            rmse = meta.get("rmse") if isinstance(meta, dict) else None
            if rmse is None:
                out.append(BarRow(case, int(seed), outcome == "crash", float("nan"), error=True))
            else:
                out.append(BarRow(case, int(seed), outcome == "crash", float(rmse)))
    return out


def _checked(rows: Sequence[BarRow], who: str) -> dict:
    want = _expected_keys()
    got = [(r.case, int(r.seed)) for r in rows]
    if len(got) != len(want):
        raise BarMismatch(f"{who}: {len(got)} rows, the bar has {len(want)}")
    if len(set(got)) != len(got):
        raise BarMismatch(f"{who}: a (case, seed) row appears twice")
    if set(got) != set(want):
        raise BarMismatch(f"{who}: the rows are not the bar's 12 calm cells x 100 seeds under salt {BAR_SALT}")
    by_key = {}
    for r in rows:
        if r.error or not math.isfinite(float(r.rmse)):
            raise BarMismatch(f"{who}: row {r.case} seed {r.seed} is an error row; the bar is read only on 1,200 scored episodes")
        by_key[(r.case, int(r.seed))] = r
    return by_key


def read_bar(policy_rows: Sequence[BarRow], pid_rows: Sequence[BarRow]) -> BarReading:
    policy = _checked(policy_rows, "policy")
    pid = _checked(pid_rows, "PID")
    keys = _expected_keys()
    crashes = sum(1 for k in keys if policy[k].crashed)
    pid_crashes = sum(1 for k in keys if pid[k].crashed)
    median = float(np.median([policy[k].rmse for k in keys]))
    pid_median = float(np.median([pid[k].rmse for k in keys]))
    w = wilson(crashes, BAR_EPISODES)
    ratio = median / pid_median if pid_median > 0 else math.inf
    passes_crashes = crashes <= MAX_CRASHES
    passes_rmse = median <= MAX_RMSE_M
    passes_ratio = median <= MAX_RMSE_RATIO * pid_median
    return BarReading(crashes=crashes, episodes=BAR_EPISODES, wilson=(float(w.lo), float(w.hi)), median_rmse=median,
                      pid_median_rmse=pid_median, pid_crashes=pid_crashes, ratio=ratio, passes_crashes=passes_crashes,
                      passes_rmse=passes_rmse, passes_ratio=passes_ratio,
                      passed=bool(passes_crashes and passes_rmse and passes_ratio))


def select_lineages(readings: Sequence) -> Selection:
    """readings: (nominal seed, BarReading) in seed order. Seeds are tried in order until 16 pass; every failure is
    reported; a ninth failure stops the study (StudyStop), wherever it falls."""
    lineages, failures, tried = [], [], 0
    for seed, reading in readings:
        if len(lineages) >= LINEAGES:
            break
        tried += 1
        if reading.passed:
            lineages.append(int(seed))
        else:
            failures.append(int(seed))
            if len(failures) > MAX_FAILURES:
                raise StudyStop(f"{len(failures)} nominal seeds failed the bar ({failures}); more than {MAX_FAILURES} "
                                f"stops the study for a ruling")
    return Selection(tuple(lineages), tuple(failures), tried)
