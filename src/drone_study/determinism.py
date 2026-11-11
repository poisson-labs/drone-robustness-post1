"""The determinism branch of spec section 6, as a measurement that prints its table (inv. 9).

Each row is one configuration (worker count, threads per worker) replaying the SAME (condition, seed) episodes as
the reference configuration. `max_abs_dev_m` is the largest position difference from the reference over those
episodes and all times; `n_flips` counts episodes whose outcome class differs.

classify(rows) applies section 6's table:
  "identical"            every row within 1e-9 m
  "pin-one-thread"       not identical, but every row with threads_per_worker == 1 is within 1e-9 m
  "report-noise-floor"   otherwise, and total flips <= 0.5% of total replayed episodes
  "pairing-buys-less"    otherwise, flips > 0.5%

How `measure` runs. The reference configuration is ONE worker process with one thread (the numerical libraries' thread
variables set to 1 before the process starts). Every other configuration is a fresh pool of `workers` spawned processes
whose thread variables are set to `threads_per_worker` before they start, so each replay is a different process from the
reference's (a (1, 1) row is the same-configuration replay in another process). Every episode is flown by the same
controller (default: the geometric controller on the normal stream) in evaluation mode, and its 100 Hz position record and
its `score()` class come back to the parent, which compares them with the reference's.
"""
from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import numpy as np

BRANCHES = ("identical", "pin-one-thread", "report-noise-floor", "pairing-buys-less")
IDENTICAL_M = 1e-9
FLIP_RATE_NUM, FLIP_RATE_DEN = 1, 200            # 0.5%, in integers so the boundary is exact
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                    "NUMEXPR_NUM_THREADS")


@dataclass(frozen=True)
class DeterminismRow:
    label: str
    workers: int
    threads_per_worker: int
    n_episodes: int
    max_abs_dev_m: float
    n_flips: int


@dataclass(frozen=True)
class DeterminismReport:
    rows: tuple
    branch: str

    def table_text(self) -> str:
        head = ("configuration", "workers", "threads", "episodes", "max |dx| vs reference (m)", "outcome flips")
        body = [(r.label, str(r.workers), str(r.threads_per_worker), str(r.n_episodes), f"{r.max_abs_dev_m:.3e}",
                 f"{r.n_flips}") for r in self.rows]
        widths = [max(len(h), *(len(b[i]) for b in body)) if body else len(h) for i, h in enumerate(head)]
        fmt = lambda cells: "  ".join(c.ljust(w) for c, w in zip(cells, widths)).rstrip()
        lines = [fmt(head), fmt(tuple("-" * w for w in widths))] + [fmt(b) for b in body]
        total = sum(r.n_episodes for r in self.rows)
        flips = sum(r.n_flips for r in self.rows)
        lines.append("")
        lines.append(f"flips: {flips} of {total} replayed episodes; worst deviation "
                     f"{max((r.max_abs_dev_m for r in self.rows), default=0.0):.3e} m")
        lines.append(f"branch (spec section 6): {self.branch}")
        return "\n".join(lines)


def classify(rows) -> str:
    rows = list(rows)
    if all(r.max_abs_dev_m <= IDENTICAL_M for r in rows):
        return "identical"
    one_thread = [r for r in rows if r.threads_per_worker == 1]
    if one_thread and all(r.max_abs_dev_m <= IDENTICAL_M for r in one_thread):
        return "pin-one-thread"
    total = sum(r.n_episodes for r in rows)
    flips = sum(r.n_flips for r in rows)
    if flips * FLIP_RATE_DEN <= total * FLIP_RATE_NUM:
        return "report-noise-floor"
    return "pairing-buys-less"


# ---- the measurement ------------------------------------------------------------------------------------------------------
def _fly_one(task):
    """Top level so a spawned worker can import it. Returns the 100 Hz positions and the score() class."""
    from drone_study.score import score
    from drone_study.stack import fly

    condition, seed, controller = task
    record = fly(condition, seed, controller)
    return np.array(record.position), score(record.trajectory()).crash_class


@contextmanager
def _threads(n: int):
    saved = {k: os.environ.get(k) for k in THREAD_VARIABLES}
    for k in THREAD_VARIABLES:
        os.environ[k] = str(n)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _run(tasks, workers: int, threads: int):
    ctx = multiprocessing.get_context("spawn")
    with _threads(threads), ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        return list(pool.map(_fly_one, tasks, chunksize=1))


def measure(conditions, seeds, configs, *, controller=None) -> DeterminismReport:
    """Fly every (condition, seed) under the reference configuration and under each (workers, threads) in configs,
    in separate worker processes, and return the report."""
    if controller is None:
        from drone_study.controllers import GeometricController

        controller = GeometricController()
    tasks = [(c, int(s), controller) for c in conditions for s in seeds]
    reference = _run(tasks, workers=1, threads=1)
    rows = []
    for workers, threads in configs:
        replay = _run(tasks, workers=workers, threads=threads)
        dev, flips = 0.0, 0
        for (p_ref, cls_ref), (p, cls) in zip(reference, replay):
            n = min(len(p_ref), len(p))
            dev = max(dev, float(np.max(np.abs(p_ref[:n] - p[:n]))) if n else 0.0)
            if len(p_ref) != len(p):
                dev = max(dev, float("inf"))
            flips += int(cls_ref != cls)
        label = f"{workers} worker{'s' if workers != 1 else ''}, {threads} thread{'s' if threads != 1 else ''}"
        rows.append(DeterminismRow(label, workers, threads, len(tasks), dev, flips))
    return DeterminismReport(tuple(rows), classify(rows))
