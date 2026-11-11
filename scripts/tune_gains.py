"""Pick the geometric controller's gains on the frozen grid (spec section 17.2; ruling 4A-16, pick B).

    uv run python scripts/tune_gains.py OUTDIR [--workers N]

Flies every grid point (75) on the 12 calm cases x 8 seeds of the `tuning` salt (7,200 flights of the plain geometric
controller, no policy, no wind), in parallel processes, and writes OUTDIR/tuning-flights.json (every flight: gains, case,
seed, RMSE, crash class), OUTDIR/tuning-pick.json (the pick, its median RMSE and every point's median and crash count) and
prints the pick. The pick becomes `controllers.DEFAULT_GAINS`, shared by the geometric controller, the wind-only reference
and the feasibility oracle; the session records it, no test can know it before the flights.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from drone_study.controllers import GeometricController, GeometricGains
from drone_study.training.tuning import GAIN_GRID, TUNING_REPLICATES, TUNING_SALT, pick_gains, tuning_conditions, tuning_figure, tuning_seeds


def _fly(args):
    gains, jobs = args
    from drone_study.rollout import tracking_rmse
    from drone_study.score import score
    from drone_study.stack import fly

    out = []
    ctl = GeometricController(GeometricGains(*gains))
    for case_name, cond, seed in jobs:
        rec = fly(cond, int(seed), ctl)
        o = score(rec.trajectory())
        out.append({"kp": gains[0], "kv": gains[1], "ki": gains[2], "case": case_name, "seed": int(seed),
                    "rmse": tracking_rmse(rec), "crashed": not o.passed, "crash_class": o.crash_class,
                    "crash_time": o.crash_time, "steps": int(rec.n_steps), "end": rec.end_reason})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 16)
    a = ap.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    conds = tuning_conditions()
    jobs = [(c.case, c, s) for c in conds for s in tuning_seeds(c)]
    assert len(jobs) == 12 * TUNING_REPLICATES
    tasks = [((g.kp, g.kv, g.ki), jobs) for g in GAIN_GRID]
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for i, part in enumerate(pool.map(_fly, tasks)):
            rows.extend(part)
            g = tasks[i][0]
            med = float(np.median([r["rmse"] for r in part]))
            crashes = sum(r["crashed"] for r in part)
            print(f"({g[0]:4.1f}, {g[1]:4.1f}, {g[2]:3.1f}) median {100 * med:5.2f} cm crashes {crashes:2d}   "
                  f"[{i + 1}/{len(tasks)}, {time.time() - t0:.0f} s]", flush=True)
    (out / "tuning-flights.json").write_text(json.dumps({"salt": TUNING_SALT, "replicates": TUNING_REPLICATES,
                                                        "flights": rows}, indent=0))
    results = {}
    for r in rows:
        results.setdefault(GeometricGains(r["kp"], r["kv"], r["ki"]), []).append((r["rmse"], r["crashed"]))
    pick = pick_gains(results)
    figure = tuning_figure(results)
    table = []
    for g in GAIN_GRID:
        rs = results[g]
        table.append({"kp": g.kp, "kv": g.kv, "ki": g.ki, "median_rmse_m": float(np.median([x for x, _ in rs])),
                      "mean_rmse_m": float(np.mean([x for x, _ in rs])), "crashes": int(sum(c for _, c in rs)),
                      "episodes": len(rs)})
    summary = {"pick": {"kp": pick.kp, "kv": pick.kv, "ki": pick.ki}, "tuning_figure_m": figure, "salt": TUNING_SALT,
               "replicates": TUNING_REPLICATES, "flights": len(rows), "seconds": time.time() - t0, "grid": table}
    (out / "tuning-pick.json").write_text(json.dumps(summary, indent=1))
    print(f"\nPICK ({pick.kp}, {pick.kv}, {pick.ki}) median RMSE {100 * figure:.2f} cm over 96 calm episodes; "
          f"{len(rows)} flights in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
