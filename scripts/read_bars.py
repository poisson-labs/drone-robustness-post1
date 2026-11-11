"""Read the bar for every flyer swept into a store, pick `B0` from the pilots, and select the lineages (spec section 2).

    uv run python scripts/read_bars.py STORE OUT.json [--pid pid] [--pilots 9001,9002,9003,9004] [--rungs 5000000,...] [--nominal 1,2,3,...]

Pilot runs are named pilot-<seed>-<steps>, nominal runs pi0-seed-<seed>; the PID's run is `pid`. The output holds every
reading (4A-11), `B0` by `b0_from_pilots` (4A-15: no rung with all four at the bar stops for a ruling) when every pilot
reading exists, and `select_lineages` over the nominal readings in seed order (a ninth failure stops for a ruling).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from poisson_prove.store import Store

from drone_study.training.bar import BarMismatch, StudyStop, read_bar, rows_from_store, select_lineages
from drone_study.training.budget import RUNGS, NoRungPasses, b0_from_pilots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("store")
    ap.add_argument("out")
    ap.add_argument("--pid", default="pid")
    ap.add_argument("--pilots", default="9001,9002,9003,9004")
    ap.add_argument("--rungs", default=",".join(str(r) for r in RUNGS))
    ap.add_argument("--nominal", default="")
    a = ap.parse_args()
    store = Path(a.store)
    st = Store(store)
    names = {st.manifest(r).name for r in st.runs()}
    pid_rows = rows_from_store(store, a.pid)
    result = {"pid_run": a.pid, "readings": {}, "b0": None, "selection": None, "stops": []}
    pilots = [int(x) for x in a.pilots.split(",") if x]
    rungs = [int(x) for x in a.rungs.split(",") if x]
    passed = {}
    for rung in rungs:
        flags = []
        for seed in pilots:
            name = f"pilot-{seed}-{rung}"
            if name in names:
                reading = read_bar(rows_from_store(store, name), pid_rows)
                result["readings"][name] = reading.to_dict()
                flags.append(reading.passed)
        if len(flags) == len(pilots):
            passed[rung] = flags
    if len(passed) == len(rungs):
        try:
            result["b0"] = b0_from_pilots(passed)
        except NoRungPasses as exc:
            result["stops"].append(str(exc))
    result["pilot_pass_table"] = {str(k): v for k, v in passed.items()}
    readings = []
    for seed in [int(x) for x in a.nominal.split(",") if x]:
        name = f"pi0-seed-{seed}"
        if name in names:
            reading = read_bar(rows_from_store(store, name), pid_rows)
            result["readings"][name] = reading.to_dict()
            readings.append((seed, reading))
        else:
            break                                                       # seeds are read in order; stop at the first not yet swept
    if readings:
        try:
            sel = select_lineages(readings)
            result["selection"] = {"lineages": list(sel.lineages), "failures": list(sel.failures), "tried": sel.tried}
        except StudyStop as exc:
            result["stops"].append(str(exc))
    Path(a.out).write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "readings"}, indent=1))
    for name, r in result["readings"].items():
        print(f"{name:22s} crashes {r['crashes']:3d} median {100 * r['median_rmse_m']:6.2f} cm  PID {100 * r['pid_median_rmse_m']:5.2f} cm "
              f"ratio {r['ratio']:.3f}  -> {'PASS' if r['passed'] else 'fail'} ({r['passes_crashes']}, {r['passes_rmse']}, {r['passes_ratio']})")


if __name__ == "__main__":
    main()
