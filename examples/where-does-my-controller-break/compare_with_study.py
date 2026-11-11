"""Does what you flew equal what the study stored, row by row?

    uv run python examples/where-does-my-controller-break/compare_with_study.py --flyer pid [--store prove-store] [--out compare-pid.json]

For a flyer the study flew (pid, geometric or pi0-<seed>), every episode in the example's Prove store is looked up in the
study's stored rows (data/rows) by its cell and seed and compared field by field: the outcome, its time, the crash class,
every threshold's first crossing, the tracking RMSE, the rotor effort, the end reason, the step count and the motor-limit
readings. A classical controller is deterministic here, so every row should be equal; a trained policy is flown through
PyTorch, and the report says how many rows are equal rather than requiring it. Read-only.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from poisson_prove.store import Store

from drone_study.boundary import sweeps as W

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flyer", default="pid")
    ap.add_argument("--store", default=str(HERE / "prove-store"))
    ap.add_argument("--rows", default=str(ROOT / "data" / "rows"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    study_name = {"pid": "pid", "geometric": "geometric"}.get(a.flyer) or (f"pi0-seed-{a.flyer.split('-')[1]}" if re.fullmatch(r"pi0-\d+", a.flyer) else None)
    if study_name is None:
        raise SystemExit("--flyer is pid, geometric or pi0-<seed>: the flyers the study stored")
    tag = a.flyer.replace(":", "-").replace(".", "-")
    mine = Store(Path(a.store))
    theirs = Store(Path(a.rows))
    their_runs = {theirs.manifest(r).name: r for r in theirs.runs()}
    stored_rows: dict = {}                      # (cell name, seed) -> the study's row, read once per run
    for zone in ("zone-b", "zone-b-extension"):
        for s in theirs.read(their_runs[f"{zone}-{study_name}"], meta=True).to_dict("records"):
            stored_rows[(s["config"]["cell"], int(s["seed"]))] = s
    report = {"flyer": a.flyer, "study_flyer": study_name, "runs": [], "rows": 0, "equal": 0, "same_outcome": 0, "fields": {}}
    for run in sorted(mine.runs(), key=lambda r: mine.manifest(r).name or r):
        name = mine.manifest(run).name or ""
        if not name.startswith(f"where-{tag}-"):
            continue
        rows = mine.read(run, meta=True).to_dict("records")
        n_equal = 0
        for row in rows:
            cfg = row["config"]
            s = stored_rows.get((cfg["cell"], int(row["seed"])))
            if s is None:
                raise SystemExit(f"{name}: no stored row for {cfg} seed {row['seed']}")
            fields = {"outcome": same(row["outcome"], s["outcome"]), "t": same(float(row["t"]) if row["t"] is not None else None, float(s["t"]) if s["t"] is not None else None)}
            for k in sorted(set(row["meta"]) | set(s["meta"])):
                fields[f"meta.{k}"] = k in row["meta"] and k in s["meta"] and same(row["meta"][k], s["meta"][k])
            for k, ok in fields.items():
                f = report["fields"].setdefault(k, {"equal": 0, "differ": 0})
                f["equal" if ok else "differ"] += 1
            report["rows"] += 1
            report["same_outcome"] += fields["outcome"]
            if all(fields.values()):
                report["equal"] += 1
                n_equal += 1
        report["runs"].append({"name": name, "rows": len(rows), "equal": n_equal})
        print(f"{name}: {n_equal} of {len(rows)} rows equal to the study's on every field")
    if not report["rows"]:
        raise SystemExit(f"no run of {a.flyer} in {a.store}; fly it first with where_does_my_controller_break.py")
    print(f"\n{a.flyer}: {report['equal']} of {report['rows']} rows equal on every field; {report['same_outcome']} of {report['rows']} with the same outcome")
    differ = {k: v["differ"] for k, v in report["fields"].items() if v["differ"]}
    print("every field equal" if not differ else f"fields that differ on some rows: {differ}")
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1) + "\n")
        print(f"wrote {a.out}")
    return 0 if report["equal"] == report["rows"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
