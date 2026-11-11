"""Ruling 4DB-2's re-read: every nominal seed tried (pi0 1 to 21), the 0.032 pilots at every rung and the two controllers flown
again through the bar's Prove sweep, now that the rollout writes the sub-step share, into a store of their own; then both motor
readings side by side, a check that every flight is the stored bar's flight bit for bit, and the selection on the line's reading.

    uv run python scripts/reread_4db2.py sweep [--parallel 6]
    uv run python scripts/reread_4db2.py read                  # writes .work/4d/4db2/readings.json and bar-match.json
    uv run python scripts/reread_4db2.py lineages              # adds the sub-step readings to data/4d-lineages.json

The sweeps are the bar's own (12 calm cells x 100 seeds, the bar's salt), so each episode is the one 4D-B stored in
.work/4d/bar-store; reading the substeps changes nothing in the flight, and `read` checks that on every row (outcome, crash
time, RMSE, rotor effort, the four heading numbers and the end-of-step share, compared exactly; a field the old run
never stored, the end-of-step share on runs swept before 4DB-1, is listed per run instead). A sweep already complete in
the new store is not flown again.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
WORK = ROOT / ".work" / "4d"
OLD = WORK / "bar-store"
NEW = WORK / "bar-store-4db2"
OUT = WORK / "4db2"
RUNS = WORK / "runs"
NOMINAL = range(1, 22)
PILOTS = (9001, 9002, 9003, 9004)
RUNGS = (5_000_000, 10_000_000, 20_000_000, 40_000_000)
LINEAGES_FILE = ROOT / "data" / "4d-lineages.json"
COMPARED = ("rmse", "rotor_effort", "end_reason", "steps", "crash_class", "yaw_rate_rms_deg_s", "heading_rate_rms_deg_s",
            "yaw_drift_deg_s", "max_abs_yaw_deg", "motor_limit_share")


def flyers() -> list:
    out = [("pid", "pid", None), ("geometric", "geometric", None)]
    out += [(f"pi0-seed-{s}", "policy", RUNS / f"nominal-{s}" / "checkpoint-5000000.zip") for s in NOMINAL]
    out += [(f"pilot-0.032-{s}-{r}", "policy", RUNS / f"pilot-0.032-{s}" / f"checkpoint-{r}.zip") for r in RUNGS for s in PILOTS]
    for name, _, ck in out:
        if ck is not None and not ck.exists():
            raise SystemExit(f"{name}: no checkpoint at {ck}")
    return out


def complete_runs(store: Path) -> dict:
    from poisson_prove.store import Store

    if not store.exists():
        return {}
    st = Store(store)
    out = {}
    for rid in st.runs():
        man = st.manifest(rid)
        if getattr(man, "status", None) in (None, "complete", "completed", "done", "finished"):
            out[man.name] = rid
    return out


def sweep(parallel: int) -> None:
    done = complete_runs(NEW)
    todo = [f for f in flyers() if f[0] not in done]
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{len(done)} complete, {len(todo)} to fly, {parallel} at a time", flush=True)

    def one(f):
        name, target, ck = f
        cmd = ["uv", "run", "python", "scripts/bar_sweep.py", str(NEW), "--name", name, "--target", target]
        if ck is not None:
            cmd += ["--checkpoint", str(ck)]
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        (OUT / f"sweep-{name}.log").write_text(r.stdout + "\n" + r.stderr)
        print(f"{'swept' if r.returncode == 0 else 'FAILED'} {name}", flush=True)
        return r.returncode

    with ThreadPoolExecutor(parallel) as ex:
        codes = list(ex.map(one, todo))
    if any(codes):
        raise SystemExit("a sweep failed; see .work/4d/4db2/sweep-*.log")


def _rows(store: Path, rid: str) -> dict:
    from poisson_prove.store import Store

    from drone_study.condition import condition_from_cfg

    df = Store(store).read(rid, meta=True)
    out = {}
    for cfg, seed, outcome, t, error, meta in zip(df["config"], df["seed"], df["outcome"], df["t"], df["error"], df["meta"]):
        out[(condition_from_cfg(cfg).case, int(seed))] = (outcome, t, error, meta)
    return out


def _same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def read() -> None:
    from drone_study.training.bar import read_bar, rows_from_store
    from drone_study.training.behaviour import (motor_acceptance, motor_rows_from_store, motor_substep_rows_from_store,
                                                yaw_acceptance, yaw_rows_from_store)

    old, new = complete_runs(OLD), complete_runs(NEW)
    readings, match = {}, {}
    pid_rows = rows_from_store(NEW, "pid")
    for name, _, _ in flyers():
        if name not in new:
            raise SystemExit(f"{name} is not swept into {NEW}")
        r = {"bar": read_bar(rows_from_store(NEW, name), pid_rows).to_dict(),
             "yaw": yaw_acceptance(yaw_rows_from_store(NEW, name)).to_dict(),
             "motor_end_of_step": motor_acceptance(motor_rows_from_store(NEW, name)).to_dict(),
             "motor_substep": motor_acceptance(motor_substep_rows_from_store(NEW, name)).to_dict()}
        readings[name] = r
        a, b = _rows(OLD, old[name]), _rows(NEW, new[name])
        diffs, not_stored = [], set()
        if set(a) != set(b):
            diffs.append("the rows' (case, seed) keys differ")
        for k in sorted(set(a) & set(b)):
            oa, ta, ea, ma = a[k]
            ob, tb, eb, mb = b[k]
            if oa != ob or not _same(ta, tb) or ea != eb:
                diffs.append(f"{k}: outcome/t/error")
                continue
            for f in COMPARED:
                if f not in (ma or {}):                          # the old run was swept before this field existed (4DB-1)
                    not_stored.add(f)
                    continue
                if not _same((ma or {}).get(f), (mb or {}).get(f)):
                    diffs.append(f"{k}: {f} {(ma or {}).get(f)!r} != {(mb or {}).get(f)!r}")
        match[name] = {"rows": len(b), "old_run": old[name], "new_run": new[name], "differences": len(diffs),
                       "first": diffs[:5], "compared": [f for f in COMPARED if f not in not_stored],
                       "not_in_old_run": sorted(not_stored)}
        print(f"{name}: end-of-step {100 * r['motor_end_of_step']['motor_limit_share']:.4f}%  "
              f"any-substep {100 * r['motor_substep']['motor_limit_share']:.4f}%  "
              f"line {'pass' if r['motor_substep']['passed'] else 'FAIL'}  bar-match diffs {len(diffs)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "readings.json").write_text(json.dumps(readings, indent=1, sort_keys=True) + "\n")
    (OUT / "bar-match.json").write_text(json.dumps(match, indent=1, sort_keys=True) + "\n")
    bad = [n for n, m in match.items() if m["differences"]]
    print(f"bar match: {len(match) - len(bad)} of {len(match)} runs identical row for row" + (f"; DIFFER: {bad}" if bad else ""))


def lineages() -> None:
    from drone_study.training.bar import BarReading
    from drone_study.training.behaviour import MOTOR_LINE_READING, MotorReading, YawReading
    from drone_study.training.yaw_weight import select_lineages_with_yaw

    readings = json.loads((OUT / "readings.json").read_text())
    rows = []
    for s in NOMINAL:
        r = readings[f"pi0-seed-{s}"]
        rows.append((s, BarReading(passed=bool(r["bar"]["passed"])), YawReading(passed=bool(r["yaw"]["passed"])),
                     MotorReading(passed=bool(r["motor_substep"]["passed"]), reading=MOTOR_LINE_READING)))
    sel = select_lineages_with_yaw(rows)
    doc = json.loads(LINEAGES_FILE.read_text())
    before = doc["selection"]
    doc["motor_line_reading"] = MOTOR_LINE_READING
    doc["motor_readings_substep"] = {f"pi0-seed-{s}": readings[f"pi0-seed-{s}"]["motor_substep"] for s in NOMINAL}
    doc["selection_end_of_step"] = before
    doc["selection"] = {"lineages": list(sel.lineages), "failures": list(sel.failures), "tried": sel.tried,
                        "failed_on": {str(k): v for k, v in sel.failed_on.items()}}
    LINEAGES_FILE.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    same = before["lineages"] == doc["selection"]["lineages"]
    print(f"selection on {MOTOR_LINE_READING}: lineages {list(sel.lineages)} failed_on {dict(sel.failed_on)} tried {sel.tried}"
          f" ({'unchanged' if same else 'CHANGED'} from end-of-step)")
    if len(sel.lineages) < 16:
        raise SystemExit(f"only {len(sel.lineages)} lineages pass on {MOTOR_LINE_READING}: the next seed trains")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["sweep", "read", "lineages"])
    ap.add_argument("--parallel", type=int, default=6)
    a = ap.parse_args()
    {"sweep": lambda: sweep(a.parallel), "read": read, "lineages": lineages}[a.command]()


if __name__ == "__main__":
    main()
