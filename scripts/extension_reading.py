"""The extension's first reading (spec 3 "Grid extension", N10; ruling 5A-8, 5A-13): the mean pi0 crash rate at 12 m/s on each of
the 16 directions, from Zone B's rows alone, and whether any is strictly under 80%.

    uv run python scripts/extension_reading.py --store STORE --out extension-reading.json

Reads only the 16 pi0 Zone B runs and prints only the 12 m/s rates, the verdict and the row counts it stood on: no other speed, no
controller, no anchor row, no contrast. It refuses to read a run that is not complete. Run it before anything else of Zone B is read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from drone_study.boundary import results, sweeps, writer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", action="append", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pi0 = [f.name for f in sweeps.flyers() if f.target == "policy"]
    rows = writer.load_rows(args.store, flyers=pi0, zones=("zone-b",))
    rows = rows[rows["cell"].map(lambda c: float(c[2]) == 12.0)]          # nothing but the 12 m/s rows is kept in hand
    base = tuple(float(v) for v in sweeps.ZONE_B_SPEEDS)
    grids = []
    for f in pi0:
        sub = rows[rows["flyer"] == f]
        # flyer_grid wants every speed; the other speeds are filled with a placeholder cell count that this reading never touches
        g = results.flyer_grid(sub, f, speeds=(12.0,))
        grids.append(results.FlyerGrid(f, (12.0,), g.directions, g.crashes, g.n, g.classes))
    reading = results.n10_extension(grids)
    out = {
        "rule": "add 16 m/s if, on any of the 16 directions, the mean pi0 crash rate at 12 m/s is strictly under 80% (spec 3; ruling 5A-8)",
        "pi0": pi0, "rows_read": int(len(rows)), "scored_rows_read": int((~rows["error"]).sum()), "error_rows": int(rows["error"].sum()),
        "rate_at_12": {f"az {d[0]:g} el {d[1]:g}": float(r) for d, r in zip(reading["directions"], reading["rate_at_12"])},
        "lowest_rate": float(np.min(reading["rate_at_12"])), "fires": bool(reading["fires"]),
        "speeds_after": (list(base) + [sweeps.EXTENSION_SPEED]) if reading["fires"] else list(base),
    }
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
