"""Verify post 1's pairing, and that every stored row carries the seed the salt derives (spec 6, 12 row 5B "pairing verified by
Prove on every pair"; ruling 5A-13).

    uv run python scripts/verify_pairing.py --store STORE --out pairing.json [--zones zone-b anchor zone-b-extension]

Two checks per zone. (1) For every run: it is complete, it holds every cell for the zone's replicates, and each row's seed is
`derive_seed(salt, config_hash, replicate_index)` (`sweeps.cell_seeds`), so the seeds are the zone's and the same for every flyer.
(2) For every pair of flyers (190 pairs of the 20), Prove's own seed check: `poisson-prove diff A B --method mcnemar_bh --json`, of which
only two flags are kept, `seed_identity_match` (must be true) and `unpaired_analysis` (must be false). Nothing else of the diff is read or
written: post 1 prints no flag, word or p-value from it (spec 20.2, 20.4). Exit status 0 only if every check holds.
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from poisson_prove.store import Store, config_hash

from drone_study.boundary import sweeps
from drone_study.condition import condition_to_cfg

ZONES = {
    "zone-b": (sweeps.zone_b_conditions, sweeps.ZONE_B_REPLICATES),
    "anchor": (sweeps.anchor_conditions, sweeps.ANCHOR_REPLICATES),
    "zone-b-extension": (sweeps.extension_conditions, sweeps.ZONE_B_REPLICATES),
}


def check_run(store: Store, zone: str, flyer: str) -> dict:
    name = sweeps.run_name(zone, flyer)
    ids = [r for r in store.runs() if store.manifest(r).name == name]
    rec = {"run": name, "found": len(ids)}
    if len(ids) != 1:
        return {**rec, "ok": False, "why": f"{len(ids)} runs named {name}"}
    run_id = ids[0]
    manifest = store.manifest(run_id)
    conds, reps = ZONES[zone]
    conds = conds()
    want = {}
    for c in conds:
        h = config_hash(condition_to_cfg(c))
        for k, s in enumerate(sweeps.cell_seeds(sweeps.salt_for(zone), c, reps)):
            want[(h, k)] = s
    df = store.read(run_id, meta=False)
    got = {(h, int(k)): int(s) for h, k, s in zip(df["config_hash"], df["replicate_index"], df["seed"])}
    errors = int(df["error"].notna().sum()) if "error" in df.columns else 0
    rec.update(run_id=run_id, status=str(manifest.status), rows=len(df), expected_rows=len(want), error_rows=errors,
               salt=str(manifest.run_salt), cells=len(conds), replicates=reps)
    complete = str(manifest.status) in ("complete", "completed")
    same_keys = set(got) == set(want) and len(got) == len(df)
    same_seeds = same_keys and all(got[k] == want[k] for k in want)
    salt_ok = str(manifest.run_salt) == sweeps.salt_for(zone)
    return {**rec, "ok": bool(complete and same_keys and same_seeds and salt_ok),
            "complete": complete, "keys_exact": same_keys, "seeds_are_the_derived_ones": same_seeds, "salt_is_the_zones": salt_ok}


def check_pair(store_path: Path, zone: str, a: str, b: str) -> dict:
    cmd = [sys.executable, "-m", "poisson_prove", "--store", str(store_path), "diff", sweeps.run_name(zone, a), sweeps.run_name(zone, b),
           "--method", "mcnemar_bh", "--json", "--plain"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return {"zone": zone, "a": a, "b": b, "ok": False, "why": f"diff exited {r.returncode}: {r.stderr[-200:]}"}
    payload = json.loads(r.stdout)
    match, unpaired = payload.get("seed_identity_match"), payload.get("unpaired_analysis")
    return {"zone": zone, "a": a, "b": b, "seed_identity_match": match, "unpaired_analysis": unpaired, "ok": match is True and unpaired is False}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--zones", nargs="+", default=["zone-b", "anchor"], choices=sorted(ZONES))
    ap.add_argument("--flyers", nargs="*")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    store_path = Path(args.store).resolve()
    store = Store(store_path)
    names = args.flyers or [f.name for f in sweeps.flyers()]
    runs, pairs = [], []
    for zone in args.zones:
        runs += [{"zone": zone, **check_run(store, zone, f)} for f in names]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futs = [pool.submit(check_pair, store_path, z, a, b) for z in args.zones for a, b in itertools.combinations(names, 2)]
        pairs = [f.result() for f in futs]
    summary = {"zones": args.zones, "flyers": len(names), "runs_checked": len(runs), "runs_ok": sum(r["ok"] for r in runs),
               "pairs_checked": len(pairs), "pairs_ok": sum(p["ok"] for p in pairs)}
    Path(args.out).write_text(json.dumps({"summary": summary, "runs": runs, "pairs": pairs}, indent=1) + "\n")
    print(json.dumps(summary))
    return 0 if summary["runs_ok"] == summary["runs_checked"] and summary["pairs_ok"] == summary["pairs_checked"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
