"""SHA-256 of every stored run of post 1, and the raw-output manifest P1a recomputes from (spec 12 row 5B, 20.6).

    uv run python scripts/hash_runs.py --store STORE --out runs.json

For each run in the store: its name, id, status, row count and the SHA-256 of every file in its folder (manifest, event log, episode
parts), and one hash over the run's files in path order. Read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from poisson_prove.store import Store


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.store).resolve()
    store = Store(root)
    runs = []
    for run_id in sorted(store.runs(), key=lambda r: store.manifest(r).name or r):
        m = store.manifest(run_id)
        files = {str(p.relative_to(root / run_id)): sha256(p) for p in sorted((root / run_id).rglob("*")) if p.is_file()}
        combined = hashlib.sha256("".join(f"{k}:{v}\n" for k, v in files.items()).encode()).hexdigest()
        runs.append({"name": m.name, "run_id": run_id, "status": str(m.status), "salt": str(m.run_salt), "rows": int(store.count(run_id)),
                     "files": files, "run_sha256": combined})
    Path(args.out).write_text(json.dumps({"store": str(root), "runs": runs}, indent=1) + "\n")
    print(f"{len(runs)} runs, {sum(r['rows'] for r in runs)} rows hashed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
