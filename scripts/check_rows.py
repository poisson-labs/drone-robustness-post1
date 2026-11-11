"""Check the stored rows in data/rows against the hashes session 5B recorded when it flew them (data/rows-hashes.json).

    uv run python scripts/check_rows.py [--store data/rows] [--hashes data/rows-hashes.json]

Every file of every run (manifest, event log, episode parts) is hashed and compared; a run missing, a file missing, a file
changed or a file added inside a run is reported, and the exit status is 1 on any of them. Read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="data/rows")
    ap.add_argument("--hashes", default="data/rows-hashes.json")
    a = ap.parse_args()
    root = Path(a.store)
    want = json.loads(Path(a.hashes).read_text())["runs"]
    bad, files, rows = [], 0, 0
    for run in sorted(want, key=lambda r: r["name"]):
        folder = root / run["run_id"]
        if not folder.is_dir():
            bad.append(f"{run['name']}: run folder {run['run_id']} is missing")
            continue
        present = {str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file()}
        for rel, digest in run["files"].items():
            files += 1
            if rel not in present:
                bad.append(f"{run['name']}: {rel} is missing")
            elif sha256(folder / rel) != digest:
                bad.append(f"{run['name']}: {rel} differs from the stored hash")
        for rel in sorted(present - set(run["files"])):
            bad.append(f"{run['name']}: {rel} was not in the run when it was hashed")
        rows += int(run["rows"])
    print(f"{len(want)} runs, {files} files, {rows} rows: " + ("all match the hashes of 2026-11-09" if not bad else f"{len(bad)} problems"))
    for line in bad:
        print("  " + line)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
