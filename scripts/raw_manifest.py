"""List the raw outputs session 4B keeps out of git, with sizes and SHA-256s (spec section 15; the brief's manifest).

    uv run python scripts/raw_manifest.py ROOT OUT.json

Walks ROOT (the git-excluded .work/4b/) and writes every file's relative path, size and SHA-256, so the vault can name
what exists where without carrying checkpoints or Prove stores.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    rows = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.name.startswith("."):
            h = hashlib.sha256()
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            rows.append({"path": str(p.relative_to(root)), "bytes": p.stat().st_size, "sha256": h.hexdigest()})
    out.write_text(json.dumps({"root": str(root), "files": len(rows), "bytes": sum(r["bytes"] for r in rows), "rows": rows}, indent=1))
    print(f"{len(rows)} files, {sum(r['bytes'] for r in rows) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
