"""tests/data/4d-lineages-shape.json: a stored lineage file in session 4D-B's shape, for 5A's N9 tests (the yaw addendum of
2026-11-08, 4DR-5). It is 4B's `bars.json` (tests/data/4b-bars.json, 4B's bar readings verbatim) with the fields 4D-B's file adds:
`k_yaw`, `crash_penalty`, `yaw_readings` (synthetic, one per lineage, every line passing) and `selection.failed_on`. It is a
fixture of "a stored lineage file", not a 4D-B result: 4D-B writes the real one to data/4d-lineages.json.

  python scripts/make_4d_lineages_shape.py
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = json.loads((ROOT / "tests" / "data" / "4b-bars.json").read_text())
out = {k: src[k] for k in ("pid_run", "readings", "b0", "selection", "stops", "pilot_pass_table")}
out["k_yaw"] = 0.008
out["crash_penalty"] = 20.0
out["selection"] = dict(out["selection"], failed_on={})
out["yaw_readings"] = {}
for seed in range(1, 17):
    out["yaw_readings"][f"pi0-seed-{seed}"] = {
        "episodes": 1200, "yaw_rate_rms_median_deg_s": round(4.0 + 0.1 * seed, 3), "heading_rate_rms_median_deg_s": round(6.0 + 0.3 * seed, 3),
        "drift_abs_median_deg_s": round(0.03 + 0.01 * seed, 3),
        "past_45": 1 if seed % 7 == 0 else 0, "past_90": 0, "max_heading_median_deg": round(6.0 + 0.2 * seed, 3),
        "passes_rate": True, "passes_drift": True, "passes_45": True, "passes_90": True, "passes_heading": True, "passed": True,
    }
path = ROOT / "tests" / "data" / "4d-lineages-shape.json"
path.write_text(json.dumps(out, indent=1, sort_keys=True))
print(path, path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
