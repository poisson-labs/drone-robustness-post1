"""Where does my controller break?

Sweep a quadrotor controller through wind gusts from every side with Poisson Prove, and draw the gust speed at which half of
its flights crash, direction by direction: a boundary map. It runs from a clone of this repository, on the study's own grid,
air and seeds, and sets the map it draws beside the sixteen trained policies' and the cascaded PID's maps from RESULTS-P1.md.

    uv sync                                   # once, at the repository's root
    uv run python examples/where-does-my-controller-break/where_does_my_controller_break.py --flyer pid --quick --jobs 2
    uv run python examples/where-does-my-controller-break/where_does_my_controller_break.py --flyer my_controller:rollout --quick --jobs 2
    uv run python examples/where-does-my-controller-break/where_does_my_controller_break.py --flyer pid --jobs 8

  --flyer   pid or geometric (the study's two classical controllers), pi0-<seed> (one of the sixteen trained policies, which
            needs `uv sync --group gate` for PyTorch), or "module:name" of your own rollout (see my_controller.py)
  --quick   two directions, a headwind and a tailwind: 1,056 episodes, a few minutes on two cores
  (default) the eight level directions: 4,224 episodes
  --all     all sixteen directions, the study's whole grid: 8,448 episodes
  --jobs    how many directions to fly at once, one process each
  --store   where Prove keeps the episodes (default: prove-store beside this script); flying again picks up what is stored

In simulation, with these limits: one drone model, one air model, one trajectory and these gusts. A speed in m/s does not
carry to a real drone or to another air model, which is why every speed is printed with its flat force ratio. Twenty-four
episodes a cell cannot show that a direction is safe, and nothing here is a test between two controllers.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from poisson_prove.store import Store

from drone_study.analysis import estimator
from drone_study.boundary import sweeps as W
from drone_study.condition import Condition, condition_from_cell, sweep_spec_toml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT / "RESULTS-P1.md"
SPEEDS = tuple(W.ZONE_B_SPEEDS) + (W.EXTENSION_SPEED,)         # 2 to 14 m/s, and the 16 m/s row the extension added
TIMINGS = tuple(W.ZONE_B_PHASES)
LEVEL = tuple(d for d in W.DIRECTIONS if d[1] == 0.0)
EPISODES = W.ZONE_B_REPLICATES
TARGETS = "drone_study.training.targets"
NAMES = {"pid": "cascaded PID", "geometric": "geometric controller"}


# ---- the study's stored numbers, read from RESULTS-P1.md -----------------------------------------------------------------------------
def _section(text: str, name: str) -> list:
    body = text.split(f"\n## {name}\n", 1)[1].split("\n## ", 1)[0]
    return [[c.strip() for c in line.strip().strip("|").split("|")] for line in body.splitlines() if line.startswith("| ") and not line.startswith("| flyer") and not line.startswith("| speed")]


def _num(cell: str) -> float:
    return float(re.match(r"-?\d+(\.\d+)?", cell).group(0))


def study() -> dict:
    """The force ratio per speed, N1's crash counts per flyer and cell, and N2's pooled v50 per flyer and direction."""
    text = RESULTS.read_text()
    force = {_num(r[0]): r[1] for r in _section(text, "Speeds")}
    counts: dict = {}
    for r in _section(text, "N1"):
        counts.setdefault(r[0], {})[(_num(r[1]), _num(r[2]), _num(r[3]), _num(r[4]))] = (int(r[5]), int(r[6]))
    v50: dict = {}
    for r in _section(text, "N2"):
        v50.setdefault(r[0], {})[(_num(r[1]), _num(r[2]))] = (_num(r[3]), "top" in r[3] or "bottom" in r[3])
    import hashlib
    return {"sha256": hashlib.sha256(RESULTS.read_bytes()).hexdigest(), "force": force, "counts": counts, "v50": v50}


# ---- flying --------------------------------------------------------------------------------------------------------------------------
def run_name(tag: str, az: float, el: float) -> str:
    return f"where-{tag}-az{az:g}-el{el:g}"


def fly_direction(store: Path, target: str, env: dict, tag: str, az: float, el: float, episodes: int) -> tuple:
    """One Prove sweep over one direction's cells (11 speeds x 2 timings). A complete stored run is left alone; a partial one
    is resumed. Prove derives each episode's seed from the cell and the salt, so the seeds are the study's."""
    name = run_name(tag, az, el)
    st = Store(store)
    hit = [(r, str(st.manifest(r).status)) for r in st.runs() if st.manifest(r).name == name]
    if hit and hit[0][1] in ("complete", "completed"):
        return name, 0, "stored"
    conditions = [Condition(case=W.ZONE_B_CASE, az=float(az), el=float(el), speed=float(v), phase=float(ph)) for v in SPEEDS for ph in TIMINGS]
    spec = store / f"{name}.toml"
    spec.write_text(sweep_spec_toml(conditions, target=target, target_id="where-does-my-controller-break", replicates=int(episodes)))
    argv = [sys.executable, "-m", "poisson_prove", "--store", str(store), "sweep", str(spec), "--target", target, "--yes", "--max-mde", "1", "--plain"]
    argv += ["--resume", hit[0][0]] if hit else ["--salt", W.ZONE_B_SALT, "--name", name]
    with open(store / f"{name}.log", "ab") as log:
        rc = subprocess.run(argv, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    return name, rc, "flown"


def read_counts(store: Path, tag: str, directions) -> dict:
    """(az, el, speed, timing) -> [crashes, scored episodes], from the stored rows. An error row is in no count."""
    st = Store(store)
    want = {run_name(tag, az, el) for az, el in directions}
    out: dict = {}
    for run in st.runs():
        if st.manifest(run).name not in want:
            continue
        for row in st.read(run, meta=False).to_dict("records"):
            if row["outcome"] not in ("ok", "crash"):
                continue
            c = condition_from_cell(row["config"]["cell"])
            cell = out.setdefault((c.az, c.el, c.speed, c.phase), [0, 0])
            cell[0] += row["outcome"] == "crash"
            cell[1] += 1
    return out


def boundary(counts: dict, directions) -> list:
    """Per direction: v50 pooled over the two timings (the study's estimator), each timing alone, and the pooled counts."""
    rows = []
    for az, el in directions:
        per_t = {ph: ([counts[(az, el, v, ph)][0] for v in SPEEDS], [counts[(az, el, v, ph)][1] for v in SPEEDS]) for ph in TIMINGS}
        k = [sum(per_t[ph][0][i] for ph in TIMINGS) for i in range(len(SPEEDS))]
        n = [sum(per_t[ph][1][i] for ph in TIMINGS) for i in range(len(SPEEDS))]
        rows.append({"az": az, "el": el, "pooled": estimator.v50(k, n, SPEEDS), "by_timing": {ph: estimator.v50(*per_t[ph], SPEEDS) for ph in TIMINGS},
                     "crashes": k, "episodes": n})
    return rows


def show(edge) -> str:
    mark = {None: "", "top": " (never reached on the grid)", "bottom": " (already past it at the lowest speed)"}[edge.censored]
    return f"{edge.value:5.2f}{mark}"


# ---- the map -------------------------------------------------------------------------------------------------------------------------
def polar_svg(rows: list, title: str, ref: dict, quick: bool) -> str:
    """The level directions as a map around the drone: a headwind at the top, travel up the page. The sixteen trained
    policies' and the PID's v50 are drawn from RESULTS-P1.md; what you flew is drawn over them."""
    size, cx, cy, radius, top = 620, 310, 320, 230, SPEEDS[-1]
    at = lambda az, v: (cx - radius * v / top * math.sin(math.radians(az)), cy - radius * v / top * math.cos(math.radians(az)))  # noqa: E731
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size + 60}" font-family="Helvetica, Arial, sans-serif" font-size="12">',
           f'<rect width="{size}" height="{size + 60}" fill="#fff"/>', f'<text x="16" y="24" font-size="14" font-weight="600">{title}</text>',
           '<text x="16" y="42" fill="#555">gust speed at which half of flights crash (v50), by gust direction. In simulation.</text>']
    for v in (4, 8, 12, 16):
        out.append(f'<circle cx="{cx}" cy="{cy}" r="{radius * v / top:.1f}" fill="none" stroke="#ddd"/>')
        x, y = at(202.5, v)
        out.append(f'<text x="{x:.1f}" y="{y + 4:.1f}" fill="#777" text-anchor="middle">{v} m/s · {ref["force"][float(v)]}</text>')
    for az, el in LEVEL:
        x, y = at(az, top)
        lx, ly = at(az, top * 1.09)
        out.append(f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="#eee"/>')
        out.append(f'<text x="{lx:.1f}" y="{ly + 4:.1f}" fill="#555" text-anchor="middle">{az:g}°</text>')

    def ring(points, stroke, width=2.0, dash="", dots=True, closed=True):
        pts = [at(az, v) for az, v, _ in points]
        if closed and len(pts) > 2:
            d = "M" + "L".join(f"{x:.1f} {y:.1f}" for x, y in pts) + "Z"
            out.append(f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{width}" stroke-dasharray="{dash}"/>')
        if dots:
            for (x, y), (_, _, censored) in zip(pts, points):
                out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{"#fff" if censored else stroke}" stroke="{stroke}" stroke-width="1.5"/>')

    pi0 = sorted(f for f in ref["v50"] if f.startswith("pi0-seed-"))
    for f in pi0:
        ring([(az, *ref["v50"][f][(az, el)]) for az, el in LEVEL], "#c9cfe3", 1.0, dots=False)
    ring([(az, *ref["v50"]["pid"][(az, el)]) for az, el in LEVEL], "#8a8a8a", 2.0, "6 3", dots=False)
    level = [r for r in rows if r["el"] == 0]
    ring([(r["az"], r["pooled"].value, r["pooled"].censored) for r in level], "#2a4dd0", 2.5, closed=not quick)
    out.append(f'<text x="16" y="{size + 14}" fill="#2a4dd0">● what you flew{" (two directions; the ring needs all eight)" if quick else ""}</text>')
    out.append(f'<text x="16" y="{size + 32}" fill="#777">┄ the cascaded PID, and ─ each of the sixteen trained policies, as stored in RESULTS-P1.md</text>')
    out.append(f'<text x="16" y="{size + 50}" fill="#555">0° is a headwind, 90° a gust from the left of travel. Ring labels: speed · flat force ratio. An open dot is off the grid.</text>')
    out.append("</svg>")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Where does my controller break? A boundary map of one flyer under wind gusts, with Poisson Prove.")
    ap.add_argument("--flyer", default="pid", help='pid, geometric, pi0-<seed>, or "module:name" of your own rollout')
    ap.add_argument("--quick", action="store_true", help="a headwind and a tailwind only")
    ap.add_argument("--all", action="store_true", help="all sixteen directions")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--store", default=str(HERE / "prove-store"))
    ap.add_argument("--out", default=None, help="output name without extension (default: boundary-<flyer> beside this script)")
    ap.add_argument("--episodes", type=int, default=EPISODES, help="episodes a cell (the study flew 24; its seeds are the first 24)")
    a = ap.parse_args()

    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    env["PYTHONPATH"] = os.pathsep.join([str(HERE), os.getcwd(), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)   # so "module:name" in this folder imports
    stored_name = None
    if ":" in a.flyer:
        target, label = a.flyer, a.flyer
    elif a.flyer in ("pid", "geometric"):
        target, label, stored_name = f"{TARGETS}:{a.flyer}", NAMES[a.flyer], a.flyer
    elif re.fullmatch(r"pi0-\d+", a.flyer):
        seed = int(a.flyer.split("-")[1])
        checkpoint = ROOT / "data" / "pi0" / f"nominal-{seed}" / f"checkpoint-{W.b0_from(ROOT / W.PICK_FILE)}.zip"
        if seed not in W.lineage_seeds(ROOT / W.LINEAGES_FILE) or not checkpoint.exists():
            raise SystemExit(f"{a.flyer} is not one of the sixteen trained policies: pi0-<seed> with seed in {list(W.lineage_seeds(ROOT / W.LINEAGES_FILE))}")
        target, label, stored_name = f"{TARGETS}:policy", f"trained policy π0 {seed}", f"pi0-seed-{seed}"
        env[W.CHECKPOINT_ENV] = str(checkpoint)
        try:
            import stable_baselines3  # noqa: F401
        except ImportError:
            raise SystemExit("flying a trained policy needs PyTorch and Stable-Baselines3: run `uv sync --group gate` first") from None
    else:
        raise SystemExit('--flyer is pid, geometric, pi0-<seed>, or module:name')
    tag = a.flyer.replace(":", "-").replace(".", "-")
    directions = [(0.0, 0.0), (180.0, 0.0)] if a.quick else list(W.DIRECTIONS if a.all else LEVEL)
    store = Path(a.store).resolve()
    store.mkdir(parents=True, exist_ok=True)
    cells = len(directions) * len(SPEEDS) * len(TIMINGS)
    print(f"{label}: {len(directions)} directions x {len(SPEEDS)} speeds x {len(TIMINGS)} timings = {cells} cells, "
          f"{a.episodes} episodes each, {cells * a.episodes:,} episodes; store {store}")
    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as pool:
        results = list(pool.map(lambda d: fly_direction(store, target, env, tag, d[0], d[1], a.episodes), directions))
    bad = [n for n, rc, _ in results if rc != 0]
    if bad:
        raise SystemExit(f"these sweeps did not finish; see their .log files in {store}: {bad}")

    counts = read_counts(store, tag, directions)
    missing = [(az, el, v, ph) for az, el in directions for v in SPEEDS for ph in TIMINGS if (az, el, v, ph) not in counts]
    if missing:
        raise SystemExit(f"{len(missing)} cells have no scored episode, the first {missing[0]}")
    rows = boundary(counts, directions)
    ref = study()

    beside = None
    if stored_name and a.episodes == EPISODES:
        stored = {k: ref["counts"][stored_name][k] for k in counts}
        same = sum(1 for k in stored if tuple(stored[k]) == tuple(counts[k]))
        apart = sum(abs(stored[k][0] - counts[k][0]) for k in stored)
        beside = {"same_cells": same, "cells": len(stored), "episodes_apart": apart, "episodes": sum(v[1] for v in stored.values())}
        print(f"\nBeside the study's stored counts (N1 of RESULTS-P1.md {ref['sha256'][:12]}): {same} of {len(stored)} cells have the same crash count; "
              f"the counts differ by {apart} episodes in all, of {beside['episodes']:,}.")

    print(f"\n{'direction':<16}{'v50, m/s':<40}{'timing 0':<10}{'timing 0.125':<14}{'PID':<8}{'16 policies, lowest to highest':<34}" + ("stored" if stored_name else ""))
    pi0 = sorted(f for f in ref["v50"] if f.startswith("pi0-seed-"))
    for r in rows:
        key = (r["az"], r["el"])
        theirs = sorted(ref["v50"][f][key][0] for f in pi0)
        line = f"az {r['az']:g}° el {r['el']:g}°".ljust(16) + f"{show(r['pooled']):<40}{r['by_timing'][0.0].value:<10.2f}{r['by_timing'][0.125].value:<14.2f}"
        line += f"{ref['v50']['pid'][key][0]:<8.2f}{f'{theirs[0]:.2f} to {theirs[-1]:.2f}':<34}"
        print(line + (f"{ref['v50'][stored_name][key][0]:.2f}" if stored_name else ""))
    print("\ncrashes of episodes at each speed, both timings pooled (speed m/s · flat force ratio):")
    print(" " * 16 + "".join(f"{v:g}·{ref['force'][v]}".ljust(10) for v in SPEEDS))
    for r in rows:
        print(f"az {r['az']:g}° el {r['el']:g}°".ljust(16) + "".join(f"{k}/{n}".ljust(10) for k, n in zip(r["crashes"], r["episodes"])))

    out = Path(a.out) if a.out else HERE / f"boundary-{tag}"
    doc = {"flyer": a.flyer, "salt": W.ZONE_B_SALT, "case": W.ZONE_B_CASE, "episodes_a_cell": a.episodes, "results_sha256": ref["sha256"], "beside_stored_counts": beside,
           "directions": [{"az": r["az"], "el": r["el"], "v50": r["pooled"].value, "censored": r["pooled"].censored,
                           "v50_timing_0": r["by_timing"][0.0].value, "v50_timing_0.125": r["by_timing"][0.125].value,
                           "crashes": r["crashes"], "episodes": r["episodes"],
                           "pid_v50": ref["v50"]["pid"][(r["az"], r["el"])][0], "pi0_v50": {f: ref["v50"][f][(r["az"], r["el"])][0] for f in pi0}} for r in rows]}
    out.with_suffix(".json").write_text(json.dumps(doc, indent=1) + "\n")
    out.with_suffix(".svg").write_text(polar_svg(rows, f"Where does {label} break?", ref, a.quick))
    print(f"\nwrote {out.with_suffix('.svg')} and {out.with_suffix('.json')}")
    print("In simulation: one drone model, one air model, one trajectory, these gusts. 24 episodes a cell cannot show a direction is safe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
