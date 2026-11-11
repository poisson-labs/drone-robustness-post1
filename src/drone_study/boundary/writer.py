"""RESULTS-P1.md and CHECK-P1.md, and the one command that regenerates them from raw rows (spec 12 row 5B, 20.1, 20.6).

RESULTS-P1.md (P1b's figures are fed only by it). Headings in this order: `# ...` the title, `## Run` (a line `Prove: poisson-prove
0.1.2`, the zones' salts, the episode counts, the grid's top), `## Speeds` (a table: speed | force ratio), then `## N1` to `## N10`
and no other number (20.3, point 11). Tables are Markdown pipe tables with a header row; a section may hold several. Headers:

  Speeds  speed | force ratio                                   speed `{v:g}`, ratio `.2f`: the flat force ratio of 17.3's primary model
  N1      flyer | az | el | speed | phase | crashes | n | ground | tilt | radius | rate | lo | hi | force ratio
  N2      flyer | az | el | v50 | v10 | v50 t0 | v50 t0.125 | v10 t0 | v10 t0.125
          an edge is `{v:.2f}` and, when censored, followed by ` [censored top]` or ` [censored bottom]`
  N3      flyer | az | el | integrated
  N4      az | el | mean v50 | v50 censored top | v50 censored bottom | mean v10 | v10 censored top | v10 censored bottom | mean integrated
          az | el | measure | seed <n> ... (one column per lineage, headed by its nominal seed; measure: v50, v10, integrated)
  N5      controller | az | el | measure | mean | lowest | highest | censored pairs | seed <n> ...   (measure: v50, integrated)
          a seed's difference is `{d:.2f}`, with a trailing `*` where either edge is censored; the section holds the line N5_NOTE
  N6      pi0 | controller | az | el | speed | phase | both | only pi0 | only controller | neither
          counts only; the flip rate of 3B's determinism branch, when one is given, is a line `flip rate: <x>`
  N7      flyer | az | phase | speed | crashes | n | rate | grid crashes | grid n | grid rate
          flyer | mean rate ;  seed | lineage mean (the 16 pi0) and a line `pi0 mean: <x>`
  N8      flyer | limit | az | el | v50 | v10 | v50 t0 | v50 t0.125 | v10 t0 | v10 t0.125 | integrated
  N9      seed | crashes | episodes | wilson lo | wilson hi | median rmse m | pid median rmse m | ratio | passed
          seed | yaw rate rms deg/s | heading rate rms deg/s | drift deg/s | past 45 | past 90 | max heading median deg | yaw passed
          and lines `B0: ...`, `k_yaw: ...`, `crash penalty: ...`, `seeds tried: ...`, `seeds failed: ...`, `failed the bar: ...`,
          `failed the yaw gate: ...`, `failed the motor line: ...` (each a comma-separated seed list or `none`), `PID: ... crashes, median RMSE ... m`,
          `geometric tuning figure: ... m`   (the yaw addendum of 2026-11-08, 4DR-5; the motor line of 2026-11-09, 4DB-1)
  N10     az | el | rate at 12  and a line `16 m/s added: yes` or `16 m/s added: no`
Words of section 8 (`safer`, `worse`, `trade`, `no significant change`, `too small to tell`), "significant" and "p-value" appear nowhere.

CHECK-P1.md (P1a's sheet, 20.6). Headings `## N1` to `## N10`; each holds a line `rows read: <n>` (the stored episodes the number reads: Zone B's
for N1 to N6 and N8, the anchor's for N7, none for N9, the pi0's rows at 12 m/s for N10), the definition's section of the spec, the
salts and seeds rule, and how to recompute it. Before them: `Prove: poisson-prove 0.1.2`, `Zone B salt: ...`, `Anchor salt: ...`,
`RESULTS-P1.md sha256: <hex>` of the results text it checks, the one command that regenerates both, and a `## Rules` list naming
rulings 5A-1, 5A-2 and 5A-3 (what section 9 leaves to code) so independent code reads the same."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from drone_study.boundary import pins as _pins
from drone_study.boundary import results as _res
from drone_study.boundary import sweeps as _sweeps
from drone_study.condition import condition_from_cfg

N_HEADINGS = tuple(f"N{i}" for i in range(1, 11))
CONTROLLERS = ("pid", "geometric", "wind_only", "oracle")
COLUMNS = ["flyer", "zone", "cell", "seed", "outcome", "error", "crash_class", "crossings", "rmse", "effort"]
N5_NOTE = _res.N5_NOTE


# ---- the loader --------------------------------------------------------------------------------------------------------------
def _find_run(stores: list, name: str, *, required: bool):
    """(store, run id) of the one complete run called `name`, or None if there is none and `required` is False."""
    found = []
    for store in stores:
        for run_id in store.runs():
            manifest = store.manifest(run_id)
            if manifest.name == name:
                found.append((store, run_id, manifest))
    if not found:
        if required:
            raise LookupError(f"no run named {name!r} in the stores given")
        return None
    if len(found) > 1:
        raise LookupError(f"{len(found)} runs are named {name!r}; the loader needs exactly one")
    store, run_id, manifest = found[0]
    if str(manifest.status) not in ("complete", "completed"):
        raise LookupError(f"run {name!r} is {manifest.status}, not complete: resume it before reading")
    return store, run_id


def discover_flyers(stores: list) -> list:
    """The flyers that have a Zone B run in the stores: the π0 by nominal seed, then the four controllers in the fixed order."""
    prefix = _sweeps.run_name("zone-b", "")
    found = set()
    for store in stores:
        for run_id in store.runs():
            name = store.manifest(run_id).name
            if name and name.startswith(prefix) and not name.startswith(_sweeps.run_name("zone-b-extension", "")):
                found.add(name[len(prefix):])
    pi0 = sorted((f for f in found if f.startswith("pi0-seed-")), key=lambda f: int(f.rsplit("-", 1)[1]))
    return pi0 + [c for c in CONTROLLERS if c in found]


def load_rows(store_paths: Sequence, *, flyers: Optional[Sequence[str]] = None, zones: Optional[Sequence[str]] = None):
    """Every stored episode of the post 1 runs in the Prove stores at `store_paths`, as the row table of the package docstring.
    Runs are found by `sweeps.run_name`; a flyer or zone with no run raises. The extension's run (`zone-b-extension-<flyer>`)
    is read when it exists, and its rows are Zone B rows (zone "B"); a run that is not complete raises. `zones` (default all of
    "zone-b", "anchor", "zone-b-extension") reads only those runs: the extension's first reading needs Zone B alone."""
    from poisson_prove.store import Store

    stores = [Store(Path(p)) for p in store_paths]
    names = list(flyers) if flyers is not None else discover_flyers(stores)
    frames = []
    for flyer in names:
        for zone, tag, required in (("zone-b", "B", True), ("anchor", "anchor", True), ("zone-b-extension", "B", False)):
            if zones is not None and zone not in zones:
                continue
            found = _find_run(stores, _sweeps.run_name(zone, flyer), required=required)
            if found is None:
                continue
            store, run_id = found
            df = store.read(run_id, meta=True)
            records = []
            for cfg, seed, outcome, err, meta in zip(df["config"], df["seed"], df["outcome"], df["error"], df["meta"]):
                cond = condition_from_cfg(cfg)
                cell = (cond.az, cond.el, cond.speed, cond.phase) if tag == "B" else (cond.az, cond.phase)
                if err is not None or outcome is None or not isinstance(meta, dict):
                    records.append((flyer, tag, cell, int(seed), "error", True, None, None, float("nan"), float("nan")))
                else:
                    records.append((flyer, tag, cell, int(seed), str(outcome), False, meta.get("crash_class"), meta.get("crossings"),
                                    float(meta["rmse"]), float(meta["rotor_effort"])))
            frames.append(pd.DataFrame(records, columns=COLUMNS))
    if not frames:
        raise LookupError("no rows: no flyers")
    out = pd.concat(frames, ignore_index=True)
    out["cell"] = out["cell"].astype(object)
    out["crossings"] = out["crossings"].astype(object)
    return out


# ---- text helpers ------------------------------------------------------------------------------------------------------------
def _g(v) -> str:
    return f"{float(v):g}"


def _edge(e) -> str:
    mark = {None: "", "top": " [censored top]", "bottom": " [censored bottom]"}[e.censored]
    return f"{_res.format_exact(_res.edge_exact(e), 2)}{mark}"


def _table(headers: Sequence[str], body: Sequence[Sequence]) -> list:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in body]
    return out


def _seed_of(flyer: str) -> str:
    return flyer.rsplit("-", 1)[1]


def _flyer_lists(rows) -> tuple:
    """(π0 flyers in the order their rows first appear, the controllers present in the fixed order)."""
    zone_b = rows[rows["zone"] == "B"]
    seen = list(dict.fromkeys(zone_b["flyer"].tolist()))
    pi0 = [f for f in seen if f.startswith("pi0-")]
    controllers = [c for c in CONTROLLERS if c in seen]
    if not pi0 or len(controllers) != len(CONTROLLERS):
        raise _res.MissingCell(f"post 1 needs the π0 and all four controllers; the rows hold {seen}")
    return pi0, controllers


def _resolve_speeds(rows, speeds) -> tuple:
    """The speeds the numbers read: those given, or Zone B's ten, plus 16 m/s when N10's first reading fires (then every
    flyer's 16 m/s rows must be there, and a missing one stops the write: ruling 5A-13)."""
    if speeds is not None:
        return tuple(float(v) for v in speeds)
    pi0, _ = _flyer_lists(rows)
    base = tuple(float(v) for v in _sweeps.ZONE_B_SPEEDS)
    reading = _res.n10_extension([_res.flyer_grid(rows, f, speeds=base) for f in pi0])
    return tuple(reading["speeds"])


# ---- RESULTS-P1.md -----------------------------------------------------------------------------------------------------------
def render_results_p1(rows, *, bars: dict, tuning: dict, speeds: Optional[Sequence[float]] = None,
                      prove_version: Optional[str] = None, flip_rate: Optional[float] = None) -> str:
    """The text of RESULTS-P1.md. `speeds` None: Zone B's ten, plus 16 m/s when N10 fires (then the 16 m/s rows must exist)."""
    pinned = _pins.require_prove_pin()
    version = prove_version if prove_version is not None else pinned
    pi0, controllers = _flyer_lists(rows)
    everyone = pi0 + controllers
    speeds = _resolve_speeds(rows, speeds)
    directions = _sweeps.DIRECTIONS
    grids = {f: _res.flyer_grid(rows, f, speeds=speeds, directions=directions) for f in everyone}
    pi0_grids = [grids[f] for f in pi0]
    n10 = _res.n10_extension(pi0_grids)
    added = 16.0 in speeds
    top = _g(speeds[-1])
    lines = []
    out = lines.append

    out("# Post 1: results (RESULTS-P1)")
    out("")
    out("Written by `python -m drone_study.boundary.writer` from the stored Prove rows. N1 to N10 and no other number (spec §20.3, point 11).")
    out("In simulation, with the limits of §17.1: this drone model, this air model, this trajectory and these gusts. Descriptive only (§20.2).")
    out("")
    out("## Run")
    out("")
    out(f"Prove: poisson-prove {version}")
    out(f"Zone B salt: {_sweeps.ZONE_B_SALT}")
    out(f"Anchor salt: {_sweeps.ANCHOR_SALT}")
    zb_rows = int(((rows["zone"] == "B") & rows["cell"].map(lambda c: len(c) == 4 and float(c[2]) in speeds)).sum())
    out(f"Zone B: {len(directions)} directions, {len(speeds)} speeds from {_g(speeds[0])} to {top} m/s, timings 0 and 0.125, case {_sweeps.ZONE_B_CASE}, "
        f"{_sweeps.ZONE_B_REPLICATES} episodes a cell; episodes read: {zb_rows}")
    out(f"Anchor: 16 cells, {_sweeps.ANCHOR_REPLICATES} episodes a cell; episodes read: {int((rows['zone'] == 'anchor').sum())}")
    out(f"Flyers: {len(pi0)} π0 ({', '.join(_seed_of(f) for f in pi0)}) and the cascaded PID, the geometric controller, the wind-only reference and the feasibility oracle")
    out(f"Grid top: {top} m/s")
    out("")

    out("## Speeds")
    out("")
    out("Every speed with its flat force ratio under §17.3's primary air model: the force of a horizontal gust at hover rotor speed over the lateral limit (0.534 N).")
    out("")
    lines += _table(["speed", "force ratio"], [(_g(v), f"{_res.flat_force_ratio(v):.2f}") for v in speeds])
    out("")

    # N1
    out("## N1")
    out("")
    out("The crash rate in every Zone B cell, for each flyer: crashes out of n scored episodes, split by crash class, with Prove's own 95% Wilson interval for the cell, one cell at a time (§20.1).")
    out("")
    body = []
    for c in _res.n1_cells(rows, everyone, speeds=speeds, directions=directions):
        body.append((c["flyer"], _g(c["az"]), _g(c["el"]), _g(c["speed"]), _g(c["phase"]), c["crashes"], c["n"], c["ground"], c["tilt"],
                     c["radius"], _res.format_exact(c["value_exact"], 4), f"{c['lo']:.4f}", f"{c['hi']:.4f}", f"{c['force_ratio']:.2f}"))
    lines += _table(["flyer", "az", "el", "speed", "phase", "crashes", "n", "ground", "tilt", "radius", "rate", "lo", "hi", "force ratio"], body)
    out("")

    # N2, N3
    edges = {f: _res.n2_edges(grids[f]) for f in everyone}
    integrated_x = {f: _res.n3_integrated_exact(grids[f]) for f in everyone}
    t_labels = [f"t{_g(p)}" for p in _sweeps.ZONE_B_PHASES]
    out("## N2")
    out("")
    out("v50 and v10 per direction, for each flyer, by §9's estimator, pooled over the two timings; then the same fit on each timing alone, at 24 episodes a speed. A censored edge is marked.")
    out("")
    body = []
    for f in everyone:
        for d, (az, el) in enumerate(directions):
            e = edges[f][d]
            body.append((f, _g(az), _g(el), _edge(e.v50), _edge(e.v10), *(_edge(x) for x in e.v50_by_timing), *(_edge(x) for x in e.v10_by_timing)))
    lines += _table(["flyer", "az", "el", "v50", "v10"] + [f"v50 {t}" for t in t_labels] + [f"v10 {t}" for t in t_labels], body)
    out("")

    out("## N3")
    out("")
    out("The integrated crash rate per direction, for each flyer: the area under the raw crash curve pooled over the timings, by the trapezoid rule, over the range from the lowest speed to the grid's top (§9). Nothing censors it.")
    out("")
    body = [(f, _g(az), _g(el), _res.format_exact(integrated_x[f][d], 4)) for f in everyone for d, (az, el) in enumerate(directions)]
    lines += _table(["flyer", "az", "el", "integrated"], body)
    out("")

    # N4
    n4 = _res.n4_across_lineages(pi0_grids)
    seed_cols = [f"seed {_seed_of(f)}" for f in pi0]
    out("## N4")
    out("")
    out("Across the π0, per direction: the mean of N2 and of N3, all the values, and how many are censored. A censored value counts as the grid's end it sits at in the mean (§9, ruling 5A-7).")
    out("")
    body = []
    for d, (az, el) in enumerate(directions):
        body.append((_g(az), _g(el), _res.format_exact(n4["v50"]["mean_exact"][d], 2), int(n4["v50"]["censored_top"][d]), int(n4["v50"]["censored_bottom"][d]),
                     _res.format_exact(n4["v10"]["mean_exact"][d], 2), int(n4["v10"]["censored_top"][d]), int(n4["v10"]["censored_bottom"][d]),
                     _res.format_exact(n4["integrated"]["mean_exact"][d], 4)))
    lines += _table(["az", "el", "mean v50", "v50 censored top", "v50 censored bottom", "mean v10", "v10 censored top", "v10 censored bottom", "mean integrated"], body)
    out("")
    body = []
    for measure, places in (("v50", 2), ("v10", 2), ("integrated", 4)):
        for d, (az, el) in enumerate(directions):
            body.append((_g(az), _g(el), measure, *(_res.format_exact(v, places) for v in n4[measure]["values_exact"][d])))
    lines += _table(["az", "el", "measure"] + seed_cols, body)
    out("")

    # N5
    out("## N5")
    out("")
    out("π0 beside each controller, per direction: each lineage's value minus the controller's, the mean of those, the lowest and the highest. A difference marked `*` has a censored edge on either side. The integrated rate is read the same way.")
    out("")
    out(N5_NOTE)
    out("")
    body = []
    for c in controllers:
        n5 = _res.n5_against_controller(pi0_grids, grids[c])
        for measure, places in (("v50", 2), ("integrated", 4)):
            part = n5[measure]
            for d, (az, el) in enumerate(directions):
                cells = []
                for k in range(len(pi0)):
                    star = "*" if measure == "v50" and bool(part["censored_pair"][d, k]) else ""
                    cells.append(f"{_res.format_exact(part['differences_exact'][d][k], places)}{star}")
                pairs = int(part["censored_pair"][d].sum()) if measure == "v50" else 0
                body.append((c, _g(az), _g(el), measure, _res.format_exact(part["mean_exact"][d], places), _res.format_exact(part["lowest_exact"][d], places),
                             _res.format_exact(part["highest_exact"][d], places), pairs, *cells))
    lines += _table(["controller", "az", "el", "measure", "mean", "lowest", "highest", "censored pairs"] + seed_cols, body)
    out("")

    # N6
    out("## N6")
    out("")
    out("Paired counts, cell by cell, for each π0 with each controller on their shared seeds: both crash, only the π0 crashes, only the controller crashes, neither. Counts only (§20.2).")
    out("")
    if flip_rate is not None:
        out(f"flip rate: {flip_rate:g}")
        out("")
    body = []
    for p in pi0:
        for c in controllers:
            for r in _res.n6_paired_counts(rows, p, c, speeds=speeds, directions=directions):
                body.append((p, c, _g(r["az"]), _g(r["el"]), _g(r["speed"]), _g(r["phase"]), r["both"], r["only_pi0"], r["only_controller"], r["neither"]))
    lines += _table(["pi0", "controller", "az", "el", "speed", "phase", "both", "only pi0", "only controller", "neither"], body)
    out("")

    # N7
    n7 = _res.n7_anchor(rows, everyone, pi0)
    out("## N7")
    out("")
    out("The anchor slice, for each flyer: crashes out of n in each of its 16 cells and the mean rate over them, beside the same cell's counts on the Zone B grid. For the π0, the mean of the lineages and all the values.")
    out("")
    body = []
    for f in everyone:
        for c in n7["flyers"][f]["cells"]:
            body.append((f, _g(c["az"]), _g(c["phase"]), _g(c["speed"]), c["crashes"], c["n"], _res.format_exact(c["rate_exact"], 4), c["grid_crashes"], c["grid_n"], _res.format_exact(c["grid_rate_exact"], 4)))
    lines += _table(["flyer", "az", "phase", "speed", "crashes", "n", "rate", "grid crashes", "grid n", "grid rate"], body)
    out("")
    lines += _table(["flyer", "mean rate"], [(f, _res.format_exact(n7["flyers"][f]["mean_exact"], 4)) for f in everyone])
    out("")
    lines += _table(["seed", "lineage mean"], [(_seed_of(f), _res.format_exact(v, 4)) for f, v in zip(pi0, n7["pi0"]["lineage_means_exact"])])
    out("")
    out(f"pi0 mean: {_res.format_exact(n7['pi0']['mean_exact'], 4)}")
    out("")

    # N8
    out("## N8")
    out("")
    out("§10's S6 re-read of N2 and N3 under each of its crash limits, for each flyer, from the crossings the flights stored. No flight.")
    out("")
    body = []
    for f in everyone:
        n8 = _res.n8_limits(rows, f, speeds=speeds, directions=directions)
        for limit, part in n8.items():
            for d, (az, el) in enumerate(directions):
                body.append((f, limit, _g(az), _g(el), _edge(part["v50"][d]), _edge(part["v10"][d]),
                             *(_edge(x) for x in part["v50_by_timing"][d]), *(_edge(x) for x in part["v10_by_timing"][d]), _res.format_exact(part["integrated_exact"][d], 4)))
    lines += _table(["flyer", "limit", "az", "el", "v50", "v10"] + [f"v50 {t}" for t in t_labels] + [f"v10 {t}" for t in t_labels] + ["integrated"], body)
    out("")

    # N9
    n9 = _res.n9_calm(bars, tuning)
    out("## N9")
    out("")
    out("Calm air, from the bar of §2 and not from Zone D, as stored by session 4D-B. Read from the stored lineage file and never recomputed. The 16 passed the bar by selection; this is a gate's reading on them, with the seeds that failed beside it.")
    out("")
    lineages = list(n9["lineages"])
    body = []
    for s in lineages:
        r = n9["lineages"][s]
        body.append((s, r["crashes"], r["episodes"], f"{r['wilson_lo']:.4f}", f"{r['wilson_hi']:.4f}", f"{r['median_rmse_m']:.4f}",
                     f"{r['pid_median_rmse_m']:.4f}", f"{r['ratio']:.4f}", "yes" if r["passed"] else "no"))
    lines += _table(["seed", "crashes", "episodes", "wilson lo", "wilson hi", "median rmse m", "pid median rmse m", "ratio", "passed"], body)
    out("")
    body = []
    for s in lineages:
        y = n9["yaw"][s]
        body.append((s, f"{y['yaw_rate_rms_median_deg_s']:.2f}", f"{y['heading_rate_rms_median_deg_s']:.2f}", f"{y['drift_abs_median_deg_s']:.2f}",
                     y["past_45"], y["past_90"], f"{y['max_heading_median_deg']:.2f}", "yes" if y["passed"] else "no"))
    lines += _table(["seed", "yaw rate rms deg/s", "heading rate rms deg/s", "drift deg/s", "past 45", "past 90", "max heading median deg", "yaw passed"], body)
    out("")

    def failed(*gates) -> str:
        seeds = sorted(s for s, g in n9["failed_on"].items() if any(x in str(g).replace("both", "bar+yaw").split("+") for x in gates))
        return ", ".join(str(s) for s in seeds) if seeds else "none"

    out(f"B0: {int(n9['b0'])}")
    out(f"k_yaw: {n9['k_yaw']:g}")
    out(f"crash penalty: {n9['crash_penalty']:g}")
    out(f"seeds tried: {n9['tried']}")
    out(f"seeds failed: {len(n9['failures'])}")
    out(f"failed the bar: {failed('bar')}")
    out(f"failed the yaw gate: {failed('yaw')}")
    out(f"failed the motor line: {failed('motor')}")
    out(f"PID: {n9['pid']['crashes']} crashes, median RMSE {n9['pid']['median_rmse_m']:.4f} m")
    out(f"geometric tuning figure: {n9['geometric_tuning_figure_m']:.4f} m ({n9['geometric_label']}: the smallest on a gain grid, picked in calm air, so it flatters)")
    out("")

    # N10
    out("## N10")
    out("")
    out("The extension's first reading (§3): the mean π0 crash rate at 12 m/s on each of the 16 directions, and whether 16 m/s was added. The rule adds it when any direction is strictly under 80%.")
    out("")
    lines += _table(["az", "el", "rate at 12"], [(_g(az), _g(el), _res.format_exact(n10["rate_at_12_exact"][d], 4)) for d, (az, el) in enumerate(directions)])
    out("")
    out(f"16 m/s added: {'yes' if added else 'no'}")
    out("")
    return "\n".join(lines)


# ---- CHECK-P1.md -------------------------------------------------------------------------------------------------------------
def render_check_p1(rows, results_text: str, *, speeds: Optional[Sequence[float]] = None) -> str:
    """The text of CHECK-P1.md for the results text given (it names that text's SHA-256)."""
    pinned = _pins.require_prove_pin()
    speeds = _resolve_speeds(rows, speeds)
    zone_b = rows[rows["zone"] == "B"]
    in_grid = zone_b["cell"].map(lambda c: len(c) == 4 and float(c[2]) in speeds)
    b_rows = int(in_grid.sum())
    anchor_rows = int((rows["zone"] == "anchor").sum())
    at_12 = int((zone_b["flyer"].str.startswith("pi0-") & zone_b["cell"].map(lambda c: len(c) == 4 and float(c[2]) == 12.0)).sum())
    digest = hashlib.sha256(results_text.encode()).hexdigest()
    cmd = ("python -m drone_study.boundary.writer --store <STORE> [--store <STORE> ...] --bars data/4d-lineages.json "
           "--tuning <tuning-pick.json> --out <DIR>")
    lines = []
    out = lines.append
    out("# Post 1: check sheet (CHECK-P1)")
    out("")
    out("For session P1a (spec §20.6): recompute N1 to N10 of RESULTS-P1.md from the raw rows with independent code, working from this sheet. A disagreement is named and settled by §20.3, point 14.")
    out("")
    out(f"Prove: poisson-prove {pinned}")
    out(f"Zone B salt: {_sweeps.ZONE_B_SALT}")
    out(f"Anchor salt: {_sweeps.ANCHOR_SALT}")
    out(f"RESULTS-P1.md sha256: {digest}")
    out(f"Grid top: {_g(speeds[-1])} m/s")
    out("")
    out("Regenerate both files from the stores, by one command:")
    out("")
    out("```")
    out(cmd)
    out("```")
    out("")
    out("## Where the rows are")
    out("")
    out("Prove runs, one per flyer and zone, named `zone-b-<flyer>`, `anchor-<flyer>` and (only if the extension fired) `zone-b-extension-<flyer>`. "
        "Flyers: `pi0-seed-<nominal seed>` for the lineages of `data/4d-lineages.json`, then `pid`, `geometric`, `wind_only`, `oracle`.")
    out("A row's `config` is `{\"cell\": <name>}`, the name `c=<case>;az=..;el=..;v=<speed>;ph=<phase>;tb=..;pl=..;mi=..;ms=..;dl=..;gl=..` "
        "(`drone_study.condition.cell_name`). Zone B and the anchor read az, el, v and ph; every other field is the default. "
        "The outcome is `ok` or `crash`; the crash class, every threshold's first crossing, the RMSE and the rotor effort are in `meta`. "
        "A Prove error row is in no count (ruling 5A-6).")
    out(f"Seeds: `derive_seed(salt, config_hash(cfg), k)` for k in 0..23 (Zone B and the extension, salt {_sweeps.ZONE_B_SALT}) and k in 0..15 (the anchor, salt {_sweeps.ANCHOR_SALT}), "
        "from `poisson_prove.executor`; one salt per zone gives every flyer the same seeds on a cell. Two flyers' rows are paired on (cell, seed).")
    out("")

    def section(name, count, spec, text):
        out(f"## {name}")
        out("")
        out(f"rows read: {count}")
        out("")
        out(f"Defined in {spec}.")
        out("")
        out(text)
        out("")

    section("N1", b_rows, "§20.1 and §20.2", "Per flyer and Zone B cell (direction, speed, timing): crashes and scored episodes, the crash classes among the crashes, the rate, and Prove's 95% Wilson interval for the cell "
            "(`poisson_prove.stats.share(crashes, n)`). The force ratio is the primary air model's flat force at hover rotor speed over the lateral limit, `.2f`.")
    section("N2", b_rows, "§9 and §20.1", "Per flyer and direction: pool crashes and episodes over the two timings at each speed, fit a monotone curve by pooled adjacent violators, "
            "v50 is where it first reaches 0.5 and v10 where it first reaches 0.1, by straight-line interpolation; the same fit on each timing alone. Rules 5A-1 to 5A-3 below.")
    section("N3", b_rows, "§9 and §20.1", "Per flyer and direction: the area under the raw pooled rate (crashes over scored episodes at each speed), trapezoid rule over the grid's speeds, divided by the range from the lowest speed to the top.")
    section("N4", b_rows, "§20.1 (N4)", "Across the π0, per direction: the mean of N2's v50 and v10 and of N3 over the lineages, every lineage's value, and the number censored at the top and at the bottom. A censored value counts as the grid's end it sits at (ruling 5A-7).")
    section("N5", b_rows, "§20.1 (N5)", "For each of the four non-π0 flyers: each lineage's value minus the controller's, for v50 and for the integrated rate, the mean, the lowest and the highest; a pair with a censored edge on either side is marked `*` (ruling 5A-9).")
    section("N6", b_rows, "§20.1 (N6) and §20.2", "For each π0 and each of the four controllers, per Zone B cell, on pairs matched by seed: both crash, only the π0 crashes, only the controller crashes, neither. A pair with an error row on either side is out (ruling 5A-6). Counts only.")
    section("N7", anchor_rows, "§3 (the anchor), §20.1 (N7) and ruling 5A-11", "Per flyer, the anchor's 16 cells: crashes and scored episodes (of 16), beside the same cell's Zone B counts at (az, 0, the anchor's speed, timing); the mean of the 16 rates; for the π0, each lineage's mean, each cell's mean over the lineages, and the mean of the lineage means.")
    section("N8", b_rows, "§10 (S6), §20.1 (N8) and ruling 5A-10", "N2 and N3 again, per flyer, under each of S6's six limits: tilt 75° and 90° instantaneous, radius 0.3, 0.75 and 1.00 m, and `ground` and `tilt` only. "
            "Each stored row is re-scored from the crossings it stored by `drone_study.score.rescore`; nothing is flown.")
    section("N9", 0, "§2, §20.1 (N9) and the yaw addendum (4DR-5)", "Read from `data/4d-lineages.json` and the geometric controller's tuning pick, and not recomputed from rows. The bar's seeds are the `bar` salt's.")
    section("N10", at_12, "§3 (grid extension), §20.1 (N10) and ruling 5A-8", "Per direction: the mean over the π0 of each lineage's plain mean of its two 12 m/s cells' rates, on exact fractions. 16 m/s is added when any direction is strictly under 0.8.")

    out("## Rules")
    out("")
    out("What §9 leaves to code, and how this build reads it (all ruled yes by Taylor on 2026-11-07):")
    out("")
    out("- 5A-1. A fitted curve that sits at exactly the level (0.5 for v50, 0.1 for v10) over a stretch crosses at the stretch's lowest speed, and the edge is not censored.")
    out("- 5A-2. \"Reaches\" and \"above\" are read on the counts, in exact arithmetic: a pooled block's rate is a ratio of integers, so a block that is exactly one half is one half.")
    out("- 5A-3. A speed with no scored episodes is left out of that fit, and a fit with none left is an error. Censoring: never reaches the level, the grid's top marked top; starts above the level, the lowest speed marked bottom.")
    out("- 5A-6. An error row is in no count; a cell with no scored row stops the read; a pair with an error row on either side is out of N6.")
    out("- 5A-7. A mean over the lineages (N4) and a difference (N5) use a censored value as the grid's end it sits at, with the count of censored values printed.")
    out("- 5A-8. N10's rate is the mean over the π0 of each lineage's plain mean of its two 12 m/s cells' rates; \"under 80%\" is strictly under.")
    out("- 5A-12. The flat force ratio is the primary model's force on a horizontal gust of that speed at hover rotor speed over the lateral limit (0.534 N), printed to two decimals.")
    out("- P1A-2 (print rule, ruled yes by Taylor on 2026-11-09). Every published number is the exact value rounded half to even to the printed precision. "
        "A value built from integer counts (a rate, an edge, an integrated rate, and a mean, difference, lowest or highest of them) is a ratio of integers, so it is computed as an exact fraction "
        "and rounded once, at print: 2 decimals for an edge, 4 for a rate. Where it sits exactly on a half in the last printed place, the even digit wins "
        "(9/32 prints 0.2812, 7.175 prints 7.18, 7.025 prints 7.02). Do not round through a float: a float of a non-dyadic tie is off the tie and rounds the wrong way.")
    out("")
    return "\n".join(lines)


def build_post1(rows, *, bars_json, tuning_json, out_dir, speeds: Optional[Sequence[float]] = None,
                flip_rate: Optional[float] = None) -> tuple:
    """Write RESULTS-P1.md and CHECK-P1.md into `out_dir`; return their paths."""
    bars = json.loads(Path(bars_json).read_text())
    tuning = json.loads(Path(tuning_json).read_text())
    speeds = _resolve_speeds(rows, speeds)
    results = render_results_p1(rows, bars=bars, tuning=tuning, speeds=speeds, flip_rate=flip_rate)
    check = render_check_p1(rows, results, speeds=speeds)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "RESULTS-P1.md").write_text(results)
    (out / "CHECK-P1.md").write_text(check)
    return (out / "RESULTS-P1.md", out / "CHECK-P1.md")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """`python -m drone_study.boundary.writer --store STORE [--store STORE ...] --bars bars.json --tuning tuning-pick.json --out DIR`."""
    ap = argparse.ArgumentParser(description="Regenerate RESULTS-P1.md and CHECK-P1.md from Prove's stores.")
    ap.add_argument("--store", action="append", required=True, help="a Prove store holding post 1 runs (repeatable)")
    ap.add_argument("--bars", default=_sweeps.LINEAGES_FILE, help="4D-B's lineage file (default: data/4d-lineages.json)")
    ap.add_argument("--tuning", required=True, help="4B's tuning-pick.json (the geometric controller's tuning figure)")
    ap.add_argument("--out", required=True, help="the directory the two files are written into")
    ap.add_argument("--flip-rate", type=float, default=None, help="3B's determinism flip rate, printed beside N6 when given")
    args = ap.parse_args(argv)
    rows = load_rows(args.store)
    stored = sorted({int(f.rsplit("-", 1)[1]) for f in rows["flyer"].unique() if f.startswith("pi0-seed-")})
    listed = sorted(int(x) for x in json.loads(Path(args.bars).read_text())["selection"]["lineages"])
    if stored != listed:
        print(f"WARNING: the stores hold π0 seeds {stored} and {args.bars} lists the lineages {listed}", file=sys.stderr)
    results_path, check_path = build_post1(rows, bars_json=args.bars, tuning_json=args.tuning, out_dir=args.out, flip_rate=args.flip_rate)
    print(f"wrote {results_path} and {check_path} from {len(rows)} stored episodes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
