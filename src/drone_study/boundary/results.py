"""Post 1's numbers, N1 to N8 and N10, as functions from stored rows; N9 from 4B's stored readings (spec 20.1).

"Post 1 reports these and no other number." One function per quantity: the estimator is
`drone_study.analysis.estimator`, the integrated crash rate `drone_study.analysis.boundary.integrated_rate`, and S6's
re-score `drone_study.analysis.sensitivities.s6_rescore`, which this module calls through its module (5B makes those
three real; they are 8A's stubs and post 2 reads the same ones).

Counting rules (as 8A-10, ruling 5A-6): an error row is in no count; a cell with no scored row raises `MissingCell`; a
pair of rows with an error on either side is out of N6's counts. Every function takes the speeds it is to read, so a 16 m/s
top (section 3's extension) is the same call with one more speed.

Directions are (az, el), in `sweeps.DIRECTIONS` order. Differences are π0 minus the controller. A rate is a fraction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Mapping, Optional, Sequence

import numpy as np

from drone_study.analysis import boundary as _ab
from drone_study.analysis import estimator as _est
from drone_study.analysis import sensitivities as _sens
from drone_study.analysis.estimator import DirectionEdges, Edge
from drone_study.boundary.sweeps import ANCHOR_SPEEDS, DIRECTIONS, ZONE_B_PHASES, ZONE_B_SPEEDS, anchor_conditions

N5_NOTE = "All 16 differences share one controller's estimate, so its error moves them together."
CRASH_CLASSES = ("ground", "tilt", "radius")
_NDIR = 16      # the stubs' shapes only: sixteen directions


def format_exact(x, places: int) -> str:
    """`x` (an exact Fraction, or anything exact such as an int) to `places` decimals, rounded half to even on the exact value.
    A float is taken as its exact binary value, so a float that sits on a tie prints as Python's own `format` does. Used for every
    printed number built from integer counts (P1a-fix, ruling P1A-1): a difference that lands on an exact half in the last
    printed place rounds to the even digit whatever a float sum would have said. The sign is the exact value's sign, as
    `format` gives it (a negative that rounds to zero prints `-0.0000`)."""
    x = Fraction(x)
    scaled = abs(x) * 10 ** places
    q, r = divmod(scaled.numerator, scaled.denominator)
    twice = 2 * r
    if twice > scaled.denominator or (twice == scaled.denominator and q % 2 == 1):
        q += 1
    digits = str(q).rjust(places + 1, "0")
    body = digits[:-places] + "." + digits[-places:] if places else digits
    return ("-" if x < 0 else "") + body


def is_tie(x, places: int) -> bool:
    """True when the exact value `x` sits exactly on a half in the last of `places` decimals."""
    scaled = abs(Fraction(x)) * 10 ** places * 2
    return scaled.denominator == 1 and scaled.numerator % 2 == 1


def edge_exact(e) -> Fraction:
    """An Edge's exact value: the `exact` the estimator stored, or the exact value of its float where there is none."""
    return e.exact if getattr(e, "exact", None) is not None else Fraction(float(e.value))


class MissingCell(ValueError):
    """A cell the read needs has no scored row."""


class PairingError(ValueError):
    """Two flyers' rows on a cell do not carry the same seeds."""


@dataclass(frozen=True)
class FlyerGrid:
    """One flyer's Zone B counts. `crashes` and `n` are (directions, speeds, timings); `n` counts scored rows only.
    `classes` maps each of CRASH_CLASSES to a (directions, speeds, timings) count."""
    flyer: str
    speeds: tuple
    directions: tuple
    crashes: np.ndarray
    n: np.ndarray
    classes: Mapping = field(default_factory=dict)


def flyer_grid(rows, flyer: str, *, speeds: Sequence[float] = ZONE_B_SPEEDS, directions: Sequence[tuple] = DIRECTIONS) -> FlyerGrid:
    """The flyer's counts on Zone B's cells from the rows (zone "B"). Raises MissingCell for a cell with no scored row."""
    speeds = tuple(float(v) for v in speeds)
    directions = tuple((float(a), float(e)) for a, e in directions)
    d_ix = {d: i for i, d in enumerate(directions)}
    s_ix = {v: i for i, v in enumerate(speeds)}
    t_ix = {float(p): i for i, p in enumerate(ZONE_B_PHASES)}
    shape = (len(directions), len(speeds), len(t_ix))
    crashes = np.zeros(shape, dtype=int)
    n = np.zeros(shape, dtype=int)
    classes = {c: np.zeros(shape, dtype=int) for c in CRASH_CLASSES}
    sub = rows[(rows["flyer"] == flyer) & (rows["zone"] == "B")]
    for cell, outcome, err, cls in zip(sub["cell"].tolist(), sub["outcome"].tolist(), sub["error"].tolist(), sub["crash_class"].tolist()):
        az, el, speed, phase = cell
        i = d_ix.get((float(az), float(el)))
        j = s_ix.get(float(speed))
        k = t_ix.get(float(phase))
        if i is None or j is None or k is None or err:
            continue
        n[i, j, k] += 1
        if outcome == "crash":
            crashes[i, j, k] += 1
            if cls in classes:
                classes[cls][i, j, k] += 1
    empty = np.argwhere(n == 0)
    if len(empty):
        i, j, k = (int(v) for v in empty[0])
        raise MissingCell(f"{flyer}: {len(empty)} Zone B cell(s) have no scored row, first {directions[i]} at {speeds[j]:g} m/s, timing {ZONE_B_PHASES[k]:g}")
    return FlyerGrid(flyer, speeds, directions, crashes, n, classes)


def flat_force_ratio(speed: float) -> float:
    """§17.3's primary air model, flat at hover rotor speed: the air force of a horizontal gust of `speed` m/s over the
    lateral limit (0.534 N). 12 m/s is 0.565 and 16 m/s is 0.95."""
    from drone_study import vehicle
    from drone_study.air import air_wrench_scalars

    lateral_limit = vehicle.WEIGHT * (vehicle.THRUST_TO_WEIGHT ** 2 - 1.0) ** 0.5
    fx, _, _, _, _ = air_wrench_scalars(float(speed), 0.0, 0.0, 4.0 * vehicle.OMEGA_HOVER)
    return float(fx / lateral_limit)


def n1_cells(rows, flyers: Sequence[str], *, speeds: Sequence[float] = ZONE_B_SPEEDS, directions: Sequence[tuple] = DIRECTIONS) -> list:
    """N1: one dict per flyer and cell, flyers in the order given, then the cells in direction, speed, timing order:
    flyer, az, el, speed, phase, crashes, n, ground, tilt, radius, value (crashes / n), lo, hi, level, method (Prove's own
    interval for the cell, `poisson_prove.stats.share(crashes, n)`: Wilson 95%), force_ratio (flat_force_ratio(speed))."""
    from poisson_prove import stats as prove_stats

    out = []
    ratio = {float(v): flat_force_ratio(v) for v in speeds}
    intervals = {}
    for flyer in flyers:
        g = flyer_grid(rows, flyer, speeds=speeds, directions=directions)
        for i, (az, el) in enumerate(g.directions):
            for j, speed in enumerate(g.speeds):
                for k, phase in enumerate(ZONE_B_PHASES):
                    c, m = int(g.crashes[i, j, k]), int(g.n[i, j, k])
                    if (c, m) not in intervals:
                        intervals[(c, m)] = prove_stats.share(c, m)
                    iv = intervals[(c, m)]
                    out.append({"flyer": flyer, "az": az, "el": el, "speed": speed, "phase": float(phase), "crashes": c, "n": m,
                                "ground": int(g.classes["ground"][i, j, k]), "tilt": int(g.classes["tilt"][i, j, k]),
                                "radius": int(g.classes["radius"][i, j, k]), "value": c / m, "value_exact": Fraction(c, m), "lo": iv["lo"], "hi": iv["hi"],
                                "level": iv["level"], "method": iv["method"], "force_ratio": ratio[speed]})
    return out


def n2_edges(grid: FlyerGrid) -> list:
    """N2: one `DirectionEdges` per direction of the grid: v50 and v10 pooled over the two timings, and each timing alone, from
    `drone_study.analysis.estimator.direction_edges` (called through the module)."""
    return [_est.direction_edges(grid.crashes[d], grid.n[d], grid.speeds) for d in range(grid.crashes.shape[0])]


def n3_integrated(grid: FlyerGrid) -> np.ndarray:
    """N3: the integrated crash rate per direction, from the pooled raw counts (analysis.boundary.integrated_rate)."""
    return np.array([_ab.integrated_rate(grid.crashes[d].sum(axis=-1), grid.n[d].sum(axis=-1), grid.speeds)
                     for d in range(grid.crashes.shape[0])], dtype=float)


def n3_integrated_exact(grid: FlyerGrid) -> list:
    """N3 on exact fractions: one Fraction per direction, the `.exact` of what analysis.boundary.integrated_rate returns (so it still reads
    the one function), and what N3 prints."""
    return [_ab.exact_of(_ab.integrated_rate(grid.crashes[d].sum(axis=-1), grid.n[d].sum(axis=-1), grid.speeds))
            for d in range(grid.crashes.shape[0])]


def _mean(xs) -> Fraction:
    xs = list(xs)
    return sum(xs, Fraction(0)) / len(xs)


def _edge_stats(edges_by_lineage: list, which: str) -> dict:
    k, d = len(edges_by_lineage), len(edges_by_lineage[0])
    vals = np.array([[getattr(edges_by_lineage[i][j], which).value for i in range(k)] for j in range(d)], dtype=float)
    exact = [[edge_exact(getattr(edges_by_lineage[i][j], which)) for i in range(k)] for j in range(d)]
    top = np.array([sum(getattr(edges_by_lineage[i][j], which).censored == "top" for i in range(k)) for j in range(d)], dtype=int)
    bottom = np.array([sum(getattr(edges_by_lineage[i][j], which).censored == "bottom" for i in range(k)) for j in range(d)], dtype=int)
    mean_exact = [_mean(row) for row in exact]
    return {"values": vals, "mean": np.array([float(m) for m in mean_exact], dtype=float), "censored_top": top, "censored_bottom": bottom,
            "values_exact": exact, "mean_exact": mean_exact}


def n4_across_lineages(grids: Sequence[FlyerGrid]) -> dict:
    """N4, over the 16 π0 grids in the order given, edges from `n2_edges` and rates from `n3_integrated`: {"directions": ..., "v50": A, "v10": A, "integrated": B} with
    A = {"values": (D, 16), "mean": (D,), "censored_top": (D,), "censored_bottom": (D,)} and B = {"values", "mean"}.
    A mean is over the 16 values as they stand (a censored value counts as the grid's end it sits at), with the count
    of censored values printed beside it (ruling 5A-7)."""
    edges = [n2_edges(g) for g in grids]
    integrated = np.array([n3_integrated(g) for g in grids], dtype=float).T          # (D, lineages)
    exact = [list(row) for row in zip(*(n3_integrated_exact(g) for g in grids))]       # (D, lineages), Fractions
    mean_exact = [_mean(row) for row in exact]
    return {"directions": tuple(grids[0].directions), "v50": _edge_stats(edges, "v50"), "v10": _edge_stats(edges, "v10"),
            "integrated": {"values": integrated, "mean": np.array([float(m) for m in mean_exact], dtype=float),
                           "values_exact": exact, "mean_exact": mean_exact}}


def _differences(values: np.ndarray, exact: list) -> dict:
    """`values` the float differences as they stand; `exact` the same (D, lineages) differences as Fractions. The mean, lowest and
    highest are taken on the exact ones (and the floats given are those rounded once), so a printed tie is read exactly."""
    mean_exact = [_mean(row) for row in exact]
    low, high = [min(row) for row in exact], [max(row) for row in exact]
    return {"differences": values, "mean": np.array([float(m) for m in mean_exact], dtype=float),
            "lowest": np.array([float(x) for x in low], dtype=float), "highest": np.array([float(x) for x in high], dtype=float),
            "differences_exact": exact, "mean_exact": mean_exact, "lowest_exact": low, "highest_exact": high}


def n5_against_controller(pi0_grids: Sequence[FlyerGrid], controller: FlyerGrid) -> dict:
    """N5, edges from `n2_edges` and rates from `n3_integrated`: {"v50": A, "integrated": B, "note": N5_NOTE} with A = {"differences": (D, 16), "mean": (D,), "lowest": (D,),
    "highest": (D,), "censored_pair": (D, 16) bool: True where either edge is censored} and B the same without the mask.
    Each difference is a lineage's value minus the controller's."""
    c_edges = n2_edges(controller)
    c_rate_x = n3_integrated_exact(controller)
    d = len(c_edges)
    v50 = np.zeros((d, len(pi0_grids)))
    mask = np.zeros((d, len(pi0_grids)), dtype=bool)
    rate = np.zeros((d, len(pi0_grids)))
    v50_x = [[None] * len(pi0_grids) for _ in range(d)]
    rate_x = [[None] * len(pi0_grids) for _ in range(d)]
    for k, g in enumerate(pi0_grids):
        edges, rates_x = n2_edges(g), n3_integrated_exact(g)
        for j in range(d):
            v50_x[j][k] = edge_exact(edges[j].v50) - edge_exact(c_edges[j].v50)
            v50[j, k] = float(v50_x[j][k])
            mask[j, k] = edges[j].v50.censored is not None or c_edges[j].v50.censored is not None
            rate_x[j][k] = rates_x[j] - c_rate_x[j]
            rate[j, k] = float(rate_x[j][k])
    return {"v50": {**_differences(v50, v50_x), "censored_pair": mask}, "integrated": _differences(rate, rate_x), "note": N5_NOTE}


def _cell_outcomes(rows, flyer: str, speeds, directions) -> dict:
    """{(az, el, speed, phase): {seed: (crashed, error)}} for the flyer's Zone B rows on the cells asked for."""
    want = {(float(a), float(e), float(v), float(p)) for a, e in directions for v in speeds for p in ZONE_B_PHASES}
    out = {}
    sub = rows[(rows["flyer"] == flyer) & (rows["zone"] == "B")]
    for cell, seed, outcome, err in zip(sub["cell"].tolist(), sub["seed"].tolist(), sub["outcome"].tolist(), sub["error"].tolist()):
        key = tuple(float(x) for x in cell)
        if key in want:
            out.setdefault(key, {})[int(seed)] = (outcome == "crash", bool(err))
    return out


def n6_paired_counts(rows, pi0: str, controller: str, *, speeds: Sequence[float] = ZONE_B_SPEEDS,
                     directions: Sequence[tuple] = DIRECTIONS) -> list:
    """N6: one dict per cell, direction, speed, timing order: az, el, speed, phase, both, only_pi0, only_controller, neither.
    Rows are matched on (cell, seed); a pair with an error row on either side is out; the two flyers must carry the same
    seeds on the cell or PairingError is raised. Counts only: nothing of Prove's `diff` is called or printed."""
    a, b = _cell_outcomes(rows, pi0, speeds, directions), _cell_outcomes(rows, controller, speeds, directions)
    out = []
    for az, el in directions:
        for speed in speeds:
            for phase in ZONE_B_PHASES:
                key = (float(az), float(el), float(speed), float(phase))
                if key not in a or key not in b:
                    raise MissingCell(f"{pi0} / {controller}: no rows on cell {key}")
                if set(a[key]) != set(b[key]):
                    raise PairingError(f"{pi0} and {controller} carry different seeds on cell {key}")
                both = only_pi0 = only_ctl = neither = 0
                for seed, (ca, ea) in a[key].items():
                    cb, eb = b[key][seed]
                    if ea or eb:
                        continue
                    if ca and cb:
                        both += 1
                    elif ca:
                        only_pi0 += 1
                    elif cb:
                        only_ctl += 1
                    else:
                        neither += 1
                out.append({"az": float(az), "el": float(el), "speed": float(speed), "phase": float(phase), "both": both,
                            "only_pi0": only_pi0, "only_controller": only_ctl, "neither": neither})
    return out


def n7_anchor(rows, flyers: Sequence[str], pi0: Sequence[str] = ()) -> dict:
    """N7: {"flyers": {name: {"cells": [16 dicts: az, phase, speed, crashes, n, rate, grid_crashes, grid_n, grid_rate],
    "mean": mean of the 16 cell rates}}, "pi0": {"cell_means": 16 values (each cell's mean rate over the lineages),
    "lineage_means": one mean per π0 in the order given, "mean": their mean}}. `grid_*` are the same cell's Zone B counts
    pooled over nothing: the Zone B row at (az, 0, speed, phase). The "pi0" entry is present only when `pi0` is non-empty."""
    anchor = rows[rows["zone"] == "anchor"]
    zone_b = rows[rows["zone"] == "B"]
    cells = [(float(c.az), float(c.phase)) for c in anchor_conditions()]
    out = {"flyers": {}}
    rates = {}
    rates_x = {}
    for flyer in list(dict.fromkeys(list(flyers) + list(pi0))):
        a = anchor[anchor["flyer"] == flyer]
        counts = {}
        for cell, outcome, err in zip(a["cell"].tolist(), a["outcome"].tolist(), a["error"].tolist()):
            if err:
                continue
            c, m = counts.get(tuple(float(x) for x in cell), (0, 0))
            counts[tuple(float(x) for x in cell)] = (c + (outcome == "crash"), m + 1)
        zb = zone_b[zone_b["flyer"] == flyer]
        grid = {}
        for cell, outcome, err in zip(zb["cell"].tolist(), zb["outcome"].tolist(), zb["error"].tolist()):
            if err:
                continue
            key = tuple(float(x) for x in cell)
            c, m = grid.get(key, (0, 0))
            grid[key] = (c + (outcome == "crash"), m + 1)
        rec = []
        for az, ph in cells:
            speed = float(ANCHOR_SPEEDS[(az, ph)])
            c, m = counts.get((az, ph), (0, 0))
            gc, gm = grid.get((az, 0.0, speed, ph), (0, 0))
            if m == 0:
                raise MissingCell(f"{flyer}: no scored anchor row on cell (az {az:g}, phase {ph:g})")
            if gm == 0:
                raise MissingCell(f"{flyer}: no scored Zone B row on cell (az {az:g}, el 0, {speed:g} m/s, phase {ph:g})")
            rec.append({"az": az, "phase": ph, "speed": speed, "crashes": int(c), "n": int(m), "rate": c / m,
                        "grid_crashes": int(gc), "grid_n": int(gm), "grid_rate": gc / gm,
                        "rate_exact": Fraction(int(c), int(m)), "grid_rate_exact": Fraction(int(gc), int(gm))})
        rates[flyer] = np.array([r["rate"] for r in rec])
        rates_x[flyer] = [r["rate_exact"] for r in rec]
        if flyer in flyers:
            mean_x = _mean(rates_x[flyer])
            out["flyers"][flyer] = {"cells": rec, "mean": float(mean_x), "mean_exact": mean_x}
    out["flyers"] = {f: out["flyers"][f] for f in flyers}
    if len(pi0):
        mx = [rates_x[f] for f in pi0]
        cell_means_x = [_mean(col) for col in zip(*mx)]
        lineage_means_x = [_mean(row) for row in mx]
        mean_x = _mean(lineage_means_x)
        out["pi0"] = {"cell_means": np.array([float(x) for x in cell_means_x]), "lineage_means": np.array([float(x) for x in lineage_means_x]),
                      "mean": float(mean_x), "cell_means_exact": cell_means_x, "lineage_means_exact": lineage_means_x, "mean_exact": mean_x}
    return out


def n8_limits(rows, flyer: str, *, speeds: Sequence[float] = ZONE_B_SPEEDS, directions: Sequence[tuple] = DIRECTIONS) -> dict:
    """N8: {limit name: {"v50": [Edge per direction], "v10": [...], "v50_by_timing": [(Edge, Edge) per direction], "v10_by_timing": [...],
    "integrated": (D,) array}} for each name in analysis.sensitivities.S6_LIMITS, in its order, from
    `sensitivities.s6_rescore(rows, name)` on this flyer's Zone B rows. No flight."""
    mine = rows[(rows["flyer"] == flyer) & (rows["zone"] == "B")]
    out = {}
    for name in list(_sens.S6_LIMITS):
        g = flyer_grid(_sens.s6_rescore(mine, name), flyer, speeds=speeds, directions=directions)
        edges = n2_edges(g)
        out[name] = {"v50": [e.v50 for e in edges], "v10": [e.v10 for e in edges],
                     "v50_by_timing": [tuple(e.v50_by_timing) for e in edges], "v10_by_timing": [tuple(e.v10_by_timing) for e in edges],
                     "integrated": n3_integrated(g), "integrated_exact": n3_integrated_exact(g)}
    return out


def n9_calm(bars: Mapping, tuning: Mapping) -> dict:
    """N9, read from 4D-B's stored lineage file (`data/4d-lineages.json`: 4B's `bars.json` shape with `k_yaw`, `crash_penalty`,
    `yaw_readings` and `selection.failed_on`) and 4B's `tuning-pick.json`, never recomputed (`training.bar.read_bar` and
    `training.behaviour.yaw_acceptance` are not called): {"b0", "k_yaw", "crash_penalty", "tried", "failures", "failed_on":
    {seed: "bar" | "yaw" | "both"}, "lineages": {nominal seed: the stored bar reading dict}, "yaw": {nominal seed: the stored
    yaw reading dict}, "pid": {"median_rmse_m", "crashes"}, "geometric_tuning_figure_m", "geometric_label": "tuning figure"}
    (the yaw addendum of 2026-11-08, 4DR-5)."""
    selection = bars["selection"]
    seeds = [int(s) for s in selection["lineages"]]
    readings = bars["readings"]
    yaw_readings = bars.get("yaw_readings", {})
    lineages = {s: readings[f"pi0-seed-{s}"] for s in seeds}
    yaw = {s: yaw_readings[f"pi0-seed-{s}"] for s in seeds}
    first = lineages[seeds[0]] if seeds else {}
    pid = {"median_rmse_m": first.get("pid_median_rmse_m", float("nan")), "crashes": int(first.get("pid_crashes", -1))}
    return {"b0": int(bars["b0"]), "k_yaw": float(bars["k_yaw"]), "crash_penalty": float(bars["crash_penalty"]),
            "tried": int(selection["tried"]), "failures": [int(s) for s in selection["failures"]],
            "failed_on": {int(k): str(v) for k, v in dict(selection.get("failed_on", {})).items()},
            "lineages": lineages, "yaw": yaw, "pid": pid,
            "geometric_tuning_figure_m": float(tuning["tuning_figure_m"]), "geometric_label": "tuning figure"}


def n10_extension(pi0_grids: Sequence[FlyerGrid]) -> dict:
    """N10: {"directions", "rate_at_12": (D,) the mean π0 crash rate at 12 m/s (the mean over the 16 π0 of each lineage's plain mean of its two
    12 m/s cells' rates, on exact fractions: ruling 5A-8), "fires": True when it is under 0.8 on any direction (strictly),
    "speeds": the grid's speeds with 16 added when it fires (and not already there). The speeds must include 12."""
    from fractions import Fraction

    speeds = tuple(pi0_grids[0].speeds)
    if 12.0 not in speeds:
        raise ValueError("the extension's first reading is on the 12 m/s rows, and the grid has none")
    s12 = speeds.index(12.0)
    d_count = pi0_grids[0].crashes.shape[0]
    rates = []
    for d in range(d_count):
        total = Fraction(0)
        for g in pi0_grids:
            cells = [Fraction(int(g.crashes[d, s12, t]), int(g.n[d, s12, t])) for t in range(g.crashes.shape[2])]
            total += sum(cells) / len(cells)
        rates.append(total / len(pi0_grids))
    fires = any(r < Fraction(4, 5) for r in rates)
    top = speeds if (not fires or 16.0 in speeds) else speeds + (16.0,)
    return {"directions": tuple(pi0_grids[0].directions), "rate_at_12": np.array([float(r) for r in rates]), "fires": bool(fires), "speeds": tuple(top),
            "rate_at_12_exact": rates}
