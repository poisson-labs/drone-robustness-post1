"""P1a-fix audit: for every printed number built from integer counts, how many values there are, how many sit exactly on a half in
the last printed place (a tie), and how many of those a float print gets differently from the exact half-even print.

    python scripts/p1a_fix_audit.py --store STORE [--store STORE ...] [--bars data/4d-lineages.json]

Read-only. Reads the stored rows through the loader and the exact values through `drone_study.boundary.results`.
"""
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

from drone_study.boundary import results as R
from drone_study.boundary import sweeps as S
from drone_study.boundary import writer as W


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", action="append", required=True)
    ap.add_argument("--bars", default=S.LINEAGES_FILE)
    args = ap.parse_args()
    rows = W.load_rows(args.store)
    pi0, controllers = W._flyer_lists(rows)
    everyone = pi0 + controllers
    speeds = W._resolve_speeds(rows, None)
    dirs = S.DIRECTIONS
    cats: "OrderedDict[str, list]" = OrderedDict()

    def add(name, places, values):
        cats.setdefault(name, [places, []])[1].extend(values)

    grids = {f: R.flyer_grid(rows, f, speeds=speeds, directions=dirs) for f in everyone}
    pg = [grids[f] for f in pi0]
    add("N1 rate", 4, [c["value_exact"] for c in R.n1_cells(rows, everyone, speeds=speeds, directions=dirs)])
    for f in everyone:
        for e in R.n2_edges(grids[f]):
            add("N2 v50 / v10 pooled", 2, [R.edge_exact(e.v50), R.edge_exact(e.v10)])
            add("N2 v50 / v10 by timing", 2, [R.edge_exact(x) for x in (*e.v50_by_timing, *e.v10_by_timing)])
        add("N3 integrated", 4, R.n3_integrated_exact(grids[f]))
    n4 = R.n4_across_lineages(pg)
    for m, p in (("v50", 2), ("v10", 2), ("integrated", 4)):
        add(f"N4 {m} mean", p, n4[m]["mean_exact"])
        add(f"N4 {m} values", p, [v for row in n4[m]["values_exact"] for v in row])
    for c in controllers:
        n5 = R.n5_against_controller(pg, grids[c])
        for m, p in (("v50", 2), ("integrated", 4)):
            part = n5[m]
            add(f"N5 {m} differences", p, [v for row in part["differences_exact"] for v in row])
            for k in ("mean", "lowest", "highest"):
                add(f"N5 {m} {k}", p, part[f"{k}_exact"])
    n7 = R.n7_anchor(rows, everyone, pi0)
    for f in everyone:
        add("N7 cell rate and grid rate", 4, [x for c in n7["flyers"][f]["cells"] for x in (c["rate_exact"], c["grid_rate_exact"])])
        add("N7 flyer mean", 4, [n7["flyers"][f]["mean_exact"]])
    add("N7 lineage means and pi0 mean", 4, list(n7["pi0"]["lineage_means_exact"]) + [n7["pi0"]["mean_exact"]])
    for f in everyone:
        n8 = R.n8_limits(rows, f, speeds=speeds, directions=dirs)
        for part in n8.values():
            add("N8 edges", 2, [R.edge_exact(x) for k in ("v50", "v10") for x in part[k]]
                + [R.edge_exact(x) for k in ("v50_by_timing", "v10_by_timing") for pair in part[k] for x in pair])
            add("N8 integrated", 4, part["integrated_exact"])
    add("N10 rate at 12", 4, R.n10_extension(pg)["rate_at_12_exact"])

    print(f"{'category':36s} {'places':>6s} {'values':>8s} {'ties':>6s} {'float(exact) print differs':>20s}")
    tot = [0, 0, 0]
    for name, (places, vals) in cats.items():
        ties = [v for v in vals if R.is_tie(v, places)]
        differs = [v for v in ties if format(float(v), f".{places}f") != R.format_exact(v, places)]
        print(f"{name:36s} {places:6d} {len(vals):8d} {len(ties):6d} {len(differs):20d}")
        tot = [tot[0] + len(vals), tot[1] + len(ties), tot[2] + len(differs)]
    print(f"{'total':36s} {'':6s} {tot[0]:8d} {tot[1]:6d} {tot[2]:20d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
