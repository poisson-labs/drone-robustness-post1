# Post 1: check sheet (CHECK-P1)

For session P1a (spec §20.6): recompute N1 to N10 of RESULTS-P1.md from the raw rows with independent code, working from this sheet. A disagreement is named and settled by §20.3, point 14.

Prove: poisson-prove 0.1.2
Zone B salt: 9c3d4a1e317d59fa
Anchor salt: 644f7723497d4cae
RESULTS-P1.md sha256: db173c86fbcffe86290ba67a68fbd572f992602d120292dbb92139e97d405bec
Grid top: 16 m/s

Regenerate both files from the stores, by one command:

```
python -m drone_study.boundary.writer --store <STORE> [--store <STORE> ...] --bars data/4d-lineages.json --tuning <tuning-pick.json> --out <DIR>
```

## Where the rows are

Prove runs, one per flyer and zone, named `zone-b-<flyer>`, `anchor-<flyer>` and (only if the extension fired) `zone-b-extension-<flyer>`. Flyers: `pi0-seed-<nominal seed>` for the lineages of `data/4d-lineages.json`, then `pid`, `geometric`, `wind_only`, `oracle`.
A row's `config` is `{"cell": <name>}`, the name `c=<case>;az=..;el=..;v=<speed>;ph=<phase>;tb=..;pl=..;mi=..;ms=..;dl=..;gl=..` (`drone_study.condition.cell_name`). Zone B and the anchor read az, el, v and ph; every other field is the default. The outcome is `ok` or `crash`; the crash class, every threshold's first crossing, the RMSE and the rotor effort are in `meta`. A Prove error row is in no count (ruling 5A-6).
Seeds: `derive_seed(salt, config_hash(cfg), k)` for k in 0..23 (Zone B and the extension, salt 9c3d4a1e317d59fa) and k in 0..15 (the anchor, salt 644f7723497d4cae), from `poisson_prove.executor`; one salt per zone gives every flyer the same seeds on a cell. Two flyers' rows are paired on (cell, seed).

## N1

rows read: 168960

Defined in §20.1 and §20.2.

Per flyer and Zone B cell (direction, speed, timing): crashes and scored episodes, the crash classes among the crashes, the rate, and Prove's 95% Wilson interval for the cell (`poisson_prove.stats.share(crashes, n)`). The force ratio is the primary air model's flat force at hover rotor speed over the lateral limit, `.2f`.

## N2

rows read: 168960

Defined in §9 and §20.1.

Per flyer and direction: pool crashes and episodes over the two timings at each speed, fit a monotone curve by pooled adjacent violators, v50 is where it first reaches 0.5 and v10 where it first reaches 0.1, by straight-line interpolation; the same fit on each timing alone. Rules 5A-1 to 5A-3 below.

## N3

rows read: 168960

Defined in §9 and §20.1.

Per flyer and direction: the area under the raw pooled rate (crashes over scored episodes at each speed), trapezoid rule over the grid's speeds, divided by the range from the lowest speed to the top.

## N4

rows read: 168960

Defined in §20.1 (N4).

Across the π0, per direction: the mean of N2's v50 and v10 and of N3 over the lineages, every lineage's value, and the number censored at the top and at the bottom. A censored value counts as the grid's end it sits at (ruling 5A-7).

## N5

rows read: 168960

Defined in §20.1 (N5).

For each of the four non-π0 flyers: each lineage's value minus the controller's, for v50 and for the integrated rate, the mean, the lowest and the highest; a pair with a censored edge on either side is marked `*` (ruling 5A-9).

## N6

rows read: 168960

Defined in §20.1 (N6) and §20.2.

For each π0 and each of the four controllers, per Zone B cell, on pairs matched by seed: both crash, only the π0 crashes, only the controller crashes, neither. A pair with an error row on either side is out (ruling 5A-6). Counts only.

## N7

rows read: 5120

Defined in §3 (the anchor), §20.1 (N7) and ruling 5A-11.

Per flyer, the anchor's 16 cells: crashes and scored episodes (of 16), beside the same cell's Zone B counts at (az, 0, the anchor's speed, timing); the mean of the 16 rates; for the π0, each lineage's mean, each cell's mean over the lineages, and the mean of the lineage means.

## N8

rows read: 168960

Defined in §10 (S6), §20.1 (N8) and ruling 5A-10.

N2 and N3 again, per flyer, under each of S6's six limits: tilt 75° and 90° instantaneous, radius 0.3, 0.75 and 1.00 m, and `ground` and `tilt` only. Each stored row is re-scored from the crossings it stored by `drone_study.score.rescore`; nothing is flown.

## N9

rows read: 0

Defined in §2, §20.1 (N9) and the yaw addendum (4DR-5).

Read from `data/4d-lineages.json` and the geometric controller's tuning pick, and not recomputed from rows. The bar's seeds are the `bar` salt's.

## N10

rows read: 12288

Defined in §3 (grid extension), §20.1 (N10) and ruling 5A-8.

Per direction: the mean over the π0 of each lineage's plain mean of its two 12 m/s cells' rates, on exact fractions. 16 m/s is added when any direction is strictly under 0.8.

## Rules

What §9 leaves to code, and how this build reads it (all ruled yes by Taylor on 2026-11-07):

- 5A-1. A fitted curve that sits at exactly the level (0.5 for v50, 0.1 for v10) over a stretch crosses at the stretch's lowest speed, and the edge is not censored.
- 5A-2. "Reaches" and "above" are read on the counts, in exact arithmetic: a pooled block's rate is a ratio of integers, so a block that is exactly one half is one half.
- 5A-3. A speed with no scored episodes is left out of that fit, and a fit with none left is an error. Censoring: never reaches the level, the grid's top marked top; starts above the level, the lowest speed marked bottom.
- 5A-6. An error row is in no count; a cell with no scored row stops the read; a pair with an error row on either side is out of N6.
- 5A-7. A mean over the lineages (N4) and a difference (N5) use a censored value as the grid's end it sits at, with the count of censored values printed.
- 5A-8. N10's rate is the mean over the π0 of each lineage's plain mean of its two 12 m/s cells' rates; "under 80%" is strictly under.
- 5A-12. The flat force ratio is the primary model's force on a horizontal gust of that speed at hover rotor speed over the lateral limit (0.534 N), printed to two decimals.
- P1A-2 (print rule, ruled yes by Taylor on 2026-11-09). Every published number is the exact value rounded half to even to the printed precision. A value built from integer counts (a rate, an edge, an integrated rate, and a mean, difference, lowest or highest of them) is a ratio of integers, so it is computed as an exact fraction and rounded once, at print: 2 decimals for an edge, 4 for a rate. Where it sits exactly on a half in the last printed place, the even digit wins (9/32 prints 0.2812, 7.175 prints 7.18, 7.025 prints 7.02). Do not round through a float: a float of a non-dyadic tie is off the tie and rounds the wrong way.
