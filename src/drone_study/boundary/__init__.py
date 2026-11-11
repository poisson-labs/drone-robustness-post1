"""Boundary before: Zone B and the anchor slice on the 20 post 1 flyers, and every post 1 number (spec 3, 9, 20).

Session 5A (boundary tests) ships this package as STUBS. Every function has the signature the frozen tests call and
returns an inert value of the right shape, so the tests import, run, and fail on the assertion each was written for.
Session 5B replaces the bodies. Names, argument order and the meaning of each field are the interface 5B inherits;
changing one is a ruling (docs/boundary-invariant-table.md, section 5).

  pins        PROVE_VERSION for post 1
  sweeps      the sweep files, the salts, the 20 flyers, the Prove commands, the row counts
  results     N1 to N8 and N10 as functions from stored rows, N9 from 4B's stored readings, the flat force ratio
  writer      RESULTS-P1.md and CHECK-P1.md, and the one command that regenerates them

The one-policy estimator is `drone_study.analysis.estimator`; the integrated crash rate is
`drone_study.analysis.boundary.integrated_rate` (session 8A pinned it). Post 2 uses the same functions.

Row table for post 1, one row per stored episode (a pandas DataFrame):
  flyer        one of the 20 flyer names: "pi0-seed-1" to "pi0-seed-16", "pid", "geometric", "wind_only", "oracle"
  zone         "B" (Zone B, including its 16 m/s extension rows) or "anchor"
  cell         B        (az, el, speed, phase)       floats
               anchor   (az, phase)                  floats; the anchor's speed is fixed by section 3's table
  seed         int, Prove's seed for the row (the same for every flyer on the same cell and replicate)
  outcome      "ok" or "crash"
  error        bool; True for a Prove error row (never a crash, never in a count)
  crash_class  "ground", "tilt", "radius", or None for a pass and for an error row
  crossings    dict, what drone_study.score.rescore reads (S6); may be None where S6 is not read
  rmse         float, metres
  effort       float, rotor effort
`outcome`, `error` and `crossings` are 8A's columns, so analysis.sensitivities.s6_rescore reads this table unchanged.
Units, everywhere: a crash rate is a fraction; speed is m/s; a force ratio is a fraction of the lateral limit.
"""
