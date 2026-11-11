"""Zone B's grid constants and the integrated crash rate (spec section 9).

The one-policy estimator of section 9 (pooled adjacent violators, v50 by straight-line interpolation, the censoring marks) is
`drone_study.analysis.estimator`, frozen by session 5A. This public slice carries from this module the integrated crash rate
(one of the three functions ruling 5A-5 handed to post 1) and the grid constants; the study's across-lineage contrasts, which
read retrained arms and belong to its second post, are not in this repository and are removed from this module."""
from __future__ import annotations

from fractions import Fraction
from typing import Sequence

# (azimuth, elevation) in degrees. Section 3: 8 at elevation 0, plus 0, 90, 180, 270 at +35 and at -35.
HORIZON_DIRECTIONS = tuple((float(az), 0.0) for az in (0, 45, 90, 135, 180, 225, 270, 315))
OFF_HORIZON_DIRECTIONS = tuple((float(az), el) for el in (35.0, -35.0) for az in (0, 90, 180, 270))
ALL_DIRECTIONS = HORIZON_DIRECTIONS + OFF_HORIZON_DIRECTIONS
ZONE_B_SPEEDS = (2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0)
ZONE_B_PHASES = (0.0, 0.125)


def integrated_rate_exact(crashes: Sequence[float], n: Sequence[float], speeds: Sequence[float]) -> Fraction:
    """`integrated_rate` on exact fractions. The counts are whole numbers and the speeds are the grid's (whole m/s), so the
    trapezoid over the grid is a rational number; nothing here rounds. Same rules as `integrated_rate`: a speed with no scored
    episode is left out, and fewer than two speeds left is an error. Each input is taken as the exact value of its float."""
    if not (len(crashes) == len(n) == len(speeds)):
        raise ValueError("crashes, n and speeds differ in length")
    pts = []
    for c, m, v in zip(crashes, n, speeds):
        fm = Fraction(float(m))
        if fm <= 0:
            continue
        pts.append((Fraction(float(v)), Fraction(float(c)) / fm))
    if len(pts) < 2:
        raise ValueError("the integrated rate needs at least two speeds with scored episodes")
    area = sum(((r0 + r1) / 2) * (s1 - s0) for (s0, r0), (s1, r1) in zip(pts[:-1], pts[1:]))
    return area / (pts[-1][0] - pts[0][0])


class ExactFloat(float):
    """A float that also carries the exact fraction it was rounded from (`.exact`). It is a float in every use; the printers read
    `.exact`, so a value that sits on a tie at the printed precision is read exactly (P1a-fix)."""
    exact: Fraction


def exact_of(x) -> Fraction:
    """The exact value behind `x`: its `.exact` when it has one (what `integrated_rate` returns), else the exact value of the float."""
    e = getattr(x, "exact", None)
    return e if e is not None else Fraction(float(x))


def integrated_rate(crashes: Sequence[float], n: Sequence[float], speeds: Sequence[float]) -> float:
    """The area under the raw crash curve (crashes / n at each speed, timings already pooled), by the trapezoid
    rule from the lowest speed to the grid's top, divided by that range (section 9). Nothing censors it, and the fit
    is never used: the raw rate at each speed is crashes over scored episodes. A speed with no scored episode is left out
    (as 5A-3 does for the edges); if fewer than two speeds remain there is no area and the call is an error.
    Computed on exact fractions (`integrated_rate_exact`) and rounded once, to the nearest float (P1a-fix, ruling P1A-1). The
    float returned carries the exact fraction as `.exact` (an `ExactFloat`); `exact_of` reads it."""
    exact = integrated_rate_exact(crashes, n, speeds)
    out = ExactFloat(exact)
    out.exact = exact
    return out
