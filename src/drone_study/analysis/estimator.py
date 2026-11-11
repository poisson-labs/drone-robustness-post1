"""Section 9's estimator for one policy and one direction (session 5A freezes it; spec 20.3, point 1).

"Pool crash counts across the two timings at each of the 10 speeds. Fit a monotone curve by pooled adjacent violators.
`v50` is where it crosses 0.5, by straight-line interpolation. Same for `v10`, the safe-side edge. ... If the curve never
reaches 0.5, `v50` is the grid's top, marked as censored. If it starts above 0.5, `v50` is 2 m/s, marked likewise."

Post 2 calls these same functions (20.3, point 1): `analysis.boundary.boundary_contrast` reads its edges from `edge` and
its rates from `analysis.boundary.integrated_rate`, which session 8A froze and this module does not repeat.

What section 9 leaves to code, picked once here (rulings 5A-1 to 5A-3, docs/boundary-invariant-table.md):
  - a fitted curve that sits at exactly `level` over a stretch crosses where it first reaches it: the edge is the lowest
    speed of the stretch, and it is not censored (5A-1);
  - "reaches" and "above" are read on the counts, in exact arithmetic: a pooled block's rate is a ratio of integers, so
    a block that is exactly one half is one half whatever floats would say (5A-2);
  - a speed with no scored episodes is left out of the fit; a fit with none left is an error (5A-3, as 8A-18).
Units: a rate is a fraction; speed is m/s. Arrays are (..., speeds, timings) for the counts the callers hold.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional, Sequence

import numpy as np

TOP = "top"
BOTTOM = "bottom"


@dataclass(frozen=True)
class Edge:
    """One edge of one curve. `censored` is None, "top" (the curve never reached the level: the value is the grid's top)
    or "bottom" (the curve starts above the level: the value is the grid's lowest speed). `n_per_speed` is the number of
    scored episodes the fit stood on at each speed it used, so a reader sees 24 or 48. `exact` is the same edge as an exact
    fraction (the crossing is rational: the counts are integers and the speeds are the grid's); it is what the printed numbers
    are built from (P1a-fix) and is left out of equality, so an Edge compares as it always did."""
    value: float
    censored: Optional[str] = None
    n_per_speed: tuple = ()
    exact: Optional[Fraction] = field(default=None, compare=False)


def pool_timings(crashes, n, axis: int = -1):
    """Counts summed over the timings axis: (crashes, n). Pooling is on counts, never on rates."""
    return np.asarray(crashes).sum(axis=axis), np.asarray(n).sum(axis=axis)


def _as_int(v) -> int:
    f = float(v)
    i = int(round(f))
    if abs(f - i) > 1e-9:
        raise ValueError(f"a count must be a whole number, got {v!r}")
    return i


def _exact_blocks(crashes: Sequence[float], n: Sequence[float]) -> list:
    """Pooled adjacent violators in exact arithmetic: a list of [crashes, n, width] blocks, each rate a ratio of integers."""
    if len(crashes) != len(n):
        raise ValueError("crashes and n differ in length")
    blocks = []
    for c, m in zip(crashes, n):
        c, m = _as_int(c), _as_int(m)
        if m <= 0:
            raise ValueError("every n must be positive")
        blocks.append([c, m, 1])
    i = 0
    while i < len(blocks) - 1:
        a, b = blocks[i], blocks[i + 1]
        if Fraction(a[0], a[1]) > Fraction(b[0], b[1]):
            blocks[i:i + 2] = [[a[0] + b[0], a[1] + b[1], a[2] + b[2]]]
            i = max(i - 1, 0)
        else:
            i += 1
    return blocks


def _exact_fit(crashes, n) -> list:
    out = []
    for c, m, k in _exact_blocks(crashes, n):
        out += [Fraction(c, m)] * k
    return out


def monotone_fit(crashes: Sequence[float], n: Sequence[float]) -> np.ndarray:
    """Pooled adjacent violators on crashes / n, weights n, non-decreasing in speed. Every n must be positive."""
    return np.array([float(f) for f in _exact_fit(crashes, n)], dtype=float)


def edge(crashes: Sequence[float], n: Sequence[float], speeds: Sequence[float], level: float = 0.5) -> Edge:
    """The speed where the fitted curve first reaches `level`, by straight-line interpolation between the grid speed
    below and the first one at or above it. `crashes` and `n` are one number per speed, timings already pooled.
    A speed with no scored episodes is left out of the fit (5A-3); a fit with none left is a ValueError. A curve that
    sits at exactly `level` over a stretch crosses at the stretch's lowest speed and is not censored (5A-1). "Reaches" and
    "above" are read on the counts, in exact arithmetic (5A-2)."""
    crashes = np.asarray(crashes)
    n = np.asarray(n)
    speeds = np.asarray(speeds, dtype=float)
    if not (len(crashes) == len(n) == len(speeds)):
        raise ValueError("crashes, n and speeds differ in length")
    keep = [j for j in range(len(n)) if _as_int(n[j]) > 0]
    if not keep:
        raise ValueError("no speed has a scored episode: there is nothing to fit")
    c = [crashes[j] for j in keep]
    m = [n[j] for j in keep]
    s = [float(speeds[j]) for j in keep]
    fs = [Fraction(v) for v in s]
    used = tuple(_as_int(v) for v in m)
    lvl = Fraction(str(level))
    iso = _exact_fit(c, m)
    if iso[-1] < lvl:
        return Edge(s[-1], TOP, used, fs[-1])
    if iso[0] > lvl:
        return Edge(s[0], BOTTOM, used, fs[0])
    i = next(j for j, v in enumerate(iso) if v >= lvl)
    if i == 0:
        return Edge(s[0], None, used, fs[0])
    exact_frac = (lvl - iso[i - 1]) / (iso[i] - iso[i - 1])
    frac = float(exact_frac)
    return Edge(s[i - 1] + frac * (s[i] - s[i - 1]), None, used, fs[i - 1] + exact_frac * (fs[i] - fs[i - 1]))


def v50(crashes: Sequence[float], n: Sequence[float], speeds: Sequence[float]) -> Edge:
    return edge(crashes, n, speeds, 0.5)


def v10(crashes: Sequence[float], n: Sequence[float], speeds: Sequence[float]) -> Edge:
    return edge(crashes, n, speeds, 0.1)


@dataclass(frozen=True)
class DirectionEdges:
    """N2 for one direction of one flyer: the pooled edges, and the same fit on each timing alone (24 episodes a speed)."""
    v50: Edge
    v10: Edge
    v50_by_timing: tuple = ()
    v10_by_timing: tuple = ()


def direction_edges(crashes, n, speeds: Sequence[float]) -> DirectionEdges:
    """`crashes` and `n` are (speeds, timings) for one direction. Pooled over timings, and each timing alone."""
    crashes = np.asarray(crashes)
    n = np.asarray(n)
    pc, pn = pool_timings(crashes, n, axis=-1)
    timings = crashes.shape[-1]
    return DirectionEdges(
        v50(pc, pn, speeds), v10(pc, pn, speeds),
        tuple(v50(crashes[:, t], n[:, t], speeds) for t in range(timings)),
        tuple(v10(crashes[:, t], n[:, t], speeds) for t in range(timings)),
    )
