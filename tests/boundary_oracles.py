"""Closed forms and exact arithmetic the boundary tests compare against, written from the spec's text and never imported
from drone_study (the tests must not borrow their expected values from the code they test)."""
from fractions import Fraction

import numpy as np

import spec_numbers as S


# ---- section 9's estimator, in exact arithmetic ---------------------------------------------------------------------
def pav_exact(crashes, n):
    """Pooled adjacent violators on crashes / n with weights n, every block a ratio of integers. Returns Fractions."""
    blocks = [[int(c), int(m), 1] for c, m in zip(crashes, n)]
    i = 0
    while i < len(blocks) - 1:
        a, b = blocks[i], blocks[i + 1]
        if Fraction(a[0], a[1]) > Fraction(b[0], b[1]):
            blocks[i:i + 2] = [[a[0] + b[0], a[1] + b[1], a[2] + b[2]]]
            i = max(i - 1, 0)
        else:
            i += 1
    out = []
    for c, m, k in blocks:
        out += [Fraction(c, m)] * k
    return out


def edge_exact(crashes, n, speeds, level=0.5):
    """(value, censored): the speed where the fitted curve first reaches `level`, interpolated between the grid speed
    below it and the first one at or above; censored 'top' if the curve never reaches it, 'bottom' if it starts above it.
    A speed with no scored episodes is left out. The curve at exactly `level` over a stretch gives the stretch's lowest speed."""
    crashes, n, speeds = (np.asarray(v) for v in (crashes, n, speeds))
    keep = n > 0
    c, m, s = crashes[keep], n[keep], speeds[keep].astype(float)
    lvl = Fraction(str(level))
    iso = pav_exact(c, m)
    if iso[-1] < lvl:
        return float(s[-1]), "top"
    if iso[0] > lvl:
        return float(s[0]), "bottom"
    i = next(j for j, v in enumerate(iso) if v >= lvl)
    if i == 0:
        return float(s[0]), None
    return float(s[i - 1] + float((lvl - iso[i - 1]) / (iso[i] - iso[i - 1])) * (s[i] - s[i - 1])), None


def integrated(crashes, n, speeds):
    """Trapezoid over the raw rates, divided by the range (section 9)."""
    r = np.asarray(crashes, dtype=float) / np.asarray(n, dtype=float)
    s = np.asarray(speeds, dtype=float)
    return float(((r[1:] + r[:-1]) / 2 * np.diff(s)).sum() / (s[-1] - s[0]))


# ---- section 17.3's flat force, closed form ----------------------------------------------------------------------------
def flat_force_n(speed):
    """Rotor and frame linear drag at hover rotor speed plus the body's quadratic drag in the rotor plane, newtons."""
    return S.K_LIN_XY * S.SUM_OMEGA_HOVER * speed + 0.5 * S.RHO * S.CDA_XY * speed * speed


def flat_force_ratio(speed):
    return flat_force_n(speed) / S.LATERAL_LIMIT_N
