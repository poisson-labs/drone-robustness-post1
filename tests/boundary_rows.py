"""Synthetic post 1 rows with known answers, in the row table drone_study.boundary documents (one row per episode).

Every builder is deterministic. A cell's seeds are the same for every flyer, as one Prove salt per zone makes them
(spec section 6). A flyer with k crashes in a cell crashes on the cell's first k replicates, so two flyers' crashes are
nested. A crash's class is CRASH_CLASSES[(j + d) % 3] for its j-th crash in direction d, so every direction splits."""
import zlib

import numpy as np
import pandas as pd

import boundary_numbers as B

COLUMNS = ["flyer", "zone", "cell", "seed", "outcome", "error", "crash_class", "crossings", "rmse", "effort"]
SHAPE = (len(B.DIRECTIONS), len(B.SPEEDS), len(B.PHASES))


def cell_seed(zone, cell, k):
    return (zlib.crc32(repr((zone, cell)).encode()) % 1_000_003) * 1000 + k


def crash_class(j, d):
    return B.CRASH_CLASSES[(j + d) % 3]


def class_counts(crashes, d):
    """(ground, tilt, radius) counts for `crashes` crashes in direction index d, by the rule above."""
    out = [0, 0, 0]
    for j in range(int(crashes)):
        out[(j + d) % 3] += 1
    return tuple(out)


def frame(rows):
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["cell"] = df["cell"].astype(object)
    df["crossings"] = df["crossings"].astype(object)
    return df


def zone_b_rows(flyer, crashes, n=B.ZONE_B_REPLICATES, *, speeds=B.SPEEDS, directions=B.DIRECTIONS, phases=B.PHASES,
                errors=None, crossings=None):
    """Rows for Zone B cells from (directions, speeds, timings) crash counts. `n` is an int or an array of that shape (scored
    rows per cell); `errors` an array of that shape (error rows added after them); `crossings` a function
    (direction index, speed index, timing index, k) -> dict or None."""
    crashes = np.asarray(crashes)
    n = np.broadcast_to(np.asarray(n), crashes.shape)
    errors = np.zeros(crashes.shape, int) if errors is None else np.broadcast_to(np.asarray(errors), crashes.shape)
    rows = []
    for d, (az, el) in enumerate(directions):
        for s, speed in enumerate(speeds):
            for t, phase in enumerate(phases):
                cell = (float(az), float(el), float(speed), float(phase))
                c = int(crashes[d, s, t])
                for k in range(int(n[d, s, t])):
                    x = None if crossings is None else crossings(d, s, t, k)
                    crashed = k < c
                    rows.append((flyer, "B", cell, cell_seed("B", cell, k), "crash" if crashed else "ok", False,
                                 crash_class(k, d) if crashed else None, x, 0.05, 1.0))
                for k in range(int(n[d, s, t]), int(n[d, s, t]) + int(errors[d, s, t])):
                    rows.append((flyer, "B", cell, cell_seed("B", cell, k), "error", True, None, None, float("nan"), float("nan")))
    return rows


def anchor_cells():
    """The anchor's 16 (az, phase) in section 3's table order: timing 0 across the azimuths, then timing 0.125."""
    return [(az, ph) for ph in B.PHASES for az in B.HORIZON_AZ]


def anchor_rows(flyer, crashes, n=B.ANCHOR_REPLICATES, errors=None):
    """Rows for the anchor's 16 cells from 16 crash counts in `anchor_cells()` order."""
    rows = []
    for i, (az, ph) in enumerate(anchor_cells()):
        cell = (float(az), float(ph))
        c = int(crashes[i])
        for k in range(n):
            crashed = k < c
            rows.append((flyer, "anchor", cell, cell_seed("anchor", cell, k), "crash" if crashed else "ok", False,
                         crash_class(k, i) if crashed else None, None, 0.05, 1.0))
        for k in range(n, n + (0 if errors is None else int(errors[i]))):
            rows.append((flyer, "anchor", cell, cell_seed("anchor", cell, k), "error", True, None, None, float("nan"), float("nan")))
    return rows


# ---- curves ---------------------------------------------------------------------------------------------------------------
def steps(threshold, n=B.ZONE_B_REPLICATES, shape=SHAPE):
    """(D, S, T) crash counts for curves that are 0 below speed index threshold[d] and n from it on."""
    out = np.zeros(shape, int)
    for d, t in enumerate(np.broadcast_to(np.asarray(threshold), (shape[0],))):
        out[d, int(t):, :] = n
    return out


def per_speed(counts_by_speed, shape=SHAPE):
    """(D, S, T) counts from one length-S list of per-timing counts used in every direction and timing."""
    return np.broadcast_to(np.asarray(counts_by_speed)[None, :, None], shape).astype(int).copy()
