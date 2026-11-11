"""Session 4C's behaviour checks as frozen functions: the heading numbers of a flight, the yaw acceptance of the yaw addenda
(2026-11-07 and 2026-11-08; rulings 4DA-4 and 4DR-4), and the altitude offset (reported, not gated).

Definitions, as session 4C's `common.py` made them and the 2026-11-08 addendum extends them, on the TRUE state of the
episode record, one row per control step:
  heading         the Z-Y-X yaw of the true attitude: atan2(2 (w z + x y), 1 - 2 (y^2 + z^2)) from the quaternion, equal to
                  atan2(R[1, 0], R[0, 0]) of the rotation; unwrapped along the episode (numpy.unwrap), degrees
  yaw-rate RMS    root mean square of the true body z rate over the rows, deg/s (4C's number; the rate line reads it)
  heading rate    the Euler yaw rate, (q sin(roll) + r cos(roll)) / cos(pitch) from the true body rates and the Z-Y-X roll and
                  pitch; its RMS over the rows, deg/s. Reported beside the body z rate and not gated: 4DR-4 asked for the rate
                  line on it, and the 2026-11-08 re-fly of 4C's episodes read the geometric controller at 13.5 deg/s of heading
                  rate against 5.8 of body z rate (its heading wobbles within 6 degrees while it pitches and rolls), so a 12 deg/s
                  line on the heading rate fails a controller; the body z rate separates the controllers (3.3, 5.8) from every
                  pi0 (23.2 to 62.4) by a factor of four, the heading rate (4.2, 13.5 against 24.4 to 62.8) by less than two.
                  Ruling 4DA-10 puts the reversal to Taylor; the numbers are in evidence/drone-robustness/4d-yaw/heading-rate-check.md
  heading drift   the slope of a least-squares straight line through the unwrapped heading against time, deg/s
  largest |heading|   the largest absolute unwrapped heading over the rows, deg
  altitude bias   the mean of (true z - 3.05 m) over the rows, cm; and the same over the last second

The yaw acceptance, for one policy over the bar's 1,200 calm episodes (12 cases x 100 seeds, salt 76e7b463820f72ad), all
five lines, each inclusive ("at most"):
  - the median over the episodes of the yaw-rate RMS (body z rate) is at most 12 deg/s;
  - the median over the episodes of |heading drift| is at most 1 deg/s;
  - at most 12 of the 1,200 episodes (1%) have a largest |heading| past 45 deg;
  - at most 3 of the 1,200 (0.25%) have one past 90 deg;
  - the median over the episodes of the largest |heading| is at most 15 deg (4DR-4: a steady crab is not a held heading).
On session 4C's episodes (re-flown read-only for the heading rate, bit for bit the same: evidence/drone-robustness/4d-yaw/
heading-rate-check.md) every one of 4B's sixteen pi0 fails (all sixteen on the rate line, fifteen on the fifth, 13.9 degrees
for pi0 6) and the cascaded PID and the geometric controller pass every line (2.3 and 6.1 degrees on the fifth).

The bar's Prove rows carry the four heading numbers in `meta` beside `rmse` (the rollout writes them: ruling 4DA-8), so the
acceptance is read from the same stored sweep as the bar, on the same seeds, and nothing flies twice.

Interface (4D-B builds these; stubs here):
  YAW_RATE_RMS_MAX_DEG_S, YAW_DRIFT_MAX_DEG_S, PAST_45_MAX, PAST_90_MAX, HEADING_45_DEG, HEADING_90_DEG,
  MAX_HEADING_MEDIAN_MAX_DEG, YAW_META_FIELDS
  heading_deg(quat) -> (n,) unwrapped heading, degrees;  heading_rate_deg_s(quat, body_rate) -> (n,) Euler yaw rate, deg/s
  EpisodeYaw;  yaw_metrics(record) -> EpisodeYaw
  EpisodeAltitude;  altitude_metrics(record) -> EpisodeAltitude
  YawRow(case, seed, yaw: EpisodeYaw, error=False);  yaw_rows_from_store(store_path, run) -> list[YawRow]
  yaw_acceptance(rows) -> YawReading   rows: the bar's 1,200 YawRows (YawMismatch otherwise)

The motor line (ruling 4DB-1, 2026-11-09; added by 4D-B as an additive change). A third lineage line beside the bar and the yaw
gate, read from the same stored bar rows: over the bar's 1,200 calm episodes, the share of control steps on which any motor's
command is at its limit (0 or 65,535 units) is at most 0.5% (4C's hard-limit mark; the cascaded PID and the geometric controller
read 0.000% and 0.005%, 4B's sixteen pi0 0.005% to 0.42%). The reading is the record's mixer output at the end of each control
step: row k of `EpisodeRecord.mixer_output` is the last 500 Hz substep of step k - 1, so rows 1 .. n - 1 are the ends of steps
0 .. n - 2 and row 0 (the reset's zeros, no step flown yet) is not a reading. The rollout writes the episode's share into each
row's `meta` as `motor_limit_share`; the line pools the episodes' steps (each episode weighted by its readings, `steps` - 1).
  MOTOR_LIMIT_SHARE_MAX, MOTOR_META_FIELD, UNITS_LIMIT
  motor_limit_share(record) -> float;  MotorRow(case, seed, share, steps, error=False)
  motor_rows_from_store(store_path, run) -> list[MotorRow];  motor_acceptance(rows) -> MotorReading (MotorMismatch otherwise)

The line's sub-step reading (ruling 4DB-2, 2026-11-09; additive). The line is re-based on the reading 4C's numbers came from when
they set the 0.5%: a control step counts when any motor's command is at 0 or 65,535 units on ANY of its five 500 Hz substeps.
The stack's record carries it (`EpisodeRecord.motor_limit_substep`, one entry per control step flown, step k at index k); the
rollout writes the episode's share into `meta` as `motor_limit_share_substep`, with the steps flown as `motor_steps_flown`; the
line pools the episodes' steps flown. The end-of-step reading above stays in the record and in `meta` and is reported beside it.
  MOTOR_SUBSTEP_META_FIELD, MOTOR_STEPS_META_FIELD, READING_END_OF_STEP, READING_SUBSTEP, MOTOR_LINE_READING
  motor_limit_share_substep(record) -> float;  motor_substep_rows_from_store(store_path, run) -> list[MotorRow]
  motor_acceptance(rows) reads either kind of row (a MotorRow's `reading`), never a mix, and says which in its reading
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

YAW_RATE_RMS_MAX_DEG_S: float = 12.0      # on the body z rate (4DA-4; the heading rate is reported beside it, 4DA-10)
YAW_DRIFT_MAX_DEG_S: float = 1.0
PAST_45_MAX: int = 12                     # of 1,200
PAST_90_MAX: int = 3                      # of 1,200
HEADING_45_DEG: float = 45.0
HEADING_90_DEG: float = 90.0
MAX_HEADING_MEDIAN_MAX_DEG: float = 15.0  # the fifth line (4DR-4)
YAW_META_FIELDS: tuple = ("yaw_rate_rms_deg_s", "heading_rate_rms_deg_s", "yaw_drift_deg_s", "max_abs_yaw_deg")
HEIGHT_M: float = 3.05                    # the reference height (section 2); the altitude offset reads true z against the reference's z
LAST_SECOND_S: float = 1.0
MOTOR_LIMIT_SHARE_MAX: float = 0.005     # the motor line (4DB-1): at most 0.5% of the bar's control steps with any motor at a limit
MOTOR_META_FIELD: str = "motor_limit_share"
UNITS_LIMIT: float = 65535.0              # the firmware motor command's full scale (vehicle.UNITS_MAX); the floor is 0
READING_END_OF_STEP: str = "end-of-step"  # 4DB-1's reading: the command at the end of each control step
READING_SUBSTEP: str = "any-substep"      # 4DB-2's reading: the command on any of the step's five 500 Hz substeps (4C's)
MOTOR_LINE_READING: str = READING_SUBSTEP # the reading the motor line is ruled on, from ruling 4DB-2 (2026-11-09)
MOTOR_SUBSTEP_META_FIELD: str = "motor_limit_share_substep"
MOTOR_STEPS_META_FIELD: str = "motor_steps_flown"


class YawMismatch(ValueError):
    """The rows are not the bar's 1,200 scored episodes with their heading numbers."""


@dataclass(frozen=True)
class EpisodeYaw:
    yaw_rate_rms_deg_s: float = 0.0
    heading_rate_rms_deg_s: float = 0.0
    drift_deg_s: float = 0.0
    max_abs_yaw_deg: float = 0.0
    final_yaw_deg: float = 0.0


@dataclass(frozen=True)
class EpisodeAltitude:
    bias_cm: float = 0.0
    last_second_bias_cm: float = 0.0
    rms_cm: float = 0.0
    lowest_cm: float = 0.0


@dataclass(frozen=True)
class YawRow:
    case: str
    seed: int
    yaw: EpisodeYaw
    error: bool = False


@dataclass(frozen=True)
class YawReading:
    episodes: int = 0
    yaw_rate_rms_median: float = 0.0
    heading_rate_rms_median: float = 0.0       # reported, not gated
    drift_abs_median: float = 0.0
    past_45: int = 0
    past_90: int = 0
    max_heading_median: float = 0.0
    passes_rate: bool = False
    passes_drift: bool = False
    passes_45: bool = False
    passes_90: bool = False
    passes_heading: bool = False
    passed: bool = False

    def to_dict(self) -> dict:
        return {"episodes": self.episodes, "yaw_rate_rms_median_deg_s": self.yaw_rate_rms_median,
                "heading_rate_rms_median_deg_s": self.heading_rate_rms_median,
                "drift_abs_median_deg_s": self.drift_abs_median, "past_45": self.past_45, "past_90": self.past_90,
                "max_heading_median_deg": self.max_heading_median, "passes_rate": self.passes_rate,
                "passes_drift": self.passes_drift, "passes_45": self.passes_45, "passes_90": self.passes_90,
                "passes_heading": self.passes_heading, "passed": self.passed}


def _quat_parts(quat):
    q = np.asarray(quat, dtype=float).reshape(-1, 4)
    return q[:, 0], q[:, 1], q[:, 2], q[:, 3]


def _yaw_rad(quat) -> np.ndarray:
    """Session 4C's formula: the Z-Y-X yaw of (w, x, y, z), wrapped, radians."""
    w, x, y, z = _quat_parts(quat)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def heading_deg(quat) -> np.ndarray:
    """(n, 4) quaternions (w, x, y, z) -> (n,) unwrapped Z-Y-X yaw, degrees."""
    return np.degrees(np.unwrap(_yaw_rad(quat)))


def heading_rate_deg_s(quat, body_rate) -> np.ndarray:
    """(n, 4) quaternions and (n, 3) true body rates -> (n,) Euler yaw rate, (q sin roll + r cos roll) / cos pitch, deg/s."""
    w, x, y, z = _quat_parts(quat)
    omega = np.asarray(body_rate, dtype=float).reshape(-1, 3)
    # the rotation's third row: R[2, 0] = 2 (x z - w y), R[2, 1] = 2 (y z + w x), R[2, 2] = 1 - 2 (x^2 + y^2)
    r20 = 2.0 * (x * z - w * y)
    r21 = 2.0 * (y * z + w * x)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(r21, r22)
    pitch = np.arcsin(np.clip(-r20, -1.0, 1.0))
    rate = (omega[:, 1] * np.sin(roll) + omega[:, 2] * np.cos(roll)) / np.cos(pitch)
    return np.degrees(rate)


def yaw_metrics(record) -> EpisodeYaw:
    t = np.asarray(record.t, dtype=float)
    quat = np.asarray(record.quat, dtype=float)
    omega = np.asarray(record.body_rate, dtype=float)
    yd = heading_deg(quat)
    body_z = np.degrees(math.sqrt(float(np.mean(omega[:, 2] ** 2))))
    hr = heading_rate_deg_s(quat, omega)
    heading_rate_rms = float(math.sqrt(float(np.mean(hr ** 2))))
    drift = float(np.polyfit(t, yd, 1)[0]) if len(t) > 1 else float("nan")
    return EpisodeYaw(float(body_z), heading_rate_rms, drift, float(np.max(np.abs(yd))), float(yd[-1]))


def altitude_metrics(record) -> EpisodeAltitude:
    t = np.asarray(record.t, dtype=float)
    dz = np.asarray(record.position, dtype=float)[:, 2] - np.asarray(record.reference_position, dtype=float)[:, 2]
    last = t >= t[-1] - LAST_SECOND_S + 1e-9
    return EpisodeAltitude(float(100.0 * dz.mean()), float(100.0 * dz[last].mean()),
                           float(100.0 * math.sqrt(float(np.mean(dz ** 2)))), float(100.0 * dz.min()))


def yaw_rows_from_store(store_path, run: str) -> list:
    """A stored Prove run's rows as YawRows, from the four heading fields in `meta`; a row without them, or an error
    row, reads back as an error."""
    from poisson_prove.store import Store

    from drone_study.condition import condition_from_cfg

    store = Store(Path(store_path))
    ids = [r for r in store.runs() if store.manifest(r).name == run or r == run]
    if len(ids) != 1:
        raise YawMismatch(f"store {store_path} has {len(ids)} runs named {run!r}")
    df = store.read(ids[0], meta=True)
    out = []
    for cfg, seed, outcome, error, meta in zip(df["config"], df["seed"], df["outcome"], df["error"], df["meta"]):
        case = condition_from_cfg(cfg).case
        if error is not None or outcome is None or not isinstance(meta, dict) or any(meta.get(f) is None for f in YAW_META_FIELDS):
            out.append(YawRow(case, int(seed), EpisodeYaw(), error=True))
            continue
        out.append(YawRow(case, int(seed), EpisodeYaw(float(meta["yaw_rate_rms_deg_s"]), float(meta["heading_rate_rms_deg_s"]),
                                                     float(meta["yaw_drift_deg_s"]), float(meta["max_abs_yaw_deg"]), float("nan"))))
    return out


def _bar_keys() -> list:
    from drone_study.training.bar import bar_conditions, bar_seeds

    return [(c.case, int(s)) for c in bar_conditions() for s in bar_seeds(c)]


def yaw_acceptance(rows: Sequence[YawRow]) -> YawReading:
    """The five lines on the bar's 1,200 rows (YawMismatch for anything else: wrong cells or seeds, a missing or error row)."""
    want = _bar_keys()
    got = [(r.case, int(r.seed)) for r in rows]
    if len(got) != len(want):
        raise YawMismatch(f"{len(got)} rows, the bar has {len(want)}")
    if len(set(got)) != len(got):
        raise YawMismatch("a (case, seed) row appears twice")
    if set(got) != set(want):
        raise YawMismatch("the rows are not the bar's 12 calm cells x 100 seeds under the bar's salt")
    rate, hrate, drift, biggest = [], [], [], []
    for r in rows:
        y = r.yaw
        vals = (y.yaw_rate_rms_deg_s, y.heading_rate_rms_deg_s, y.drift_deg_s, y.max_abs_yaw_deg)
        if r.error or not all(math.isfinite(float(v)) for v in vals):
            raise YawMismatch(f"row {r.case} seed {r.seed} carries no heading numbers; the gate is read only on 1,200 scored episodes")
        rate.append(float(y.yaw_rate_rms_deg_s))
        hrate.append(float(y.heading_rate_rms_deg_s))
        drift.append(abs(float(y.drift_deg_s)))
        biggest.append(float(y.max_abs_yaw_deg))
    rate_med = float(np.median(rate))
    hrate_med = float(np.median(hrate))
    drift_med = float(np.median(drift))
    heading_med = float(np.median(biggest))
    past_45 = int(sum(1 for b in biggest if b > HEADING_45_DEG))
    past_90 = int(sum(1 for b in biggest if b > HEADING_90_DEG))
    passes_rate = rate_med <= YAW_RATE_RMS_MAX_DEG_S
    passes_drift = drift_med <= YAW_DRIFT_MAX_DEG_S
    passes_45 = past_45 <= PAST_45_MAX
    passes_90 = past_90 <= PAST_90_MAX
    passes_heading = heading_med <= MAX_HEADING_MEDIAN_MAX_DEG
    return YawReading(episodes=len(rows), yaw_rate_rms_median=rate_med, heading_rate_rms_median=hrate_med, drift_abs_median=drift_med,
                      past_45=past_45, past_90=past_90, max_heading_median=heading_med, passes_rate=passes_rate,
                      passes_drift=passes_drift, passes_45=passes_45, passes_90=passes_90, passes_heading=passes_heading,
                      passed=bool(passes_rate and passes_drift and passes_45 and passes_90 and passes_heading))


# ---- the motor line (ruling 4DB-1) --------------------------------------------------------------------------------------------
class MotorMismatch(ValueError):
    """The rows are not the bar's 1,200 scored episodes with their motor-limit share."""


@dataclass(frozen=True)
class MotorRow:
    case: str
    seed: int
    share: float = float("nan")
    steps: int = 0
    error: bool = False
    reading: str = READING_END_OF_STEP         # end-of-step: `steps` is the record's rows; any-substep: the steps flown (4DB-2)


@dataclass(frozen=True)
class MotorReading:
    episodes: int = 0
    share: float = 0.0                         # pooled over the episodes' control steps
    episode_share_median: float = 0.0
    episode_share_max: float = 0.0
    episodes_over: int = 0                     # episodes whose own share is past the line (reported, not gated)
    passed: bool = False
    reading: str = READING_END_OF_STEP

    def to_dict(self) -> dict:
        return {"episodes": self.episodes, "motor_limit_share": self.share, "episode_share_median": self.episode_share_median,
                "episode_share_max": self.episode_share_max, "episodes_over": self.episodes_over,
                "limit": MOTOR_LIMIT_SHARE_MAX, "passed": self.passed, "reading": self.reading}


def motor_limit_share(record) -> float:
    """The share of the episode's control steps that end with any motor's command at 0 or 65,535 units (rows 1 .. n - 1 of the
    record's mixer output; row 0 is the reset's zeros). 0.0 for an episode with no step read."""
    mix = np.asarray(record.mixer_output, dtype=float).reshape(-1, 4)[1:]
    if len(mix) == 0:
        return 0.0
    at = (mix <= 0.0) | (mix >= UNITS_LIMIT)
    return float(np.mean(at.any(axis=1)))


def motor_limit_share_substep(record) -> float:
    """The share of the control steps flown on which any motor's command is at 0 or 65,535 units on any of the step's five
    substeps (ruling 4DB-2), from the record's `motor_limit_substep`. 0.0 for an episode with no step flown; MotorMismatch for a
    record that does not carry the reading."""
    flags = getattr(record, "motor_limit_substep", None)
    if flags is None:
        raise MotorMismatch("the record carries no sub-step motor reading")
    flags = np.asarray(flags, dtype=bool).reshape(-1)
    if len(flags) == 0:
        return 0.0
    return float(np.mean(flags))


def motor_rows_from_store(store_path, run: str) -> list:
    """A stored Prove run's rows as MotorRows, from `meta["motor_limit_share"]` and `meta["steps"]`; a row without them, or an
    error row, reads back as an error."""
    from poisson_prove.store import Store

    from drone_study.condition import condition_from_cfg

    store = Store(Path(store_path))
    ids = [r for r in store.runs() if store.manifest(r).name == run or r == run]
    if len(ids) != 1:
        raise MotorMismatch(f"store {store_path} has {len(ids)} runs named {run!r}")
    df = store.read(ids[0], meta=True)
    out = []
    for cfg, seed, outcome, error, meta in zip(df["config"], df["seed"], df["outcome"], df["error"], df["meta"]):
        case = condition_from_cfg(cfg).case
        if (error is not None or outcome is None or not isinstance(meta, dict) or meta.get(MOTOR_META_FIELD) is None
                or meta.get("steps") is None):
            out.append(MotorRow(case, int(seed), error=True))
            continue
        out.append(MotorRow(case, int(seed), float(meta[MOTOR_META_FIELD]), int(meta["steps"])))
    return out


def motor_substep_rows_from_store(store_path, run: str) -> list:
    """A stored Prove run's rows as sub-step MotorRows (ruling 4DB-2), from `meta["motor_limit_share_substep"]` and
    `meta["motor_steps_flown"]`; a row without them (a run swept before 4DB-2), or an error row, reads back as an error."""
    from poisson_prove.store import Store

    from drone_study.condition import condition_from_cfg

    store = Store(Path(store_path))
    ids = [r for r in store.runs() if store.manifest(r).name == run or r == run]
    if len(ids) != 1:
        raise MotorMismatch(f"store {store_path} has {len(ids)} runs named {run!r}")
    df = store.read(ids[0], meta=True)
    out = []
    for cfg, seed, outcome, error, meta in zip(df["config"], df["seed"], df["outcome"], df["error"], df["meta"]):
        case = condition_from_cfg(cfg).case
        if (error is not None or outcome is None or not isinstance(meta, dict) or meta.get(MOTOR_SUBSTEP_META_FIELD) is None
                or meta.get(MOTOR_STEPS_META_FIELD) is None):
            out.append(MotorRow(case, int(seed), error=True, reading=READING_SUBSTEP))
            continue
        out.append(MotorRow(case, int(seed), float(meta[MOTOR_SUBSTEP_META_FIELD]), int(meta[MOTOR_STEPS_META_FIELD]),
                            reading=READING_SUBSTEP))
    return out


def motor_acceptance(rows: Sequence[MotorRow]) -> MotorReading:
    """The motor line on the bar's 1,200 rows (MotorMismatch for anything else: wrong cells or seeds, a missing or error row, a
    share outside [0, 1])."""
    want = _bar_keys()
    got = [(r.case, int(r.seed)) for r in rows]
    if len(got) != len(want):
        raise MotorMismatch(f"{len(got)} rows, the bar has {len(want)}")
    if len(set(got)) != len(got):
        raise MotorMismatch("a (case, seed) row appears twice")
    if set(got) != set(want):
        raise MotorMismatch("the rows are not the bar's 12 calm cells x 100 seeds under the bar's salt")
    kinds = {getattr(r, "reading", READING_END_OF_STEP) for r in rows}
    if len(kinds) != 1 or not kinds <= {READING_END_OF_STEP, READING_SUBSTEP}:
        raise MotorMismatch(f"the rows mix or misname their readings: {sorted(kinds)}")
    kind = kinds.pop()
    shares, weights = [], []
    for r in rows:
        share = float(r.share)
        if r.error or not math.isfinite(share) or not 0.0 <= share <= 1.0 or int(r.steps) < 1:
            raise MotorMismatch(f"row {r.case} seed {r.seed} carries no motor-limit share; the line is read only on 1,200 scored episodes")
        shares.append(share)
        weights.append(int(r.steps) if kind == READING_SUBSTEP else max(int(r.steps) - 1, 0))
    total = float(sum(weights))
    pooled = float(sum(s * w for s, w in zip(shares, weights)) / total) if total > 0 else 0.0
    return MotorReading(episodes=len(rows), share=pooled, episode_share_median=float(np.median(shares)),
                        episode_share_max=float(max(shares)), episodes_over=int(sum(1 for s in shares if s > MOTOR_LIMIT_SHARE_MAX)),
                        passed=bool(pooled <= MOTOR_LIMIT_SHARE_MAX), reading=kind)
