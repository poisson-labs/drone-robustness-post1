"""score(trajectory) -> outcome: a pure function of the recorded trajectory (spec section 4; inv. 10).

Interface 3B inherits:
  Trajectory(t, position, quat, reference, floor_contact)   quat is w,x,y,z; tilt = angle between body z and world z
  Limits  primary: ground 0.05 m; tilt (60 deg held 1.0 s) or (101.5 deg held 0.1 s); radius 0.5 m
          S6 variants: tilt_rules=((75.0, 0.0),) instantaneous, ((90.0, 0.0),); radius 0.3, 0.75, 1.0; radius=None
          (ground and tilt only)
  A hold of D seconds is a run of consecutive samples above the threshold from t_a to a sample at t >= t_a + D; the
  crash time is that sample's time. 3A-5 pins this reading. "Over" and "more than" are strict; "at or under" is not.
  score(trajectory, limits=PRIMARY_LIMITS) -> Outcome
  Outcome.crossings: JSON-ready dict of every threshold's first crossing time (None if never), enough for rescore
  rescore(crossings, limits) -> Outcome        equal to score(trajectory, limits) for any S6 variant
  NonFiniteState                               raised for a NaN or infinite value: an error row, never a crash

How it is built. `score` computes the first crossing of every threshold the study ever uses (the ground line; tilt rules
60/1.0, 101.5/0.1, 75/0.0 and 90/0.0; radii 0.3, 0.5, 0.75 and 1.0 m) and of any extra threshold in the limits it is given,
stores them, and `rescore` reads a set of limits off them. `score` is `rescore(crossings, limits)`, so a re-score from the
stored crossings is the direct calculation by construction. Keys: `ground`, `tilt:<deg>:<hold>`, `radius:<m>`. When two classes
cross at the same sample the order is ground, tilt, radius. A hold compares times with a tolerance of 1e-9 s so that
`t - t_a >= D` is not lost to rounding in the sample times (the tests' times are exact; an environment's are `k * 0.01`).

`ScoreTracker` is the same rule one sample at a time, for a training episode that ends at its first crash class.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

HOLD_TOL_S = 1e-9
GROUND_HEIGHT_M = 0.05
STORED_TILT_RULES = ((60.0, 1.0), (101.5, 0.1), (75.0, 0.0), (90.0, 0.0))
STORED_RADII = (0.3, 0.5, 0.75, 1.0)
_ORDER = ("ground", "tilt", "radius")


class NonFiniteState(ValueError):
    pass


@dataclass(frozen=True)
class Trajectory:
    t: np.ndarray
    position: np.ndarray
    quat: np.ndarray
    reference: np.ndarray
    floor_contact: np.ndarray


@dataclass(frozen=True)
class Limits:
    ground_height: Optional[float] = GROUND_HEIGHT_M
    tilt_rules: Optional[tuple] = ((60.0, 1.0), (101.5, 0.1))   # (degrees, seconds held); None = tilt class off
    radius: Optional[float] = 0.5


PRIMARY_LIMITS = Limits()


@dataclass(frozen=True)
class Outcome:
    passed: bool
    crash_class: Optional[str]
    crash_time: Optional[float]
    crossings: dict = field(default_factory=dict)


def tilt_key(deg: float, hold: float) -> str:
    return f"tilt:{float(deg)!r}:{float(hold)!r}"


def radius_key(r: float) -> str:
    return f"radius:{float(r)!r}"


def tilt_degrees(quat) -> np.ndarray:
    """Angle between the body z axis and the world z axis, degrees, from quaternions w,x,y,z of shape (..., 4)."""
    q = np.asarray(quat, dtype=float)
    cos_tilt = 1.0 - 2.0 * (q[..., 1] ** 2 + q[..., 2] ** 2)
    return np.degrees(np.arccos(np.clip(cos_tilt, -1.0, 1.0)))


def _tilt_degrees_scalar(w: float, x: float, y: float, z: float) -> float:
    return math.degrees(math.acos(min(1.0, max(-1.0, 1.0 - 2.0 * (x * x + y * y)))))


def _first_hold(t: np.ndarray, over: np.ndarray, hold: float) -> Optional[float]:
    """Time of the first sample that completes a hold: a run of consecutive `over` samples from t_a to t >= t_a + hold."""
    if not over.any():
        return None
    edges = np.diff(np.concatenate([[0], over.astype(np.int8), [0]]))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1                      # inclusive
    for a, b in zip(starts, ends):
        j = a + int(np.searchsorted(t[a:b + 1], t[a] + hold - HOLD_TOL_S, side="left"))
        if j <= b:
            return float(t[j])
    return None


def _check_finite(traj: Trajectory) -> None:
    for name in ("t", "position", "quat", "reference"):
        if not np.all(np.isfinite(np.asarray(getattr(traj, name), dtype=float))):
            raise NonFiniteState(f"non-finite value in trajectory.{name}")


def crossings_of(traj: Trajectory, limits: Limits = PRIMARY_LIMITS) -> dict:
    _check_finite(traj)
    t = np.asarray(traj.t, dtype=float)
    pos = np.asarray(traj.position, dtype=float)
    out: dict = {}
    low = pos[:, 2] <= GROUND_HEIGHT_M
    low |= np.asarray(traj.floor_contact, dtype=bool)
    idx = np.flatnonzero(low)
    out["ground"] = float(t[idx[0]]) if idx.size else None
    tilt = tilt_degrees(traj.quat)
    rules = list(STORED_TILT_RULES)
    for rule in (limits.tilt_rules or ()):
        if (float(rule[0]), float(rule[1])) not in [(float(a), float(b)) for a, b in rules]:
            rules.append(rule)
    for deg, hold in rules:
        out[tilt_key(deg, hold)] = _first_hold(t, tilt > deg, hold)
    dist = np.linalg.norm(pos - np.asarray(traj.reference, dtype=float), axis=-1)
    radii = list(STORED_RADII)
    if limits.radius is not None and float(limits.radius) not in [float(r) for r in radii]:
        radii.append(limits.radius)
    for r in radii:
        idx = np.flatnonzero(dist > r)
        out[radius_key(r)] = float(t[idx[0]]) if idx.size else None
    return out


def rescore(crossings: dict, limits: Limits) -> Outcome:
    if limits.ground_height is not None and float(limits.ground_height) != GROUND_HEIGHT_M:
        raise ValueError("only the 0.05 m ground line is stored; S6 does not vary it")
    candidates = []                                             # (time, class order, class)
    if limits.ground_height is not None and crossings.get("ground") is not None:
        candidates.append((crossings["ground"], 0, "ground"))
    if limits.tilt_rules is not None:
        times = []
        for deg, hold in limits.tilt_rules:
            key = tilt_key(deg, hold)
            if key not in crossings:
                raise ValueError(f"the crossing {key} was not stored, so it cannot be re-scored")
            if crossings[key] is not None:
                times.append(crossings[key])
        if times:
            candidates.append((min(times), 1, "tilt"))
    if limits.radius is not None:
        key = radius_key(limits.radius)
        if key not in crossings:
            raise ValueError(f"the crossing {key} was not stored, so it cannot be re-scored")
        if crossings[key] is not None:
            candidates.append((crossings[key], 2, "radius"))
    stored = dict(crossings)
    if not candidates:
        return Outcome(True, None, None, stored)
    time, _, cls = min(candidates)
    return Outcome(False, cls, float(time), stored)


def score(trajectory: Trajectory, limits: Limits = PRIMARY_LIMITS) -> Outcome:
    return rescore(crossings_of(trajectory, limits), limits)


class ScoreTracker:
    """The primary limits, one sample at a time. `update` returns (class, time) at the first sample where any class fires
    (the same sample and the same class `score` finds), else None. Samples must be fed in time order."""

    def __init__(self, limits: Limits = PRIMARY_LIMITS):
        self.limits = limits
        self._run_start: dict = {}
        self.fired: Optional[tuple] = None

    def update(self, t: float, position, quat, reference, floor_contact: bool) -> Optional[tuple]:
        if self.fired is not None:
            return self.fired
        lim = self.limits
        if not (math.isfinite(t) and all(math.isfinite(float(v)) for v in position)
                and all(math.isfinite(float(v)) for v in quat) and all(math.isfinite(float(v)) for v in reference)):
            raise NonFiniteState("non-finite value in the state")
        if lim.ground_height is not None and (floor_contact or position[2] <= lim.ground_height):
            self.fired = ("ground", float(t))
            return self.fired
        if lim.tilt_rules is not None:
            tilt = _tilt_degrees_scalar(*(float(v) for v in quat))
            best = None
            for deg, hold in lim.tilt_rules:
                key = (float(deg), float(hold))
                if tilt > deg:
                    start = self._run_start.setdefault(key, float(t))
                    if t - start >= hold - HOLD_TOL_S:
                        best = float(t)
                else:
                    self._run_start.pop(key, None)
            if best is not None:
                self.fired = ("tilt", best)
                return self.fired
        if lim.radius is not None:
            d = math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(position, reference)))
            if d > lim.radius:
                self.fired = ("radius", float(t))
                return self.fired
        return None
