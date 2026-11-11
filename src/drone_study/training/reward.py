"""The reward: tracking error, a heading term, a heading-rate damping term, a crash penalty and an action-smoothness term
(spec section 2; the yaw addenda of 2026-11-07 and 2026-11-08 in its log), the same in every arm.

The numbers are ruling 4A-1 (Taylor, 2026-11-05) plus the yaw addenda (session 4D-A; rulings 4DA-1 to 4DA-9, 4DR-2 and
4DR-3), per control step
  tracking      -0.1 x (true distance from the reference / 0.5 m)
  heading       -k_yaw x H(wrap(true heading - reference heading)) / pi         (Huber: quadratic inside the band, linear outside)
  heading rate  -k_yaw x YAW_RATE_WEIGHT x min(heading rate, YAW_RATE_CAP)^2 / YAW_RATE_CAP^2   (damping on the Euler yaw rate)
  smoothness    -0.005 x (squared change of the action from the step before, in its +-1 units, summed over the four)
  crash         -crash_penalty_for(k_yaw) on the step that ends the episode in a crash class
and no other term. H is the Huber function with band YAW_HUBER_BAND_RAD (5 degrees): H(e) = e^2 / (2 band) for |e| <= band and
|e| - band / 2 outside, so the restoring slope goes to zero at zero error (no cusp over the firmware's underdamped yaw rate
loop, ruling 4DR-2) and is k_yaw / pi per radian outside the band, as the 2026-11-07 term's was everywhere. The heading is
the yaw of the body-to-world rotation, atan2(R[1, 0], R[0, 0]) (the Z-Y-X Euler yaw, what session 4C's quaternion formula
reads). The reference heading is 0 on every case and every training-family member. The error is wrapped into (-pi, pi], so
the heading term is bounded by k_yaw x (1 - band / (2 pi)). The heading rate is the Euler yaw rate of the true state,
(q sin(roll) + r cos(roll)) / cos(pitch) with (p, q, r) the true body rates, so a banked turn's body z rate is not counted;
it is clipped at YAW_RATE_CAP (2 rad/s) before squaring, so the damping term is bounded by k_yaw x YAW_RATE_WEIGHT (a quarter
of the heading term's weight: "small", 4DR-2) and a tumble is priced by the crash penalty, not by it.

The weight k_yaw is one of YAW_WEIGHT_LADDER, the base ladder (0.004, 0.008, 0.016) extended by (0.032, 0.064) when no base
weight passes (4DA-3 as amended), picked by session 4D-B by the addendum's rule (`training.yaw_weight`), never a free number.
The crash penalty is per weight (4DR-3): the smallest whole number strictly above the most a flight can lose by staying up for
ever at the frozen discount, (0.1 + 0.005 x 16 + k_yaw (1 - band / 2 pi) + k_yaw YAW_RATE_WEIGHT) / (1 - 0.99), and never
under 4A-1's 20: 20 at 0.004, 0.008 and 0.016 (18.49, 18.99, 19.98 for ever), 22 at 0.032 (21.96), 26 at 0.064 (25.91). So
ending an episode early never pays, at any weight.

Interface (4A-19, extended by the yaw addenda; session 4D-B builds the heading parts, which are stubs here):
  reward_terms(tracking_error, action, previous_action, crashed, *, heading_error, heading_rate, k_yaw) -> RewardTerms
        tracking_error   metres between the drone's true position and the reference at that moment
        action, previous_action   the policy's commands in the +-1 action units (4,), this step's and the one before
        crashed          True on the step that ends the episode in a crash class
        heading_error    radians, the true heading minus the reference heading, wrapped by this function
        heading_rate     rad/s, the true Euler yaw rate, clipped by this function
        k_yaw            the heading term's weight, a member of YAW_WEIGHT_LADDER (ValueError otherwise)
  RewardTerms(tracking, heading, heading_rate, smoothness, crash)   .total is their sum, and there is no sixth term
  wrap_angle(a)                 radians into (-pi, pi]
  huber(e, band=YAW_HUBER_BAND_RAD)    the Huber function above
  heading_error(rotation, reference_yaw=0.0)   the wrapped heading error of a body-to-world rotation (3, 3)
  heading_rate(rotation, body_rate)   the Euler yaw rate, rad/s, of a body-to-world rotation and the body rates (3,)
  crash_penalty_for(k_yaw)      the penalty for a ladder weight (ValueError off the ladder)
  worst_alive_per_step(k_yaw)   0.18 + k_yaw (1 - band / 2 pi) + k_yaw YAW_RATE_WEIGHT, the arithmetic the penalty rests on
  K_TRACK, K_SMOOTH, CRASH_PENALTY (4A-1's 20, the floor), TRACK_SCALE_M, YAW_SCALE_RAD, YAW_WEIGHT_LADDER, YAW_WEIGHT_LADDER_BASE,
  YAW_WEIGHT_LADDER_EXTENSION, REFERENCE_YAW, YAW_HUBER_BAND_RAD, YAW_RATE_WEIGHT, YAW_RATE_CAP_RAD_S, GAMMA   the frozen constants
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

K_TRACK: float = 0.1
TRACK_SCALE_M: float = 0.5
K_SMOOTH: float = 0.005
CRASH_PENALTY: float = 20.0               # 4A-1's penalty: the floor of crash_penalty_for
GAMMA: float = 0.99                       # 4A-2's discount, which the penalty's arithmetic reads
YAW_SCALE_RAD: float = math.pi
YAW_WEIGHT_LADDER_BASE: tuple = (0.004, 0.008, 0.016)        # 4DA-2, smallest first
YAW_WEIGHT_LADDER_EXTENSION: tuple = (0.032, 0.064)          # 4DA-3 as amended: tried in order only when no base weight passes
YAW_WEIGHT_LADDER: tuple = YAW_WEIGHT_LADDER_BASE + YAW_WEIGHT_LADDER_EXTENSION
REFERENCE_YAW: float = 0.0                # section 2: reference yaw 0 on every case
YAW_HUBER_BAND_RAD: float = math.pi / 36.0   # 5 degrees, ruling 4DR-2 / 4DA-11
YAW_RATE_WEIGHT: float = 0.25             # the damping term's weight as a share of k_yaw (4DR-2 / 4DA-11)
YAW_RATE_CAP_RAD_S: float = 2.0           # where the damping term saturates (4DR-2 / 4DA-11)
_TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class RewardTerms:
    tracking: float = 0.0
    heading: float = 0.0
    heading_rate: float = 0.0
    smoothness: float = 0.0
    crash: float = 0.0

    @property
    def total(self) -> float:
        return self.tracking + self.heading + self.heading_rate + self.smoothness + self.crash


def wrap_angle(a: float) -> float:
    """Radians into (-pi, pi]: pi is kept, -pi is sent to pi."""
    w = (float(a) + math.pi) % _TWO_PI - math.pi
    if w <= -math.pi:
        w += _TWO_PI
    return w


def huber(e: float, band: float = YAW_HUBER_BAND_RAD) -> float:
    """e^2 / (2 band) for |e| <= band, |e| - band / 2 outside: continuous with a continuous slope at the band."""
    e = abs(float(e))
    band = float(band)
    if e <= band:
        return e * e / (2.0 * band)
    return e - band / 2.0


def heading_error(rotation, reference_yaw: float = REFERENCE_YAW) -> float:
    """The wrapped heading error, radians, of a body-to-world rotation (3, 3): atan2(R[1, 0], R[0, 0]) - reference_yaw, wrapped."""
    R = np.asarray(rotation, dtype=float)
    return wrap_angle(math.atan2(R[1, 0], R[0, 0]) - float(reference_yaw))


def heading_rate(rotation, body_rate) -> float:
    """The Euler (Z-Y-X) yaw rate, rad/s: (q sin(roll) + r cos(roll)) / cos(pitch), unclipped."""
    R = np.asarray(rotation, dtype=float)
    roll = math.atan2(R[2, 1], R[2, 2])
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    p, q, r = (float(v) for v in np.asarray(body_rate, dtype=float).reshape(3))
    return (q * math.sin(roll) + r * math.cos(roll)) / math.cos(pitch)


def _check_weight(k_yaw: float) -> float:
    k = float(k_yaw)
    for w in YAW_WEIGHT_LADDER:
        if k == w:
            return w
    raise ValueError(f"k_yaw must be one of the ladder {YAW_WEIGHT_LADDER}, got {k_yaw!r}")


def worst_alive_per_step(k_yaw: float) -> float:
    """The most a flight can lose in one step while up: 0.1 + 0.08 + the heading term's bound + the damping term's bound."""
    k = _check_weight(k_yaw)
    return K_TRACK + K_SMOOTH * 16.0 + k * (1.0 - YAW_HUBER_BAND_RAD / _TWO_PI) + k * YAW_RATE_WEIGHT


def crash_penalty_for(k_yaw: float) -> float:
    """The smallest whole number strictly above worst_alive_per_step(k_yaw) / (1 - GAMMA), and not under CRASH_PENALTY (20)."""
    forever = worst_alive_per_step(k_yaw) / (1.0 - GAMMA)
    return max(CRASH_PENALTY, float(math.floor(forever) + 1))


def reward_terms(tracking_error: float, action, previous_action, crashed: bool, *, heading_error: float, heading_rate: float,
                 k_yaw: float) -> RewardTerms:
    k = _check_weight(k_yaw)
    delta = np.asarray(action, dtype=float) - np.asarray(previous_action, dtype=float)
    tracking = -K_TRACK * float(tracking_error) / TRACK_SCALE_M
    heading = -k * huber(wrap_angle(heading_error)) / YAW_SCALE_RAD
    rate = min(abs(float(heading_rate)), YAW_RATE_CAP_RAD_S)
    damping = -k * YAW_RATE_WEIGHT * rate * rate / (YAW_RATE_CAP_RAD_S * YAW_RATE_CAP_RAD_S)
    smoothness = -K_SMOOTH * float(delta @ delta)
    crash = -crash_penalty_for(k) if crashed else 0.0
    return RewardTerms(tracking, heading, damping, smoothness, crash)
