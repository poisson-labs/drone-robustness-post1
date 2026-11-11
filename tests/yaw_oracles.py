"""Closed forms for the yaw addendum's tests, written from the addendum and session 4C's definitions, not from the package.

  wrap(a)                      radians into (-pi, pi]
  heading_of_rotation(R)       atan2(R[1, 0], R[0, 0]) of a body-to-world rotation
  heading_of_quat(q)           the Z-Y-X yaw from (w, x, y, z), session 4C's formula
  huber(e, band)               e^2 / (2 band) inside the band, |e| - band / 2 outside
  heading_term(e, k)           -k x huber(wrap(e)) / pi            (the 2026-11-08 addendum; the 2026-11-07 term was -k |wrap e| / pi)
  rate_term(rate, k)           -k x 0.25 x min(|rate|, 2)^2 / 2^2  (the damping term on the Euler yaw rate)
  euler_zyx(R) / euler_rate(R, body_rate)   roll, pitch, yaw of a rotation; the Euler yaw rate (q sin roll + r cos roll) / cos pitch
  body_rates_of(Rs, dt)        the body angular velocity of a rotation history, by central differences ([w]x = R^T dR/dt)
  rot_z(psi), rot_zyx(yaw, pitch, roll), quat_from_rotation(R)
  constructed_record(...)      an EpisodeRecord-shaped object with a scripted heading history, for yaw_metrics
  yaw_numbers(t, heading_deg, r_rad_s)   the per-episode numbers by 4C's definitions (RMS, slope, largest |heading|)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def wrap(a: float) -> float:
    """(-pi, pi]: the same as numpy's angle wrapping, with pi kept and -pi sent to pi."""
    a = float(a)
    w = (a + math.pi) % (2.0 * math.pi) - math.pi
    if w <= -math.pi:
        w += 2.0 * math.pi
    return w


def heading_of_rotation(R) -> float:
    R = np.asarray(R, dtype=float)
    return math.atan2(R[1, 0], R[0, 0])


def heading_of_quat(q) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


HUBER_BAND = math.pi / 36.0
RATE_WEIGHT = 0.25
RATE_CAP = 2.0


def huber(e: float, band: float = HUBER_BAND) -> float:
    e = abs(float(e))
    return e * e / (2.0 * band) if e <= band else e - band / 2.0


def heading_term(error: float, k: float) -> float:
    return -float(k) * huber(wrap(error)) / math.pi


def rate_term(rate: float, k: float) -> float:
    r = min(abs(float(rate)), RATE_CAP)
    return -float(k) * RATE_WEIGHT * r * r / (RATE_CAP * RATE_CAP)


def euler_zyx(R):
    """(roll, pitch, yaw) of a body-to-world rotation with R = Rz(yaw) Ry(pitch) Rx(roll)."""
    R = np.asarray(R, dtype=float)
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def euler_rate(R, body_rate) -> float:
    roll, pitch, _ = euler_zyx(R)
    p, q, r = (float(v) for v in body_rate)
    return (q * math.sin(roll) + r * math.cos(roll)) / math.cos(pitch)


def body_rates_of(Rs, dt: float) -> np.ndarray:
    """(n, 3, 3) rotations -> (n, 3) body angular velocities, from [w]x = R^T dR/dt by central differences (one-sided at the ends)."""
    Rs = np.asarray(Rs, dtype=float)
    dR = np.gradient(Rs, dt, axis=0)
    out = np.zeros((len(Rs), 3))
    for i in range(len(Rs)):
        W = Rs[i].T @ dR[i]
        out[i] = [W[2, 1], W[0, 2], W[1, 0]]
    return out


def rot_z(psi: float) -> np.ndarray:
    c, s = math.cos(psi), math.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_y(th: float) -> np.ndarray:
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_x(ph: float) -> np.ndarray:
    c, s = math.cos(ph), math.sin(ph)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_zyx(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Body-to-world rotation with Z-Y-X Euler angles: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def quat_from_rotation(R) -> np.ndarray:
    """(w, x, y, z) of a rotation matrix (Shepperd's method)."""
    R = np.asarray(R, dtype=float)
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2.0
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    if i == 0:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    if i == 1:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
    s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
    return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])


@dataclass
class Record:
    """The fields of `drone_study.stack.EpisodeRecord` that the behaviour functions read."""
    t: np.ndarray
    position: np.ndarray
    quat: np.ndarray
    body_rate: np.ndarray
    reference_position: np.ndarray
    end_reason: str = "time"

    @property
    def n_steps(self) -> int:
        return len(self.t)


def constructed_record(heading_rad, *, r_rad_s=None, pitch_rad=0.0, roll_rad=0.0, dz_m=None, height_m=3.05, dt=0.01,
                       body_rate=None) -> Record:
    """A record whose true heading follows `heading_rad` (n,), with a tilt held constant (a scalar each, or (n,) histories),
    a body z rate `r_rad_s` (n,) (by default the heading's own finite difference, with no other component), or a full
    `body_rate` (n, 3), and a height offset `dz_m` (n,) from the reference (zero by default)."""
    psi = np.asarray(heading_rad, dtype=float)
    n = len(psi)
    t = np.arange(n) * dt
    pitch = np.broadcast_to(np.asarray(pitch_rad, dtype=float), (n,))
    roll = np.broadcast_to(np.asarray(roll_rad, dtype=float), (n,))
    quat = np.array([quat_from_rotation(rot_zyx(p, th, ph)) for p, th, ph in zip(psi, pitch, roll)])
    if body_rate is not None:
        body_rate = np.asarray(body_rate, dtype=float)
    else:
        if r_rad_s is None:
            r = np.gradient(psi, dt) if n > 1 else np.zeros(n)
        else:
            r = np.asarray(r_rad_s, dtype=float)
        body_rate = np.stack([np.zeros(n), np.zeros(n), r], axis=1)
    dz = np.zeros(n) if dz_m is None else np.asarray(dz_m, dtype=float)
    ref = np.stack([np.cos(t), np.sin(t), np.full(n, height_m)], axis=1)
    pos = ref.copy()
    pos[:, 2] += dz
    return Record(t, pos, quat, body_rate, ref)


def yaw_numbers(t, heading_deg_unwrapped, r_rad_s):
    """Session 4C's three numbers: yaw-rate RMS deg/s of the body z rate, the slope of a straight-line fit of the heading
    against time deg/s, the largest |heading| deg."""
    t = np.asarray(t, dtype=float)
    yd = np.asarray(heading_deg_unwrapped, dtype=float)
    r = np.asarray(r_rad_s, dtype=float)
    rms = float(np.degrees(math.sqrt(float(np.mean(r ** 2)))))
    slope = float(np.polyfit(t, yd, 1)[0]) if len(t) > 1 else float("nan")
    return rms, slope, float(np.abs(yd).max())


def rotation_of_quat(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def heading_rate_rms_of(rec) -> float:
    """The 2026-11-08 addendum's rate number: RMS over the rows of the Euler yaw rate of the true state, deg/s."""
    rates = [euler_rate(rotation_of_quat(q), w) for q, w in zip(rec.quat, rec.body_rate)]
    return float(np.degrees(math.sqrt(float(np.mean(np.square(rates))))))
