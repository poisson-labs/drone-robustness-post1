"""Sensing: the on-board gyro (rate loop only) and the 100 Hz outer stream (spec sections 2, 17.4; inv. 6, 7).

Two streams, two jobs. The 500 Hz rate loop reads only the on-board gyro: the true body rate plus white noise at the
sensor's density, through the firmware's 80 Hz two-pole low-pass (run at 500 Hz), with no added delay. The 100 Hz outer
stream is what the policy and every classical outer loop read, and it alone carries the estimator noise, the latencies and
the delay axis. Truth never reaches it except through these.

Interface 3B inherits:
  SensingConfig(gyro_scale=1.0, outer_scale=1.0)   multiply the tapes' noise AT USE; the tapes themselves do not change
  TrueState(position, velocity, rotation, body_rate)     world-frame position and velocity; rotation is body-to-world
                                                         (3, 3); body_rate is the body-frame angular velocity (p, q, r)
  OuterReading(position, velocity, rotation, body_rate)  same fields: what the policy and every outer loop may read
  OnboardGyro(tape, scale=1.0, dt=0.002).measure(true_rate)  one 500 Hz tick (advances an internal index);
        the reading is lfilter(butter(2, 80 Hz at fs 500), true_rate + scale * tape.gyro_white[j]); no added delay
  OuterStream(tape, delay_steps, scale=1.0)
        .push(truth) once per 100 Hz control step k, then .read(): the measurement of truth at step
        k - 1 - delay_steps (one step of sensor latency, plus the delay axis) with this step's noise from the tape:
        position + pos_offset + pos_ar[k]; velocity + vel_ar[k]; rotation Exp(att_ar[k]) @ rotation;
        body_rate + rate_white[k]. Before enough history exists the earliest truth is held.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import signal

from drone_study.streams import SensorTape

GYRO_LPF_HZ = 80.0
PHYS_HZ = 500.0
SENSOR_LATENCY_STEPS = 1
_B, _A = signal.butter(2, GYRO_LPF_HZ / (PHYS_HZ / 2.0))
_B0, _B1, _B2 = (float(x) for x in _B)
_A1, _A2 = float(_A[1]), float(_A[2])


@dataclass(frozen=True)
class SensingConfig:
    gyro_scale: float = 1.0
    outer_scale: float = 1.0


@dataclass(frozen=True)
class TrueState:
    position: np.ndarray
    velocity: np.ndarray
    rotation: np.ndarray
    body_rate: np.ndarray


@dataclass(frozen=True)
class OuterReading:
    position: np.ndarray
    velocity: np.ndarray
    rotation: np.ndarray
    body_rate: np.ndarray


class OnboardGyro:
    """The BMI088 as the rate loop sees it: white noise on the true rate, then the 80 Hz low-pass (direct form II
    transposed, the form `scipy.signal.lfilter` runs, so the two agree to rounding)."""

    def __init__(self, tape: SensorTape, *, scale: float = 1.0, dt: float = 0.002):
        self.tape, self.scale, self.dt = tape, scale, dt
        self._noise = scale * tape.gyro_white
        self._j = 0
        self._z1 = np.zeros(3)
        self._z2 = np.zeros(3)

    def measure(self, true_rate) -> np.ndarray:
        x = np.asarray(true_rate, dtype=float) + self._noise[self._j]
        self._j += 1
        y = _B0 * x + self._z1
        self._z1 = _B1 * x - _A1 * y + self._z2
        self._z2 = _B2 * x - _A2 * y
        return y


def exp_so3(v) -> np.ndarray:
    """Rotation matrix of a rotation vector (Rodrigues)."""
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    th = math.sqrt(x * x + y * y + z * z)
    if th < 1e-15:
        return np.eye(3)
    kx, ky, kz = x / th, y / th, z / th
    K = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]])
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


class OuterStream:
    def __init__(self, tape: SensorTape, *, delay_steps: int, scale: float = 1.0):
        self.tape, self.delay_steps, self.scale = tape, int(delay_steps), scale
        self._history: list[TrueState] = []

    def push(self, truth: TrueState) -> None:
        self._history.append(truth)

    def read(self) -> OuterReading:
        k = len(self._history) - 1
        if k < 0:
            raise RuntimeError("read() before any push()")
        src = self._history[max(k - SENSOR_LATENCY_STEPS - self.delay_steps, 0)]
        t = self.tape
        kk = min(k, len(t.pos_ar) - 1)
        s = self.scale
        rotation = src.rotation if s == 0.0 else exp_so3(s * t.att_ar[kk]) @ src.rotation
        return OuterReading(
            position=src.position + s * (t.pos_offset + t.pos_ar[kk]),
            velocity=src.velocity + s * t.vel_ar[kk],
            rotation=np.array(rotation),
            body_rate=src.body_rate + s * t.rate_white[kk],
        )
