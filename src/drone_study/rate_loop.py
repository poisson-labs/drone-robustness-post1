"""The rate loop (spec section 17.4): the firmware's rate PID, run at 500 Hz on the on-board gyro.

Units: the firmware works in DEGREES per second, so a rate error of 1 rad/s is 57.29578 deg/s into the gains.
Output units are the mixer's (section 12 inv. 5 and 6). 3A-4 pins this reading.

Interface 3B inherits:
  RateLoop(dt=0.002)
     .step(rate_cmd, measured) -> (roll, pitch, yaw) units; both inputs in rad/s, shape (3,)
     .terms -> dict with the last 'p', 'i', 'd' per axis, shape (3,) each, in units
     .reset()
  The loop's two inputs are the command and the on-board gyro reading. It takes nothing else.

The structure is the firmware's `pid.c` at e828ed2: error = desired - measured; P on the error; D on the MEASURED value
(`-(measured - previous) / dt`, so a step in the command does not kick); the integral state accumulates `error dt` and is
clamped to its limit BEFORE the gain multiplies it (roll and pitch +-33.3, yaw +-166.7); the sum saturates to
+-32,767 (`saturateSignedInt16`, 3A-4). Gains: roll and pitch 250/500/2.5, yaw 120/16.7/0. The 30 Hz D filter is off, as
shipped. The integral starts at zero and the previous measurement at 0.
"""
from __future__ import annotations

import math

import numpy as np

KP = np.array([250.0, 250.0, 120.0])
KI = np.array([500.0, 500.0, 16.7])
KD = np.array([2.5, 2.5, 0.0])
I_LIMIT = np.array([33.3, 33.3, 166.7])
INT16_MAX = 32767.0
_DEG = math.degrees(1.0)


class RateLoop:
    def __init__(self, dt: float = 0.002):
        self.dt = dt
        self.reset()

    def reset(self) -> None:
        self._integral = np.zeros(3)
        self._prev = np.zeros(3)                                  # previous measurement, deg/s
        self.terms = {"p": np.zeros(3), "i": np.zeros(3), "d": np.zeros(3)}

    def step(self, rate_cmd, measured) -> np.ndarray:
        desired = np.asarray(rate_cmd, dtype=float) * _DEG
        meas = np.asarray(measured, dtype=float) * _DEG
        err = desired - meas
        p = KP * err
        d = KD * (-(meas - self._prev) / self.dt)
        self._integral = np.clip(self._integral + err * self.dt, -I_LIMIT, I_LIMIT)
        i = KI * self._integral
        self._prev = meas
        self.terms = {"p": p, "i": i, "d": d}
        return np.clip(p + i + d, -INT16_MAX, INT16_MAX)
