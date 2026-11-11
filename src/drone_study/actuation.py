"""Motors and the mixer (spec sections 12 inv. 5 and 6, 17.4).

Firmware units: a motor command u is in [0, 65535], 65535 being the motor's maximum. THE MAPPING (a ruling,
3A-3 in docs/invariant-table.md): u is linear in commanded rotor speed, omega_cmd = u / 65535 * omega_max, so
thrust goes as u squared and omega_max is the speed at which four motors give thrust-to-weight 2.25.

Interface 3B inherits:
  MotorBank(dt, scales=(1,1,1,1), omega0=None)
     .omega        four rotor speeds, rad/s (initial: omega0, default the hover speed)
     .step(omega_cmd) -> thrust  advance one dt: first-order lag, tau 72 ms, exact discrete update; returns the
                          four rotor thrusts in N = scale_i * kf * omega_i**2
  mix(thrust, roll, pitch, yaw) -> four units   the firmware mixer and cap (no scale argument: it cannot know)
  ScaleAwareMixer(scales).mix(...)               the oracle's mixer
  units_to_omega(u), omega_to_units(w)
  Actuation(dt, scales, mixer=None).step(thrust, roll, pitch, yaw) -> thrusts; .mixer_output holds the last units

The weak motor's scale multiplies THAT motor's thrust only, after the mixer, which does not know it. The rotor speed itself
(and so the linear drag's sum of speeds) is not scaled.
"""
from __future__ import annotations

import math

import numpy as np

from drone_study.vehicle import KF, MOTOR_TAU_S, OMEGA_HOVER, OMEGA_MAX, UNITS_MAX

_UNITS_TO_OMEGA = OMEGA_MAX / UNITS_MAX

# The simulator's motor layout (CF2X) and the firmware's signs: thrust +- roll/2 +- pitch/2 +- yaw.
ROLL_S = np.array([1.0, 1.0, -1.0, -1.0])
PITCH_S = np.array([-1.0, 1.0, 1.0, -1.0])
YAW_S = np.array([-1.0, 1.0, -1.0, 1.0])


def units_to_omega(units):
    return np.asarray(units, dtype=float) * _UNITS_TO_OMEGA


def omega_to_units(omega):
    return np.asarray(omega, dtype=float) / _UNITS_TO_OMEGA


def mix(thrust: float, roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Firmware mixer (CF2X): thrust +- roll/2 +- pitch/2 +- yaw, then the cap. Motors in the simulator's order.

    A motor over the maximum takes that overshoot off all four, so thrust gives way first (torques are untouched); the
    floor is zero, so thrust never goes negative."""
    m = thrust + ROLL_S * (roll / 2.0) + PITCH_S * (pitch / 2.0) + YAW_S * yaw
    over = m.max() - UNITS_MAX
    if over > 0.0:
        m = m - over
    return np.maximum(m, 0.0)


class ScaleAwareMixer:
    """Knows each motor's true scale. Keeps roll and pitch, gives up yaw first, then collective.

    Work in EFFECTIVE units e_i = sqrt(s_i) u_i (the thrust a motor at scale s_i gives at command u_i is that of a healthy
    motor at e_i, thrust going as units squared), so motor i can deliver e_i in [0, cap_i = sqrt(s_i) 65535]. The command is
    then u_i = e_i / sqrt(s_i). The request is e = thrust + roll/2 ROLL + pitch/2 PITCH + yaw YAW. If it fits it is
    delivered whole. If not, in this order:
      1. keep collective, roll and pitch, scale yaw by the largest alpha in [0, 1] that fits;
      2. if even alpha = 0 does not fit, drop yaw, keep roll and pitch and move the collective to the nearest value that
         fits: down when a motor is over its cap (collective gives way), up when one would go below zero (roll and pitch
         first, as an attitude-priority mixer does; the spec is silent here);
      3. if roll and pitch alone do not fit at any collective, scale them together by the largest beta that does.
    """

    def __init__(self, scales):
        self.scales = np.asarray(scales, dtype=float)
        self._root = np.sqrt(self.scales)
        self._cap = self._root * UNITS_MAX

    def _range(self, rp: np.ndarray):
        return float(np.max(-rp)), float(np.min(self._cap - rp))

    def mix(self, thrust: float, roll: float, pitch: float, yaw: float) -> np.ndarray:
        rp = ROLL_S * (roll / 2.0) + PITCH_S * (pitch / 2.0)
        yv = YAW_S * yaw
        e = thrust + rp + yv
        if np.all(e >= 0.0) and np.all(e <= self._cap):
            return e / self._root
        lower, upper = self._range(rp)
        if lower > upper:                                                   # roll and pitch alone do not fit (step 3)
            lo_b, hi_b = 0.0, 1.0
            for _ in range(40):
                mid = 0.5 * (lo_b + hi_b)
                lo, up = self._range(mid * rp)
                lo_b, hi_b = (mid, hi_b) if lo <= up else (lo_b, mid)
            rp = lo_b * rp
            lower, upper = self._range(rp)
            lower, upper = min(lower, upper), max(lower, upper)
            collective = min(max(thrust, lower), upper)
            return (collective + rp) / self._root
        if lower <= thrust <= upper:                                         # step 1: only yaw is in the way
            base = thrust + rp
            alpha = 1.0
            for b, y, cap in zip(base, yv, self._cap):
                if y > 0.0:
                    alpha = min(alpha, (cap - b) / y)
                elif y < 0.0:
                    alpha = min(alpha, (0.0 - b) / y)
            alpha = max(alpha, 0.0)
            return (base + alpha * yv) / self._root
        collective = min(max(thrust, lower), upper)                          # step 2
        return (collective + rp) / self._root


class MotorBank:
    def __init__(self, dt: float, scales=(1.0, 1.0, 1.0, 1.0), omega0=None):
        self.dt = dt
        self.scales = np.asarray(scales, dtype=float)
        self.omega = np.full(4, OMEGA_HOVER) if omega0 is None else np.asarray(omega0, dtype=float).copy()
        self._decay = math.exp(-dt / MOTOR_TAU_S)                            # exact first-order step

    def step(self, omega_cmd) -> np.ndarray:
        cmd = np.asarray(omega_cmd, dtype=float)
        self.omega = cmd + (self.omega - cmd) * self._decay
        return self.scales * KF * self.omega ** 2


class Actuation:
    def __init__(self, dt: float, scales=(1.0, 1.0, 1.0, 1.0), mixer=None):
        self.dt = dt
        self.motors = MotorBank(dt, scales)
        self.mixer = mixer
        self.mixer_output = np.zeros(4)

    def step(self, thrust: float, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """thrust, roll, pitch, yaw in firmware units. Returns the four rotor thrusts, N."""
        units = self.mixer.mix(thrust, roll, pitch, yaw) if self.mixer is not None else mix(thrust, roll, pitch, yaw)
        self.mixer_output = units
        return self.motors.step(units * _UNITS_TO_OMEGA)
