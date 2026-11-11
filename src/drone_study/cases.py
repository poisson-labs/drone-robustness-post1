"""The twelve cases (spec section 2): four shapes at three tiers, flown at 3.05 m, reference yaw 0.

`u = omega * t`; a lap is one period `T = 2 pi / omega`. Peak reference speed is exactly 1.5, 1.8 and 2.2 m/s by tier,
and omega is solved from it: fig8 `v / sqrt(1^2 + 4 (0.5)^2)`, circle `v`, oval `v / 1.5`, slalom `sqrt(v^2 - 1) / 0.8`
(its x runs at 1 m/s, so the lateral weave takes what is left). Phase 0 is at t = 0 and at each whole lap.

Session 4B (ruling 4A-19): `case()` also accepts a training-family member's name, `<shape>~<peak speed>~<scale>` (section 2,
"Training draws from the same shape family"; `drone_study.training.family`): the same formula with every horizontal
coordinate multiplied by the scale, the slalom's forward run included, and omega solved from the drawn peak speed. The
twelve named cases are the members at scale 1; their values are unchanged (the factor 1.0 is exact).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from drone_study.vehicle import HEIGHT_M

SHAPES = ("fig8", "circle", "oval", "slalom")
TIERS = ("medium", "fast", "faster")
CASE_NAMES = tuple(f"{s}-{t}" for s in SHAPES for t in TIERS)
LEVEL_PHASES = (0.0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375)
PEAK_SPEED = {"medium": 1.5, "fast": 1.8, "faster": 2.2}
FIRST_LAP_AFTER_S = 3.0


def _omega(shape: str, peak: float) -> float:
    if shape == "fig8":
        return peak / math.sqrt(1.0 ** 2 + 4 * 0.5 ** 2)
    if shape == "circle":
        return peak / 1.0
    if shape == "oval":
        return peak / 1.5
    if shape == "slalom":
        return math.sqrt(peak * peak - 1.0) / 0.8
    raise ValueError(f"unknown shape {shape!r}")


FAMILY_SEP = "~"


@dataclass(frozen=True)
class Case:
    name: str
    period: float = 1.0
    peak_speed: float = 0.0
    shape: str = ""
    omega: float = 0.0
    scale: float = 1.0

    def _state(self, t):
        """(position, velocity, acceleration), each (..., 3)."""
        t = np.asarray(t, dtype=float)
        w = self.omega
        u = w * t
        zero = np.zeros_like(t)
        height = np.full_like(t, HEIGHT_M)
        if self.shape == "fig8":
            a, b = 1.0, 0.5
            p = (a * np.sin(u), b * np.sin(2 * u))
            v = (a * w * np.cos(u), 2 * b * w * np.cos(2 * u))
            acc = (-a * w * w * np.sin(u), -4 * b * w * w * np.sin(2 * u))
        elif self.shape == "circle":
            p = (np.cos(u), np.sin(u))
            v = (-w * np.sin(u), w * np.cos(u))
            acc = (-w * w * np.cos(u), -w * w * np.sin(u))
        elif self.shape == "oval":
            a, b = 1.5, 0.75
            p = (a * np.sin(u), b * np.cos(u))
            v = (a * w * np.cos(u), -b * w * np.sin(u))
            acc = (-a * w * w * np.sin(u), -b * w * w * np.cos(u))
        else:
            b = 0.8
            p = (1.0 * t, b * np.sin(u))
            v = (zero + 1.0, b * w * np.cos(u))
            acc = (zero, -b * w * w * np.sin(u))
        k = self.scale
        stack = lambda xy, z: np.stack([k * xy[0], k * xy[1], z], axis=-1)
        return stack(p, height), stack(v, zero), stack(acc, zero)

    def position(self, t):
        return self._state(t)[0]

    def velocity(self, t):
        return self._state(t)[1]

    def acceleration(self, t):
        return self._state(t)[2]


_CACHE: dict[str, Case] = {}


def _family_member(name: str) -> tuple:
    """(shape, peak speed, scale) of a training-family member's name, or None if the name is not one."""
    parts = name.split(FAMILY_SEP)
    if len(parts) != 3 or parts[0] not in SHAPES:
        return None
    try:
        return parts[0], float(parts[1]), float(parts[2])
    except ValueError:
        return None


def case(name: str) -> Case:
    if name not in _CACHE:
        if name in CASE_NAMES:
            shape, tier = name.rsplit("-", 1)
            peak, scale = PEAK_SPEED[tier], 1.0
        else:
            member = _family_member(name)
            if member is None:
                raise ValueError(f"unknown case {name!r}; the twelve are {CASE_NAMES}, or a training-family member "
                                 f"<shape>{FAMILY_SEP}<peak speed>{FAMILY_SEP}<scale>")
            shape, peak, scale = member
            if not (peak > 0.0 and scale > 0.0):
                raise ValueError(f"a family member needs a positive peak speed and scale, got {name!r}")
        w = _omega(shape, peak / scale)                 # the scaled shape's lap rate, solved from the drawn peak speed
        _CACHE[name] = Case(name=name, period=2 * math.pi / w, peak_speed=peak, shape=shape, omega=w, scale=scale)
    return _CACHE[name]


def gust_peak_time(name: str, phase: float) -> float:
    """The first whole lap at or after 3.0 s, plus phase x lap (spec section 3, Timing)."""
    period = case(name).period
    return math.ceil(FIRST_LAP_AFTER_S / period) * period + phase * period
