"""The 1 - cos gust (spec section 3). Air velocity in the world frame, uniform over the drone, gust only.

Direction names where the air comes FROM, measured from the reference's direction of travel at the gust's peak: 0 degrees
is a headwind, azimuth counts anticlockwise seen from above (90 comes from the left of travel), and elevation is up
(3A-2: a gust at +elevation comes from above and blows downward). The air VELOCITY is minus that source direction. It
reads the reference and never the drone, so it is the same for every policy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from drone_study.cases import case as get_case, gust_peak_time
from drone_study.condition import Condition


@dataclass(frozen=True)
class Gust:
    speed: float
    azimuth_deg: float
    elevation_deg: float
    peak_time: float
    travel: tuple[float, float, float]  # the reference's velocity at the gust's peak, world frame
    length: float = 1.0

    @property
    def window(self) -> tuple[float, float]:
        return (self.peak_time - self.length / 2, self.peak_time + self.length / 2)

    def direction(self) -> np.ndarray:
        """Unit vector of the air velocity at the peak."""
        tx, ty = float(self.travel[0]), float(self.travel[1])
        norm = math.hypot(tx, ty)
        t_hat = np.array([tx / norm, ty / norm, 0.0])
        left = np.cross([0.0, 0.0, 1.0], t_hat)
        az, el = math.radians(self.azimuth_deg), math.radians(self.elevation_deg)
        source = math.cos(el) * (math.cos(az) * t_hat + math.sin(az) * left) + math.sin(el) * np.array([0.0, 0.0, 1.0])
        return -source

    def magnitude(self, t):
        t = np.asarray(t, dtype=float)
        start = self.peak_time - self.length / 2
        inside = (t >= start) & (t <= start + self.length)
        return np.where(inside, 0.5 * self.speed * (1 - np.cos(2 * np.pi * (t - start) / self.length)), 0.0)

    def air_velocity(self, t):
        """Gust air velocity, world frame, m/s, at time t (scalar or array). Zero outside the window."""
        t = np.asarray(t, dtype=float)
        return self.magnitude(t)[..., None] * self.direction()

    @classmethod
    def from_condition(cls, condition: Condition) -> "Gust":
        case = get_case(condition.case)
        if not condition.has_gust:
            return cls(0.0, 0.0, 0.0, 0.0, (1.0, 0.0, 0.0), condition.gust_length)
        peak = gust_peak_time(condition.case, float(condition.phase))
        vx, vy, vz = case.velocity(np.array(peak))
        return cls(float(condition.speed), float(condition.az), float(condition.el), peak,
                   (float(vx), float(vy), float(vz)), condition.gust_length)
