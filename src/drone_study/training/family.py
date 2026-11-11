"""The training family of reference trajectories (spec section 2, 'Cases', last paragraph).

"Training draws from the same shape family: peak speed uniform over 1.5 to 2.2 m/s, geometry scaled uniformly by 0.8
to 1.2, and a draw over 5.5 m/s^2 is drawn again. So the twelve are a sample of what it flew."

Readings (ruling 4A-3): the four shapes are equally likely; one factor, uniform on 0.8 to 1.2, multiplies every
horizontal coordinate of the shape's formula (the slalom's forward run included); the lap rate is solved from the drawn
peak speed; height 3.05 m as always; a draw over the bound is redrawn whole, shape included.

A member's name, `<shape>~<peak speed>~<scale>` with the floats by `repr`, is a case name `drone_study.cases.case`
accepts (it is injective, and it holds no ';' or '=', so it sits inside a Prove cell name). The twelve named cases are
members too (`member_of`), under their own names.

Interface (4A-19):
  FamilyDraw(shape, peak_speed, scale)  .name
  family_case(draw)                     the reference: .period, .omega, .peak_speed, .position(t), .velocity(t), .acceleration(t)
  peak_acceleration(draw)               the largest reference acceleration over a lap, m/s^2 (closed form)
  draw_family(rng)                      one accepted draw from a numpy Generator
  member_of(case_name)                  the FamilyDraw one of the twelve named cases is
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from drone_study import cases

SPEED_RANGE: tuple = (1.5, 2.2)          # m/s
SCALE_RANGE: tuple = (0.8, 1.2)
MAX_ACCELERATION: float = 5.5            # m/s^2
SHAPES = cases.SHAPES
_SEP = "~"


@dataclass(frozen=True)
class FamilyDraw:
    shape: str
    peak_speed: float
    scale: float

    @property
    def name(self) -> str:
        return f"{self.shape}{_SEP}{float(self.peak_speed)!r}{_SEP}{float(self.scale)!r}"


def is_family_name(name: str) -> bool:
    return name.count(_SEP) == 2 and name.split(_SEP)[0] in SHAPES


def parse_family_name(name: str) -> FamilyDraw:
    shape, v, k = name.split(_SEP)
    if shape not in SHAPES:
        raise ValueError(f"unknown shape {shape!r} in family name {name!r}")
    return FamilyDraw(shape, float(v), float(k))


def family_case(draw: FamilyDraw) -> cases.Case:
    """The reference of a member: section 2's formula with x and y multiplied by the scale, its lap rate solved from the
    peak speed. A named case for a draw with scale 1 at a tier's speed is the same reference under the case's own name."""
    return cases.case(draw.name)


def peak_acceleration(draw: FamilyDraw) -> float:
    """The largest reference acceleration over a lap, in closed form from the shape's formula.

    fig8: |a| = k w^2 sqrt(sin^2 u + 4 sin^2 2u), largest 2.125 k w^2 with w = v / (k sqrt 2): 1.0625 v^2 / k.
    circle: k w^2 with w = v / k: v^2 / k.  oval: 1.5 k w^2 with w = v / (1.5 k): v^2 / (1.5 k).
    slalom: 0.8 k w^2 with w = sqrt((v / k)^2 - 1) / 0.8: (v^2 / k - k) / 0.8."""
    v, k = float(draw.peak_speed), float(draw.scale)
    if draw.shape == "fig8":
        return 1.0625 * v * v / k
    if draw.shape == "circle":
        return v * v / k
    if draw.shape == "oval":
        return v * v / (1.5 * k)
    if draw.shape == "slalom":
        return (v * v / k - k) / 0.8
    raise ValueError(f"unknown shape {draw.shape!r}")


def draw_family(rng: np.random.Generator) -> FamilyDraw:
    """One accepted draw: shape, peak speed and scale each uniform; a draw over the bound is drawn again whole."""
    while True:
        shape = SHAPES[int(rng.integers(0, len(SHAPES)))]
        v = float(rng.uniform(SPEED_RANGE[0], SPEED_RANGE[1]))
        k = float(rng.uniform(SCALE_RANGE[0], SCALE_RANGE[1]))
        draw = FamilyDraw(shape, v, k)
        if peak_acceleration(draw) <= MAX_ACCELERATION:
            return draw


def member_of(case_name: str) -> FamilyDraw:
    if case_name in cases.CASE_NAMES:
        shape, tier = case_name.rsplit("-", 1)
        return FamilyDraw(shape, cases.PEAK_SPEED[tier], 1.0)
    if is_family_name(case_name):
        return parse_family_name(case_name)
    raise ValueError(f"{case_name!r} is neither one of the twelve cases nor a family member's name")
