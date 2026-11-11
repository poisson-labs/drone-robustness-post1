"""What the nominal arm trains on (spec section 5's table; section 2 for calm air): calm air only, section 2's training
family, no wind of any kind.

This public slice carries the nominal arm alone. The study's other arms, which retrain a policy on what was found for it, are
part of its second post and are not in this repository; their samplers are removed from this module and nothing else in it
is changed. Readings (4A-4): each episode's kind is drawn on its own at the arm's share, which for the nominal arm is calm.

Interface (4A-19), as the slice carries it:
  ARMS, CALM_SHARE
  EpisodeSpec(condition, source, family, finding, table_index)   source "calm"
  make_sampler(arm) -> Sampler;  Sampler.draw(rng) -> EpisodeSpec;  .arm  .calm_share
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from drone_study.condition import Condition
from drone_study.training.family import FamilyDraw, draw_family

ARMS: tuple = ("nominal",)
CALM_SHARE: dict = {"nominal": 1.0}


@dataclass(frozen=True)
class EpisodeSpec:
    condition: Condition
    source: str
    family: Optional[FamilyDraw] = None
    finding: Optional[str] = None
    table_index: Optional[int] = None


class Sampler:
    def __init__(self, arm: str = "", findings: Sequence = (), table=None):
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}")
        self.arm = arm
        self.calm_share = CALM_SHARE[arm]

    def _calm(self, rng: np.random.Generator) -> EpisodeSpec:
        member = draw_family(rng)
        return EpisodeSpec(Condition.calm(member.name), "calm", family=member)

    def draw(self, rng: np.random.Generator) -> EpisodeSpec:
        if self.calm_share >= 1.0:
            return self._calm(rng)
        raise ValueError(f"the slice carries the nominal arm only; {self.arm!r} draws at a calm share under 1")


def make_sampler(arm: str, *, findings: Sequence = (), table=None) -> Sampler:
    return Sampler(arm, findings, table)
