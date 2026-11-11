"""Synthetic flyers as Prove --target rollouts for the boundary tests: no simulator, a crash rate per cell.

The module has the attribute names drone_study.training.targets has (`policy`, `pid`, `geometric`, `wind_only`, `oracle`).
`policy` reads DRONE_STUDY_CHECKPOINT as the real one does and takes its seed from the path (".../nominal-<n>/..."), so each
pi0 has its own curve. A flyer crashes when a number fixed by the seed falls under its rate for the cell's speed, so two
flyers on one salt crash on nested seeds. Nothing about the flyer is in cfg."""
import os
import re

from poisson_prove import Episode

from drone_study.condition import condition_from_cfg
from drone_study.score import radius_key, tilt_key

CLASSES = ("ground", "tilt", "radius")
OFFSET = {"pid": 1.0, "geometric": 2.0, "wind_only": 3.0, "oracle": 5.0}


def rate(speed, offset):
    return min(1.0, max(0.0, (speed - 4.0 - offset) / 8.0))


def crossings(crashed):
    """Every threshold the real score stores, crossed at 4 s by a crash and never by a pass, so S6's re-score reads it."""
    t = 4.0 if crashed else None
    keys = {"ground": t}
    keys.update({tilt_key(d, h): t for d, h in ((60.0, 1.0), (101.5, 0.1), (75.0, 0.0), (90.0, 0.0))})
    keys.update({radius_key(r): t for r in (0.3, 0.5, 0.75, 1.0)})
    return keys


def make(offset_of):
    def rollout(cfg, seed):
        cond = condition_from_cfg(cfg)
        u = ((seed * 2654435761) % 4294967296) / 4294967296.0
        crash = u < rate(cond.speed, offset_of())
        return Episode(outcome="crash" if crash else "ok", t=4.0 if crash else 10.0, horizon=10.0,
                       meta={"crash_class": CLASSES[seed % 3] if crash else None, "crossings": crossings(crash), "rmse": 0.04 + 0.02 * u,
                             "rotor_effort": 1.0})
    return rollout


def _policy_offset():
    m = re.search(r"nominal-(\d+)", os.environ["DRONE_STUDY_CHECKPOINT"])
    return -1.0 + 0.25 * (int(m.group(1)) % 4)


policy = make(_policy_offset)
pid = make(lambda: OFFSET["pid"])
geometric = make(lambda: OFFSET["geometric"])
wind_only = make(lambda: OFFSET["wind_only"])
oracle = make(lambda: OFFSET["oracle"])
