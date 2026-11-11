"""The flyers as Prove `--target`s (spec section 13: `--target` names the policy; the cell carries nothing about it).

A target is `(cfg, seed) -> Episode`, built by `drone_study.rollout.rollout_for`. The policy's checkpoint is named by the
environment variable DRONE_STUDY_CHECKPOINT, read once per process when `policy` is first called, so one sweep process
flies one checkpoint: `poisson-prove sweep bar.toml --target drone_study.training.targets:policy --salt <bar salt>`.

  policy       the trained network at DRONE_STUDY_CHECKPOINT, as `PolicyController` (mean action, 4A-18)
  pid          the cascaded PID (section 17.2)
  geometric    the geometric controller at DEFAULT_GAINS (the 4B pick)
  wind_only    the wind-only reference
  oracle       the feasibility oracle
"""
from __future__ import annotations

import os

from drone_study.controllers import FeasibilityOracle, GeometricController, WindOnlyReference
from drone_study.pid import CascadedPID
from drone_study.rollout import rollout_for

_POLICY = {}


def _policy_rollout():
    if "rollout" not in _POLICY:
        from drone_study.training.policy import load_policy

        path = os.environ.get("DRONE_STUDY_CHECKPOINT")
        if not path:
            raise RuntimeError("DRONE_STUDY_CHECKPOINT names the checkpoint the `policy` target flies")
        _POLICY["path"] = path
        _POLICY["rollout"] = rollout_for(load_policy(path))
    return _POLICY["rollout"]


def policy(cfg, seed):
    return _policy_rollout()(cfg, seed)


pid = rollout_for(CascadedPID())
geometric = rollout_for(GeometricController())
wind_only = rollout_for(WindOnlyReference())
oracle = rollout_for(FeasibilityOracle())
