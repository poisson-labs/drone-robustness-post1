"""The rollout Prove calls: (cfg, seed) -> Episode (spec sections 4, 6, 13; inv. 12).

cfg is {"cell": <name>} and nothing else: the policy is never in it. The policy is closed over by `rollout_for`,
and Prove's `--target` names the target that does so.

An episode is flown in evaluation mode (it flies on past the tilt and radius thresholds to 10 s or floor contact) and then
scored by `score()`. Prove sees `ok` or `crash`; the crash class, every threshold's first crossing, the tracking RMSE, the
rotor effort, the four heading numbers of the yaw addenda (body z rate RMS, heading-rate RMS, drift, largest |heading|) and the
motor-limit share (ruling 4DB-1) go in `meta`, so the yaw gate and the motor line are read from the stored bar sweep itself (4DA-8). A non-finite state raises `NonFiniteState`, which Prove stores as an error row, never a crash.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from poisson_prove import Episode

from drone_study.condition import condition_from_cfg
from drone_study.score import score
from drone_study.stack import Controller, EpisodeRecord, fly
from drone_study.training.behaviour import (MOTOR_META_FIELD, MOTOR_STEPS_META_FIELD, MOTOR_SUBSTEP_META_FIELD, motor_limit_share,
                                            motor_limit_share_substep, yaw_metrics)
from drone_study.vehicle import EPISODE_S, OMEGA_HOVER


def tracking_rmse(record: EpisodeRecord) -> float:
    """Root mean square distance from the reference over the rows flown, m."""
    err = np.linalg.norm(record.position - record.reference_position, axis=1)
    return float(np.sqrt(np.mean(err ** 2)))


def rotor_effort(record: EpisodeRecord) -> float:
    """Mean over the rows of the mean squared rotor speed, as a multiple of hover's (thrust goes as speed squared)."""
    return float(np.mean(record.rotor_omega ** 2) / OMEGA_HOVER ** 2)


def rollout_for(policy: Controller):
    def rollout(cfg: Mapping[str, Any], seed: int) -> Episode:
        condition = condition_from_cfg(cfg)
        record = fly(condition, int(seed), policy)
        outcome = score(record.trajectory())
        meta = {
            "crash_class": outcome.crash_class,
            "crossings": outcome.crossings,
            "rmse": tracking_rmse(record),
            "rotor_effort": rotor_effort(record),
            "end_reason": record.end_reason,
            "steps": int(record.n_steps),
        }
        yaw = yaw_metrics(record)                       # the four heading numbers beside rmse (ruling 4DA-8; the gate reads them)
        meta.update({"yaw_rate_rms_deg_s": yaw.yaw_rate_rms_deg_s, "heading_rate_rms_deg_s": yaw.heading_rate_rms_deg_s,
                     "yaw_drift_deg_s": yaw.drift_deg_s, "max_abs_yaw_deg": yaw.max_abs_yaw_deg})
        meta[MOTOR_META_FIELD] = motor_limit_share(record)   # the motor line's share (ruling 4DB-1): steps ending with a motor at 0 or 65,535
        meta[MOTOR_SUBSTEP_META_FIELD] = motor_limit_share_substep(record)   # the line's reading (4DB-2): a motor at a limit on any substep
        meta[MOTOR_STEPS_META_FIELD] = int(len(record.motor_limit_substep))  # the steps flown it pools over
        if outcome.passed:
            return Episode(outcome="ok", t=min(float(record.t[-1]) + 0.01, EPISODE_S), horizon=EPISODE_S, meta=meta)
        return Episode(outcome="crash", t=float(outcome.crash_time), horizon=EPISODE_S, meta=meta)

    return rollout
