"""Reset and episode-length logging, and a run's manifest (spec section 5, "Equal steps aren't equal resets").

"Each fine-tune's manifest logs its episode count, mean episode length and the share of its steps within 0.5 s of a
reset, and session 7 reports them per arm beside the learning curves." Readings (4A-12): a step is within 0.5 s of a
reset when it starts less than 0.5 s after it, the first 50 control steps of an episode; the episode still running when
the budget ends counts in the steps and in that share, and not in the episode count or the mean length; the report
pools an arm's runs by episodes and by steps.

Interface (4A-19):
  ResetLog()  .add(steps, finished=True)  .episodes .steps .mean_episode_steps .mean_episode_seconds .steps_near_reset
              .share_near_reset
  merge(logs) -> ResetLog
  run_manifest(arm, lineage, seed, budget, log, checkpoint_sha256, k_yaw) -> dict (JSON-ready); its "reward" block carries
              the reward's constants with the run's k_yaw and that weight's crash penalty (the yaw addenda of 2026-11-07 and
              2026-11-08: k_track, track_scale_m, k_smooth, crash_penalty, k_yaw, yaw_scale_rad, reference_yaw,
              yaw_huber_band_rad, yaw_rate_weight, yaw_rate_cap_rad_s), so a checkpoint says what it trained on
  reset_report(manifests) -> {arm: {"runs", "episodes", "steps", "mean_episode_steps", "share_near_reset"}}
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

RESET_WINDOW_S: float = 0.5
CTRL_HZ = 100
RESET_WINDOW_STEPS = round(RESET_WINDOW_S * CTRL_HZ)


class ResetLog:
    def __init__(self):
        self.episodes = 0            # finished episodes
        self.steps = 0               # every step flown, finished or not
        self.steps_near_reset = 0
        self._finished_steps = 0

    def add(self, steps: int, finished: bool = True) -> None:
        steps = int(steps)
        self.steps += steps
        self.steps_near_reset += min(steps, RESET_WINDOW_STEPS)
        if finished:
            self.episodes += 1
            self._finished_steps += steps

    @property
    def mean_episode_steps(self) -> float:
        return self._finished_steps / self.episodes if self.episodes else 0.0

    @property
    def mean_episode_seconds(self) -> float:
        return self.mean_episode_steps / CTRL_HZ

    @property
    def share_near_reset(self) -> float:
        return self.steps_near_reset / self.steps if self.steps else 0.0

    def to_dict(self) -> dict:
        return {"episodes": self.episodes, "steps": self.steps, "steps_near_reset": self.steps_near_reset,
                "finished_steps": self._finished_steps}

    @classmethod
    def from_dict(cls, d: Mapping) -> "ResetLog":
        log = cls()
        log.episodes, log.steps = int(d["episodes"]), int(d["steps"])
        log.steps_near_reset, log._finished_steps = int(d["steps_near_reset"]), int(d["finished_steps"])
        return log


def merge(logs: Iterable[ResetLog]) -> ResetLog:
    out = ResetLog()
    for log in logs:
        out.episodes += log.episodes
        out.steps += log.steps
        out.steps_near_reset += log.steps_near_reset
        out._finished_steps += log._finished_steps
    return out


def run_manifest(*, arm: str, lineage: int, seed: int, budget: int, log: ResetLog, checkpoint_sha256: str, k_yaw: float) -> dict:
    from drone_study.training import reward as R
    from drone_study.training.hyper import HYPERPARAMETERS
    reward = {"k_track": R.K_TRACK, "track_scale_m": R.TRACK_SCALE_M, "k_smooth": R.K_SMOOTH,
              "crash_penalty": R.crash_penalty_for(k_yaw), "k_yaw": float(k_yaw), "yaw_scale_rad": R.YAW_SCALE_RAD,
              "reference_yaw": R.REFERENCE_YAW, "yaw_huber_band_rad": R.YAW_HUBER_BAND_RAD,
              "yaw_rate_weight": R.YAW_RATE_WEIGHT, "yaw_rate_cap_rad_s": R.YAW_RATE_CAP_RAD_S}
    return {
        "arm": arm, "lineage": int(lineage), "seed": int(seed), "budget": int(budget),
        "steps": int(log.steps), "episodes": int(log.episodes),
        "mean_episode_steps": float(log.mean_episode_steps), "mean_episode_seconds": float(log.mean_episode_seconds),
        "steps_near_reset": int(log.steps_near_reset), "share_near_reset": float(log.share_near_reset),
        "checkpoint_sha256": str(checkpoint_sha256),
        "reward": reward,
        "hyperparameters": {k: (list(v) if isinstance(v, tuple) else v) for k, v in HYPERPARAMETERS.items()},
    }


def reset_report(manifests: Sequence[Mapping]) -> dict:
    out: dict = {}
    for m in manifests:
        arm = m["arm"]
        a = out.setdefault(arm, {"runs": 0, "episodes": 0, "steps": 0, "steps_near_reset": 0, "_finished_steps": 0})
        a["runs"] += 1
        a["episodes"] += int(m["episodes"])
        a["steps"] += int(m["steps"])
        a["steps_near_reset"] += int(m["steps_near_reset"])
        # the finished episodes' steps: mean length x count (an unfinished tail is in "steps" and not in the mean)
        a["_finished_steps"] += float(m["mean_episode_steps"]) * int(m["episodes"])
    for arm, a in out.items():
        a["mean_episode_steps"] = a["_finished_steps"] / a["episodes"] if a["episodes"] else 0.0
        a["share_near_reset"] = a["steps_near_reset"] / a["steps"] if a["steps"] else 0.0
        del a["_finished_steps"]
    return out
