"""The training environment: one drone, one arm's sampler, the stack in training mode (spec sections 2, 4, 5).

Each episode: the sampler names a condition, a fresh episode seed builds the tapes, and `StackEnv(condition, seed,
mode="training")` flies it, so the episode ends at its first crash class or at 10 s, the same in every arm. The
observation is `drone_study.observation`'s 319 numbers at their fixed physical scales (the `PolicyObserver`'s frame
history of the outer stream, the last action being the one the policy chose the step before, the hover hold on the first
step); nothing between the stack and the network keeps a running statistic. The action is in +-1 units (4A-10), mapped
onto thrust 0 to T_max and body rates +-10 rad/s and clipped. The reward is rule 1's on the true distance from the
reference after the step, the true heading's wrapped error from the reference heading and the true Euler yaw rate (the yaw
addenda of 2026-11-07 and 2026-11-08; the weight `k_yaw` is handed to the environment and recorded in the run's manifest,
never a default; the crash penalty is the weight's, `reward.crash_penalty_for`), and the change of the clipped action.

Seeds (4A-13): the environment's generator is seeded by the run's seed and the environment's index together, and every
episode draws its own seed under 2^32 from it (after the sampler's draw), so a run repeats and two environments never fly
the same episode.

Interface (4A-19):
  to_physical(action) / to_normalised(physical) / HOLD_NORMALISED / N_ENVS
  TrainingEnv(sampler, seed, k_yaw, env_index=0)   a gymnasium.Env; .k_yaw; .log the ResetLog; info keys spec, episode_seed,
        t, position, reference_position, tracking_error, heading_error (radians, wrapped, the truth), heading_rate (rad/s, the
        true Euler yaw rate, unclipped), crash_class, and from step reward_terms and action
  make_envs(sampler, seed, k_yaw, n_envs=N_ENVS)   a DummyVecEnv of n_envs TrainingEnvs, env_index 0 .. n_envs - 1
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from drone_study.observation import CLIP, OBS_SIZE, PolicyObserver
from drone_study.stack import HOLD_ACTION, RATE_LIMIT, StackEnv
from drone_study.training.manifest import ResetLog
from drone_study.training.reward import RewardTerms, heading_error, heading_rate, reward_terms
from drone_study.training.sampler import EpisodeSpec
from drone_study.vehicle import T_MAX

N_ENVS: int = 25
ACTION_LO = np.array([0.0, -RATE_LIMIT, -RATE_LIMIT, -RATE_LIMIT])
ACTION_HI = np.array([T_MAX, RATE_LIMIT, RATE_LIMIT, RATE_LIMIT])
_HALF_RANGE = 0.5 * (ACTION_HI - ACTION_LO)


def to_physical(action) -> np.ndarray:
    """+-1 action units (4,) -> [collective thrust N in 0..Tmax, p, q, r in +-10 rad/s], clipped at the range."""
    a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
    return ACTION_LO + (a + 1.0) * _HALF_RANGE


def to_normalised(physical) -> np.ndarray:
    return 2.0 * (np.asarray(physical, dtype=float) - ACTION_LO) / (ACTION_HI - ACTION_LO) - 1.0


HOLD_PHYSICAL = HOLD_ACTION.copy()
HOLD_NORMALISED = to_normalised(HOLD_PHYSICAL)


class TrainingEnv(gym.Env):
    metadata: dict = {"render_modes": []}

    def __init__(self, sampler, seed: int, k_yaw: float, *, env_index: int = 0):
        self.sampler, self.seed, self.env_index = sampler, int(seed), int(env_index)
        self.k_yaw = float(k_yaw)
        self.observation_space = spaces.Box(-CLIP, CLIP, (OBS_SIZE,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)
        self.log = ResetLog()
        self._rng = np.random.default_rng([self.seed, self.env_index])
        self._observer = PolicyObserver()
        self._stack = None
        self._spec = None
        self._episode_seed = None
        self._reading = None
        self._last_physical = HOLD_PHYSICAL.copy()
        self._last_normalised = HOLD_NORMALISED.copy()
        self._steps = 0
        self._done = True
        self._crash_class = None

    # ---- bookkeeping -----------------------------------------------------------------------------------------------------
    @property
    def steps_in_progress(self) -> int:
        """Control steps of the episode still running (0 when none is): the trainer logs them unfinished at a budget's end."""
        return 0 if self._done else self._steps

    def close_log(self) -> None:
        """Count the running episode's steps as unfinished (4A-12). Called by the trainer when the budget ends."""
        if not self._done and self._steps > 0:
            self.log.add(self._steps, finished=False)
            self._done = True

    def _observe(self) -> np.ndarray:
        return self._observer.push(self._reading, self._stack.reference_now(), self._stack.reference_ahead(),
                                   self._last_physical).astype(np.float32)

    def _info(self, terms=None, action=None) -> dict:
        truth = self._stack.truth
        reference = self._stack.reference_now()
        info = {
            "spec": self._spec, "episode_seed": self._episode_seed, "t": self._stack.time,
            "position": np.array(truth.position), "reference_position": np.array(reference),
            "tracking_error": float(np.linalg.norm(truth.position - reference)), "crash_class": self._crash_class,
            "heading_error": heading_error(truth.rotation), "heading_rate": heading_rate(truth.rotation, truth.body_rate),
        }
        if terms is not None:
            info["reward_terms"] = terms
            info["action"] = action
        return info

    # ---- gymnasium ---------------------------------------------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if not self._done and self._steps > 0:
            self.log.add(self._steps, finished=False)
        self._spec = self.sampler.draw(self._rng)
        self._episode_seed = int(self._rng.integers(0, 2 ** 32))
        self._stack = StackEnv(self._spec.condition, self._episode_seed, mode="training")
        self._observer.reset()
        self._reading = self._stack.reset()
        self._last_physical = HOLD_PHYSICAL.copy()
        self._last_normalised = HOLD_NORMALISED.copy()
        self._steps, self._done, self._crash_class = 0, False, None
        return self._observe(), self._info()

    def step(self, action):
        if self._done:
            raise RuntimeError("the episode has ended; call reset()")
        clipped = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        physical = to_physical(clipped)
        result = self._stack.step(physical)
        self._steps += 1
        self._reading = result.reading
        self._last_physical = physical
        truth = self._stack.truth
        error = float(np.linalg.norm(truth.position - self._stack.reference_now()))
        terminated = truncated = False
        if result.done:
            reason = result.end_reason or ""
            if reason == "time":
                truncated = True
            else:
                terminated = True
                self._crash_class = reason.split(":", 1)[1] if reason.startswith("training-crash:") else reason
            self._done = True
            self.log.add(self._steps, finished=True)
        psi = heading_error(truth.rotation)                               # the true heading after the step, wrapped
        psi_dot = heading_rate(truth.rotation, truth.body_rate)           # the true Euler yaw rate after the step, unclipped
        terms = reward_terms(error, clipped, self._last_normalised, terminated, heading_error=psi, heading_rate=psi_dot, k_yaw=self.k_yaw)
        self._last_normalised = clipped
        obs = self._observe()
        return obs, float(terms.total), terminated, truncated, self._info(terms, clipped)


def make_envs(sampler, seed: int, k_yaw: float, n_envs: int = N_ENVS):
    from stable_baselines3.common.vec_env import DummyVecEnv

    return DummyVecEnv([(lambda i=i: TrainingEnv(sampler, seed, k_yaw, env_index=i)) for i in range(int(n_envs))])
