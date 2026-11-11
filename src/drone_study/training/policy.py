"""A trained network as a flyer (spec sections 2 and 17.9; the bar and every zone fly it through `rollout_for`).

"The policy's actions (deterministic mean at evaluation, so a crash isn't from dice in the network)". The flyer reads
what the training environment handed the network: the 3B observer's 319 numbers at their fixed scales from the outer
stream, the last action being the one it chose the step before (the hover hold on the first step), and it returns the
network's mean action mapped from +-1 units onto thrust and body rates (4A-18).

Interface (4A-19):
  PolicyController(model)     a `drone_study.stack.Controller` with no privilege and the firmware mixer
  load_policy(checkpoint)     the PolicyController of a checkpoint saved by `save_checkpoint`
"""
from __future__ import annotations

import numpy as np

from drone_study.observation import PolicyObserver
from drone_study.stack import Controller, ControllerView
from drone_study.training.env import HOLD_PHYSICAL, to_physical


class PolicyController(Controller):
    privileges = frozenset()

    def __init__(self, model=None):
        self.model = model
        self._observer = PolicyObserver()
        self._last = HOLD_PHYSICAL.copy()

    def reset(self) -> None:
        self._observer.reset()
        self._last = HOLD_PHYSICAL.copy()

    def act(self, view: ControllerView) -> np.ndarray:
        obs = self._observer.push(view.reading, view.reference[0], view.reference, self._last).astype(np.float32)
        action, _ = self.model.predict(obs, deterministic=True)
        physical = np.asarray(to_physical(action), dtype=float)
        self._last = physical
        return physical


def load_policy(checkpoint) -> PolicyController:
    from stable_baselines3 import PPO

    from drone_study.training.hyper import HYPERPARAMETERS, set_torch_threads

    set_torch_threads()
    return PolicyController(PPO.load(str(checkpoint), device=HYPERPARAMETERS["device"]))
