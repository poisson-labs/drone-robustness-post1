"""A stand-in environment with the training environment's shapes and nothing behind them, for the tests that build the
network and never fly."""
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class Stand(gym.Env):
    observation_space = spaces.Box(-5.0, 5.0, (319,), dtype=np.float32)
    action_space = spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        return np.zeros(319, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(319, dtype=np.float32), 0.0, False, False, {}


def stand_envs(n=25):
    from stable_baselines3.common.vec_env import DummyVecEnv
    return DummyVecEnv([Stand for _ in range(n)])
