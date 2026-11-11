"""PPO's frozen settings and the network (spec section 2, "Architecture and training"; section 5, "The same for every
fine-tune"; ruling 4A-2). Stable-Baselines3 2.9.2 and torch 2.15.0, pinned in pyproject.toml's `gate` group.

25 environments x 1,250 steps between updates (31,250 a rollout, which divides every rung, every B_ft and every quarter
of one), minibatch 1,250 (25 a pass), 10 passes, one torch thread; learning rate 3e-4 for the nominal run and 1e-4 for a
fine-tune, constant; discount 0.99, GAE 0.95, clip 0.2, no value clipping, advantages normalised, no entropy bonus, value
weight 0.5, gradient norm 0.5, no KL target, no gSDE; actor and critic each their own two layers of 256 tanh units,
orthogonal init; the log-std starts at -1; Adam with eps 1e-5; CPU.

A fine-tune (section 5; 4A-9) starts from pi0's checkpoint: actor, critic and log-std as saved, the fine-tune learning
rate, a fresh optimiser and a step count of zero, every other setting the same.

Interface (4A-19):
  HYPERPARAMETERS, NOMINAL_LEARNING_RATE, FINE_TUNE_LEARNING_RATE
  make_ppo(env, seed)                     a PPO for a nominal run (sets torch to one thread)
  fine_tune_ppo(checkpoint, env, seed)    a PPO that starts from a saved nominal checkpoint
  save_checkpoint(model, path) -> the file's SHA-256 (the path ends in .zip)
"""
from __future__ import annotations

import hashlib
from pathlib import Path

HYPERPARAMETERS: dict = {
    "n_envs": 25, "n_steps": 1250, "batch_size": 1250, "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95,
    "clip_range": 0.2, "clip_range_vf": None, "normalize_advantage": True, "ent_coef": 0.0, "vf_coef": 0.5,
    "max_grad_norm": 0.5, "target_kl": None, "use_sde": False, "net_arch_pi": (256, 256), "net_arch_vf": (256, 256),
    "activation": "tanh", "ortho_init": True, "log_std_init": -1.0, "optimizer": "Adam", "adam_eps": 1e-5,
    "nominal_learning_rate": 3e-4, "fine_tune_learning_rate": 1e-4, "device": "cpu", "torch_threads": 1,
}
NOMINAL_LEARNING_RATE: float = HYPERPARAMETERS["nominal_learning_rate"]
FINE_TUNE_LEARNING_RATE: float = HYPERPARAMETERS["fine_tune_learning_rate"]
N_ENVS = HYPERPARAMETERS["n_envs"]
ROLLOUT_STEPS = HYPERPARAMETERS["n_envs"] * HYPERPARAMETERS["n_steps"]


def set_torch_threads() -> None:
    import torch

    torch.set_num_threads(HYPERPARAMETERS["torch_threads"])


def _build(env, *, seed: int, learning_rate: float):
    import torch
    from stable_baselines3 import PPO

    set_torch_threads()
    h = HYPERPARAMETERS
    return PPO(
        "MlpPolicy", env,
        learning_rate=learning_rate, n_steps=h["n_steps"], batch_size=h["batch_size"], n_epochs=h["n_epochs"],
        gamma=h["gamma"], gae_lambda=h["gae_lambda"], clip_range=h["clip_range"], clip_range_vf=h["clip_range_vf"],
        normalize_advantage=h["normalize_advantage"], ent_coef=h["ent_coef"], vf_coef=h["vf_coef"],
        max_grad_norm=h["max_grad_norm"], use_sde=h["use_sde"], target_kl=h["target_kl"],
        policy_kwargs=dict(net_arch=dict(pi=list(h["net_arch_pi"]), vf=list(h["net_arch_vf"])), activation_fn=torch.nn.Tanh,
                           ortho_init=h["ortho_init"], log_std_init=h["log_std_init"],
                           optimizer_class=torch.optim.Adam, optimizer_kwargs=dict(eps=h["adam_eps"])),
        seed=int(seed), device=h["device"], verbose=0,
    )


def make_ppo(env, *, seed: int):
    return _build(env, seed=seed, learning_rate=NOMINAL_LEARNING_RATE)


def fine_tune_ppo(checkpoint, env, *, seed: int):
    """pi0's actor, critic and log-std, the fine-tune learning rate, a fresh optimiser, a step count of zero."""
    from stable_baselines3 import PPO

    model = _build(env, seed=seed, learning_rate=FINE_TUNE_LEARNING_RATE)
    saved = PPO.load(str(checkpoint), device=HYPERPARAMETERS["device"])
    model.policy.load_state_dict(saved.policy.state_dict())
    model.num_timesteps = 0
    return model


def save_checkpoint(model, path) -> str:
    path = Path(path)
    if path.suffix != ".zip":
        raise ValueError(f"a checkpoint's path ends in .zip, got {path}")
    model.save(str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()
