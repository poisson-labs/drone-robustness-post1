"""Synthetic flyers for the motor line's Prove runs (ruling 4DB-1): each is a `(cfg, seed) -> Episode` with a known answer and
no simulator. `bar_motor_policy` writes the four heading numbers and the motor-limit share in `meta`, as the rollout does;
`bar_no_motor` writes the heading numbers only (a run swept before the motor line); `bar_motor_buzz` pegs a motor on a fifth
of its steps, as pilot 9001 at 0.032 did. Exposed as Prove --targets."""
from poisson_prove import Episode

import yaw_prove_targets as Y

STEPS = 1001                                   # the rows of a full calm episode: 1,000 control steps and the start


def motor_share(seed: int) -> float:
    """The stand-in policy's share of steps with a motor at a limit, a function of the seed alone: 0 on most seeds, a few
    steps on some, well under the line when pooled."""
    s = int(seed)
    return (s % 7) / 1000.0 if s % 5 == 0 else 0.0


def episode(seed: int, kind: str):
    ep = Y.episode(seed, "all")
    meta = dict(ep.meta, steps=STEPS)
    if kind == "motor":
        meta["motor_limit_share"] = motor_share(seed)
    elif kind == "buzz":
        meta["motor_limit_share"] = 0.20
    return Episode(outcome=ep.outcome, t=ep.t, horizon=ep.horizon, meta=meta)


def bar_motor_policy(cfg, seed):
    return episode(seed, "motor")


def bar_no_motor(cfg, seed):
    return episode(seed, "none")


def bar_motor_buzz(cfg, seed):
    return episode(seed, "buzz")
