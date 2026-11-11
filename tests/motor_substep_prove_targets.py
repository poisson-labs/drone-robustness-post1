"""Synthetic flyers for the motor line's sub-step Prove runs (ruling 4DB-2): each is a `(cfg, seed) -> Episode` with a known answer
and no simulator. `bar_substep_policy` writes the heading numbers, the end-of-step share and the sub-step share with the steps
flown in `meta`, as the rollout does since 4DB-2; `bar_substep_hidden` is a flyer whose ends of steps sit inside the line while
its substeps do not (pilots 9003 and 9004 at 40M: 0.2% to 0.3% end-of-step, 1.2% to 1.9% on any substep). The end-of-step-only
run of 4DB-1 is `motor_prove_targets.bar_motor_policy`. Exposed as Prove --targets."""
from poisson_prove import Episode

import yaw_prove_targets as Y

ROWS = 1000                                    # a timed-out episode's rows, 0 .. 999
FLOWN = 1000                                   # and the control steps it flew


def substep_share(seed: int) -> float:
    """The stand-in policy's sub-step share, a function of the seed alone: 0 on most seeds, a few steps on some."""
    s = int(seed)
    return (s % 11) / 1000.0 if s % 3 == 0 else 0.0


def episode(seed: int, end: float, sub: float):
    ep = Y.episode(seed, "all")
    meta = dict(ep.meta, steps=ROWS, motor_limit_share=end, motor_limit_share_substep=sub, motor_steps_flown=FLOWN)
    return Episode(outcome=ep.outcome, t=ep.t, horizon=ep.horizon, meta=meta)


def bar_substep_policy(cfg, seed):
    return episode(seed, 0.0, substep_share(seed))


def bar_substep_hidden(cfg, seed):
    return episode(seed, 0.002, 0.012)
