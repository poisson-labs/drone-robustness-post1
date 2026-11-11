"""Synthetic flyers for the bar's Prove runs: each is a `(cfg, seed) -> Episode` with a known answer and no simulator,
so a real `poisson-prove sweep` of 1,200 episodes takes seconds. Exposed as Prove --targets."""
from poisson_prove import Episode


def policy_outcome(seed: int):
    """(crashed, rmse in metres) for the stand-in policy: 1 seed in 400 crashes."""
    return seed % 400 == 7, 0.050 + (seed % 1000) * 1e-5


def pid_outcome(seed: int):
    return False, 0.045 + (seed % 700) * 1e-5


def _episode(crashed, rmse):
    meta = {"rmse": rmse, "crash_class": "radius" if crashed else None}
    if crashed:
        return Episode(outcome="crash", t=4.0, horizon=10.0, meta=meta)
    return Episode(outcome="ok", t=10.0, horizon=10.0, meta=meta)


def bar_policy(cfg, seed):
    return _episode(*policy_outcome(int(seed)))


def bar_pid(cfg, seed):
    return _episode(*pid_outcome(int(seed)))


def bar_faulty(cfg, seed):
    if int(seed) % 300 == 11:
        raise FloatingPointError("non-finite state")
    return _episode(*policy_outcome(int(seed)))
