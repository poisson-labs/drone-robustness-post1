"""Synthetic flyers for the yaw acceptance's Prove runs: each is a `(cfg, seed) -> Episode` with a known answer and no
simulator. `bar_yaw_policy` writes the four heading numbers in `meta` beside `rmse`, as the addenda ask the rollout to;
`bar_body_z_only` writes the 2026-11-07 addendum's three (no heading rate); `bar_no_yaw` writes a bar row without any.
Exposed as Prove --targets."""
from poisson_prove import Episode


def yaw_outcome(seed: int):
    """(body z rate RMS deg/s, heading-rate RMS deg/s, drift deg/s, largest |heading| deg) for the stand-in policy, a function of the seed alone."""
    s = int(seed)
    return 4.0 + (s % 97) * 0.05, 5.0 + (s % 89) * 0.1, ((s % 11) - 5) * 0.03, 8.0 + (s % 53) * 0.5


def episode(seed: int, fields: str = "all"):
    s = int(seed)
    crashed = s % 400 == 7
    meta = {"rmse": 0.050 + (s % 1000) * 1e-5, "crash_class": "radius" if crashed else None}
    rate, heading_rate, drift, biggest = yaw_outcome(s)
    if fields == "all":
        meta.update(yaw_rate_rms_deg_s=rate, heading_rate_rms_deg_s=heading_rate, yaw_drift_deg_s=drift, max_abs_yaw_deg=biggest)
    elif fields == "body-z":
        meta.update(yaw_rate_rms_deg_s=rate, yaw_drift_deg_s=drift, max_abs_yaw_deg=biggest)
    if crashed:
        return Episode(outcome="crash", t=4.0, horizon=10.0, meta=meta)
    return Episode(outcome="ok", t=10.0, horizon=10.0, meta=meta)


def bar_yaw_policy(cfg, seed):
    return episode(seed, "all")


def bar_body_z_only(cfg, seed):
    return episode(seed, "body-z")


def bar_no_yaw(cfg, seed):
    return episode(seed, "none")
