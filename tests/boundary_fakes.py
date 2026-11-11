"""Stand-ins for the 8A pieces post 1's N8 reads, which session 5B makes real: S6's limits and its re-score (8A's stubs)."""
import numpy as np

import boundary_numbers as B
from drone_study.analysis import sensitivities as S

NAMES = list(B.S6_NAMES)


def fake_rescore_by_seed(rows, name):
    """A crash stays a crash under limit NAMES[j] when seed % 7 >= j. Error rows stay error rows."""
    j = NAMES.index(name)
    out = rows.copy()
    scored = out["outcome"] != "error"
    crash = [bool(sc and o == "crash" and (int(seed) % 7) >= j) for sc, o, seed in zip(scored, out["outcome"], out["seed"])]
    out["outcome"] = np.where(~scored, "error", np.where(crash, "crash", "ok"))
    return out


def patch_s6(monkeypatch):
    monkeypatch.setattr(S, "S6_LIMITS", {n: object() for n in NAMES})
    monkeypatch.setattr(S, "s6_rescore", fake_rescore_by_seed)
