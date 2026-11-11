"""Training seeds (spec sections 2 and 5; ruling 4A-13): the nominal seeds in order, the pilots, and each run's derived seed.

The nominal seeds are tried in the order 1, 2, 3, ...; the four pilots are 9001 to 9004 (never lineages); a run's seed is
the first 8 hex digits of sha256("drone-robustness-2026-11-04:train:<role>:<nominal seed>"), the salts' formula with a
word for the run: "nominal" or "pilot" in this public slice.

Interface (4A-19): LINEAGE_SEEDS_START, PILOT_SEEDS; nominal_seed(i); run_seed(role, nominal)
"""
from __future__ import annotations

import hashlib

LINEAGE_SEEDS_START: int = 1
PILOT_SEEDS: tuple = (9001, 9002, 9003, 9004)


def nominal_seed(i: int) -> int:
    return LINEAGE_SEEDS_START + int(i)


def run_seed(role: str, nominal: int) -> int:
    digest = hashlib.sha256(f"drone-robustness-2026-11-04:train:{role}:{int(nominal)}".encode()).hexdigest()
    return int(digest[:8], 16)
