"""A condition: one point of the spec's condition space (section 3), as a plain record.

Field names are the columns of the frozen held-out file (zone-c-conditions.csv). A gust is present
when `speed` > 0; then `az`, `el` and `phase` are set. `turb` is the Dryden sigma_u in m/s: 0.3 for every
condition but the calm control (0.0). `payload` is grams, `delay` is milliseconds on the outer stream.

The cell name is what Prove sees: one categorical axis, `cell`, whose value names the condition and nothing else (spec
6 and 12 inv. 12). It is written field by field, floats by `repr` of the float (shortest text that reads back to the same
number) so it is injective, and it carries no word of the policy: the policy is closed over by the rollout.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

CONDITION_FIELDS = (
    "case", "az", "el", "speed", "phase", "turb", "payload", "motor_index", "motor_scale", "delay", "gust_length",
)


@dataclass(frozen=True)
class Condition:
    case: str
    az: float | None = None
    el: float | None = None
    speed: float = 0.0
    phase: float | None = None
    turb: float = 0.3
    payload: float = 0.0
    motor_index: int | None = None
    motor_scale: float = 1.0
    delay: int = 0
    gust_length: float = 1.0

    @property
    def has_gust(self) -> bool:
        return self.speed > 0.0

    @classmethod
    def calm(cls, case: str) -> "Condition":
        return cls(case=case, turb=0.0)

    def motor_scales(self) -> np.ndarray:
        """The four motors' thrust scales: 1.0 everywhere but the weak motor."""
        scales = np.ones(4)
        if self.motor_index is not None:
            scales[int(self.motor_index)] = float(self.motor_scale)
        return scales


# ---- the cell name ------------------------------------------------------------------------------------------------------
_KEYS = (("c", "case"), ("az", "az"), ("el", "el"), ("v", "speed"), ("ph", "phase"), ("tb", "turb"), ("pl", "payload"),
         ("mi", "motor_index"), ("ms", "motor_scale"), ("dl", "delay"), ("gl", "gust_length"))


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return str(int(value))
    return repr(float(value))


def cell_name(condition: Condition) -> str:
    """A canonical, injective, policy-free name for the condition, used as the value of Prove's one axis."""
    parts = []
    for key, attr in _KEYS:
        v = getattr(condition, attr)
        if attr in ("motor_index", "delay") and v is not None:
            v = int(v)
        elif attr not in ("case",) and v is not None:
            v = float(v)
        parts.append(f"{key}={_fmt(v)}")
    return ";".join(parts)


def condition_from_cell(name: str) -> Condition:
    fields = dict(p.split("=", 1) for p in name.split(";"))
    kw: dict[str, Any] = {}
    for key, attr in _KEYS:
        raw = fields[key]
        if attr == "case":
            kw[attr] = raw
        elif raw == "-":
            kw[attr] = None
        elif attr in ("motor_index", "delay"):
            kw[attr] = int(raw)
        else:
            kw[attr] = float(raw)
    return Condition(**kw)


def condition_to_cfg(condition: Condition) -> dict[str, Any]:
    """What Prove sees as the cell: {"cell": <canonical name>}. Never carries a policy (spec 6, 12 inv. 12)."""
    return {"cell": cell_name(condition)}


def condition_from_cfg(cfg: Mapping[str, Any]) -> Condition:
    """The inverse of condition_to_cfg."""
    return condition_from_cell(str(cfg["cell"]))


def sweep_spec_toml(conditions: list[Condition], *, target: str, target_id: str, replicates: int) -> str:
    """Text of a Prove sweep spec whose single categorical axis `cell` names each condition (spec 3, 13)."""
    names = []
    for c in conditions:
        n = cell_name(c)
        if n not in names:
            names.append(n)
    quoted = ", ".join(json.dumps(n) for n in names)
    return (
        f"target = {json.dumps(target)}\n"
        f"target_id = {json.dumps(target_id)}\n"
        f't_unit = "s"\n'
        f"replicates = {int(replicates)}\n\n"
        f"[axes.cell]\n"
        f'type = "categorical"\n'
        f"values = [{quoted}]\n\n"
        f"[declare]\n"
        f'failures = ["crash"]\n'
    )
