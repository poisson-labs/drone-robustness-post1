"""The sweep files, salts and flyers of session 5B (spec 3, 6, 10 item 6, 12, 14, 20.1; the yaw addendum of 2026-11-08, 4DR-5).

Zone B: 16 directions x 10 speeds (2 to 14 m/s) x timings 0 and 0.125 on `fig8-faster`, 24 episodes a cell. The anchor slice:
16 of those cells, 16 episodes each, on its own salt. Both fly the 20 flyers. The extension (section 3) adds 16 m/s on
every direction and timing, for every flyer, if its first reading fires. One Prove sweep per flyer and zone: the spec file
is the same for every flyer, `--target` names the flyer, and the policy's checkpoint rides in an environment variable, so
nothing about a flyer reaches `cfg` (inv. 12). The sixteen π0 are 4D-B's lineages, read from its lineage file
(`data/4d-lineages.json`, the selection's `lineages` in order), and B0 is read from its pick file (`data/yaw-weight.json`),
never from seeds 1 to 16 or a 5M constant (4DR-5): a seed that failed the yaw gate is not a flyer, and a B0 other than 5M
names other checkpoint files.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

from drone_study.condition import Condition

ZONE_B_SALT: str = "9c3d4a1e317d59fa"      # first 16 hex of sha256("drone-robustness-2026-11-04:B") (spec 10, item 6)
ANCHOR_SALT: str = "644f7723497d4cae"      # ... ":anchor"
ZONE_B_CASE: str = "fig8-faster"
ZONE_B_REPLICATES: int = 24
ANCHOR_REPLICATES: int = 16
ZONE_B_SPEEDS: tuple = (2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0)
EXTENSION_SPEED: float = 16.0
ZONE_B_PHASES: tuple = (0.0, 0.125)
DIRECTIONS: tuple = (tuple((float(az), 0.0) for az in (0, 45, 90, 135, 180, 225, 270, 315))
                     + tuple((float(az), el) for el in (35.0, -35.0) for az in (0, 90, 180, 270)))   # 16 (az, el): the 8 on the horizon, then +35 and -35 at 0, 90, 180, 270
LINEAGES_FILE: str = "data/4d-lineages.json"   # 4D-B's bars file with the yaw readings and the selection (4DR-5)
PICK_FILE: str = "data/yaw-weight.json"        # 4D-B's pick: weight, B0, crash penalty (4DR-5)
ANCHOR_SPEEDS: Mapping = {    # (az, phase) -> speed, section 3's table
    (0.0, 0.0): 4.0, (45.0, 0.0): 6.0, (90.0, 0.0): 8.0, (135.0, 0.0): 10.0,
    (180.0, 0.0): 4.0, (225.0, 0.0): 6.0, (270.0, 0.0): 8.0, (315.0, 0.0): 10.0,
    (0.0, 0.125): 8.0, (45.0, 0.125): 10.0, (90.0, 0.125): 4.0, (135.0, 0.125): 6.0,
    (180.0, 0.125): 8.0, (225.0, 0.125): 10.0, (270.0, 0.125): 4.0, (315.0, 0.125): 6.0,
}


@dataclass(frozen=True)
class Flyer:
    name: str                              # "pi0-seed-3", "pid", "geometric", "wind_only", "oracle"
    target: str                            # the attribute of drone_study.training.targets: "policy", "pid", ...
    checkpoint: Optional[Path] = None      # the π0's checkpoint at B0; None for a controller


RUNS_ROOT: str = ".work/4d/runs"               # session 4D-B's nominal runs (the yaw addendum of 2026-11-07; 4B's are superseded)


def checkpoint_path(seed: int, runs_root, b0: int) -> Path:
    """4B's format, kept by 4D-B: <runs_root>/nominal-<seed>/checkpoint-<b0>.zip. b0 is required: it comes from the pick file."""
    return Path(runs_root) / f"nominal-{int(seed)}" / f"checkpoint-{int(b0)}.zip"


def lineage_seeds(lineages_path=None) -> tuple:
    """The 16 nominal seeds of 4D-B's selection, in the file's order (`selection.lineages`; by default LINEAGES_FILE). A file
    without sixteen, or with a seed listed under its failures, is refused (ValueError)."""
    import json

    path = Path(lineages_path) if lineages_path is not None else Path(LINEAGES_FILE)
    doc = json.loads(path.read_text())
    selection = doc.get("selection") or {}
    seeds = tuple(int(s) for s in selection.get("lineages", ()))
    failures = {int(s) for s in selection.get("failures", ())}
    if len(seeds) != 16:
        raise ValueError(f"{path}: {len(seeds)} lineages, the study has sixteen")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{path}: a seed is listed twice among the lineages")
    both = sorted(set(seeds) & failures)
    if both:
        raise ValueError(f"{path}: seeds {both} are listed as lineages and as failures")
    return seeds


def b0_from(pick_path=None) -> int:
    """B0 from 4D-B's pick file (by default PICK_FILE), through `training.yaw_weight.load_yaw_weight`, so a pick the rule would
    not give is refused."""
    from drone_study.training.yaw_weight import load_yaw_weight

    path = Path(pick_path) if pick_path is not None else Path(PICK_FILE)
    return int(load_yaw_weight(path).b0)


def flyers(runs_root=None, *, lineages_path=None, pick_path=None) -> list:
    """The 20 flyers in order: the 16 π0 (`lineage_seeds(lineages_path)`, named "pi0-seed-<seed>", each at
    `checkpoint_path(seed, runs_root, b0_from(pick_path))`; runs_root by default RUNS_ROOT), then pid, geometric, wind_only,
    oracle. A missing file is an error, never seeds 1 to 16 at 5M."""
    root = Path(runs_root) if runs_root is not None else Path(RUNS_ROOT)
    seeds = lineage_seeds(lineages_path)
    b0 = b0_from(pick_path)
    out = [Flyer(f"pi0-seed-{s}", "policy", checkpoint_path(s, root, b0)) for s in seeds]
    out += [Flyer("pid", "pid"), Flyer("geometric", "geometric"), Flyer("wind_only", "wind_only"), Flyer("oracle", "oracle")]
    return out


def _cell(az, el, speed, phase) -> Condition:
    return Condition(case=ZONE_B_CASE, az=float(az), el=float(el), speed=float(speed), phase=float(phase))


def zone_b_conditions(speeds: Sequence[float] = ZONE_B_SPEEDS) -> list:
    """Every Zone B cell, as Conditions on `fig8-faster`: directions x speeds x timings."""
    return [_cell(az, el, v, ph) for az, el in DIRECTIONS for v in speeds for ph in ZONE_B_PHASES]


def anchor_conditions() -> list:
    """The anchor's 16 cells: Zone B cells at the speed section 3's table gives each (azimuth, timing). Timing 0 across the
    eight azimuths, then timing 0.125."""
    return [_cell(az, 0.0, ANCHOR_SPEEDS[(float(az), float(ph))], ph) for ph in ZONE_B_PHASES for az in (0, 45, 90, 135, 180, 225, 270, 315)]


def extension_conditions() -> list:
    """The extension's cells: every direction and timing at 16 m/s (32 cells)."""
    return [_cell(az, el, EXTENSION_SPEED, ph) for az, el in DIRECTIONS for ph in ZONE_B_PHASES]


CHECKPOINT_ENV = "DRONE_STUDY_CHECKPOINT"     # the variable drone_study.training.targets reads
TARGET_MODULE = "drone_study.training.targets"

# The spec file carries no flyer: its `target` is the module's generic name and `--target` wins on the command line, and its
# `target_id` names the zone. So the text is the same for every flyer (inv. 12) and the cfg Prove hands the rollout is the cell.
_SPEC_TARGET = f"{TARGET_MODULE}:rollout"


def _spec(conditions, replicates: int, zone_id: str) -> str:
    from drone_study.condition import sweep_spec_toml

    return sweep_spec_toml(conditions, target=_SPEC_TARGET, target_id=f"drone-robustness-5b:{zone_id}", replicates=replicates)


def zone_b_sweep_spec(speeds: Sequence[float] = ZONE_B_SPEEDS) -> str:
    """The Prove sweep spec for Zone B at `speeds` (24 replicates). The same text for every flyer."""
    return _spec(zone_b_conditions(speeds), ZONE_B_REPLICATES, "zone-b")


def anchor_sweep_spec() -> str:
    return _spec(anchor_conditions(), ANCHOR_REPLICATES, "anchor")


def extension_sweep_spec() -> str:
    """Only the 16 m/s cells, for the sweep that adds the extension once the addendum is filed."""
    return _spec(extension_conditions(), ZONE_B_REPLICATES, "zone-b-extension")


_ZONES = {"zone-b": ZONE_B_SALT, "anchor": ANCHOR_SALT, "zone-b-extension": ZONE_B_SALT}


def run_name(zone: str, flyer: str) -> str:
    """The Prove run's name: "zone-b-<flyer>", "anchor-<flyer>" or "zone-b-extension-<flyer>"; zone is "zone-b", "anchor" or "zone-b-extension"."""
    if zone not in _ZONES:
        raise ValueError(f"{zone!r} is not a post 1 zone: {sorted(_ZONES)}")
    return f"{zone}-{flyer}"


def salt_for(zone: str) -> str:
    if zone not in _ZONES:
        raise ValueError(f"{zone!r} is not a post 1 zone: {sorted(_ZONES)}")
    return _ZONES[zone]


def cell_seeds(salt: str, condition: Condition, replicates: int) -> tuple:
    """The seeds Prove derives for the cell under `salt`: derive_seed(salt, config_hash(cfg), k) for k in range(replicates)."""
    from poisson_prove.executor import derive_seed
    from poisson_prove.store import config_hash

    from drone_study.condition import condition_to_cfg

    h = config_hash(condition_to_cfg(condition))
    return tuple(derive_seed(salt, h, k) for k in range(int(replicates)))


def sweep_argv(*, store, spec_path, zone: str, flyer: Flyer, python: str = "python", target_module: Optional[str] = None) -> list:
    """The Prove command that flies `flyer` through `zone`'s spec: `python -m poisson_prove --store S sweep SPEC --target
    <target_module>:<flyer.target> --salt <zone's salt> --yes --max-mde 1 --name <run name> --plain`, with no cfg override
    and nothing of the flyer but the target and the run name. `target_module` defaults to TARGET_MODULE."""
    module = target_module if target_module is not None else TARGET_MODULE
    return [python, "-m", "poisson_prove", "--store", str(store), "sweep", str(spec_path),
            "--target", f"{module}:{flyer.target}", "--salt", salt_for(zone), "--yes", "--max-mde", "1",
            "--name", run_name(zone, flyer.name), "--plain"]


def sweep_env(flyer: Flyer, base: Optional[Mapping[str, str]] = None) -> dict:
    """The environment for the sweep: `base` plus DRONE_STUDY_CHECKPOINT for a π0, and no checkpoint variable for a controller."""
    import os

    env = dict(os.environ if base is None else base)
    if flyer.checkpoint is None:
        env.pop(CHECKPOINT_ENV, None)
    else:
        env[CHECKPOINT_ENV] = str(Path(flyer.checkpoint).resolve())
    return env


def row_counts() -> dict:
    """The episode counts of section 14 and 20.1: zone_b (153,600), zone_b_pi0 (122,880), zone_b_pid_geometric (15,360),
    zone_b_wind_only (7,680), zone_b_oracle (7,680), anchor (5,120), extension (15,360)."""
    cells = len(DIRECTIONS) * len(ZONE_B_SPEEDS) * len(ZONE_B_PHASES)
    per_flyer = cells * ZONE_B_REPLICATES
    out = {"zone_b_pi0": 16 * per_flyer, "zone_b_pid_geometric": 2 * per_flyer, "zone_b_wind_only": per_flyer, "zone_b_oracle": per_flyer}
    out["zone_b"] = out["zone_b_pi0"] + out["zone_b_pid_geometric"] + out["zone_b_wind_only"] + out["zone_b_oracle"]
    out["anchor"] = 20 * len(anchor_conditions()) * ANCHOR_REPLICATES
    out["extension"] = 20 * len(extension_conditions()) * ZONE_B_REPLICATES
    return out
