"""Fly a flyer through the calm-air bar's Prove sweep and read it (spec section 2; N9 of section 20.1; 4A-11).

    uv run python scripts/bar_sweep.py STORE --name pid --target pid
    uv run python scripts/bar_sweep.py STORE --name pi0-seed-1 --target policy --checkpoint runs/nominal-1/checkpoint-20000000.zip
    uv run python scripts/bar_sweep.py STORE --read pi0-seed-1 --pid pid

A sweep is `poisson-prove sweep bar.toml --target drone_study.training.targets:<target> --salt <bar salt> --name <name>`:
12 calm cells x 100 replicates on the bar's seeds, one run per flyer in STORE. `--read NAME --pid PIDNAME` reads the bar
from two stored runs and the yaw gate (the yaw addenda of 2026-11-07 and 2026-11-08) from the policy's rows, and prints both
readings as JSON: {"bar": ..., "yaw": ..., "motor": ..., "motor_substep": ...}; "motor" is 4DB-1's end-of-step reading,
"motor_substep" the any-substep reading the line is ruled on since 4DB-2 ("motor_line_reading" names it). The rollout writes the four heading numbers and the motor-limit share
(ruling 4DB-1) into every row's meta, so the gate and the motor line are read from the same stored sweep as the bar.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from drone_study.training.bar import BAR_SALT, bar_sweep_spec, read_bar, rows_from_store
from drone_study.training.behaviour import (MOTOR_LINE_READING, MotorMismatch, motor_acceptance, motor_rows_from_store,
                                            motor_substep_rows_from_store, yaw_acceptance, yaw_rows_from_store)


def sweep(store: Path, name: str, target: str, checkpoint: str | None) -> None:
    store.mkdir(parents=True, exist_ok=True)
    spec = store / f"bar-{name}.toml"
    spec.write_text(bar_sweep_spec(target=f"drone_study.training.targets:{target}", target_id=f"drone-robustness-4d:{name}"))
    env = dict(os.environ)
    if checkpoint:
        env["DRONE_STUDY_CHECKPOINT"] = str(Path(checkpoint).resolve())
    cmd = [sys.executable, "-m", "poisson_prove", "--store", str(store), "sweep", str(spec), "--target",
           f"drone_study.training.targets:{target}", "--salt", BAR_SALT, "--yes", "--max-mde", "1", "--name", name, "--plain"]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    (store / f"bar-{name}.log").write_text(r.stdout + "\n" + r.stderr)
    if r.returncode != 0:
        raise SystemExit(f"sweep {name} failed ({r.returncode}):\n{r.stdout}\n{r.stderr}")
    print(f"swept {name} ({target}) into {store}", flush=True)


def read(store: Path, name: str, pid_name: str) -> dict:
    bar = read_bar(rows_from_store(store, name), rows_from_store(store, pid_name)).to_dict()
    yaw = yaw_acceptance(yaw_rows_from_store(store, name)).to_dict()
    try:
        motor = motor_acceptance(motor_rows_from_store(store, name)).to_dict()
    except MotorMismatch as exc:                                   # a run swept before the motor line (4DB-1) has no share in meta
        motor = {"refused": str(exc)}
    try:                                                           # the line's reading since 4DB-2: any of the five substeps
        motor_substep = motor_acceptance(motor_substep_rows_from_store(store, name)).to_dict()
    except MotorMismatch as exc:                                   # a run swept before 4DB-2 carries no sub-step share
        motor_substep = {"refused": str(exc)}
    return {"run": name, "pid_run": pid_name, "bar": bar, "yaw": yaw, "motor": motor, "motor_substep": motor_substep,
            "motor_line_reading": MOTOR_LINE_READING}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("store")
    ap.add_argument("--name")
    ap.add_argument("--target", choices=["policy", "pid", "geometric", "wind_only", "oracle"])
    ap.add_argument("--checkpoint")
    ap.add_argument("--read")
    ap.add_argument("--pid", default="pid")
    a = ap.parse_args()
    store = Path(a.store)
    if a.name and a.target:
        sweep(store, a.name, a.target, a.checkpoint)
    if a.read:
        print(json.dumps(read(store, a.read, a.pid), indent=1))


if __name__ == "__main__":
    main()
