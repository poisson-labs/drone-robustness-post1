"""Train one nominal or pilot run in calm air (spec section 2; rulings 4A-1, 4A-2, 4A-13).

    uv run python scripts/train_run.py --role pilot --nominal-seed 9001 --k-yaw 0.004 --budget 40000000 --checkpoints 5000000,10000000,20000000,40000000 OUTDIR
    uv run python scripts/train_run.py --role nominal --nominal-seed 1 --k-yaw 0.004 --budget 5000000 --checkpoints 5000000 OUTDIR

The heading term's weight `--k-yaw` is a ladder weight (the yaw addenda of 2026-11-07 and 2026-11-08) and is required: the
environment refuses a weight off the ladder, and the manifest's `reward` block records it with that weight's crash penalty.

The run's seed is `run_seed(role, nominal seed)`; the sampler is `nominal` (calm air, the training family); the 25
environments are `make_envs`; PPO is `make_ppo`. Training runs in whole rollouts of 31,250 steps. At each listed step
count the model is saved as OUTDIR/checkpoint-<steps>.zip and hashed. The run is resumable: every `--save-every`
rollouts (and at every checkpoint) OUTDIR/resume.zip, resume-state.json and the environments' generator states are
written; started again with the same arguments, the run picks up from the last save at a rollout boundary, with fresh
episodes in the 25 environments (the ones that were running are counted as unfinished in the log), and the manifest
records every resume. The learning curve (one row per rollout) is OUTDIR/curve.csv; the manifest OUTDIR/manifest.json.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

from drone_study.training.budget import ROLLOUT_STEPS
from drone_study.training.manifest import ResetLog, merge, run_manifest
from drone_study.training.seeds import run_seed


def _git_commit():
    import subprocess

    root = Path(__file__).resolve().parents[1]
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--role", required=True, choices=["pilot", "nominal"])
    ap.add_argument("--nominal-seed", type=int, required=True)
    ap.add_argument("--k-yaw", type=float, required=True, help="the heading term's weight, a ladder weight")
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--checkpoints", default="", help="comma-separated step counts to save at")
    ap.add_argument("--save-every", type=int, default=16, help="rollouts between resume saves")
    a = ap.parse_args()
    if a.budget % ROLLOUT_STEPS:
        raise SystemExit(f"the budget must be whole rollouts of {ROLLOUT_STEPS}")
    checkpoints = sorted({int(x) for x in a.checkpoints.split(",") if x} | {a.budget})
    if any(c % ROLLOUT_STEPS for c in checkpoints):
        raise SystemExit("every checkpoint must be a whole number of rollouts")

    import torch
    from stable_baselines3 import PPO

    from drone_study.training.env import make_envs
    from drone_study.training.hyper import HYPERPARAMETERS, make_ppo, save_checkpoint, set_torch_threads
    from drone_study.training.sampler import make_sampler

    set_torch_threads()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    seed = run_seed(a.role, a.nominal_seed)
    arm = "nominal"
    sampler = make_sampler(arm)
    envs = make_envs(sampler, seed, a.k_yaw)
    state_path, resume_path, curve_path = out / "resume-state.json", out / "resume.zip", out / "curve.csv"
    state = {"role": a.role, "nominal_seed": a.nominal_seed, "run_seed": seed, "k_yaw": a.k_yaw, "budget": a.budget, "checkpoints": checkpoints,
             "steps": 0, "rollouts": 0, "resumes": [], "checkpoint_sha256": {}, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "git_commit": _git_commit(), "hyperparameters": {k: (list(v) if isinstance(v, tuple) else v) for k, v in HYPERPARAMETERS.items()},
             "machine": platform.platform(), "torch": torch.__version__, "pid": os.getpid(), "log_before_resume": None}
    if state_path.exists() and resume_path.exists():
        state = json.loads(state_path.read_text())
        if float(state.get("k_yaw", a.k_yaw)) != a.k_yaw:
            raise SystemExit(f"this run trained at k_yaw {state.get('k_yaw')}, not {a.k_yaw}")
        model = PPO.load(str(resume_path), env=envs, device="cpu")
        set_torch_threads()
        model.num_timesteps = state["steps"]
        torch.set_rng_state(torch.tensor(np.frombuffer(bytes.fromhex(state["torch_rng"]), dtype=np.uint8).copy()))
        for i, s in enumerate(state["env_rng"]):
            envs.envs[i]._rng.bit_generator.state = s
        prior = ResetLog.from_dict(state["log_before_resume"]) if state.get("log_before_resume") else ResetLog()
        state["resumes"].append({"at_steps": state["steps"], "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        print(f"resumed at {state['steps']} steps ({len(state['resumes'])} resume(s))", flush=True)
    else:
        model = make_ppo(envs, seed=seed)
        prior = ResetLog()
        curve_path.write_text("steps,rollouts,wall_s,episodes_finished,mean_episode_steps,share_near_reset,mean_reward_per_step,mean_episode_return\n")

    def current_log():
        return merge([prior] + [e.log for e in envs.envs])

    def save_resume():
        # the running episodes are counted as unfinished in the saved log; the resumed run starts them afresh
        snapshot = merge([current_log()])
        for e in envs.envs:
            if e.steps_in_progress:
                snapshot.add(e.steps_in_progress, finished=False)
        state["log_before_resume"] = snapshot.to_dict()
        state["torch_rng"] = torch.get_rng_state().numpy().tobytes().hex()
        state["env_rng"] = [e._rng.bit_generator.state for e in envs.envs]
        model.save(str(resume_path))
        state_path.write_text(json.dumps(state, indent=1, default=str))

    t0 = time.time() - state.get("wall_s", 0.0)
    done_ckpts = set(int(k) for k in state["checkpoint_sha256"])
    while state["steps"] < a.budget:
        before = model.num_timesteps
        model.learn(total_timesteps=ROLLOUT_STEPS, reset_num_timesteps=False)
        assert model.num_timesteps == before + ROLLOUT_STEPS, (model.num_timesteps, before)
        state["steps"] = model.num_timesteps
        state["rollouts"] += 1
        state["wall_s"] = time.time() - t0
        log = current_log()
        rb = model.rollout_buffer
        rewards = float(np.mean(rb.rewards)) if rb.full else float("nan")
        mean_len = log.mean_episode_steps
        with curve_path.open("a") as f:
            f.write(f"{state['steps']},{state['rollouts']},{state['wall_s']:.1f},{log.episodes},{mean_len:.2f},"
                    f"{log.share_near_reset:.5f},{rewards:.6f},{rewards * mean_len:.4f}\n")
        if state["steps"] in checkpoints and state["steps"] not in done_ckpts:
            path = out / f"checkpoint-{state['steps']}.zip"
            digest = save_checkpoint(model, path)
            state["checkpoint_sha256"][str(state["steps"])] = digest
            done_ckpts.add(state["steps"])
            print(f"checkpoint {state['steps']} sha256 {digest}  [{state['wall_s']:.0f} s]", flush=True)
            save_resume()
        elif state["rollouts"] % a.save_every == 0:
            save_resume()
        if state["rollouts"] % 8 == 0:
            print(f"{state['steps']:>10d} steps  {log.episodes:>7d} episodes  mean len {mean_len:7.1f}  reward/step {rewards:+.4f}  "
                  f"{state['wall_s']:.0f} s  {state['steps'] / max(state['wall_s'], 1e-9):.0f} steps/s", flush=True)

    for e in envs.envs:
        e.close_log()
    log = current_log()
    manifest = run_manifest(arm=arm, lineage=a.nominal_seed, seed=seed, budget=a.budget, log=log,
                            checkpoint_sha256=state["checkpoint_sha256"][str(a.budget)], k_yaw=a.k_yaw)
    manifest.update({"role": a.role, "checkpoints": {k: {"steps": int(k), "file": f"checkpoint-{k}.zip", "sha256": v}
                                                     for k, v in state["checkpoint_sha256"].items()},
                     "resumes": state["resumes"], "wall_seconds": state["wall_s"], "git_commit": state["git_commit"],
                     "started": state["started"], "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    save_resume()
    print(f"done: {state['steps']} steps, {log.episodes} episodes, mean length {log.mean_episode_steps:.1f} steps, "
          f"share near reset {log.share_near_reset:.4f}, {state['wall_s']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
