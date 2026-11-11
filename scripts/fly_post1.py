"""Fly post 1's rows: Zone B and the anchor slice on the 20 flyers (and, if the extension fires, the 16 m/s row), one Prove run per
flyer and zone, resumable by run (spec 3, 12 row 5B, 14, 20.1; rulings 5A-13, 4DR-5).

    uv run python scripts/fly_post1.py --store STORE --zones zone-b anchor --jobs 10
    uv run python scripts/fly_post1.py --store STORE --zones zone-b-extension --jobs 10     # only once the addendum is filed

For each (zone, flyer) the script looks the run up by its name (`sweeps.run_name`) in the store:
  - complete  -> nothing is flown (a stored row is never flown again);
  - partial   -> `poisson-prove sweep SPEC --resume RUN` carries on where it stopped, on the run's own salt;
  - absent    -> a fresh sweep under the zone's salt (`sweeps.sweep_argv`).
Up to --jobs sweeps run at once, each one process on one core (OMP/MKL/OPENBLAS threads pinned to 1), at a lower priority
than the machine's other work. Each sweep's output goes to <store>/logs/<run name>.log. Exit status 0 only if every run is complete.
Run it again after any interruption: it picks up from the store.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from poisson_prove.store import Store

from drone_study.boundary import pins, sweeps

SPECS = {"zone-b": sweeps.zone_b_sweep_spec, "anchor": sweeps.anchor_sweep_spec, "zone-b-extension": sweeps.extension_sweep_spec}
THREADS = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
LOCK = threading.Lock()


def log(msg: str) -> None:
    with LOCK:
        print(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}", flush=True)


def find_run(store: Store, name: str):
    """(run id, status) of the run called `name`, or None. More than one run of a name is an error: it would be flown twice."""
    hits = [(r, str(store.manifest(r).status)) for r in store.runs() if store.manifest(r).name == name]
    if len(hits) > 1:
        raise SystemExit(f"the store holds {len(hits)} runs named {name}: {hits}")
    return hits[0] if hits else None


def job_argv(store_path: Path, spec_path: Path, zone: str, flyer, existing) -> list:
    argv = sweeps.sweep_argv(store=store_path, spec_path=spec_path, zone=zone, flyer=flyer, python=sys.executable)
    if existing is not None:
        i = argv.index("--salt")
        argv[i:i + 2] = ["--resume", existing[0]]               # the salt comes from the run's own manifest
        j = argv.index("--name")
        del argv[j:j + 2]                                        # a resumed run keeps its name
    return argv


def fly(job) -> tuple:
    store_path, spec_path, zone, flyer, logdir = job
    name = sweeps.run_name(zone, flyer.name)
    store = Store(store_path)
    existing = find_run(store, name)
    if existing is not None and existing[1] in ("complete", "completed"):
        log(f"stored   {name} ({existing[0]})")
        return name, 0
    argv = job_argv(store_path, spec_path, zone, flyer, existing)
    env = sweeps.sweep_env(flyer, dict(os.environ, **THREADS))
    verb = "resume " if existing is not None else "start  "
    log(f"{verb}  {name}" + (f" ({existing[0]}, was {existing[1]})" if existing else ""))
    with open(logdir / f"{name}.log", "ab") as fh:
        fh.write(f"\n=== {time.strftime('%Y-%m-%dT%H:%M:%S')} {' '.join(argv)}\n".encode())
        fh.flush()
        rc = subprocess.run(argv, env=env, cwd=Path(__file__).resolve().parents[1], stdout=fh, stderr=subprocess.STDOUT,
                            preexec_fn=lambda: os.nice(10)).returncode
    after = find_run(Store(store_path), name)
    state = after[1] if after else "absent"
    log(f"{'done    ' if rc == 0 else 'FAILED  '}{name} exit {rc}, run {state}")
    return name, rc


def watch(store_path: Path, names: list, stop: threading.Event, every: float) -> None:
    while not stop.wait(every):
        try:
            store = Store(store_path)
            done = rows = 0
            for n in names:
                hit = find_run(store, n)
                if hit:
                    rows += store.count(hit[0])
                    done += hit[1] in ("complete", "completed")
            log(f"progress {done}/{len(names)} runs complete, {rows} rows stored")
        except Exception as exc:                                 # a read while a sweep appends must never stop the watch
            log(f"progress unreadable: {exc!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--zones", nargs="+", default=["zone-b", "anchor"], choices=sorted(SPECS))
    ap.add_argument("--flyers", nargs="*", help="only these flyer names (default: all 20)")
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--runs-root", default=None, help="where the nominal-<seed>/checkpoint-<B0>.zip files are (default .work/4d/runs)")
    ap.add_argument("--watch-every", type=float, default=300.0)
    args = ap.parse_args()

    pinned = pins.require_prove_pin()
    store_path = Path(args.store).resolve()
    logdir = store_path / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    flyers = sweeps.flyers(args.runs_root)
    for f in flyers:
        if f.checkpoint is not None and not Path(f.checkpoint).is_file():
            raise SystemExit(f"checkpoint missing for {f.name}: {f.checkpoint}")
    if args.flyers:
        flyers = [f for f in flyers if f.name in set(args.flyers)]
    specs = {}
    for zone in args.zones:
        specs[zone] = store_path / f"spec-{zone}.toml"
        specs[zone].write_text(SPECS[zone]())
    jobs = [(store_path, specs[z], z, f, logdir) for z in args.zones for f in flyers]
    names = [sweeps.run_name(z, f.name) for z in args.zones for f in flyers]
    log(f"poisson-prove {pinned}; {len(jobs)} runs over zones {args.zones}, {len(flyers)} flyers, {args.jobs} at a time; store {store_path}")
    stop = threading.Event()
    threading.Thread(target=watch, args=(store_path, names, stop, args.watch_every), daemon=True).start()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(fly, jobs))
    stop.set()
    bad = [n for n, rc in results if rc != 0]
    store = Store(store_path)
    incomplete = [n for n in names if (find_run(store, n) or (None, "absent"))[1] not in ("complete", "completed")]
    log(f"finished: {len(names) - len(incomplete)}/{len(names)} runs complete" + (f"; failed: {bad}" if bad else "") + (f"; not complete: {incomplete}" if incomplete else ""))
    return 0 if not bad and not incomplete else 1


if __name__ == "__main__":
    raise SystemExit(main())
