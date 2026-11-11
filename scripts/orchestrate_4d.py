"""Session 4D-B's driver: the climb of the weight ladder, the pilots' bars, the pick, the nominal seeds and the lineages (spec
section 2; the yaw addenda of 2026-11-07 and 2026-11-08; rulings 4DA-3 and 4DA-5 as amended, 4DA-9/4DA-15, 4DR-1, 4DR-3).

    uv run python scripts/orchestrate_4d.py run [--max-train 10] [--max-sweeps 3] [--poll 30]
    uv run python scripts/orchestrate_4d.py status
    uv run python scripts/orchestrate_4d.py files          # write data/yaw-weight.json and data/4d-lineages.json from the store

Everything it knows is on disk under .work/4d (run directories, checkpoints, manifests, the Prove bar store), so it can be stopped
and started again and nothing is lost: a training run with a resume file and no live process is started again and resumes; a
checkpoint with no stored bar run is swept; readings are re-read from the store.

The climb (4DA-15). The base weights are trained in order. A weight's four pilots start; when its four 5M checkpoints are read
against the bar and the yaw gate and not all four pass, the next base weight's pilots start (a weight that passes at 5M ends the
climb: no later weight can enter the pick). Pilots of every weight tried train to 40M. The extension weight 0.032 starts only when
the three base weights' full tables pass nowhere, and 0.064 only when 0.032's does too; with 0.064 passing nowhere the driver stops
and reports (ruling 4DA-13). The pick is read rung first (4DR-1): the smallest rung at which every tried weight has its four readings
and some weight passes, then the smallest weight there. It is final as soon as it is read (a smaller rung has been read with no
pass, and no untried weight can enter), so the nominal seeds start then; the pick file is written when every tried weight's table
is complete, since the file carries the whole table and is refused otherwise.

The fail-fast check (the 4D-B brief, point 2). When the FIRST weight's four pilots reach 5M the driver stops their processes
(SIGSTOP), sweeps the four checkpoints through the bar, and waits for the file .work/4d/control/continue before it lets them run
on (SIGCONT) and starts anything else. The session runs the yaw gate, 4C's behaviour numbers and the clips on those checkpoints
meanwhile and creates the file if nothing new looks wrong.

Lineages (4DA-5 as amended). Nominal seeds 1, 2, 3, ... train at the pick to B0, each is swept through the bar when its checkpoint
exists, and a seed is a lineage when it passes the bar and the yaw gate; a failed seed is reported with the gate it failed and the
next seed is trained; there is no stop. The motor line (ruling 4DB-1, 2026-11-09) is the third: at most 0.5% of the bar's control
steps with any motor at its limit, read from the rows' meta; a seed failing it is reported with the gate "motor". Seeds are kept in flight so that the passed, the training and the unread together reach 16.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".work" / "4d"
RUNS = WORK / "runs"
STORE = WORK / "bar-store"
LOGS = WORK / "logs"
CONTROL = WORK / "control"
STATUS = WORK / "status.json"
READINGS = WORK / "readings.json"
PILOT_SEEDS = (9001, 9002, 9003, 9004)
PILOT_BUDGET = 40_000_000
UV = ["uv", "run", "python"]

sys.path.insert(0, str(ROOT / "src"))
from drone_study.training.bar import BarMismatch, read_bar, rows_from_store                       # noqa: E402
from drone_study.training.behaviour import (MOTOR_LIMIT_SHARE_MAX, MotorMismatch, YawMismatch, motor_acceptance,  # noqa: E402
                                            motor_rows_from_store, yaw_acceptance, yaw_rows_from_store)
from drone_study.training.budget import RUNGS                                                     # noqa: E402
from drone_study.training.reward import YAW_WEIGHT_LADDER, YAW_WEIGHT_LADDER_BASE, crash_penalty_for   # noqa: E402
from drone_study.training.yaw_weight import (LINEAGES, LINEAGES_FILE, YAW_WEIGHT_FILE, NoWeightPasses, PilotReading, YawPick,  # noqa: E402
                                             pick_yaw_weight, select_lineages_with_yaw, write_yaw_weight)


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
    print(line, flush=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    with (LOGS / "orchestrate.log").open("a") as f:
        f.write(line + "\n")


def wtag(w: float) -> str:
    return f"{w:g}"


def pilot_dir(w: float, seed: int) -> Path:
    return RUNS / f"pilot-{wtag(w)}-{seed}"


def nominal_dir(seed: int) -> Path:
    return RUNS / f"nominal-{seed}"


def pilot_run_name(w: float, seed: int, rung: int) -> str:
    return f"pilot-{wtag(w)}-{seed}-{rung}"


def nominal_run_name(seed: int) -> str:
    return f"pi0-seed-{seed}"


# ---- processes ----------------------------------------------------------------------------------------------------------------
def pids_matching(pattern: str) -> list:
    r = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    return [int(x) for x in r.stdout.split() if x.strip() and int(x) != os.getpid()]


def train_pattern(role: str, seed: int, w: float) -> str:
    return f"train_run.py --role {role} --nominal-seed {seed} --k-yaw {wtag(w)} "


def train_alive(role: str, seed: int, w: float) -> bool:
    return bool(pids_matching(train_pattern(role, seed, w)))


def launch_train(role: str, seed: int, w: float, budget: int, checkpoints: list, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    cmd = UV + ["scripts/train_run.py", "--role", role, "--nominal-seed", str(seed), "--k-yaw", wtag(w), "--budget", str(budget),
                "--checkpoints", ",".join(str(c) for c in checkpoints), str(outdir.relative_to(ROOT))]
    logf = (LOGS / f"{outdir.name}.log").open("a")
    subprocess.Popen(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
    log(f"launched {' '.join(cmd[3:])}")


def finished(outdir: Path, budget: int) -> bool:
    m = outdir / "manifest.json"
    if not m.exists():
        return False
    try:
        return int(json.loads(m.read_text())["budget"]) == budget and (outdir / f"checkpoint-{budget}.zip").exists()
    except Exception:                                                     # noqa: BLE001
        return False


def sweep_pattern(name: str) -> str:
    return f"bar_sweep.py .work/4d/bar-store --name {name} "


def sweep_alive(name: str) -> bool:
    return bool(pids_matching(sweep_pattern(name)))


def launch_sweep(name: str, checkpoint: Path) -> None:
    cmd = UV + ["scripts/bar_sweep.py", ".work/4d/bar-store", "--name", name, "--target", "policy", "--checkpoint", str(checkpoint)]
    logf = (LOGS / f"bar-{name}.log").open("a")
    subprocess.Popen(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
    log(f"launched sweep {name}")


# ---- the store ---------------------------------------------------------------------------------------------------------------
def store_runs() -> dict:
    """{name: run id} for every COMPLETE run in the bar store (a partial run, from a killed sweep, is left out and reported)."""
    from poisson_prove.store import Store

    if not STORE.exists():
        return {}
    st = Store(STORE)
    out, partial = {}, []
    for rid in st.runs():
        man = st.manifest(rid)
        status = getattr(man, "status", None)
        if status not in (None, "complete", "completed", "done", "finished"):
            partial.append((man.name, rid, status))
            continue
        out[man.name] = rid
    for name, rid, status in partial:
        if rid not in _PARTIAL_SEEN:                                      # said once per run: a sweep is partial while it runs
            _PARTIAL_SEEN.add(rid)
            log(f"store: run {name} ({rid}) is {status}; not read until complete")
    return out


_PARTIAL_SEEN: set = set()


_READ_CACHE: dict = {}


def read_run(name: str, pid_rows) -> dict:
    """The bar reading and the yaw gate's reading of a stored run; for a nominal seed also the motor line's (ruling 4DB-1: a
    lineage passes the bar, the yaw gate and the motor line; a pilot is read on the bar and the yaw gate, as the pick reads it).
    Cached on disk in readings.json."""
    if name in _READ_CACHE and (not name.startswith("pi0-seed-") or "motor" in _READ_CACHE[name]):
        return _READ_CACHE[name]
    try:
        bar = read_bar(rows_from_store(STORE, name), pid_rows).to_dict()
        yaw = yaw_acceptance(yaw_rows_from_store(STORE, name)).to_dict()
        motor = motor_acceptance(motor_rows_from_store(STORE, name)).to_dict() if name.startswith("pi0-seed-") else None
    except (BarMismatch, YawMismatch, MotorMismatch) as exc:
        log(f"reading {name} refused: {exc}")
        return {}
    out = {"bar": bar, "yaw": yaw, "passed": bool(bar["passed"] and yaw["passed"])}
    if motor is not None:
        out["motor"] = motor
        out["passed"] = bool(out["passed"] and motor["passed"])
    _READ_CACHE[name] = out
    return out


def load_reading_cache() -> None:
    if READINGS.exists():
        _READ_CACHE.update(json.loads(READINGS.read_text()))


def save_reading_cache() -> None:
    READINGS.write_text(json.dumps(_READ_CACHE, indent=1, sort_keys=True))


# ---- the climb ---------------------------------------------------------------------------------------------------------------
def weights_started() -> list:
    out = []
    for w in YAW_WEIGHT_LADDER:
        if any(pilot_dir(w, s).exists() for s in PILOT_SEEDS):
            out.append(w)
    return out


def pilot_table(readings: dict, weights: list) -> dict:
    """{weight: {rung: [PilotReading x 4] or None}} over the started weights, None where a reading is missing."""
    table = {}
    for w in weights:
        table[w] = {}
        for rung in RUNGS:
            rows = [readings.get(pilot_run_name(w, s, rung)) for s in PILOT_SEEDS]
            table[w][rung] = None if any(not r for r in rows) else [PilotReading(r["bar"]["passed"], r["yaw"]["passed"]) for r in rows]
    return table


def complete_table(table: dict) -> dict:
    """The weights whose four rungs are all read."""
    return {w: rungs for w, rungs in table.items() if all(v is not None for v in rungs.values())}


def weight_passes_at(table: dict, w: float, rung: int):
    rows = table[w][rung]
    return None if rows is None else all(r.passed for r in rows)


def decide_pick(table: dict):
    """(weight, rung) when the pick is readable from the partial table, else None. Rung first: at each rung every tried weight must
    be read; the first rung with a passing weight gives the pick (the smallest passing weight)."""
    for rung in RUNGS:
        flags = {w: weight_passes_at(table, w, rung) for w in table}
        if any(v is None for v in flags.values()):
            return None
        passing = [w for w, ok in flags.items() if ok]
        if passing:
            return (min(passing), rung)
    return None


def next_weight_to_start(table: dict):
    """The climb's next weight, or None: a base weight starts when the one before failed at 5M; an extension weight starts only when
    every weight before it has a complete table that passes nowhere."""
    started = list(table)
    if not started:
        return YAW_WEIGHT_LADDER[0]
    last = started[-1]
    idx = YAW_WEIGHT_LADDER.index(last)
    if idx + 1 >= len(YAW_WEIGHT_LADDER):
        return None
    nxt = YAW_WEIGHT_LADDER[idx + 1]
    if decide_pick(table) is not None:
        return None
    if nxt in YAW_WEIGHT_LADDER_BASE:
        at_5m = weight_passes_at(table, last, RUNGS[0])
        return nxt if at_5m is False else None
    full = complete_table(table)
    if all(w in full for w in started) and not any(weight_passes_at(table, w, r) for w in started for r in RUNGS):
        return nxt
    return None


def all_fail_everywhere(table: dict) -> bool:
    full = complete_table(table)
    return set(full) == set(table) and not any(weight_passes_at(table, w, r) for w in table for r in RUNGS)


# ---- the files ----------------------------------------------------------------------------------------------------------------
def write_pick_file(table: dict) -> YawPick:
    full = complete_table(table)
    pick = pick_yaw_weight(full)
    digest = write_yaw_weight(ROOT / YAW_WEIGHT_FILE, pick)
    log(f"wrote {YAW_WEIGHT_FILE}: weight {pick.weight} B0 {pick.b0} penalty {pick.crash_penalty} sha256 {digest}")
    return pick


def nominal_selection(readings: dict):
    """select_lineages_with_yaw over the contiguous prefix of nominal seeds that have readings, with the motor line (4DB-1);
    returns (selection, seeds read)."""
    from drone_study.training.bar import BarReading
    from drone_study.training.behaviour import MotorReading, YawReading

    rows, seed = [], 1
    while nominal_run_name(seed) in readings and readings[nominal_run_name(seed)]:
        r = readings[nominal_run_name(seed)]
        rows.append((seed, BarReading(passed=bool(r["bar"]["passed"])), YawReading(passed=bool(r["yaw"]["passed"])),
                     MotorReading(passed=bool(r["motor"]["passed"]))))
        seed += 1
    return select_lineages_with_yaw(rows), seed - 1


def write_lineages_file(readings: dict, pick: YawPick, table: dict) -> dict:
    sel, n_read = nominal_selection(readings)
    if len(sel.lineages) < LINEAGES:
        raise RuntimeError(f"only {len(sel.lineages)} lineages have passed; the file waits")
    doc = {
        "pid_run": "pid",
        "b0": int(pick.b0), "k_yaw": float(pick.weight), "crash_penalty": float(pick.crash_penalty),
        "readings": {name: r["bar"] for name, r in sorted(readings.items()) if r},
        "yaw_readings": {name: r["yaw"] for name, r in sorted(readings.items()) if r},
        "motor_readings": {name: r["motor"] for name, r in sorted(readings.items()) if r and "motor" in r},
        "motor_limit_share_max": MOTOR_LIMIT_SHARE_MAX,
        "selection": {"lineages": list(sel.lineages), "failures": list(sel.failures), "tried": sel.tried,
                      "failed_on": {str(k): v for k, v in sel.failed_on.items()}},
        "stops": [],
        "pilot_pass_table": {str(rung): [r.passed for r in table[pick.weight][rung]] for rung in RUNGS},
        "pilot_table": {wtag(w): {str(rung): [[r.bar_passed, r.yaw_passed] for r in rows] for rung, rows in rungs.items()}
                        for w, rungs in complete_table(table).items()},
        "pick_file": str(YAW_WEIGHT_FILE),
    }
    path = ROOT / LINEAGES_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    log(f"wrote {LINEAGES_FILE}: lineages {list(sel.lineages)} failures {list(sel.failures)} failed_on {dict(sel.failed_on)}")
    return doc


# ---- one pass of the loop -----------------------------------------------------------------------------------------------------
def one_pass(a) -> dict:
    status = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    runs = store_runs()
    pid_rows = rows_from_store(STORE, "pid") if "pid" in runs else None
    training, sweeping = 0, 0

    # pilots: keep every started weight's four pilots running or finished
    started = weights_started()
    for w in started:
        for s in PILOT_SEEDS:
            d = pilot_dir(w, s)
            if finished(d, PILOT_BUDGET):
                continue
            if train_alive("pilot", s, w):
                training += 1
            elif (d / "resume-state.json").exists() and not (CONTROL / "hold").exists():
                launch_train("pilot", s, w, PILOT_BUDGET, list(RUNGS), d)
                training += 1

    # sweeps of pilot checkpoints
    def want_sweeps():
        for w in started:
            for s in PILOT_SEEDS:
                for rung in RUNGS:
                    ck = pilot_dir(w, s) / f"checkpoint-{rung}.zip"
                    name = pilot_run_name(w, s, rung)
                    if ck.exists() and name not in runs:
                        yield name, ck
        if pick_known:
            for d in sorted(RUNS.glob("nominal-*"), key=lambda p: int(p.name.split("-")[1])):
                seed = int(d.name.split("-")[1])
                ck = d / f"checkpoint-{pick_known[1]}.zip"
                if finished(d, pick_known[1]) and nominal_run_name(seed) not in runs:
                    yield nominal_run_name(seed), ck

    # readings so far
    readings = {}
    if pid_rows is not None:
        for name in runs:
            if name.startswith("pilot-") or name.startswith("pi0-seed-"):
                readings[name] = read_run(name, pid_rows)
        save_reading_cache()
    table = pilot_table(readings, started)
    pick_known = decide_pick(table)
    status["weights_started"] = [wtag(w) for w in started]
    status["pick"] = {"weight": pick_known[0], "b0": pick_known[1]} if pick_known else None
    status["pilot_table"] = {wtag(w): {str(r): (None if v is None else [x.passed for x in v]) for r, v in rungs.items()} for w, rungs in table.items()}

    for name, ck in want_sweeps():
        if sweep_alive(name):
            sweeping += 1
    for name, ck in want_sweeps():
        if sweeping >= a.max_sweeps:
            break
        if not sweep_alive(name) and pid_rows is not None and not (LOGS / f"bar-{name}.failed").exists():
            launch_sweep(name, ck)
            sweeping += 1

    # the fail-fast pause on the first weight
    first = YAW_WEIGHT_LADDER[0]
    paused = CONTROL / "paused"
    cont = CONTROL / "continue"
    CONTROL.mkdir(parents=True, exist_ok=True)
    if first in started and not cont.exists():
        at_5m = all((pilot_dir(first, s) / f"checkpoint-{RUNGS[0]}.zip").exists() for s in PILOT_SEEDS)
        if at_5m and not paused.exists():
            for s in PILOT_SEEDS:
                for pid in pids_matching(train_pattern("pilot", s, first)):
                    os.kill(pid, signal.SIGSTOP)
            paused.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            log(f"fail-fast: the four pilots of {wtag(first)} are at 5M and STOPPED; sweeping their 5M checkpoints; waiting for {cont}")
        status["fail_fast"] = "paused" if paused.exists() else "not yet at 5M"
        STATUS.write_text(json.dumps(status, indent=1))
        return status
    if paused.exists() and cont.exists() and not (CONTROL / "resumed").exists():
        for s in PILOT_SEEDS:
            for pid in pids_matching(train_pattern("pilot", s, first)):
                os.kill(pid, signal.SIGCONT)
        (CONTROL / "resumed").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        log("fail-fast: continue file found; the pilots run on")
    status["fail_fast"] = "continued"

    # the climb
    nxt = next_weight_to_start(table) if not (CONTROL / "hold").exists() else None
    if nxt is not None and training + 4 <= a.max_train:
        log(f"climb: weight {wtag(nxt)} starts (table so far: {status['pilot_table']})")
        for s in PILOT_SEEDS:
            launch_train("pilot", s, nxt, PILOT_BUDGET, list(RUNGS), pilot_dir(nxt, s))
            training += 1
        started.append(nxt)
    elif nxt is not None:
        log(f"climb: weight {wtag(nxt)} waits for capacity ({training} training)")
    if started and all_fail_everywhere(table) and set(started) == set(YAW_WEIGHT_LADDER):
        status["stop"] = "no weight passes at any rung up to 0.064: ruling 4DA-13"
        log(status["stop"])

    # the pick file, once every tried weight's table is complete
    pick_obj = None
    if pick_known and set(complete_table(table)) == set(started):
        try:
            pick_obj = pick_yaw_weight(complete_table(table))
            if not (ROOT / YAW_WEIGHT_FILE).exists():
                write_pick_file(table)
        except NoWeightPasses:
            pick_obj = None

    # the nominal seeds
    if pick_known and not (CONTROL / "hold").exists():
        w, b0 = pick_known
        sel, n_read = nominal_selection(readings)
        passed = len(sel.lineages)
        status["lineages"] = {"passed": list(sel.lineages), "failures": list(sel.failures), "failed_on": dict(sel.failed_on), "read": n_read}
        if passed >= LINEAGES:
            if not (ROOT / LINEAGES_FILE).exists() and pick_obj is not None:
                write_lineages_file(readings, pick_obj, table)
            status["nominals"] = "sixteen passed"
        else:
            # seeds in flight: training, or finished and not yet read
            in_flight = []
            for d in RUNS.glob("nominal-*"):
                seed = int(d.name.split("-")[1])
                if nominal_run_name(seed) in readings and readings[nominal_run_name(seed)]:
                    continue
                in_flight.append(seed)
                if finished(d, b0):
                    continue
                if train_alive("nominal", seed, w):
                    training += 1
                elif (d / "resume-state.json").exists():
                    launch_train("nominal", seed, w, b0, [b0], d)
                    training += 1
            need = LINEAGES - passed - len(in_flight)
            next_seed = max([int(d.name.split("-")[1]) for d in RUNS.glob("nominal-*")] + [0]) + 1
            while need > 0 and training < a.max_train:
                launch_train("nominal", next_seed, w, b0, [b0], nominal_dir(next_seed))
                training += 1
                need -= 1
                next_seed += 1
            status["nominals"] = {"in_flight": sorted(in_flight), "need_more": max(need, 0)}
    status["training"], status["sweeping"] = training, sweeping
    STATUS.write_text(json.dumps(status, indent=1))
    return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["run", "status", "files", "once"])
    ap.add_argument("--max-train", type=int, default=10)
    ap.add_argument("--max-sweeps", type=int, default=3)
    ap.add_argument("--poll", type=int, default=30)
    a = ap.parse_args()
    load_reading_cache()
    if a.command == "status":
        print(STATUS.read_text() if STATUS.exists() else "no status yet")
        return
    if a.command == "once":
        print(json.dumps(one_pass(a), indent=1))
        return
    if a.command == "files":
        runs = store_runs()
        pid_rows = rows_from_store(STORE, "pid")
        readings = {name: read_run(name, pid_rows) for name in runs if name.startswith("pilot-") or name.startswith("pi0-seed-")}
        save_reading_cache()
        table = pilot_table(readings, weights_started())
        pick = write_pick_file(table)
        write_lineages_file(readings, pick, table)
        return
    log(f"orchestrator starts: max {a.max_train} training, {a.max_sweeps} sweeps, poll {a.poll} s")
    while True:
        try:
            st = one_pass(a)
            if st.get("nominals") == "sixteen passed" and (ROOT / LINEAGES_FILE).exists():
                # keep running until every pilot has finished and every checkpoint is swept, then stop
                if st["training"] == 0 and st["sweeping"] == 0 and (ROOT / YAW_WEIGHT_FILE).exists():
                    log("everything is done: sixteen lineages, the pick file, every pilot at 40M and swept")
                    return
            if st.get("stop"):
                return
        except Exception as exc:                                        # noqa: BLE001
            import traceback
            log("pass failed: " + traceback.format_exc().strip().splitlines()[-1])
            log(traceback.format_exc())
            _ = exc
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
