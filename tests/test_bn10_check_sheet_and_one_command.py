"""Rule 8 (spec 12 row 5B, 20.1, 20.6): CHECK-P1.md, and "every post 1 number regenerates from raw rows by one command".

P1a recomputes N1 to N10 with independent code, working from CHECK-P1.md. The sheet names what each number reads and how many rows,
the salts and the rules section 9 leaves to code, and which RESULTS-P1.md it is the sheet for. The one command reads Prove's
stores, finds the runs by `sweeps.run_name`, and writes both files."""
import hashlib
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import boundary_fakes as F
import boundary_numbers as B
import boundary_rows as R
from drone_study.boundary import sweeps as W
from drone_study.boundary import writer as Wr
from test_bn09_results_and_check_files import BARS, BARS_DICT, TUNING, TUNING_DICT, build

ROOT = Path(__file__).resolve().parents[1]
ENV = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / "tests"), os.environ.get("PYTHONPATH", "")]))


@pytest.fixture(scope="module")
def made():
    mp = pytest.MonkeyPatch()
    F.patch_s6(mp)
    try:
        rows = build()
        results = Wr.render_results_p1(rows, bars=BARS_DICT, tuning=TUNING_DICT)
        return rows, results, Wr.render_check_p1(rows, results)
    finally:
        mp.undo()


def sec(text, name):
    m = re.search(rf"^## {name}\b.*?(?=^## |\Z)", text, re.S | re.M)
    assert m, name
    return m.group(0)


def rows_read(text, name):
    m = re.search(r"^rows read: (\d+)", sec(text, name), re.M)
    assert m, f"{name} has no `rows read:` line"
    return int(m.group(1))


def test_bn10_the_sheet_has_a_section_per_number_with_the_rows_each_reads(made):
    _, _, check = made
    heads = re.findall(r"^## (N\d+)\b", check, re.M)
    assert heads == list(B.N_NUMBERS)
    for n in ("N1", "N2", "N3", "N4", "N5", "N6", "N8"):
        assert rows_read(check, n) == 153_600, n
    assert rows_read(check, "N7") == 5_120
    assert rows_read(check, "N9") == 0
    assert rows_read(check, "N10") == 16 * 16 * 2 * 24 == 12_288


def test_bn10_the_sheet_names_the_prove_version_the_salts_and_the_results_text_it_checks(made):
    _, results, check = made
    assert re.search(r"^Prove: poisson-prove 0\.1\.2$", check, re.M)
    assert re.search(rf"^Zone B salt: {B.ZONE_B_SALT}$", check, re.M) and re.search(rf"^Anchor salt: {B.ANCHOR_SALT}$", check, re.M)
    assert re.search(rf"^RESULTS-P1\.md sha256: {hashlib.sha256(results.encode()).hexdigest()}$", check, re.M)
    assert "python -m drone_study.boundary.writer" in check


def test_bn10_the_sheet_lists_what_section_9_leaves_to_code_so_independent_code_reads_the_same(made):
    _, _, check = made
    rules = sec(check, "Rules")
    for ruling in ("5A-1", "5A-2", "5A-3"):
        assert re.search(rf"^- {ruling}\b", rules, re.M), ruling
    assert "lowest speed" in rules and "exact" in rules.lower() and "no scored" in rules


def test_bn10_each_number_cites_its_definition_in_the_spec(made):
    _, _, check = made
    for n in B.N_NUMBERS:
        assert re.search(r"§\s*\d|section \d", sec(check, n)), n


def test_bn10_the_sheet_counts_the_rows_it_was_given_not_the_ledgers_numbers(made):
    rows, results, _ = made
    # a few replicates gone from every flyer's 14 m/s cells (the same seeds, so the pairs still match): a cell still has scored
    # rows, and the sheet says how many rows it read
    fewer = rows[~((rows["zone"] == "B") & rows["cell"].map(lambda c: len(c) == 4 and c[2] == 14.0) & rows["seed"].map(lambda s: s % 24 == 0))]
    mp = pytest.MonkeyPatch()
    F.patch_s6(mp)
    try:
        t = Wr.render_results_p1(fewer, bars=BARS_DICT, tuning=TUNING_DICT)
        check = Wr.render_check_p1(fewer, t)
    finally:
        mp.undo()
    assert rows_read(check, "N1") == int((fewer["zone"] == "B").sum()) and rows_read(check, "N1") < 153_600


# ---- build_post1 ---------------------------------------------------------------------------------------------------------------------
def test_bn10_build_writes_both_files_and_they_are_the_same_text_every_time(made, tmp_path, monkeypatch):
    rows, results, check = made
    F.patch_s6(monkeypatch)
    a, b = Wr.build_post1(rows, bars_json=BARS, tuning_json=TUNING, out_dir=tmp_path / "one")
    assert a.name == "RESULTS-P1.md" and b.name == "CHECK-P1.md" and a.exists() and b.exists()
    assert a.read_text() == results and b.read_text() == check
    a2, b2 = Wr.build_post1(rows, bars_json=BARS, tuning_json=TUNING, out_dir=tmp_path / "two")
    assert a2.exists() and b2.exists()
    assert a.read_bytes() == a2.read_bytes() and b.read_bytes() == b2.read_bytes()


# ---- one command from Prove's stores ----------------------------------------------------------------------------------------------------
REPLICATES = 2


_STORES = {}


def get_stores(factory):
    """20 flyers through Zone B and the anchor at 2 replicates a cell, made once, inside the tests that use it."""
    if "store" in _STORES:
        return _STORES["store"]
    tmp = factory.mktemp("one-command")
    store = tmp / "store"
    fl = W.flyers(tmp / "runs")
    jobs = []
    for zone, spec in (("zone-b", W.zone_b_sweep_spec()), ("anchor", W.anchor_sweep_spec())):
        assert spec.strip(), "the study wrote no sweep spec"
        short = tmp / f"{zone}.toml"
        short.write_text(re.sub(r"replicates = \d+", f"replicates = {REPLICATES}", spec))
        for f in fl:
            argv = W.sweep_argv(store=store, spec_path=short, zone=zone, flyer=f, python=sys.executable, target_module="boundary_prove_targets")
            assert argv, "the study built no sweep command"
            jobs.append((argv, W.sweep_env(f, ENV)))

    def run(job):
        argv, env = job
        return subprocess.run(argv, env=env, cwd=ROOT, capture_output=True, text=True)

    store.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(run, jobs))
    bad = [r for r in results if r.returncode != 0]
    assert not bad, bad[0].stdout + bad[0].stderr
    _STORES["store"] = store
    return store


def test_bn10_the_loader_reads_prove_stores_into_the_row_table(tmp_path_factory):
    rows = Wr.load_rows([get_stores(tmp_path_factory)])
    assert rows is not None
    assert list(rows.columns) == R.COLUMNS
    assert len(rows) == 20 * (320 + 16) * REPLICATES
    assert set(rows["flyer"]) == {f.name for f in W.flyers(ROOT)} and set(rows["zone"]) == {"B", "anchor"}      # 5B-1: the lineage file's flyers (4DR-5), not seeds 1 to 16
    b = rows[rows["zone"] == "B"]
    assert (b["cell"].map(len) == 4).all() and (rows[rows["zone"] == "anchor"]["cell"].map(len) == 2).all()
    one = rows[(rows["flyer"] == "pid") & (rows["zone"] == "B") & (rows["cell"] == (0.0, 0.0, 4.0, 0.0))]
    assert sorted(one["seed"]) == sorted(W.cell_seeds(B.ZONE_B_SALT, W.zone_b_conditions()[[(c.az, c.el, c.speed, c.phase) for c in W.zone_b_conditions()].index((0.0, 0.0, 4.0, 0.0))], 24)[:REPLICATES])
    assert (~rows["error"]).all() and set(rows["outcome"]) <= {"ok", "crash"}
    crashed = rows[rows["outcome"] == "crash"]
    assert len(crashed) and set(crashed["crash_class"]) <= set(B.CRASH_CLASSES) and rows[rows["outcome"] == "ok"]["crash_class"].isna().all()
    assert rows["crossings"].map(lambda x: isinstance(x, dict)).all()


def test_bn10_one_command_regenerates_both_files_from_the_stores(tmp_path_factory, tmp_path):
    stores = get_stores(tmp_path_factory)
    out = tmp_path / "out"
    cmd = [sys.executable, "-m", "drone_study.boundary.writer", "--store", str(stores), "--bars", str(BARS), "--tuning", str(TUNING), "--out", str(out)]
    r = subprocess.run(cmd, env=ENV, cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    results, check = out / "RESULTS-P1.md", out / "CHECK-P1.md"
    assert results.exists() and check.exists()
    assert rows_read(check.read_text(), "N1") == 20 * 320 * REPLICATES
    assert re.search(rf"^RESULTS-P1\.md sha256: {hashlib.sha256(results.read_bytes()).hexdigest()}$", check.read_text(), re.M)
    again = tmp_path / "again"
    subprocess.run(cmd[:-1] + [str(again)], env=ENV, cwd=ROOT, capture_output=True, text=True, check=True)
    assert (again / "RESULTS-P1.md").read_bytes() == results.read_bytes()
