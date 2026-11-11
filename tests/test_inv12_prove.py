"""Public slice (session P1c-repo, 2026-11-10): the cell-name case reads the grid cells; the held-out conditions it also read are not in this repository. No kept case is changed.

Invariant 12 (spec sections 6, 13): two policies under one salt pass Prove's own seed check on every matched row,
and the rollout never carries policy identity in cfg.

Prove is driven as the study drives it (section 13): one sweep per policy, `--target` naming the policy, one `--salt`
for the zone, then `diff --method mcnemar_bh`, which refuses rows whose stored seeds differ. Prove 0.1.2, pinned in
pyproject.toml and uv.lock.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import spec_numbers as S
from drone_study.condition import (Condition, cell_name, condition_from_cell, condition_from_cfg, condition_to_cfg,
                                   sweep_spec_toml)
from poisson_prove.store import Store

ROOT = Path(__file__).resolve().parents[1]
CELLS = [
    Condition(case="fig8-faster", az=0.0, el=0.0, speed=4.0, phase=0.0),
    Condition(case="fig8-faster", az=90.0, el=0.0, speed=8.0, phase=0.125),
    Condition(case="fig8-faster", az=180.0, el=35.0, speed=12.0, phase=0.0),
]
REPLICATES = 4
POLICY_WORDS = ("policy", "arm", "lineage", "checkpoint", "pi0", "found", "defensive", "retrain", "prove_targets", "policy_a", "policy_b")


def prove(store, *args, **kw):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / "tests"), os.environ.get("PYTHONPATH", "")]))
    return subprocess.run([sys.executable, "-m", "poisson_prove", "--store", str(store), *args, "--plain"], env=env, cwd=ROOT,
                          capture_output=True, text=True, **kw)


def sweep(store, spec, target, name):
    r = prove(store, "sweep", str(spec), "--target", target, "--salt", S.ZONE_B_SALT, "--yes", "--max-mde", "1", "--name", name)
    assert r.returncode == 0, f"sweep {name} failed:\n{r.stdout}\n{r.stderr}"


def rows_of(store, name):
    st = Store(Path(store))
    ids = [r for r in st.runs() if st.manifest(r).name == name]
    assert len(ids) == 1, (name, ids)
    return st.read(ids[0], meta=False)


_STORE = {}


def runs_store(tmp_root):
    """Three sweeps (policy a, policy b, policy a again) under one salt, made once and reused by the tests below."""
    if "store" in _STORE:
        return _STORE["store"]
    spec = tmp_root / "zone-b.toml"
    text = sweep_spec_toml(CELLS, target="prove_targets:policy_a", target_id="drone-robustness-3a", replicates=REPLICATES)
    assert text.strip(), "the study wrote no sweep spec"
    spec.write_text(text)
    store = tmp_root / "store"
    sweep(store, spec, "prove_targets:policy_a", "a")
    sweep(store, spec, "prove_targets:policy_b", "b")
    sweep(store, spec, "prove_targets:policy_a_again", "a2")
    _STORE["store"] = store
    return store


@pytest.fixture(scope="module")
def tmp_root(tmp_path_factory):
    return tmp_path_factory.mktemp("prove")


def test_inv12_two_policies_under_one_salt_pass_provs_seed_check_on_every_matched_row(tmp_root):
    runs = runs_store(tmp_root)
    r = prove(runs, "diff", "a", "b", "--method", "mcnemar_bh", "--json")
    assert r.returncode == 0, r.stdout + r.stderr          # NotPairable is raised when matched rows carry different seeds
    payload = json.loads(r.stdout)
    assert payload["seed_identity_match"] is True
    assert payload["unpaired_analysis"] is False
    a, b = rows_of(runs, "a"), rows_of(runs, "b")
    assert len(a) == len(b) == len(CELLS) * REPLICATES
    merged = a.merge(b, on=["config_hash", "replicate_index"], suffixes=("_a", "_b"))
    assert len(merged) == len(CELLS) * REPLICATES          # every row has a counterpart: no cell lost its match
    assert (merged["seed_a"] == merged["seed_b"]).all()


def test_inv12_the_cfg_prove_hands_the_rollout_names_the_cell_and_nothing_about_the_policy(tmp_root):
    runs = runs_store(tmp_root)
    for run in ("a", "b"):
        rows = rows_of(runs, run)
        configs = [json.loads(c) if isinstance(c, str) else dict(c) for c in rows["config"]]
        for cfg in configs:
            assert set(cfg) == {"cell"}
            assert not any(w in json.dumps(cfg) for w in POLICY_WORDS)
            assert condition_from_cfg(cfg) in CELLS
    ha = sorted(rows_of(runs, "a")["config_hash"].unique())
    hb = sorted(rows_of(runs, "b")["config_hash"].unique())
    assert ha == hb and len(ha) == len(CELLS)               # the same cells to Prove, so the same seeds


def test_inv12_a_policy_run_twice_under_one_salt_flies_the_same_episodes(tmp_root):
    runs = runs_store(tmp_root)
    a, a2 = rows_of(runs, "a").sort_values(["config_hash", "replicate_index"]), rows_of(runs, "a2").sort_values(["config_hash", "replicate_index"])
    assert list(a["seed"]) == list(a2["seed"])
    assert list(a["outcome"]) == list(a2["outcome"])
    assert a["t"].fillna(-1.0).tolist() == a2["t"].fillna(-1.0).tolist()


# ---- the cell naming the rollout relies on -------------------------------------------------------------------------------
def all_conditions():
    return list(CELLS)


def test_inv12_a_cell_name_is_policy_free_injective_and_reads_back_to_the_same_condition():
    conds = all_conditions()
    names = [cell_name(c) for c in conds]
    assert all(names) and len(set(names)) == len(set(conds))
    for c in conds:
        assert condition_from_cell(cell_name(c)) == c
        cfg = condition_to_cfg(c)
        assert cfg == {"cell": cell_name(c)}
        assert condition_from_cfg(cfg) == c
        assert not any(w in cell_name(c).lower() for w in POLICY_WORDS)


def test_inv12_two_conditions_that_differ_in_any_one_field_get_different_cells():
    base = Condition(case="oval-fast", az=45.0, el=0.0, speed=6.0, phase=0.25)
    variants = [Condition(**{**base.__dict__, k: v}) for k, v in
                (("case", "oval-faster"), ("az", 90.0), ("el", 35.0), ("speed", 7.0), ("phase", 0.125), ("turb", 1.5),
                 ("payload", 5.0), ("motor_index", 2), ("motor_scale", 0.85), ("delay", 20), ("gust_length", 2.0))]
    names = {cell_name(base)} | {cell_name(v) for v in variants}
    assert len(names) == 1 + len(variants)
