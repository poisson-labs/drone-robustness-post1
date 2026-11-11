"""Rule 2 (spec sections 6, 10 item 6, 12 inv. 12): the salts, and one seed set for every flyer.

"Zone B's salt and the anchor's are the frozen values in section 10 item 6. derive_seed(salt, cell, k) gives every flyer
the same seeds in a cell. Two flyers under one salt pass Prove's own seed check on every matched row." The anchor cells are
Zone B cells flown again "on fresh seeds"."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from poisson_prove.executor import derive_seed
from poisson_prove.store import Store, config_hash

import boundary_numbers as B
from drone_study.boundary import sweeps as W
from drone_study.condition import cell_name, condition_to_cfg

ROOT = Path(__file__).resolve().parents[1]
ENV = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / "tests"), os.environ.get("PYTHONPATH", "")]))


def formula(zone):
    return hashlib.sha256(f"{B.SALT_PREFIX}{zone}".encode()).hexdigest()[:16]


# ---- the salts ---------------------------------------------------------------------------------------------------------
def test_bn02_zone_bs_and_the_anchors_salts_are_the_frozen_values():
    assert W.ZONE_B_SALT == B.ZONE_B_SALT == "9c3d4a1e317d59fa"
    assert W.ANCHOR_SALT == B.ANCHOR_SALT == "644f7723497d4cae"


def test_bn02_the_frozen_values_are_section_10s_formula_and_differ_from_every_other_zones():
    for zone, salt in B.SALTS.items():
        assert formula(zone) == salt, zone
    assert W.ZONE_B_SALT == formula("B") and W.ANCHOR_SALT == formula("anchor")
    others = [s for z, s in B.SALTS.items() if z not in ("B", "anchor")]
    assert W.ZONE_B_SALT not in others and W.ANCHOR_SALT not in others and W.ZONE_B_SALT != W.ANCHOR_SALT


def test_bn02_each_sweep_zone_takes_its_own_salt_and_the_extension_takes_zone_bs():
    assert (W.salt_for("zone-b"), W.salt_for("anchor"), W.salt_for("zone-b-extension")) == (B.ZONE_B_SALT, B.ANCHOR_SALT, B.ZONE_B_SALT)


# ---- the seeds ---------------------------------------------------------------------------------------------------------
def test_bn02_a_cells_seeds_are_derive_seed_of_the_salt_the_cell_and_k_for_every_zone_b_cell():
    conds = W.zone_b_conditions(B.SPEEDS_EXTENDED)
    assert len(conds) == 352
    for c in conds:
        h = config_hash(condition_to_cfg(c))
        want = tuple(derive_seed(B.ZONE_B_SALT, h, k) for k in range(24))
        assert W.cell_seeds(B.ZONE_B_SALT, c, 24) == want, cell_name(c)
        assert len(set(want)) == 24


def test_bn02_seeds_depend_on_the_cell_and_the_salt_only_so_every_flyer_gets_the_same_ones():
    """cell_seeds has no flyer argument; its answer for a cell does not depend on which other cells are in the sweep."""
    import inspect
    assert list(inspect.signature(W.cell_seeds).parameters) == ["salt", "condition", "replicates"]
    cs = W.zone_b_conditions()
    assert len(cs) == 320
    assert W.cell_seeds(B.ZONE_B_SALT, cs[0], 24) == W.cell_seeds(B.ZONE_B_SALT, cs[0], 24)
    assert len({W.cell_seeds(B.ZONE_B_SALT, c, 24)[0] for c in cs}) == 320            # cells differ from one another
    assert W.cell_seeds(B.ZONE_B_SALT, cs[0], 24)[:16] == W.cell_seeds(B.ZONE_B_SALT, cs[0], 16)


def test_bn02_the_anchors_seeds_are_fresh_on_every_one_of_its_cells():
    cells = W.anchor_conditions()
    assert len(cells) == 16
    for c in cells:
        a = W.cell_seeds(B.ANCHOR_SALT, c, 16)
        z = W.cell_seeds(B.ZONE_B_SALT, c, 24)
        assert a == tuple(derive_seed(B.ANCHOR_SALT, config_hash(condition_to_cfg(c)), k) for k in range(16))
        assert not set(a) & set(z), cell_name(c)


# ---- Prove's own seed check, end to end -------------------------------------------------------------------------------
def prove(store, *args):
    return subprocess.run([sys.executable, "-m", "poisson_prove", "--store", str(store), *args, "--plain"], env=ENV, cwd=ROOT,
                          capture_output=True, text=True)


def fly(store, zone, spec_text, flyer, tmp):
    assert spec_text.strip(), f"the study wrote no {zone} sweep spec"
    spec = tmp / f"{zone}.toml"
    spec.write_text(spec_text)
    argv = W.sweep_argv(store=store, spec_path=spec, zone=zone, flyer=flyer, python=sys.executable, target_module="boundary_prove_targets")
    assert argv, "the study built no sweep command"
    r = subprocess.run(argv, env=W.sweep_env(flyer, ENV), cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, f"{flyer.name} {zone}:\n{r.stdout}\n{r.stderr}"


_STORE = {}


def get_store(factory):
    """Two flyers (a pi0 and the PID) swept through Zone B and the anchor under their salts, made once. Built inside the tests
    that use it, so a stub's refusal is a failed assertion in the test and not a setup error."""
    if "store" not in _STORE:
        tmp = factory.mktemp("boundary-prove")
        fl = W.flyers(tmp / "runs")
        assert len(fl) == 20, "the study lists no flyers"
        st = tmp / "store"
        for f in (fl[0], fl[16]):
            fly(st, "zone-b", W.zone_b_sweep_spec(), f, tmp)
            fly(st, "anchor", W.anchor_sweep_spec(), f, tmp)
        _STORE["store"] = (st, (fl[0], fl[16]))
    return _STORE["store"]


def rows_of(store, name):
    s = Store(Path(store))
    ids = [r for r in s.runs() if s.manifest(r).name == name]
    assert len(ids) == 1, (name, ids)
    return s.read(ids[0], meta=False)


def test_bn02_two_flyers_in_zone_b_pass_proves_seed_check_on_every_matched_row(tmp_path_factory):
    st, (a, b) = get_store(tmp_path_factory)
    na, nb = W.run_name("zone-b", a.name), W.run_name("zone-b", b.name)
    r = prove(st, "diff", na, nb, "--method", "mcnemar_bh", "--json")
    assert r.returncode == 0, r.stdout + r.stderr
    payload = json.loads(r.stdout)
    assert payload["seed_identity_match"] is True and payload["unpaired_analysis"] is False
    ra, rb = rows_of(st, na), rows_of(st, nb)
    assert len(ra) == len(rb) == 320 * 24 == 7680
    merged = ra.merge(rb, on=["config_hash", "replicate_index"], suffixes=("_a", "_b"))
    assert len(merged) == 7680 and (merged["seed_a"] == merged["seed_b"]).all()


def test_bn02_the_stored_seeds_are_the_derived_ones_and_the_cfg_is_the_cell_name_alone(tmp_path_factory):
    st, (a, _) = get_store(tmp_path_factory)
    ra = rows_of(st, W.run_name("zone-b", a.name))
    want = {}
    for c in W.zone_b_conditions():
        for k, s in enumerate(W.cell_seeds(B.ZONE_B_SALT, c, 24)):
            want[(config_hash(condition_to_cfg(c)), k)] = s
    got = {(h, k): s for h, k, s in zip(ra["config_hash"], ra["replicate_index"], ra["seed"])}
    assert got == want
    assert all(set(cfg) == {"cell"} for cfg in ra["config"])


def test_bn02_the_anchor_sweep_is_16_cells_of_16_on_the_anchors_seeds_and_pairs_too(tmp_path_factory):
    st, (a, b) = get_store(tmp_path_factory)
    na, nb = W.run_name("anchor", a.name), W.run_name("anchor", b.name)
    r = prove(st, "diff", na, nb, "--method", "mcnemar_bh", "--json")
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads(r.stdout)["seed_identity_match"] is True
    ra = rows_of(st, na)
    assert len(ra) == 256
    want = {(config_hash(condition_to_cfg(c)), k): s for c in W.anchor_conditions() for k, s in enumerate(W.cell_seeds(B.ANCHOR_SALT, c, 16))}
    assert {(h, k): s for h, k, s in zip(ra["config_hash"], ra["replicate_index"], ra["seed"])} == want
