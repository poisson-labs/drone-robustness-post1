"""Rule 1 (spec sections 3, 7, 12 row 5A, 14, 20.1; the yaw addendum of 2026-11-08, 4DR-5): the sweep files.

"Zone B is 16 directions x 10 speeds (2 to 14 m/s) x timings 0 and 0.125, on fig8-faster, with 24 episodes a cell. The
anchor slice is its 16 cells at 16 episodes each. Both take the 20 flyers: the 16 pi0 lineages, the cascaded PID, the
geometric controller, the wind-only reference and the feasibility oracle." The sixteen are session 4D-B's lineages, read from
its lineage file, at the B0 of its pick file (4DR-5; before the addendum: seeds 1 to 16 at 4B's 5M). Test the exact cell set,
the counts and the flyer list; "the rollout never carries policy identity in cfg" (inv. 12). The extension (section 3): 16 m/s
on every direction and timing, for every flyer."""
import importlib
import inspect
import json
import tomllib
from collections import Counter
from pathlib import Path

import pytest

import boundary_numbers as B
from drone_study.boundary import sweeps as W
from drone_study.condition import cell_name, condition_from_cfg, condition_from_cell, condition_to_cfg

ROOT = Path(__file__).resolve().parents[1]
POLICY_WORDS = ("policy", "pi0", "lineage", "checkpoint", "seed", "pid", "geometric", "oracle", "wind_only", "found", "arm")
SEEDS_1_TO_16 = tuple(range(1, 17))
SEEDS_WITH_A_FAILURE = (1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17)      # seed 5 failed the yaw gate, 17 is the sixteenth


def lineage_file(tmp_path, seeds=SEEDS_1_TO_16, failures=(), failed_on=None, name="4d-lineages.json"):
    """A lineage file in 4D-B's shape (4B's bars.json with the selection and the yaw readings), with these lineages."""
    path = tmp_path / name
    path.write_text(json.dumps({"selection": {"lineages": list(seeds), "failures": list(failures), "tried": len(seeds) + len(failures),
                                              "failed_on": failed_on or {}}, "k_yaw": 0.008, "crash_penalty": 20.0, "readings": {}, "yaw_readings": {}}))
    return path


def pick_file(tmp_path, weight=0.008, b0=5_000_000, name="yaw-weight.json"):
    """A pick the rule gives, written by the package's own writer (so the loader accepts it)."""
    from drone_study.training.yaw_weight import PilotReading, pick_yaw_weight, write_yaw_weight
    table = {}
    for w in B.YAW_WEIGHT_LADDER[:3]:
        table[w] = {}
        for rung in B.RUNGS:
            ok = (w >= weight) and (rung >= b0)
            table[w][rung] = [PilotReading(bar_passed=True, yaw_passed=ok) for _ in range(4)]
    pick = pick_yaw_weight(table)
    assert (pick.weight, pick.b0) == (weight, b0)
    path = tmp_path / name
    write_yaw_weight(path, pick)
    return path


def files(tmp_path, seeds=SEEDS_1_TO_16, weight=0.008, b0=5_000_000, **kw):
    return {"lineages_path": lineage_file(tmp_path, seeds, **kw), "pick_path": pick_file(tmp_path, weight, b0)}


def key(c):
    return (c.az, c.el, c.speed, c.phase)


def spec_cells(text):
    assert text.strip(), "the study wrote no sweep spec"
    spec = tomllib.loads(text)
    return spec, spec["axes"]["cell"]["values"]


# ---- Zone B's cells ------------------------------------------------------------------------------------------------------
def test_bn01_zone_b_is_320_cells_and_exactly_the_product_of_the_grid():
    conds = W.zone_b_conditions()
    assert len(conds) == B.N_CELLS == 320
    want = {(az, el, v, ph) for az, el in B.DIRECTIONS for v in B.SPEEDS for ph in B.PHASES}
    assert {key(c) for c in conds} == want and len(want) == 320
    assert len({cell_name(c) for c in conds}) == 320                     # no cell twice


def test_bn01_every_zone_b_cell_is_fig8_faster_with_only_the_gust_set():
    conds = W.zone_b_conditions()
    assert len(conds) == 320
    for c in conds:
        assert c.case == B.CASE
        assert c.turb == 0.3 and c.payload == 0.0 and c.motor_index is None and c.motor_scale == 1.0 and c.delay == 0
        assert c.gust_length == 1.0 and c.speed > 0 and c.az is not None and c.el is not None and c.phase is not None


def test_bn01_the_sixteen_directions_are_eight_on_the_horizon_and_four_azimuths_at_plus_and_minus_35():
    assert tuple(W.DIRECTIONS) == B.DIRECTIONS
    assert len(set(W.DIRECTIONS)) == 16
    horizon = [d for d in W.DIRECTIONS if d[1] == 0.0]
    off = [d for d in W.DIRECTIONS if d[1] != 0.0]
    assert sorted(a for a, _ in horizon) == sorted(B.HORIZON_AZ)
    assert sorted(off) == sorted((a, e) for e in (35.0, -35.0) for a in (0.0, 90.0, 180.0, 270.0))


def test_bn01_the_grid_constants_are_the_specs():
    assert tuple(float(v) for v in W.ZONE_B_SPEEDS) == B.SPEEDS
    assert tuple(float(v) for v in W.ZONE_B_PHASES) == B.PHASES
    assert (W.ZONE_B_CASE, W.ZONE_B_REPLICATES, W.ANCHOR_REPLICATES) == (B.CASE, 24, 16)
    assert float(W.EXTENSION_SPEED) == B.EXTENSION_SPEED
    assert W.LINEAGES_FILE == B.LINEAGES_FILE == "data/4d-lineages.json" and W.PICK_FILE == B.PICK_FILE == "data/yaw-weight.json"
    assert not hasattr(W, "B0")                                               # B0 is read from the pick file, not pinned (4DR-5)


# ---- the extension -------------------------------------------------------------------------------------------------------
def test_bn01_the_extension_is_the_32_cells_at_16_ms_on_every_direction_and_timing():
    ext = W.extension_conditions()
    assert len(ext) == 32 and {key(c) for c in ext} == {(az, el, 16.0, ph) for az, el in B.DIRECTIONS for ph in B.PHASES}
    assert all(c.case == B.CASE and c.turb == 0.3 for c in ext)
    base = {cell_name(c) for c in W.zone_b_conditions()}
    assert not base & {cell_name(c) for c in ext}                          # new cells, none flown twice


def test_bn01_zone_b_with_a_16_top_is_the_320_plus_the_extension():
    conds = W.zone_b_conditions(B.SPEEDS_EXTENDED)
    assert len(conds) == 352
    assert {cell_name(c) for c in conds} == {cell_name(c) for c in W.zone_b_conditions()} | {cell_name(c) for c in W.extension_conditions()}


# ---- the anchor ----------------------------------------------------------------------------------------------------------
def test_bn01_the_anchor_is_16_zone_b_cells_at_the_speeds_of_the_table():
    conds = W.anchor_conditions()
    assert len(conds) == 16 == B.N_ANCHOR_CELLS
    assert {(c.az, c.phase): c.speed for c in conds} == B.ANCHOR
    assert dict(W.ANCHOR_SPEEDS) == B.ANCHOR
    assert all(c.el == 0.0 and c.case == B.CASE for c in conds)
    zone_b = {cell_name(c) for c in W.zone_b_conditions()}
    assert all(cell_name(c) in zone_b for c in conds)                    # Zone B cells flown again


def test_bn01_each_anchor_speed_appears_four_times_and_each_direction_meets_two_speeds():
    conds = W.anchor_conditions()
    assert Counter(c.speed for c in conds) == {4.0: 4, 6.0: 4, 8.0: 4, 10.0: 4}
    for az in B.HORIZON_AZ:
        assert len({c.speed for c in conds if c.az == az}) == 2


# ---- the sweep specs -----------------------------------------------------------------------------------------------------
def test_bn01_zone_b_spec_names_the_320_cells_with_24_replicates_and_declares_crash():
    spec, cells = spec_cells(W.zone_b_sweep_spec())
    assert spec["replicates"] == 24 and spec["declare"]["failures"] == ["crash"]
    assert list(spec["axes"]) == ["cell"] and spec["axes"]["cell"]["type"] == "categorical"
    assert len(cells) == len(set(cells)) == 320
    assert set(cells) == {cell_name(c) for c in W.zone_b_conditions()}


def test_bn01_anchor_spec_names_the_16_cells_with_16_replicates():
    spec, cells = spec_cells(W.anchor_sweep_spec())
    assert spec["replicates"] == 16 and spec["declare"]["failures"] == ["crash"]
    assert set(cells) == {cell_name(c) for c in W.anchor_conditions()} and len(cells) == 16


def test_bn01_the_extension_spec_is_only_the_new_cells_at_24_and_a_16_top_takes_the_whole_grid():
    spec, cells = spec_cells(W.extension_sweep_spec())
    assert spec["replicates"] == 24 and len(cells) == 32
    assert set(cells) == {cell_name(c) for c in W.extension_conditions()}
    _, full = spec_cells(W.zone_b_sweep_spec(B.SPEEDS_EXTENDED))
    assert len(full) == 352 and set(cells) <= set(full)


def test_bn01_the_spec_text_is_the_same_for_every_flyer_because_no_flyer_goes_in():
    for fn in (W.zone_b_sweep_spec, W.anchor_sweep_spec, W.extension_sweep_spec):
        assert set(inspect.signature(fn).parameters) <= {"speeds"}, fn.__name__
        assert fn() == fn()
    assert W.zone_b_sweep_spec() != W.anchor_sweep_spec()


def test_bn01_the_cfg_prove_hands_the_rollout_is_the_cell_name_and_nothing_else():
    """Inv. 12: no policy identity in cfg. Every cell of every spec is a policy-free name that reads back to its condition."""
    for text in (W.zone_b_sweep_spec(B.SPEEDS_EXTENDED), W.anchor_sweep_spec()):
        _, cells = spec_cells(text)
        assert len(cells) in (352, 16)
        for name in cells:
            assert not [w for w in POLICY_WORDS if w in name.lower()], name
            cond = condition_from_cell(name)
            assert cell_name(cond) == name
            assert condition_to_cfg(cond) == {"cell": name}
            assert condition_from_cfg({"cell": name}) == cond


# ---- the flyers ----------------------------------------------------------------------------------------------------------
def test_bn01_twenty_flyers_in_order_the_16_pi0_then_the_four_controllers(tmp_path):
    """Re-frozen by 4D-A (2026-11-08, 4DR-5). Before: the 16 pi0 were seeds 1 to 16 with no file read."""
    fl = W.flyers(tmp_path, **files(tmp_path))
    assert [f.name for f in fl] == list(B.FLYERS) and len(fl) == 20
    assert [f.target for f in fl[:16]] == ["policy"] * 16
    assert [f.target for f in fl[16:]] == ["pid", "geometric", "wind_only", "oracle"]
    assert all(f.checkpoint is None for f in fl[16:])


def test_bn01_the_sixteen_pi0_are_the_lineage_files_seeds_in_its_order_so_a_failed_seed_is_no_flyer(tmp_path):
    """4DR-5: a seed that failed the yaw gate is not a flyer and the sixteenth passing seed is."""
    fl = W.flyers(tmp_path, **files(tmp_path, SEEDS_WITH_A_FAILURE, failures=(5,), failed_on={"5": "yaw"}))
    assert [f.name for f in fl[:16]] == [f"pi0-seed-{s}" for s in SEEDS_WITH_A_FAILURE]
    assert "pi0-seed-5" not in [f.name for f in fl] and "pi0-seed-17" in [f.name for f in fl]
    assert [Path(f.checkpoint) for f in fl[:16]] == [tmp_path / f"nominal-{s}" / "checkpoint-5000000.zip" for s in SEEDS_WITH_A_FAILURE]
    assert W.lineage_seeds(lineage_file(tmp_path, SEEDS_WITH_A_FAILURE, failures=(5,), failed_on={"5": "yaw"})) == SEEDS_WITH_A_FAILURE
    assert [f.target for f in fl[16:]] == ["pid", "geometric", "wind_only", "oracle"]


def test_bn01_a_lineage_file_without_sixteen_or_with_a_failed_seed_among_them_is_refused(tmp_path):
    with pytest.raises(ValueError):
        W.lineage_seeds(lineage_file(tmp_path, SEEDS_1_TO_16[:15]))
    with pytest.raises(ValueError):
        W.lineage_seeds(lineage_file(tmp_path, SEEDS_1_TO_16 + (17,)))
    with pytest.raises(ValueError):
        W.lineage_seeds(lineage_file(tmp_path, SEEDS_1_TO_16, failures=(3,), failed_on={"3": "bar"}))     # 3 is listed on both sides
    with pytest.raises(ValueError):
        W.lineage_seeds(lineage_file(tmp_path, (1, 2, 3, 3) + SEEDS_1_TO_16[4:]))                        # a seed twice
    with pytest.raises((FileNotFoundError, OSError, ValueError)):
        W.lineage_seeds(tmp_path / "missing.json")
    assert W.lineage_seeds(lineage_file(tmp_path)) == SEEDS_1_TO_16                                       # the control


def test_bn01_each_pi0_loads_from_4bs_checkpoint_format_at_the_pick_files_b0(tmp_path):
    """Re-frozen by 4D-A (2026-11-08, 4DR-5). Before: `checkpoint-5000000.zip` from a 5M constant."""
    for b0 in (5_000_000, 10_000_000, 40_000_000):
        fl = W.flyers(tmp_path, **files(tmp_path, b0=b0))
        assert len(fl) == 20
        for seed, f in zip(B.PI0_SEEDS, fl[:16]):
            assert Path(f.checkpoint) == tmp_path / f"nominal-{seed}" / f"checkpoint-{b0}.zip"
            assert W.checkpoint_path(seed, tmp_path, b0) == Path(f.checkpoint)
        assert W.b0_from(pick_file(tmp_path, b0=b0)) == b0
    assert W.checkpoint_path(3, tmp_path, 10_000_000) == tmp_path / "nominal-3" / "checkpoint-10000000.zip"
    with pytest.raises(TypeError):
        W.checkpoint_path(3, tmp_path)                                                                     # no default B0 any more


def test_bn01_a_pick_the_rule_would_not_give_or_a_missing_pick_is_refused(tmp_path):
    from drone_study.training.yaw_weight import WeightMismatch
    good = pick_file(tmp_path)
    d = json.loads(good.read_text())
    for tamper in ({"b0": 7_000_000}, {"weight": 0.05}, {"crash_penalty": 26.0}):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({**d, **tamper}))
        with pytest.raises(WeightMismatch):
            W.b0_from(bad)
        with pytest.raises(WeightMismatch):
            W.flyers(tmp_path, lineages_path=lineage_file(tmp_path), pick_path=bad)
    with pytest.raises((FileNotFoundError, OSError, WeightMismatch)):
        W.flyers(tmp_path, lineages_path=lineage_file(tmp_path), pick_path=tmp_path / "missing.json")
    assert W.b0_from(good) == 5_000_000                                                                     # the control


def test_bn01_the_default_runs_root_is_4d_bs_work_directory_and_the_default_files_are_4d_bs(tmp_path):
    """Re-frozen by 4D-A (the yaw addenda): 4B's sixteen pi0 are superseded, and the flyers load from session 4D-B's runs at
    the seeds and B0 of its files. Before: `.work/4b/runs`, seeds 1 to 16, 5M."""
    assert W.RUNS_ROOT == ".work/4d/runs"
    fl = W.flyers(**files(tmp_path, SEEDS_WITH_A_FAILURE, failures=(5,), failed_on={"5": "yaw"}, b0=10_000_000))
    assert len(fl) == 20
    f = fl[0]
    assert str(f.checkpoint).endswith(str(Path(".work/4d/runs/nominal-1/checkpoint-10000000.zip")))
    assert str(fl[15].checkpoint).endswith(str(Path(".work/4d/runs/nominal-17/checkpoint-10000000.zip")))
    assert ".work/4b" not in str(f.checkpoint) and "checkpoint-5000000" not in str(f.checkpoint)
    if not (ROOT / B.LINEAGES_FILE).exists() or not (ROOT / B.PICK_FILE).exists():
        with pytest.raises((FileNotFoundError, OSError, ValueError)):
            W.flyers()                                                                                    # no files: no flyers, never seeds 1 to 16


def test_bn01_every_target_named_is_a_flyer_of_the_training_targets_module(tmp_path):
    mod = importlib.import_module("drone_study.training.targets")
    fl = W.flyers(Path("x"), **files(tmp_path))
    assert len(fl) == 20
    for f in fl:
        assert callable(getattr(mod, f.target)), f.name
    assert W.TARGET_MODULE == "drone_study.training.targets" and W.CHECKPOINT_ENV == "DRONE_STUDY_CHECKPOINT"


def test_bn01_the_real_checkpoints_are_4d_bs_nominal_runs_at_its_b0_and_load():
    """Re-frozen by 4D-A (the yaw addenda of 2026-11-07 and 2026-11-08). 4D-B's 16 nominal checkpoints (git-excluded raw output
    under .work/4d/runs, listed with their SHA-256 in 4D-B's manifests): the sixteen are the lineage file's seeds, each file's hash
    is its manifest's, each manifest names the heading term's weight the rule picked (`data/yaw-weight.json`), that weight's crash
    penalty and 4D-B's B0, and each loads as a 319-input network at that B0. Skipped only where that raw output is absent; 5B runs
    where it is present. Before: 4B's runs under .work/4b, at 4B-2's B0 = 5M, with no weight to check."""
    import hashlib

    from drone_study.training.yaw_weight import load_yaw_weight

    root = ROOT / ".work" / "4d" / "runs"
    if not root.exists():
        pytest.skip("4D-B's raw output is not under .work/4d in this checkout (symlink or copy it, see docs/boundary-invariant-table.md)")
    from drone_study.training.policy import load_policy

    pick = load_yaw_weight(ROOT / B.PICK_FILE)                                # the weight, B0 and penalty the rule gave; refused if not the rule's
    assert pick.weight in B.YAW_WEIGHT_LADDER and pick.b0 in B.RUNGS and pick.crash_penalty == B.CRASH_PENALTIES[pick.weight]
    seeds = W.lineage_seeds(ROOT / B.LINEAGES_FILE)
    lineage_doc = json.loads((ROOT / B.LINEAGES_FILE).read_text())
    assert lineage_doc["k_yaw"] == pick.weight and lineage_doc["b0"] == pick.b0 and lineage_doc["crash_penalty"] == pick.crash_penalty
    pi0 = W.flyers(root)[:16]
    assert len(pi0) == 16 and [f.name for f in pi0] == [f"pi0-seed-{s}" for s in seeds]
    for f, seed in zip(pi0, seeds):
        path = Path(f.checkpoint)
        assert path.exists(), path
        assert path == root / f"nominal-{seed}" / f"checkpoint-{pick.b0}.zip"
        manifest = json.loads((path.parent / "manifest.json").read_text())
        assert manifest["budget"] == pick.b0 and manifest["checkpoint_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert manifest["lineage"] == seed
        assert manifest["reward"]["k_yaw"] == pick.weight and manifest["reward"]["k_track"] == 0.1
        assert manifest["reward"]["crash_penalty"] == pick.crash_penalty == B.CRASH_PENALTIES[pick.weight]
        policy = load_policy(path)
        assert policy.model.observation_space.shape == (319,) and policy.model.num_timesteps == pick.b0


# ---- how many episodes ---------------------------------------------------------------------------------------------------
def test_bn01_the_episode_counts_are_the_ledgers():
    got = W.row_counts()
    assert got == B.COUNTS
    assert B.COUNTS["zone_b_pi0"] + B.COUNTS["zone_b_pid_geometric"] + B.COUNTS["zone_b_wind_only"] + B.COUNTS["zone_b_oracle"] == B.COUNTS["zone_b"]
    assert got["zone_b"] == 20 * 320 * 24 and got["anchor"] == 20 * 16 * 16 and got["extension"] == 20 * 32 * 24


# ---- the Prove commands --------------------------------------------------------------------------------------------------
def argv_for(zone, flyer, tmp_path, **kw):
    return W.sweep_argv(store=tmp_path / "store", spec_path=tmp_path / "spec.toml", zone=zone, flyer=flyer, python="PY", **kw)


@pytest.mark.parametrize("zone,salt", [("zone-b", B.ZONE_B_SALT), ("anchor", B.ANCHOR_SALT), ("zone-b-extension", B.ZONE_B_SALT)])
def test_bn01_the_sweep_command_names_the_zones_salt_the_flyers_target_and_a_run_of_its_own(zone, salt, tmp_path):
    fl = W.flyers(tmp_path / "runs", **files(tmp_path))
    assert len(fl) == 20
    for f in (fl[0], fl[15], fl[16], fl[19]):
        a = argv_for(zone, f, tmp_path)
        assert a[:3] == ["PY", "-m", "poisson_prove"]
        assert a[a.index("--store") + 1] == str(tmp_path / "store")
        assert a[a.index("sweep") + 1] == str(tmp_path / "spec.toml")
        assert a[a.index("--target") + 1] == f"drone_study.training.targets:{f.target}"
        assert a[a.index("--salt") + 1] == salt
        assert a[a.index("--name") + 1] == W.run_name(zone, f.name)
        assert a[a.index("--max-mde") + 1] == "1"
        assert "--yes" in a and "--plain" in a
        assert not [x for x in a if x in ("--set", "--cfg", "--config", "--replicates")]      # nothing in cfg, replicates from the spec
    assert W.salt_for(zone) == salt


def test_bn01_the_target_module_can_be_swapped_and_run_names_are_unique_per_flyer_and_zone(tmp_path):
    fl = W.flyers(tmp_path, **files(tmp_path))
    assert len(fl) == 20
    f = fl[17]
    a = argv_for("zone-b", f, tmp_path, target_module="boundary_prove_targets")
    assert a[a.index("--target") + 1] == "boundary_prove_targets:geometric"
    names = [W.run_name(z, f.name) for z in ("zone-b", "anchor", "zone-b-extension") for f in fl]
    assert len(names) == len(set(names)) == 60
    assert W.run_name("zone-b", "pi0-seed-1") == "zone-b-pi0-seed-1" and W.run_name("anchor", "pid") == "anchor-pid"


def test_bn01_a_pi0_sweep_carries_its_checkpoint_in_the_environment_and_a_controller_carries_none(tmp_path):
    fl = W.flyers(tmp_path, **files(tmp_path))
    assert len(fl) == 20
    env = W.sweep_env(fl[4], {"PATH": "/bin"})
    assert env["PATH"] == "/bin" and Path(env["DRONE_STUDY_CHECKPOINT"]) == Path(fl[4].checkpoint).resolve()
    assert W.sweep_env(fl[16], {"PATH": "/bin"}) == {"PATH": "/bin"}
