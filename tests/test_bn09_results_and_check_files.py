"""Public slice (session P1c-repo, 2026-11-10): the Prove-pin case no longer cross-checks the pin against the analysis tests' numbers file, which is not in this repository. No kept assertion is changed.

Rule 7 (spec 12 row 5B, 20.1, 20.2, 20.3 points 9 and 11, 20.4, 20.6): RESULTS-P1.md and CHECK-P1.md, and Prove's pin.

"Post 1 reports these and no other number." The writer prints each speed with its flat force ratio under 17.3's primary air model;
marks censored values; says in N5 that the 16 differences share one controller's estimate; gives N6 as counts only, with no flag,
word or p-value from Prove's diff; gives N8 as S6's re-read under each of its crash limits; reads N9 from 4B's stored readings
and does not recompute it; and prints Prove's pinned version (0.1.2, ruling 5A-4). Layout: the docstring of
drone_study.boundary.writer. Every number is checked against the oracle on the synthetic counts the rows were built from."""
import hashlib
import importlib.metadata as md
import json
import re
from pathlib import Path

import numpy as np
import pytest

import boundary_fakes as F
import boundary_numbers as B
import boundary_oracles as O
import boundary_rows as R
from drone_study.boundary import pins
from drone_study.boundary import writer as Wr
from drone_study.training import bar as bar_module

DATA = Path(__file__).resolve().parent / "data"
BARS, TUNING = DATA / "4d-lineages-shape.json", DATA / "4b-tuning-pick.json"   # 4D-B's lineage-file shape on 4B's readings (4DR-5)
ROOT = Path(__file__).resolve().parents[1]
N5_NOTE = "All 16 differences share one controller's estimate, so its error moves them together."
EXT = B.SPEEDS_EXTENDED
BASE_THR = np.array([5, 6, 7, 8, 5, 6, 7, 8, 6, 7, 5, 6, 7, 5, 6, 7])          # per direction


# ---- the synthetic flyers, with counts known by construction ----------------------------------------------------------------
def ramp(thr, speeds=B.SPEEDS, shift=0):
    """Per direction: 24 crashes from speed index thr, 12 at thr - 1, none below, at timing 0; timing 0.125 breaks `shift` speeds
    later (a negative shift: earlier), so the two timings differ and a swapped column shows. Returns (D, S, 2)."""
    idx = np.arange(len(speeds))[None, :]

    def one(t):
        t = np.asarray(t)[:, None]
        return np.where(idx >= t, 24, np.where(idx == t - 1, 12, 0))
    t0 = np.asarray(thr)
    return np.stack([one(t0), one(np.maximum(t0 + shift, 0))], axis=2)


def pi0_counts(i, speeds=B.SPEEDS, fire=False):
    thr = np.full(16, 9) if fire else np.clip(BASE_THR + (i % 3) - 1, 1, 8)
    thr = thr.copy()
    if i == 0 and not fire:
        thr[0] = 0                                                                # lineage 1, direction 0: crashes at every speed
    return ramp(thr, speeds, -(i % 2))                                            # odd lineages: timing 0.125 breaks a speed earlier


CTL = {"pid": ramp(np.full(16, 7), shift=-1), "geometric": ramp(BASE_THR + 1), "wind_only": ramp(np.clip(BASE_THR - 1, 1, 9), shift=-1),
       "oracle": np.zeros(R.SHAPE, int)}                                           # the oracle never crashes: censored at the top


def counts_of(flyer, fire=False):
    return pi0_counts(int(flyer.rsplit("-", 1)[1]) - 1, fire=fire) if flyer.startswith("pi0") else CTL[flyer]


def anchor_of(flyer):
    i = B.FLYERS.index(flyer)
    return [(3 * i + j) % 17 for j in range(16)]


def build(fire=False):
    rows = []
    for f in B.FLYERS:
        c = counts_of(f, fire)
        if fire:
            top = np.full((16, 1, 2), 24)
            c = np.concatenate([c, top if f.startswith("pi0") else np.zeros((16, 1, 2), int) + 12], axis=1)
            rows += R.zone_b_rows(f, c, speeds=EXT)
        else:
            rows += R.zone_b_rows(f, c)
        rows += R.anchor_rows(f, anchor_of(f))
    return R.frame(rows)


@pytest.fixture(scope="module")
def rows():
    return build()


@pytest.fixture(scope="module")
def text(rows):
    F_patch = pytest.MonkeyPatch()
    F.patch_s6(F_patch)
    try:
        return Wr.render_results_p1(rows, bars=BARS_DICT, tuning=TUNING_DICT)
    finally:
        F_patch.undo()


BARS_DICT = json.loads(BARS.read_text())
TUNING_DICT = json.loads(TUNING.read_text())


def section(text, name):
    m = re.search(rf"^## {name}\b.*?(?=^## |\Z)", text, re.S | re.M)
    assert m, f"no section {name}"
    return m.group(0)


def tables(sec):
    """Every pipe table in the section as a list of dicts keyed by header text."""
    out, cur = [], []
    for line in sec.splitlines() + [""]:
        if line.startswith("|"):
            cur.append([c.strip() for c in line.strip().strip("|").split("|")])
        elif cur:
            hdr, body = cur[0], [r for r in cur[1:] if not set("".join(r)) <= set("-: ")]
            out.append([dict(zip(hdr, r)) for r in body])
            cur = []
    return out


def num(cell):
    return float(re.match(r"-?[\d.]+(?:e-?\d+)?", cell.replace("−", "-")).group(0))


def pooled(flyer, d):
    c = counts_of(flyer)[d]
    return c.sum(-1), np.full(10, 48)


# ---- the pin -----------------------------------------------------------------------------------------------------------------
def test_bn09_prove_for_post_1_is_pinned_at_0_1_2_the_version_8a_pinned_and_the_installed_one():
    assert pins.PROVE_VERSION == B.PROVE_VERSION == "0.1.2"
    assert pins.installed_prove_version() == md.version("poisson-prove") == "0.1.2"
    assert pins.require_prove_pin() == "0.1.2"
    assert '"poisson-prove==0.1.2"' in (ROOT / "pyproject.toml").read_text()


def test_bn09_a_prove_that_is_not_the_pinned_one_is_refused_before_anything_is_written(monkeypatch, rows):
    real = md.version
    monkeypatch.setattr(md, "version", lambda name: "0.1.3" if name == "poisson-prove" else real(name))
    with pytest.raises(pins.PinMismatch):
        pins.require_prove_pin()
    F.patch_s6(monkeypatch)
    with pytest.raises(pins.PinMismatch):
        Wr.render_results_p1(rows, bars=BARS_DICT, tuning=TUNING_DICT)


def test_bn09_results_prints_the_pinned_prove_version(text):
    assert re.search(r"^Prove: poisson-prove 0\.1\.2$", text, re.M)


# ---- layout ---------------------------------------------------------------------------------------------------------------------
def test_bn09_results_holds_n1_to_n10_in_order_and_no_other_number(text):
    heads = re.findall(r"^## (.+)$", text, re.M)
    assert heads == ["Run", "Speeds"] + list(B.N_NUMBERS)
    assert not re.search(r"\bN11\b", text)


def test_bn09_no_word_of_section_8_no_significance_and_no_p_value_anywhere(text):
    assert len(text) > 10_000, "the writer wrote no results"
    low = text.lower()
    for w in B.PROVE_WORDS + ("significant", "p-value", "p value", "mcnemar"):
        assert w not in low, w


def test_bn09_the_run_section_names_the_salts_and_the_grid(text):
    run = section(text, "Run")
    assert B.ZONE_B_SALT in run and B.ANCHOR_SALT in run
    assert re.search(r"top[^\n]*\b14\b", run) and "16 m/s" not in run.replace("16 m/s added", "")


# ---- every speed with its flat force ratio -------------------------------------------------------------------------------------
def test_bn09_each_speed_is_printed_with_its_flat_force_ratio(text):
    (tab,) = tables(section(text, "Speeds"))
    assert [num(r["speed"]) for r in tab] == list(B.SPEEDS)
    for r in tab:
        assert r["force ratio"] == f"{O.flat_force_ratio(num(r['speed'])):.2f}"


def test_bn09_the_ratios_match_17_3s_table_where_it_lists_a_speed(text):
    (tab,) = tables(section(text, "Speeds"))
    ratio = {num(r["speed"]): num(r["force ratio"]) for r in tab}
    assert (ratio[2.0], ratio[4.0], ratio[6.0], ratio[8.0], ratio[10.0], ratio[12.0], ratio[14.0]) == (0.03, 0.09, 0.17, 0.28, 0.41, 0.57, 0.75)


# ---- N1 to N3 ---------------------------------------------------------------------------------------------------------------------
def test_bn09_n1_has_a_line_per_flyer_and_cell(text):
    n1 = [r for t in tables(section(text, "N1")) for r in t]
    assert len(n1) == 20 * 320 == 6400
    r = next(x for x in n1 if x["flyer"] == "pid" and (num(x["az"]), num(x["el"]), num(x["speed"]), num(x["phase"])) == (90.0, 0.0, 8.0, 0.125))
    c = CTL["pid"][2, 6, 1]
    assert (int(r["crashes"]), int(r["n"])) == (c, 24)
    assert (int(r["ground"]), int(r["tilt"]), int(r["radius"])) == R.class_counts(c, 2)
    assert r["force ratio"] == f"{O.flat_force_ratio(8.0):.2f}" and abs(num(r["rate"]) - c / 24) < 5e-4


def test_bn09_n2_prints_the_edges_with_censored_ones_marked(text):
    n2 = [r for t in tables(section(text, "N2")) for r in t]
    assert len(n2) == 20 * 16
    oracle_row = [r for r in n2 if r["flyer"] == "oracle"]
    assert len(oracle_row) == 16
    for r in oracle_row:
        for col in ("v50", "v10", "v50 t0", "v50 t0.125", "v10 t0", "v10 t0.125"):
            assert r[col] == "14.00 [censored top]", (r["az"], col)
    first = next(r for r in n2 if r["flyer"] == "pi0-seed-1" and num(r["az"]) == 0.0 and num(r["el"]) == 0.0)
    assert first["v50"] == "2.00 [censored bottom]" and first["v10"] == "2.00 [censored bottom]"
    second = next(r for r in n2 if r["flyer"] == "pi0-seed-2" and num(r["az"]) == 0.0 and num(r["el"]) == 0.0)
    c, n = pooled("pi0-seed-2", 0)
    want, mark = O.edge_exact(c, n, B.SPEEDS, 0.5)
    assert mark is None and second["v50"] == f"{want:.2f}" and "censored" not in second["v50"]
    marks = {None: "", "top": " [censored top]", "bottom": " [censored bottom]"}
    by = {(r["flyer"], num(r["az"]), num(r["el"])): r for r in n2}
    columns = {"v50": (0.5, None), "v10": (0.1, None), "v50 t0": (0.5, 0), "v50 t0.125": (0.5, 1), "v10 t0": (0.1, 0), "v10 t0.125": (0.1, 1)}
    for f in B.FLYERS:
        for d, (az, el) in enumerate(B.DIRECTIONS):
            r = by[(f, az, el)]
            for col, (level, timing) in columns.items():
                crashes, n = (pooled(f, d) if timing is None else (counts_of(f)[d, :, timing], np.full(10, 24)))
                v, mark = O.edge_exact(crashes, n, B.SPEEDS, level)
                assert r[col] == f"{v:.2f}" + marks[mark], (f, az, el, col)


def test_bn09_n3_is_the_integrated_rate_per_flyer_and_direction(text):
    n3 = [r for t in tables(section(text, "N3")) for r in t]
    assert len(n3) == 320
    for f in ("pi0-seed-5", "pid", "oracle"):
        for d in (0, 5, 15):
            r = next(x for x in n3 if x["flyer"] == f and (num(x["az"]), num(x["el"])) == B.DIRECTIONS[d])
            assert r["integrated"] == f"{O.integrated(*pooled(f, d), B.SPEEDS):.4f}"


# ---- N4, N5 ----------------------------------------------------------------------------------------------------------------------
def test_bn09_n4_has_the_means_all_16_values_and_how_many_are_censored(text):
    t1, t2 = tables(section(text, "N4"))
    assert len(t1) == 16 and len(t2) == 48
    v50 = np.array([[O.edge_exact(*pooled(f"pi0-seed-{i}", d), B.SPEEDS)[0] for i in range(1, 17)] for d in range(16)])
    assert num(t1[0]["mean v50"]) == pytest.approx(v50[0].mean(), abs=6e-3)
    assert (int(t1[0]["v50 censored bottom"]), int(t1[0]["v50 censored top"])) == (1, 0)              # lineage 1 only, direction 0
    assert int(t1[1]["v50 censored bottom"]) == 0
    row = next(r for r in t2 if r["measure"] == "v50" and num(r["az"]) == 45.0 and num(r["el"]) == 0.0)
    assert [num(row[f"seed {i}"]) for i in range(1, 17)] == pytest.approx(list(v50[1]), abs=6e-3)


def test_bn09_n5_has_each_controller_with_the_shared_estimate_note_and_pairs_with_a_censored_side_marked(text):
    sec = section(text, "N5")
    assert sec.count(N5_NOTE) >= 1
    tabs = [r for t in tables(sec) for r in t]
    assert {r["controller"] for r in tabs} == set(B.CONTROLLERS) and len(tabs) == 4 * 16 * 2
    row = next(r for r in tabs if r["controller"] == "oracle" and r["measure"] == "v50" and num(r["az"]) == 0.0 and num(r["el"]) == 0.0)
    assert all(row[f"seed {i}"].endswith("*") for i in range(1, 17))                                 # the oracle is censored at the top
    assert int(row["censored pairs"]) == 16
    row2 = next(r for r in tabs if r["controller"] == "pid" and r["measure"] == "v50" and num(r["az"]) == 90.0 and num(r["el"]) == 0.0)
    pv = [O.edge_exact(*pooled(f"pi0-seed-{i}", 2), B.SPEEDS)[0] - O.edge_exact(*pooled("pid", 2), B.SPEEDS)[0] for i in range(1, 17)]
    assert [num(row2[f"seed {i}"]) for i in range(1, 17)] == pytest.approx(pv, abs=6e-3)
    assert num(row2["mean"]) == pytest.approx(np.mean(pv), abs=6e-3) and num(row2["lowest"]) == pytest.approx(min(pv), abs=6e-3)
    assert num(row2["highest"]) == pytest.approx(max(pv), abs=6e-3)
    assert not any(row2[f"seed {i}"].endswith("*") for i in range(1, 17))


# ---- N6: counts only --------------------------------------------------------------------------------------------------------------
def test_bn09_n6_is_counts_only_one_line_per_pi0_controller_and_cell(text):
    sec = section(text, "N6")
    n6 = [r for t in tables(sec) for r in t]
    assert len(n6) == 16 * 4 * 320 == 20480
    assert set(n6[0]) == {"pi0", "controller", "az", "el", "speed", "phase", "both", "only pi0", "only controller", "neither"}
    by = {(x["pi0"], x["controller"], num(x["az"]), num(x["el"]), num(x["speed"]), num(x["phase"])): x for x in n6}
    for pi0, ctl in (("pi0-seed-3", "geometric"), ("pi0-seed-1", "wind_only"), ("pi0-seed-9", "pid"), ("pi0-seed-16", "oracle")):
        for d, (az, el) in enumerate(B.DIRECTIONS):
            for s, v in enumerate(B.SPEEDS):
                for t, ph in enumerate(B.PHASES):
                    a, b = counts_of(pi0)[d, s, t], CTL[ctl][d, s, t]
                    both = min(a, b)
                    r = by[(pi0, ctl, az, el, v, ph)]
                    assert (int(r["both"]), int(r["only pi0"]), int(r["only controller"]), int(r["neither"])) == (both, a - both, b - both, 24 - a - b + both), (pi0, ctl, az, el, v, ph)
    assert "flip rate" not in sec


def test_bn09_n6_prints_3bs_flip_rate_beside_it_when_there_is_one(rows):
    import pytest as _p
    mp = _p.MonkeyPatch()
    F.patch_s6(mp)
    try:
        t = Wr.render_results_p1(rows, bars=BARS_DICT, tuning=TUNING_DICT, flip_rate=0.004)
    finally:
        mp.undo()
    assert re.search(r"^flip rate: 0\.004", section(t, "N6"), re.M)


# ---- N7, N8, N9, N10 --------------------------------------------------------------------------------------------------------------
def test_bn09_n7_prints_the_anchor_beside_the_grid_for_every_flyer(text):
    cells, means, lineages = tables(section(text, "N7"))
    assert len(cells) == 20 * 16 and len(means) == 20 and len(lineages) == 16
    f = "pi0-seed-2"
    j = 3
    az, ph = R.anchor_cells()[j]
    r = next(x for x in cells if x["flyer"] == f and (num(x["az"]), num(x["phase"])) == (az, ph))
    speed = B.ANCHOR[(az, ph)]
    d, s, t = B.HORIZON_AZ.index(az), B.SPEEDS.index(speed), B.PHASES.index(ph)
    assert (int(r["crashes"]), int(r["n"])) == (anchor_of(f)[j], 16) and num(r["speed"]) == speed
    assert (int(r["grid crashes"]), int(r["grid n"])) == (counts_of(f)[d, s, t], 24)
    m = next(x for x in means if x["flyer"] == f)
    assert num(m["mean rate"]) == pytest.approx(np.mean(anchor_of(f)) / 16, abs=5e-4)
    assert [num(x["lineage mean"]) for x in lineages] == pytest.approx([np.mean(anchor_of(f"pi0-seed-{i}")) / 16 for i in range(1, 17)], abs=5e-4)
    assert re.search(r"^pi0 mean: ", section(text, "N7"), re.M)


def test_bn09_n8_has_every_flyer_under_each_of_the_six_limits(text, rows):
    n8 = [r for t in tables(section(text, "N8")) for r in t]
    assert len(n8) == 20 * 6 * 16 == 1920
    assert {r["limit"] for r in n8} == set(B.S6_NAMES)
    mp = pytest.MonkeyPatch()
    F.patch_s6(mp)
    try:
        re_ = F.fake_rescore_by_seed(rows[(rows["flyer"] == "pid") & (rows["zone"] == "B")], "tilt-90")
    finally:
        mp.undo()
    c = np.zeros(R.SHAPE, int)
    for cell, outcome in zip(re_["cell"], re_["outcome"]):
        if outcome == "crash":
            c[B.DIRECTIONS.index((cell[0], cell[1])), B.SPEEDS.index(cell[2]), B.PHASES.index(cell[3])] += 1
    r = next(x for x in n8 if x["flyer"] == "pid" and x["limit"] == "tilt-90" and (num(x["az"]), num(x["el"])) == B.DIRECTIONS[4])
    v, mark = O.edge_exact(c[4].sum(-1), np.full(10, 48), B.SPEEDS)
    assert r["v50"] == f"{v:.2f}" + ({None: "", "top": " [censored top]", "bottom": " [censored bottom]"}[mark])
    assert r["integrated"] == f"{O.integrated(c[4].sum(-1), np.full(10, 48), B.SPEEDS):.4f}"


def test_bn09_n9_prints_the_stored_numbers_and_never_recomputes_them(rows, monkeypatch):
    """Re-frozen by 4D-A (2026-11-08, 4DR-5): N9 gains the yaw gate's table, k_yaw, the crash penalty and the failed seeds by gate."""
    from drone_study.training import behaviour as behaviour_module
    from drone_study.training import yaw_weight as yaw_module

    def refuse(*a, **k):
        raise AssertionError("the writer recomputed the bar or the yaw gate")
    for name in ("read_bar", "rows_from_store", "select_lineages"):
        monkeypatch.setattr(bar_module, name, refuse)
    for name in ("yaw_acceptance", "yaw_rows_from_store"):
        monkeypatch.setattr(behaviour_module, name, refuse)
    monkeypatch.setattr(yaw_module, "select_lineages_with_yaw", refuse)
    F.patch_s6(monkeypatch)
    bars = json.loads(BARS.read_text())
    bars["readings"]["pi0-seed-7"].update(median_rmse_m=0.123456789, crashes=3)
    bars["yaw_readings"]["pi0-seed-7"].update(yaw_rate_rms_median_deg_s=9.87654, past_45=2, max_heading_median_deg=11.25)
    t = Wr.render_results_p1(rows, bars=bars, tuning=TUNING_DICT)
    sec = section(t, "N9")
    bar_tab, yaw_tab = tables(sec)
    assert len(bar_tab) == 16 and len(yaw_tab) == 16
    r = next(x for x in bar_tab if int(x["seed"]) == 7)
    assert r["median rmse m"].startswith("0.1235") and int(r["crashes"]) == 3
    y = next(x for x in yaw_tab if int(x["seed"]) == 7)
    assert set(y) == {"seed", "yaw rate rms deg/s", "heading rate rms deg/s", "drift deg/s", "past 45", "past 90", "max heading median deg", "yaw passed"}
    assert y["yaw rate rms deg/s"].startswith("9.88") and int(y["past 45"]) == 2 and y["max heading median deg"].startswith("11.25") and y["yaw passed"] == "yes"
    assert [int(x["seed"]) for x in yaw_tab] == [int(x["seed"]) for x in bar_tab]
    assert re.search(r"^B0: 5000000$", sec, re.M) and re.search(r"^seeds tried: 16$", sec, re.M) and re.search(r"^seeds failed: 0$", sec, re.M)
    assert re.search(r"^k_yaw: 0\.008$", sec, re.M) and re.search(r"^crash penalty: 20$", sec, re.M)
    assert re.search(r"^failed the bar: none$", sec, re.M) and re.search(r"^failed the yaw gate: none$", sec, re.M)
    assert re.search(r"^PID: 0 crashes, median RMSE 0\.0623 m$", sec, re.M)
    assert re.search(r"^geometric tuning figure: 0\.0476 m", sec, re.M)


def test_bn09_n9_names_the_failed_seeds_by_gate_and_lists_the_sixteen_that_passed(rows, monkeypatch):
    """4DR-5: with seed 4 failed on the bar and seeds 9 and 11 on the yaw gate, N9 lists seeds 1 to 3, 5 to 8, 10 and 12 to 19."""
    F.patch_s6(monkeypatch)
    bars = json.loads(BARS.read_text())
    lineages = [1, 2, 3, 5, 6, 7, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19]
    for new in (17, 18, 19):
        bars["readings"][f"pi0-seed-{new}"] = dict(bars["readings"]["pi0-seed-16"])
        bars["yaw_readings"][f"pi0-seed-{new}"] = dict(bars["yaw_readings"]["pi0-seed-16"])
    bars["selection"] = {"lineages": lineages, "failures": [4, 9, 11], "tried": 19, "failed_on": {"4": "bar", "9": "yaw", "11": "yaw"}}
    bars["k_yaw"], bars["crash_penalty"], bars["b0"] = 0.032, 22.0, 10_000_000
    t = Wr.render_results_p1(rows, bars=bars, tuning=TUNING_DICT)
    sec = section(t, "N9")
    bar_tab, yaw_tab = tables(sec)
    assert [int(x["seed"]) for x in bar_tab] == lineages and [int(x["seed"]) for x in yaw_tab] == lineages
    assert re.search(r"^B0: 10000000$", sec, re.M) and re.search(r"^k_yaw: 0\.032$", sec, re.M) and re.search(r"^crash penalty: 22$", sec, re.M)
    assert re.search(r"^seeds tried: 19$", sec, re.M) and re.search(r"^seeds failed: 3$", sec, re.M)
    assert re.search(r"^failed the bar: 4$", sec, re.M) and re.search(r"^failed the yaw gate: 9, 11$", sec, re.M)
    low = t.lower()
    for w in B.PROVE_WORDS + ("significant", "p-value"):
        assert w not in low, w


def test_bn09_n10_prints_the_12_ms_rates_and_that_16_ms_was_not_added(text):
    sec = section(text, "N10")
    (tab,) = tables(sec)
    assert len(tab) == 16 and all(num(r["rate at 12"]) == 1.0 for r in tab)
    assert re.search(r"^16 m/s added: no$", sec, re.M)


# ---- the extension fires ------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def fired():
    mp = pytest.MonkeyPatch()
    F.patch_s6(mp)
    try:
        r = build(fire=True)
        return r, Wr.render_results_p1(r, bars=BARS_DICT, tuning=TUNING_DICT)
    finally:
        mp.undo()


def test_bn09_when_the_extension_fires_the_16_ms_rows_are_in_every_number(fired):
    _, t = fired
    sec10 = section(t, "N10")
    assert re.search(r"^16 m/s added: yes$", sec10, re.M)
    (spd,) = tables(section(t, "Speeds"))
    assert [num(r["speed"]) for r in spd] == list(EXT) and spd[-1]["force ratio"] == "0.95"
    n1 = [r for tt in tables(section(t, "N1")) for r in tt]
    assert len(n1) == 20 * 352 and sum(1 for r in n1 if num(r["speed"]) == 16.0) == 20 * 32
    n6 = [r for tt in tables(section(t, "N6")) for r in tt]
    assert len(n6) == 16 * 4 * 352
    n2 = [r for tt in tables(section(t, "N2")) for r in tt]
    pid = next(r for r in n2 if r["flyer"] == "pid" and num(r["az"]) == 0.0 and num(r["el"]) == 0.0)
    c = np.concatenate([CTL["pid"], np.zeros((16, 1, 2), int) + 12], axis=1)
    v, mark = O.edge_exact(c[0].sum(-1), np.full(11, 48), EXT)
    assert pid["v50"] == f"{v:.2f}" + ({None: "", "top": " [censored top]", "bottom": " [censored bottom]"}[mark])


def test_bn09_a_firing_reading_without_the_16_ms_rows_stops_the_write(monkeypatch):
    from drone_study.boundary.results import MissingCell
    F.patch_s6(monkeypatch)
    r = build(fire=True)
    r = r[~((r["zone"] == "B") & r["cell"].map(lambda c: len(c) == 4 and c[2] == 16.0))].reset_index(drop=True)
    with pytest.raises(MissingCell):
        Wr.render_results_p1(r, bars=BARS_DICT, tuning=TUNING_DICT)
