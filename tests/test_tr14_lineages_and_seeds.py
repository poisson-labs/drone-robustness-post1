"""Public slice (session P1c-repo, 2026-11-10): the two cases on the fine-tune roles' seeds are removed, and the derived-seed case runs on the nominal and pilot roles. No kept case is changed.

Training rule 14 (spec section 2, 'A lineage that fails the bar'; section 5, 'Lineage'): "A lineage that fails the bar
isn't retrained. Seeds are tried in order until 16 pass, every failure is reported, and more than 8 failures stop the
study for a ruling. Selection reads calm air only, so it can't favour any hunt outcome." And: "Four pilot seeds, not used
as lineages".

The spec names no seed values (ruling 4A-13): the nominal seeds tried are 1, 2, 3, ...; the pilots are 9001 to 9004; a
run's seed is the first 8 hex digits of sha256("drone-robustness-2026-11-04:train:<role>:<nominal seed>"), the salts'
formula with a word for the run.
"""
import pytest

import training_numbers as N
from drone_study.training import seeds as SD
from drone_study.training.bar import BarReading, Selection, StudyStop, select_lineages
from drone_study.training.seeds import nominal_seed, run_seed

PASS = BarReading(crashes=1, episodes=1200, median_rmse=0.06, pid_median_rmse=0.06, ratio=1.0, passes_crashes=True,
                  passes_rmse=True, passes_ratio=True, passed=True)
FAIL = BarReading(crashes=9, episodes=1200, median_rmse=0.06, pid_median_rmse=0.06, ratio=1.0, passes_crashes=False,
                  passes_rmse=True, passes_ratio=True, passed=False)


SLOW = BarReading(crashes=0, episodes=1200, median_rmse=0.14, pid_median_rmse=0.06, ratio=2.33, passes_crashes=True,
                  passes_rmse=False, passes_ratio=False, passed=False)
LOOSE = BarReading(crashes=0, episodes=1200, median_rmse=0.09, pid_median_rmse=0.06, ratio=1.5, passes_crashes=True,
                   passes_rmse=True, passes_ratio=False, passed=False)
KIND = {"P": PASS, "F": FAIL, "S": SLOW, "L": LOOSE}


def readings(pattern):
    """pattern: one letter per nominal seed in order (the seeds being 1, 2, 3, ...): P passes the bar, F fails on crashes,
    S on the 10 cm line, L only on the PID's line."""
    return [(i + 1, KIND[ch]) for i, ch in enumerate(pattern)]


def test_tr14_sixteen_seeds_that_pass_in_order_are_the_lineages():
    sel = select_lineages(readings("P" * 16))
    assert isinstance(sel, Selection)
    assert sel.lineages == tuple(range(1, 17)) and sel.failures == () and sel.tried == 16


def test_tr14_a_failed_seed_is_skipped_reported_and_the_next_is_tried():
    sel = select_lineages(readings("PPFPPPFPPPPPPPPPPPP"))
    assert sel.lineages == (1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)
    assert sel.failures == (3, 7)                                             # every failure is reported
    assert sel.tried == 18 and len(sel.lineages) == N.LINEAGES


def test_tr14_a_seed_is_a_lineage_only_if_it_clears_all_three_lines():
    sel = select_lineages(readings("PSLFP" + "P" * 14))
    assert sel.failures == (2, 3, 4)                                          # too slow, over the PID's line, too many crashes
    assert sel.lineages == (1, 5) + tuple(range(6, 20)) and sel.tried == 19


def test_tr14_no_seed_is_read_after_the_sixteenth_pass():
    sel = select_lineages(readings("P" * 16 + "FFFFFFFFFFFF" + "PP"))
    assert sel.lineages == tuple(range(1, 17)) and sel.failures == () and sel.tried == 16


def test_tr14_eight_failures_are_allowed_and_a_ninth_stops_the_study():
    eight = "F" * 8 + "P" * 16
    sel = select_lineages(readings(eight))
    assert sel.failures == tuple(range(1, 9)) and sel.lineages == tuple(range(9, 25)) and sel.tried == 24
    with pytest.raises(StudyStop):
        select_lineages(readings("F" * 9 + "P" * 16))
    with pytest.raises(StudyStop):
        select_lineages(readings("PFPFPFPFPFPFPFPFPF" + "P" * 16))            # the ninth failure, wherever it falls
    assert len(select_lineages(readings("PFPFPFPFPFPFPFPF" + "P" * 16)).lineages) == 16     # the control: eight


def test_tr14_fewer_than_sixteen_passes_so_far_is_not_yet_a_selection():
    sel = select_lineages(readings("PPFPP"))
    assert sel.lineages == (1, 2, 4, 5) and sel.failures == (3,) and sel.tried == 5
    assert len(sel.lineages) < N.LINEAGES


def test_tr14_the_nominal_seeds_are_tried_in_order_from_one_and_the_pilots_are_none_of_them():
    assert [nominal_seed(i) for i in range(24)] == list(range(N.FIRST_NOMINAL_SEED, N.FIRST_NOMINAL_SEED + 24))
    assert tuple(SD.PILOT_SEEDS) == N.PILOT_SEEDS and len(set(SD.PILOT_SEEDS)) == 4
    assert not set(SD.PILOT_SEEDS) & {nominal_seed(i) for i in range(N.LINEAGES + N.MAX_FAILURES)}   # "not used as lineages"


ROLES = ("nominal", "pilot")


@pytest.mark.parametrize("role", ROLES)
def test_tr14_each_run_has_its_own_derived_seed(role):
    for nominal in (1, 2, 16, 24, 9001):
        got = run_seed(role, nominal)
        assert got == N.run_seed(role, nominal)
        assert isinstance(got, int) and 0 <= got < 2 ** 32

