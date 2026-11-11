"""Public slice (session P1c-repo, 2026-11-10): the cases on the fine-tune budgets are removed; the rungs, the pilots' B0 and the curve marks stay. No kept case is changed.

Training rule 10 (spec section 2, 'Training budget, by a rule'; session 7's row in section 12: "Every run's steps equal, by manifest").

"Four pilot seeds, not used as lineages, train to 40M control steps with checkpoints at 5M, 10M, 20M and 40M. The budget
B0 is the smallest rung at which all four pass the bar."

Readings (rulings 4A-2, 4A-14, 4A-15): a run's steps come in whole rollouts of 31,250, a size that divides every rung,
so a manifest shows the budget exactly; if no rung has all four pilots at the bar, nothing is picked and the study waits on a ruling.
"""
import pytest

import training_numbers as N
from drone_study.training import budget as B
from drone_study.training.budget import NoRungPasses, b0_from_pilots, budget, curve_marks

ALL = [True] * 4


def passed(**at):
    """{rung: four booleans}; by default no pilot passes anywhere."""
    out = {r: [False] * 4 for r in N.RUNGS}
    for key, flags in at.items():
        out[int(key[1:]) * 1_000_000] = flags
    return out


def test_tr10_the_rungs_and_the_pilots_are_section_2s():
    assert tuple(B.RUNGS) == N.RUNGS == (5_000_000, 10_000_000, 20_000_000, 40_000_000)
    assert B.N_PILOTS == N.N_PILOTS == 4
    assert B.ROLLOUT_STEPS == N.ROLLOUT_STEPS == 31_250


@pytest.mark.parametrize("flags,b0", [
    (passed(m5=ALL, m10=ALL, m20=ALL, m40=ALL), 5_000_000),
    (passed(m10=ALL, m20=ALL, m40=ALL), 10_000_000),
    (passed(m5=[True, True, True, False], m10=[True, False, True, True], m20=ALL, m40=ALL), 20_000_000),   # all four, not three
    (passed(m40=ALL), 40_000_000),
    (passed(m10=ALL, m20=[True, True, False, True], m40=ALL), 10_000_000),                                # the smallest rung, whatever follows
])
def test_tr10_b0_is_the_smallest_rung_at_which_all_four_pilots_pass(flags, b0):
    assert b0_from_pilots(flags) == b0


def test_tr10_no_rung_with_all_four_at_the_bar_picks_nothing():
    with pytest.raises(NoRungPasses):
        b0_from_pilots(passed(m5=[True, True, True, False], m40=[False, True, True, True]))
    assert b0_from_pilots(passed(m5=[True, True, True, False], m40=ALL)) == 40_000_000       # the control: one rung with four


@pytest.mark.parametrize("b0", N.RUNGS)
def test_tr10_every_budget_and_every_learning_curve_mark_is_a_whole_number_of_rollouts(b0):
    b_ft = b0 // 4
    marks = curve_marks(b_ft)
    assert tuple(marks) == (b_ft // 4, b_ft // 2, 3 * b_ft // 4, b_ft)       # 25, 50, 75 and 100% of B_ft (session 7)
    for steps in (b0, b_ft) + tuple(marks):
        assert steps % N.ROLLOUT_STEPS == 0
    assert all(r % N.ROLLOUT_STEPS == 0 for r in N.RUNGS)                    # the pilots' checkpoints too


def manifests(b0, **steps_by_arm):
    out = []
    for lineage in (1, 2):
        for arm in N.FINE_TUNE_ARMS:
            steps = steps_by_arm.get(arm.replace("-", "_"), b0 // 4)
            out.append({"arm": arm, "lineage": lineage, "budget": b0 // 4, "steps": steps})
    return out

