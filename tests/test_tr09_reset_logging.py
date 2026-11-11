"""Public slice (session P1c-repo, 2026-11-10): the three cases that name a retrain arm are removed. No kept case is changed.

Training rule 9 (spec section 5, "Equal steps aren't equal resets"; session 7's row in section 12): "An arm that
crashes early in training resets more often, so more of its steps sit just after a reset. Each fine-tune's manifest logs
its episode count, mean episode length and the share of its steps within 0.5 s of a reset, and session 7 reports them
per arm beside the learning curves."

Readings (ruling 4A-12): a step is within 0.5 s of a reset when it starts less than 0.5 s after it, the first 50 control
steps of an episode; the episode still running when the budget ends counts in the steps and in that share, and not in
the episode count or the mean length.
"""
import numpy as np
import pytest

import spec_numbers as S
import training_numbers as N
from drone_study.condition import Condition
from drone_study.training import manifest as M
from drone_study.training.env import TrainingEnv
from drone_study.training.manifest import ResetLog, merge, reset_report, run_manifest
from drone_study.training.sampler import make_sampler

DROP = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def log_of(lengths, unfinished=None):
    log = ResetLog()
    for n in lengths:
        log.add(n)
    if unfinished is not None:
        log.add(unfinished, finished=False)
    return log


def test_tr09_the_window_is_half_a_second():
    assert M.RESET_WINDOW_S == N.RESET_WINDOW_S == 0.5
    assert N.RESET_WINDOW_STEPS == round(N.RESET_WINDOW_S * S.CTRL_HZ) == 50


def test_tr09_episode_count_mean_length_and_the_share_of_steps_near_a_reset():
    log = log_of([1000, 20, 300])
    assert (log.episodes, log.steps) == (3, 1320)
    assert log.mean_episode_steps == pytest.approx(440.0) and log.mean_episode_seconds == pytest.approx(4.4)
    assert log.steps_near_reset == 50 + 20 + 50                              # a 20-step episode has only 20 to give
    assert log.share_near_reset == pytest.approx(120 / 1320, rel=1e-12)


@pytest.mark.parametrize("lengths,near", [([1000] * 5, 250), ([50] * 4, 200), ([49, 51], 99), ([1], 1), ([1000, 1000, 10, 10, 10], 130)])
def test_tr09_the_first_fifty_control_steps_of_each_episode_are_the_ones_near_its_reset(lengths, near):
    log = log_of(lengths)
    assert log.steps == sum(lengths) and log.steps_near_reset == near
    assert log.share_near_reset == pytest.approx(near / sum(lengths), rel=1e-12)
    assert log.episodes == len(lengths) and log.mean_episode_steps == pytest.approx(sum(lengths) / len(lengths))


def test_tr09_an_arm_that_crashes_early_shows_more_of_its_steps_near_a_reset():
    steady, crashing = log_of([1000] * 10), log_of([100] * 100)
    assert steady.steps == crashing.steps == 10_000                          # equal steps
    assert steady.share_near_reset == pytest.approx(0.05) and crashing.share_near_reset == pytest.approx(0.5)   # are not equal resets
    assert crashing.episodes == 10 * steady.episodes


def test_tr09_the_episode_still_running_at_the_budgets_end_counts_its_steps_and_not_an_episode():
    log = log_of([1000, 400], unfinished=30)
    assert log.steps == 1430 and log.episodes == 2
    assert log.mean_episode_steps == pytest.approx(700.0)
    assert log.steps_near_reset == 50 + 50 + 30 and log.share_near_reset == pytest.approx(130 / 1430, rel=1e-12)
    longer = log_of([1000, 400], unfinished=600)
    assert longer.steps_near_reset == 150 and longer.episodes == 2


def test_tr09_an_empty_log_reads_as_zeros():
    log = ResetLog()
    assert (log.episodes, log.steps, log.steps_near_reset, log.mean_episode_steps, log.share_near_reset) == (0, 0, 0, 0.0, 0.0)
    log.add(10)
    assert (log.episodes, log.steps) == (1, 10)


def test_tr09_a_runs_environments_add_up():
    a, b = log_of([1000, 20]), log_of([300], unfinished=40)
    total = merge([a, b])
    assert (total.episodes, total.steps, total.steps_near_reset) == (3, 1360, 50 + 20 + 50 + 40)
    assert total.mean_episode_steps == pytest.approx(440.0) and total.share_near_reset == pytest.approx(160 / 1360, rel=1e-12)
    assert (a.episodes, a.steps) == (2, 1020)                                # the parts are left alone

