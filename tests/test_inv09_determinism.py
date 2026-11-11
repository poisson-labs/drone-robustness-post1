"""Invariant 9 (spec section 6): the determinism branch test exists and prints its table.

Two parts. `classify` is section 6's table as a pure function, tested at every branch and at each boundary. `measure`
flies a small set of episodes under several worker and thread configurations in separate processes, and the test
checks the report is complete and prints. WHICH branch the stack lands in is a measurement for 3B's gate (section 12),
not something this test decides.
"""
import pytest

import spec_numbers as S
from drone_study.condition import Condition
from drone_study.determinism import BRANCHES, DeterminismRow, classify, measure


def row(label, workers, threads, dev, flips, n=1000):
    return DeterminismRow(label, workers, threads, n, dev, flips)


CASES = [
    ("identical", [row("2 workers", 2, 1, 0.0, 0), row("4 workers, 4 threads", 4, 4, 5e-10, 0)]),
    ("identical", [row("1e-9 exactly", 2, 2, 1e-9, 0)]),
    ("pin-one-thread", [row("2 workers, 1 thread", 2, 1, 0.0, 0), row("2 workers, 4 threads", 2, 4, 3e-6, 40)]),
    ("pin-one-thread", [row("2 workers, 1 thread", 2, 1, 1e-9, 0), row("1 worker, 8 threads", 1, 8, 2e-9, 0)]),
    ("report-noise-floor", [row("2 workers, 1 thread", 2, 1, 2e-7, 3), row("2 workers, 4 threads", 2, 4, 3e-6, 2)]),
    ("report-noise-floor", [row("replay", 1, 1, 1e-6, 5)]),                          # 5 of 1000 is 0.5%: "at most"
    ("pairing-buys-less", [row("replay", 1, 1, 1e-6, 6)]),                           # 6 of 1000 is over 0.5%
    ("pairing-buys-less", [row("a", 2, 1, 1e-5, 4), row("b", 2, 4, 1e-4, 12)]),
]


@pytest.mark.parametrize("expected,rows", CASES)
def test_inv9_classify_applies_section_6s_table(expected, rows):
    assert classify(rows) == expected


def test_inv9_just_over_a_nanometre_is_not_identical():
    assert classify([row("a", 2, 1, 1.0000001e-9, 0)]) == "report-noise-floor"     # not identical, no flips


def test_inv9_the_measurement_runs_in_separate_processes_and_prints_its_table(capsys):
    conds = [Condition(case="circle-fast", az=0.0, el=0.0, speed=6.0, phase=0.0), Condition(case="fig8-faster", turb=0.3)]
    configs = [(1, 1), (2, 1), (2, 2)]
    report = measure(conds, seeds=[1, 2], configs=configs)
    assert len(report.rows) == len(configs)
    assert [(r.workers, r.threads_per_worker) for r in report.rows] == configs
    for r in report.rows:
        assert r.n_episodes == len(conds) * 2
        assert r.max_abs_dev_m >= 0.0 and r.max_abs_dev_m == r.max_abs_dev_m      # a number, not NaN
        assert 0 <= r.n_flips <= r.n_episodes
    assert report.branch in BRANCHES
    assert report.branch == classify(report.rows)
    text = report.table_text()
    print(text)
    printed = capsys.readouterr().out
    assert printed.strip() == text.strip() and text.strip()
    for r in report.rows:
        assert r.label in text
    assert report.branch in text
    for header in ("max", "flip"):
        assert header in text.lower()
