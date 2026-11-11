"""P1a-fix (ruling P1A-1): printed numbers built from integer counts are read on exact fractions and rounded half to even once, at print.

P1a's independent check found three N5 integrated-rate differences that land exactly on a half in the fourth decimal (9/32, 1/32 and
-3/32). The float trapezoid's difference sat one ulp off the tie and printed 0.2813, 0.0313 and -0.0937; the exact ratio prints 0.2812,
0.0312 and -0.0938. The counts below are the stored rows' own pooled counts (48 episodes at each of the eleven speeds, 2 to 14 m/s and the 16 m/s extension). This file only adds cases:
no frozen case is changed."""
from fractions import Fraction

import numpy as np
import pytest

import boundary_numbers as B
import boundary_rows as R
from drone_study.analysis import boundary as analysis_boundary
from drone_study.analysis import estimator as E
from drone_study.boundary import results as N
from drone_study.boundary import writer as W

SPEEDS = B.SPEEDS_EXTENDED
N48 = [48] * 11
SHAPE = (16, len(SPEEDS), 2)

# the three cells P1a found: (direction index, pi0 pooled crashes, controller pooled crashes, exact difference, printed before, printed now)
PI0_7 = [0, 0, 0, 0, 0, 0, 14, 41, 48, 48, 48]
GEOMETRIC = [0, 0, 0, 0, 0, 0, 0, 2, 6, 45, 48]
PI0_5 = [0, 0, 0, 0, 0, 1, 12, 48, 48, 48, 48]
PID_A = [0, 0, 0, 0, 0, 0, 0, 47, 48, 48, 48]
PI0_1 = [0, 0, 0, 0, 0, 0, 0, 21, 48, 48, 48]
PID_B = [0, 0, 0, 0, 0, 0, 6, 48, 48, 48, 48]
TIES = [(PI0_7, GEOMETRIC, Fraction(9, 32), "0.2812"), (PI0_5, PID_A, Fraction(1, 32), "0.0312"), (PI0_1, PID_B, Fraction(-3, 32), "-0.0938")]
# pi0-seed-13 at azimuth 270, elevation -35: its v10 edge is 287/40 = 7.175 m/s, a tie that no float holds
PI0_13 = [0, 0, 0, 0, 0, 2, 18, 48, 48, 48, 48]


def exact_rate(c):
    return analysis_boundary.integrated_rate_exact(c, N48, SPEEDS)


# ---- the printer -------------------------------------------------------------------------------------------------------------
def test_p1a_fix_format_exact_rounds_half_to_even_on_the_exact_value():
    assert [N.format_exact(Fraction(x), 4) for x in (Fraction(9, 32), Fraction(1, 32), Fraction(-3, 32), Fraction(7, 32))] == \
        ["0.2812", "0.0312", "-0.0938", "0.2188"]
    assert [N.format_exact(Fraction(x), 2) for x in (Fraction(287, 40), Fraction(281, 40), Fraction(1, 8), Fraction(3, 8))] == \
        ["7.18", "7.02", "0.12", "0.38"]


def test_p1a_fix_format_exact_is_plain_rounding_away_from_ties_and_keeps_zero_and_sign():
    assert N.format_exact(Fraction(1, 3), 4) == "0.3333" and N.format_exact(Fraction(2, 3), 4) == "0.6667"
    assert N.format_exact(Fraction(0), 4) == "0.0000" and N.format_exact(Fraction(5), 2) == "5.00"
    assert N.format_exact(Fraction(-1, 3), 2) == "-0.33"
    assert N.format_exact(Fraction(-1, 100000), 4) == "-0.0000"            # as `format` prints a small negative
    assert N.format_exact(Fraction(999995, 10 ** 6), 4) == "1.0000" and N.format_exact(Fraction(12), 0) == "12"


@pytest.mark.parametrize("num,den,places", [(1, 3, 4), (5, 8, 2), (123456, 1000, 2), (7, 16, 3), (1, 32, 4), (9, 7, 4), (22, 7, 2)])
def test_p1a_fix_format_exact_agrees_with_decimal_half_even(num, den, places):
    from decimal import ROUND_HALF_EVEN, Decimal, getcontext
    getcontext().prec = 60
    want = (Decimal(num) / Decimal(den)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_EVEN)
    assert N.format_exact(Fraction(num, den), places) == format(want, "f")


def test_p1a_fix_is_tie_names_exact_halves_only():
    assert N.is_tie(Fraction(9, 32), 4) and N.is_tie(Fraction(-3, 32), 4) and N.is_tie(Fraction(287, 40), 2)
    assert not N.is_tie(Fraction(1, 8), 4) and not N.is_tie(Fraction(1, 3), 4) and not N.is_tie(Fraction(7, 10), 2)
    assert N.is_tie(Fraction(7, 8), 2)                                         # 0.875 is a half in the second place


# ---- the integrated rate -----------------------------------------------------------------------------------------------------
def test_p1a_fix_integrated_rate_exact_is_the_trapezoid_on_fractions():
    got = exact_rate([0] * 10 + [48])                    # rate 0 up to 14 m/s and 1 at 16: a triangle of area 1 over 14 m/s
    assert isinstance(got, Fraction) and got == Fraction(1, 14)
    assert exact_rate([24] * 11) == Fraction(1, 2)


def test_p1a_fix_integrated_rate_is_the_exact_value_rounded_once():
    rng = np.random.default_rng(11)
    for _ in range(200):
        c = rng.integers(0, 49, size=11)
        assert analysis_boundary.integrated_rate(c, N48, SPEEDS) == float(exact_rate(c))


@pytest.mark.parametrize("c,want", [(PI0_7, None), (GEOMETRIC, None), (PI0_5, None), (PID_A, None), (PI0_1, None), (PID_B, None)])
def test_p1a_fix_integrated_rate_exact_agrees_with_the_old_float_trapezoid_to_within_an_ulp_or_two(c, want):
    old = float(((np.asarray(c) / 48.0)[1:] + (np.asarray(c) / 48.0)[:-1]).__truediv__(2.0).__mul__(np.diff(np.asarray(SPEEDS, float))).sum() / (SPEEDS[-1] - SPEEDS[0]))
    assert float(exact_rate(c)) == pytest.approx(old, abs=1e-15)


def test_p1a_fix_a_speed_with_no_scored_episode_is_left_out_and_fewer_than_two_is_an_error():
    c, n = [0, 6, 0, 12], [48, 48, 0, 48]
    assert analysis_boundary.integrated_rate_exact(c, n, [2, 3, 4, 5]) == analysis_boundary.integrated_rate_exact([0, 6, 12], [48, 48, 48], [2, 3, 5])
    with pytest.raises(ValueError):
        analysis_boundary.integrated_rate_exact([3, 0], [48, 0], [2, 3])
    with pytest.raises(ValueError):
        analysis_boundary.integrated_rate_exact([1, 2], [48, 48, 48], [2, 3])


# ---- the three ties P1a found ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("pi0,controller,diff,printed", TIES)
def test_p1a_fix_the_three_ties_are_exact_halves_and_print_to_the_even_digit(pi0, controller, diff, printed):
    d = exact_rate(pi0) - exact_rate(controller)
    assert d == diff and N.is_tie(d, 4)
    assert N.format_exact(d, 4) == printed


def _grids(pi0, controller):
    """Sixteen-direction grids in which direction 0 carries the pooled counts given (split over the two timings) and the rest is flat."""
    out = []
    for name, pooled in (("p", pi0), ("c", controller)):
        c = np.zeros(SHAPE, int)
        for s, k in enumerate(pooled):
            c[0, s, 0], c[0, s, 1] = k - k // 2, k // 2
        out.append(N.flyer_grid(R.frame(R.zone_b_rows(name, c, 24, speeds=SPEEDS)), name, speeds=SPEEDS))
    return out


@pytest.mark.parametrize("pi0,controller,diff,printed", TIES)
def test_p1a_fix_n5_carries_the_difference_as_a_fraction_and_the_float_is_that_rounded_once(pi0, controller, diff, printed):
    p, c = _grids(pi0, controller)
    got = N.n5_against_controller([p], c)["integrated"]
    assert got["differences_exact"][0][0] == diff
    assert float(got["differences"][0, 0]) == float(diff)
    assert got["mean_exact"][0] == diff and got["lowest_exact"][0] == diff and got["highest_exact"][0] == diff
    assert N.format_exact(got["differences_exact"][0][0], 4) == printed
    assert got["differences_exact"][1][0] == 0                                   # a direction with no crashes against one with none


def test_p1a_fix_n5_mean_lowest_and_highest_are_read_on_the_exact_differences():
    p1, c = _grids(PI0_7, GEOMETRIC)
    p2, _ = _grids(PI0_5, GEOMETRIC)
    got = N.n5_against_controller([p1, p2], c)["integrated"]
    d1, d2 = exact_rate(PI0_7) - exact_rate(GEOMETRIC), exact_rate(PI0_5) - exact_rate(GEOMETRIC)
    assert got["mean_exact"][0] == (d1 + d2) / 2 and got["lowest_exact"][0] == min(d1, d2) and got["highest_exact"][0] == max(d1, d2)


def test_p1a_fix_n3_exact_is_one_fraction_per_direction_and_its_float_is_n3():
    p, _ = _grids(PI0_7, GEOMETRIC)
    ex = N.n3_integrated_exact(p)
    assert len(ex) == 16 and ex[0] == exact_rate(PI0_7) and ex[1] == 0
    assert N.n3_integrated(p) == pytest.approx([float(x) for x in ex], abs=0, rel=0)


# ---- an edge on a tie that no float holds ------------------------------------------------------------------------------------
def test_p1a_fix_an_edge_carries_its_exact_value_and_equality_ignores_it():
    e = E.edge(PI0_13, N48, SPEEDS, 0.1)
    assert e.exact == Fraction(287, 40) and e.censored is None
    assert e.value == 7.175 and float(Fraction(287, 40)) == 7.175                # the float nearest the tie is below it:
    assert format(e.value, ".2f") == "7.17" and N.format_exact(e.exact, 2) == "7.18"
    assert E.Edge(7.175, None, e.n_per_speed) == e                              # `exact` is left out of comparison
    assert E.edge(PI0_13, N48, SPEEDS, 0.5).exact == Fraction(42, 5)


def test_p1a_fix_a_censored_edge_and_a_flat_start_carry_the_grid_speed_as_a_fraction():
    top = E.edge([0] * 11, N48, SPEEDS, 0.5)
    bottom = E.edge([48] * 11, N48, SPEEDS, 0.5)
    assert (top.censored, top.exact) == ("top", Fraction(16)) and (bottom.censored, bottom.exact) == ("bottom", Fraction(2))
    start = E.edge([24] * 11, N48, SPEEDS, 0.5)
    assert start.censored is None and start.exact == Fraction(2)


def test_p1a_fix_an_edge_built_without_an_exact_prints_as_the_exact_value_of_its_float():
    assert N.edge_exact(E.Edge(7.175, None)) == Fraction(7.175)
    assert N.format_exact(N.edge_exact(E.Edge(2.5, None)), 2) == "2.50"


# ---- the printed text --------------------------------------------------------------------------------------------------------
def test_p1a_fix_the_writer_prints_the_tie_cells_from_the_exact_values():
    assert W._edge(E.Edge(7.175, None, (), Fraction(287, 40))) == "7.18"
    assert W._edge(E.Edge(7.175, "top", (), Fraction(287, 40))) == "7.18 [censored top]"


# ---- the one function --------------------------------------------------------------------------------------------------------
def test_p1a_fix_integrated_rate_returns_a_float_that_carries_its_exact_fraction():
    got = analysis_boundary.integrated_rate(PI0_7, N48, SPEEDS)
    assert isinstance(got, float) and got == 49 / 96 and analysis_boundary.exact_of(got) == Fraction(49, 96)
    assert analysis_boundary.exact_of(0.25) == Fraction(1, 4)                     # a plain float is read as its exact binary value


def test_p1a_fix_the_exact_readers_go_through_integrated_rate(monkeypatch):
    monkeypatch.setattr(analysis_boundary, "integrated_rate", lambda c, n, s: 0.25)
    p, c = _grids(PI0_7, GEOMETRIC)
    assert N.n3_integrated_exact(p) == [Fraction(1, 4)] * 16
    assert N.n5_against_controller([p], c)["integrated"]["differences_exact"][0][0] == 0


def test_p1a_fix_the_check_sheet_states_the_print_rule_so_the_checker_applies_it_too():
    import inspect
    src = inspect.getsource(W.render_check_p1)
    assert "P1A-2" in src and "rounded half to even" in src and "Do not round through a float" in src
