"""Invariant 13 (spec sections 2, 3, 17.7): every trajectory equals section 2's formula and meets its peak speed
exactly and its acceleration bound; phase 0 lands within 0.1% of peak speed on fig8, oval and slalom; peak
acceleration is within 1% at phase 0.125 on fig8 and within 0.1% at phase 0.25 on oval and slalom; circle has
constant speed and acceleration within 0.1%; every gust's window at section 3's phases lies inside the episode.
"""
import numpy as np
import pytest

import oracles as O
import spec_numbers as S
from drone_study.cases import CASE_NAMES, LEVEL_PHASES, case as get_case, gust_peak_time

CASES = [(s, t) for s in S.SHAPES for t in S.TIERS]
LAPS = np.linspace(0.0, 1.0, 20001)


def name(shape, tier):
    return f"{shape}-{tier}"


def test_inv13_there_are_twelve_cases_each_with_its_tiers_peak_speed():
    assert tuple(CASE_NAMES) == tuple(name(s, t) for s, t in CASES)
    assert tuple(LEVEL_PHASES) == tuple(sorted(set(S.SEARCH_PHASES + S.ZONE_B_PHASES + S.ZONE_C_PHASES)))
    for s, t in CASES:
        c = get_case(name(s, t))
        assert c.name == name(s, t)
        assert c.peak_speed == S.PEAK_SPEED[t]


@pytest.mark.parametrize("shape,tier", CASES)
def test_inv13_trajectory_equals_the_section_2_formula(shape, tier):
    c = get_case(name(shape, tier))
    t = LAPS * c.period * 3 + 0.0123                  # three laps, off the grid
    pos, vel, acc = O.case_state(shape, tier, t)
    np.testing.assert_allclose(c.position(t), pos, rtol=0, atol=1e-12)
    np.testing.assert_allclose(c.velocity(t), vel, rtol=0, atol=1e-9)
    np.testing.assert_allclose(c.acceleration(t), acc, rtol=0, atol=1e-9)
    assert np.all(c.position(t)[:, 2] == S.HEIGHT_M)


@pytest.mark.parametrize("shape,tier", CASES)
def test_inv13_a_lap_is_one_period_and_matches_the_printed_lap_time(shape, tier):
    c = get_case(name(shape, tier))
    assert c.period == pytest.approx(O.lap(shape, tier), rel=1e-12)
    printed = S.LAP_S[shape][S.TIERS.index(tier)]
    assert c.period == pytest.approx(printed, abs=S.SPEC_ROUNDING_S / 10)       # printed to 3 decimals
    # a lap is one period of the reference's phase: speed and acceleration repeat exactly after T
    t = np.array([0.31, 1.07])
    np.testing.assert_allclose(c.velocity(t), c.velocity(t + c.period), atol=1e-9)
    np.testing.assert_allclose(c.acceleration(t), c.acceleration(t + c.period), atol=1e-9)


@pytest.mark.parametrize("shape,tier", CASES)
def test_inv13_the_peak_speed_is_met_exactly_and_the_acceleration_bound_holds(shape, tier):
    c = get_case(name(shape, tier))
    t = LAPS * c.period
    speed = np.linalg.norm(c.velocity(t), axis=-1)
    accel = np.linalg.norm(c.acceleration(t), axis=-1)
    v = S.PEAK_SPEED[tier]
    assert speed.max() == pytest.approx(v, abs=1e-9)
    assert speed.max() <= v + 1e-9
    assert c.peak_speed == pytest.approx(v, abs=1e-12)
    assert accel.max() <= S.MAX_ACCEL_BOUND
    printed = S.PEAK_ACCEL[shape][S.TIERS.index(tier)]
    assert accel.max() == pytest.approx(printed, abs=0.0051)                  # the table prints to 0.01


@pytest.mark.parametrize("shape,tier", [(s, t) for s, t in CASES if s in ("fig8", "oval", "slalom")])
def test_inv13_phase_zero_is_within_a_tenth_of_a_percent_of_peak_speed(shape, tier):
    c = get_case(name(shape, tier))
    speed = np.linalg.norm(c.velocity(LAPS * c.period), axis=-1).max()
    assert speed == pytest.approx(S.PEAK_SPEED[tier], abs=1e-9)
    at_zero = np.linalg.norm(c.velocity(np.array([0.0])), axis=-1)[0]
    assert at_zero >= speed * (1 - 0.001)
    for lap_no in (1, 2):
        assert np.linalg.norm(c.velocity(np.array([lap_no * c.period])), axis=-1)[0] >= speed * (1 - 0.001)


def test_inv13_peak_acceleration_phases():
    for tier in S.TIERS:
        c = get_case(name("fig8", tier))
        peak = np.linalg.norm(c.acceleration(LAPS * c.period), axis=-1).max()
        assert peak == pytest.approx(S.PEAK_ACCEL["fig8"][S.TIERS.index(tier)], abs=0.0051)
        at = np.linalg.norm(c.acceleration(np.array([0.125 * c.period])), axis=-1)[0]
        assert at >= peak * (1 - 0.01) and at <= peak + 1e-12           # within 1% at phase 0.125 on fig8
        for shape in ("oval", "slalom"):
            c = get_case(name(shape, tier))
            peak = np.linalg.norm(c.acceleration(LAPS * c.period), axis=-1).max()
            assert peak == pytest.approx(S.PEAK_ACCEL[shape][S.TIERS.index(tier)], abs=0.0051)
            at = np.linalg.norm(c.acceleration(np.array([0.25 * c.period])), axis=-1)[0]
            assert at >= peak * (1 - 0.001) and at <= peak + 1e-12      # within 0.1% at phase 0.25


@pytest.mark.parametrize("tier", S.TIERS)
def test_inv13_circle_has_constant_speed_and_acceleration(tier):
    c = get_case(name("circle", tier))
    t = LAPS * c.period
    speed = np.linalg.norm(c.velocity(t), axis=-1)
    accel = np.linalg.norm(c.acceleration(t), axis=-1)
    assert speed.min() >= speed.max() * (1 - 0.001)
    assert accel.min() >= accel.max() * (1 - 0.001)
    assert speed.max() == pytest.approx(S.PEAK_SPEED[tier], abs=1e-9)


@pytest.mark.parametrize("shape,tier", CASES)
def test_inv13_every_gust_window_at_section_3_phases_lies_inside_the_episode(shape, tier):
    for phase in LEVEL_PHASES:
        peak = gust_peak_time(name(shape, tier), phase)
        assert peak == pytest.approx(O.gust_peak(shape, tier, phase), abs=1e-9)
        start, end = peak - S.GUST_LENGTH_S / 2, peak + S.GUST_LENGTH_S / 2
        assert 0.0 <= start and end <= S.EPISODE_S
        assert start >= S.EARLIEST_GUST_START_S - S.SPEC_ROUNDING_S      # "starts at 2.86 s or later"
        assert end <= S.LATEST_GUST_END_S + S.SPEC_ROUNDING_S            # "ends by 9.53 s"


def test_inv13_s2_lengths_fit_on_fig8_faster_at_phase_zero():
    peak = gust_peak_time("fig8-faster", 0.0)
    for length in S.GUST_LENGTHS_S2:
        assert 0.0 <= peak - length / 2 and peak + length / 2 <= S.EPISODE_S
