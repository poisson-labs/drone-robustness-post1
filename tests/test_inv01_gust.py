"""Invariant 1 (spec section 12; enforces sections 3 and 17.7).

Gust: air velocity at time t equals the 1 - cos formula for the condition, to machine precision, is zero outside its
window, and peaks at section 3's time for the case and phase.
"""
import numpy as np
import pytest

import oracles as O
import spec_numbers as S
from drone_study.cases import gust_peak_time
from drone_study.condition import Condition
from drone_study.gust import Gust

CASES = [(s, t) for s in S.SHAPES for t in S.TIERS]
LEVELS = sorted(set(S.SEARCH_PHASES + S.ZONE_B_PHASES + S.ZONE_C_PHASES))   # the eight levels of section 3
SPEEDS = (2.0, 7.0, 12.0, 16.0)
TGRID = np.arange(0, S.EPISODE_S + 1e-9, S.DT_PHYS)


def gust_for(shape, tier, phase, speed, length=S.GUST_LENGTH_S):
    cond = Condition(case=f"{shape}-{tier}", az=0.0, el=0.0, speed=speed, phase=phase, gust_length=length)
    return Gust.from_condition(cond)


@pytest.mark.parametrize("shape,tier", CASES)
@pytest.mark.parametrize("phase", LEVELS)
def test_inv1_air_velocity_is_the_one_minus_cos_formula(shape, tier, phase):
    peak = O.gust_peak(shape, tier, phase)
    for speed in SPEEDS:
        g = gust_for(shape, tier, phase, speed)
        got = np.linalg.norm(g.air_velocity(TGRID), axis=-1)
        want = O.gust_magnitude(TGRID, speed, peak)
        assert want.max() > 0.9 * speed          # the grid really samples the gust
        np.testing.assert_allclose(got, want, rtol=0, atol=S.MACHINE_ATOL * max(speed, 1.0))


@pytest.mark.parametrize("shape,tier", CASES)
def test_inv1_air_velocity_is_exactly_zero_outside_the_window(shape, tier):
    peak = O.gust_peak(shape, tier, 0.125)
    g = gust_for(shape, tier, 0.125, 9.0)
    start, end = peak - S.GUST_LENGTH_S / 2, peak + S.GUST_LENGTH_S / 2
    outside = np.array([0.0, 1.0, start - 1.0, start - 1e-9, end + 1e-9, end + 1.0, S.EPISODE_S])
    assert np.all(g.air_velocity(outside) == 0.0)
    inside = np.linalg.norm(g.air_velocity(np.array([peak])), axis=-1)
    assert inside[0] == pytest.approx(9.0, abs=S.MACHINE_ATOL * 9)    # the control: the gust is there at all


@pytest.mark.parametrize("shape,tier", CASES)
@pytest.mark.parametrize("phase", LEVELS)
def test_inv1_the_gust_peaks_at_the_time_section_3_gives(shape, tier, phase):
    want = O.gust_peak(shape, tier, phase)           # first whole lap at or after 3.0 s, plus phase x lap
    g = gust_for(shape, tier, phase, 10.0)
    assert g.peak_time == pytest.approx(want, abs=1e-9)
    assert gust_peak_time(f"{shape}-{tier}", phase) == pytest.approx(want, abs=1e-9)
    fine = np.arange(want - 0.6, want + 0.6, 1e-4)
    mags = np.linalg.norm(g.air_velocity(fine), axis=-1)
    assert abs(fine[int(np.argmax(mags))] - want) <= 1.5e-4
    assert mags.max() == pytest.approx(10.0, rel=1e-6)
    assert np.linalg.norm(g.air_velocity(np.array([want])), axis=-1)[0] == pytest.approx(10.0, abs=S.MACHINE_ATOL * 10)


@pytest.mark.parametrize("length", (S.GUST_LENGTH_S,) + S.GUST_LENGTHS_S2)
def test_inv1_gust_length_is_one_second_and_s2_lengths_follow_the_same_formula(length):
    """The gust is 1.0 s long; S2 flies 0.5 s and 2 s on fig8-faster at phase 0."""
    peak = O.gust_peak("fig8", "faster", 0.0)        # S2 flies fig8-faster at phase 0 only
    g = gust_for("fig8", "faster", 0.0, 8.0, length=length)
    assert g.window == pytest.approx((peak - length / 2, peak + length / 2), abs=1e-9)
    got = np.linalg.norm(g.air_velocity(TGRID), axis=-1)
    np.testing.assert_allclose(got, O.gust_magnitude(TGRID, 8.0, peak, length), rtol=0, atol=S.MACHINE_ATOL * 8)
    assert np.all(g.air_velocity(np.array([peak - length / 2 - 1e-9, peak + length / 2 + 1e-9])) == 0.0)
