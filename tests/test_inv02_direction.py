"""Invariant 2 (spec sections 3, 6, 12): a gust from 0 degrees at a given reference velocity pushes against it;
rotating the reference rotates the force; the gust is the same whatever the drone does.
"""
import numpy as np
import pytest

import oracles as O
import spec_numbers as S
from drone_study.air import PRIMARY, air_wrench_world
from drone_study.condition import Condition
from drone_study.gust import Gust
from drone_study.stack import ScriptedController, fly
from drone_study.streams import build_tapes

CASES = [(s, t) for s in S.SHAPES for t in S.TIERS]
LEVELS = sorted(set(S.SEARCH_PHASES + S.ZONE_B_PHASES + S.ZONE_C_PHASES))
HOVER_OMEGA = np.full(4, S.OMEGA_HOVER)


def travel_at_peak(shape, tier, phase):
    peak = O.gust_peak(shape, tier, phase)
    _, vel, _ = O.case_state(shape, tier, peak)
    return peak, vel


def gust_for(shape, tier, phase, az, el, speed=8.0):
    return Gust.from_condition(Condition(case=f"{shape}-{tier}", az=az, el=el, speed=speed, phase=phase))


@pytest.mark.parametrize("shape,tier", CASES)
@pytest.mark.parametrize("phase", LEVELS)
def test_inv2_a_gust_from_zero_degrees_pushes_against_the_reference(shape, tier, phase):
    peak, vel = travel_at_peak(shape, tier, phase)
    v = gust_for(shape, tier, phase, 0.0, 0.0, 8.0).air_velocity(np.array([peak]))[0]
    assert np.linalg.norm(v) == pytest.approx(8.0, abs=1e-9)
    cosine = v @ vel / (np.linalg.norm(v) * np.linalg.norm(vel))
    assert cosine == pytest.approx(-1.0, abs=1e-9)


@pytest.mark.parametrize("shape,tier", [("fig8", "faster"), ("circle", "fast"), ("oval", "medium"), ("slalom", "faster")])
@pytest.mark.parametrize("az", (0, 22.5, 45, 90, 135, 180, 225, 270, 315, 337.5))
def test_inv2_azimuth_counts_anticlockwise_seen_from_above(shape, tier, az):
    peak, vel = travel_at_peak(shape, tier, 0.0)
    v = gust_for(shape, tier, 0.0, az, 0.0).air_velocity(np.array([peak]))[0]
    np.testing.assert_allclose(v, 8.0 * O.gust_direction(vel, az, 0.0), rtol=0, atol=1e-9)
    t_hat = np.array([vel[0], vel[1], 0.0]) / np.hypot(vel[0], vel[1])
    left = np.cross([0, 0, 1.0], t_hat)
    if az == 90:      # "90 degrees comes from the left of travel": the air moves to the right
        np.testing.assert_allclose(v, -8.0 * left, atol=1e-9)
    if az == 180:     # a tailwind: the air moves with the drone
        np.testing.assert_allclose(v, 8.0 * t_hat, atol=1e-9)
    if az == 270:     # from the right: the air moves to the left
        np.testing.assert_allclose(v, 8.0 * left, atol=1e-9)


@pytest.mark.parametrize("el", (0, 15, -15, 35, -35, 55, -55))
def test_inv2_elevation_sets_the_vertical_share_and_its_sign_flips_with_it(el):
    peak, vel = travel_at_peak("fig8", "faster", 0.0)
    v = gust_for("fig8", "faster", 0.0, 0.0, el).air_velocity(np.array([peak]))[0]
    assert np.hypot(v[0], v[1]) == pytest.approx(8.0 * np.cos(np.radians(el)), abs=1e-9)
    assert abs(v[2]) == pytest.approx(8.0 * np.sin(np.radians(abs(el))), abs=1e-9)
    v_neg = gust_for("fig8", "faster", 0.0, 0.0, -el).air_velocity(np.array([peak]))[0]
    assert v[2] == pytest.approx(-v_neg[2], abs=1e-9)


@pytest.mark.parametrize("el", (15, 35, 55))
def test_inv2_elevation_up_names_where_the_gust_comes_from(el):
    """3A-2 (a convention, not a number): direction names the side the air comes FROM for azimuth ('0 degrees is
    a headwind', '90 comes from the left'), so elevation 'up' reads the same way: a gust at +el comes from above and
    blows downward. If Taylor rules the other reading, this one test flips its sign."""
    peak, _ = travel_at_peak("fig8", "faster", 0.0)
    v = gust_for("fig8", "faster", 0.0, 0.0, el).air_velocity(np.array([peak]))[0]
    assert v[2] == pytest.approx(-8.0 * np.sin(np.radians(el)), abs=1e-9)


@pytest.mark.parametrize("theta_deg", (30.0, 90.0, 200.0))
def test_inv2_rotating_the_reference_rotates_the_gust_and_the_force(theta_deg):
    base_travel = (1.3, 0.4, 0.0)
    rot = O.rot_z(np.radians(theta_deg))
    travel = tuple(rot @ np.array(base_travel))
    g0 = Gust(8.0, 60.0, 20.0, 4.0, base_travel)
    g1 = Gust(8.0, 60.0, 20.0, 4.0, travel)
    v0, v1 = g0.air_velocity(np.array([4.0]))[0], g1.air_velocity(np.array([4.0]))[0]
    assert np.linalg.norm(v0) == pytest.approx(8.0, abs=1e-9)
    np.testing.assert_allclose(v1, rot @ v0, atol=1e-9)
    # and the force on a flat drone at rest, which section 17.3 makes the same all round in plan
    f0, m0 = air_wrench_world(v0, np.zeros(3), np.eye(3), HOVER_OMEGA, PRIMARY)
    f1, m1 = air_wrench_world(v1, np.zeros(3), np.eye(3), HOVER_OMEGA, PRIMARY)
    assert np.linalg.norm(f0) > 0.05
    np.testing.assert_allclose(f1, rot @ f0, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(m1, rot @ m0, rtol=1e-9, atol=1e-15)


@pytest.mark.parametrize("shape,tier", CASES)
def test_inv2_the_direction_is_read_from_the_reference_at_the_gusts_peak(shape, tier):
    for phase in (0.0, 0.125, 0.25):
        peak, vel = travel_at_peak(shape, tier, phase)
        g = gust_for(shape, tier, phase, 45.0, 0.0)
        t_hat = np.array([vel[0], vel[1]]) / np.hypot(vel[0], vel[1])
        got = np.array(g.travel[:2]) / np.hypot(g.travel[0], g.travel[1])
        np.testing.assert_allclose(got, t_hat, atol=1e-9)


def _hover_script(view):
    return [S.WEIGHT, 0.0, 0.0, 0.0]


def _busy_script(view):
    t = view.t
    return [S.WEIGHT * (1.0 + 0.05 * np.sin(9 * t)), 0.3 * np.sin(7 * t), 0.2 * np.cos(5 * t), 0.1 * np.sin(3 * t)]


def test_inv2_the_gust_is_the_same_whatever_the_drone_does():
    cond = Condition(case="fig8-faster", az=90.0, el=0.0, speed=8.0, phase=0.0)
    a = fly(cond, 11, ScriptedController(_hover_script))
    b = fly(cond, 11, ScriptedController(_busy_script))
    n = min(a.n_steps, b.n_steps)
    peak = O.gust_peak("fig8", "faster", 0.0)
    assert a.t[n - 1] > peak + 0.6                   # both fly right through the gust, which ends half a second after its peak
    np.testing.assert_array_equal(a.air_velocity[:n], b.air_velocity[:n])
    gust = Gust.from_condition(cond)
    tapes = build_tapes(cond, 11)
    t = a.t[:n]
    want = gust.air_velocity(t) + tapes.turbulence[(np.round(t / S.DT_PHYS)).astype(int)]
    np.testing.assert_allclose(a.air_velocity[:n], want, rtol=0, atol=S.MACHINE_ATOL * 8)
    assert np.linalg.norm(a.air_velocity[:n] - tapes.turbulence[(np.round(t / S.DT_PHYS)).astype(int)], axis=-1).max() > 7.0
