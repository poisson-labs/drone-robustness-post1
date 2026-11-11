"""The yaw addenda of 2026-11-07 and 2026-11-08 (session 4D-A), rule 1: the heading terms. The 2026-11-08 addendum (4DR-2)
replaces the 2026-11-07 term -k |wrap e| / pi with a Huber heading term, -k x H(wrap(true heading - reference heading)) / pi, H
quadratic inside a 5 degree band and linear outside, so the restoring slope goes to zero at zero error, plus a damping term on
the Euler yaw rate, -k x 0.25 x min(|rate|, 2)^2 / 4, so a banked turn's body z rate is not penalised. The heading is the yaw of
the body-to-world rotation, atan2(R[1, 0], R[0, 0]); the reference heading is 0 on every case; the error is wrapped into (-pi, pi];
k is one of the ladder's weights, base or extension, and never a free number (4DA-1 as amended by 4DR-2, 4DA-2, 4DA-3 as amended).

Closed-form cases: the wrap at its seams, the Huber function at and about its band, the heading of constructed rotations with and
without tilt, the Euler yaw rate of a banked turn against a body z rate, the terms' bounds and symmetries, a reference heading
other than zero, and the refusal of a weight off the ladder.
"""
import math

import numpy as np
import pytest

import training_numbers as N
import yaw_oracles as Y
from drone_study.training import reward as R
from drone_study.training.reward import huber, reward_terms, wrap_angle
from drone_study.training.reward import heading_error as heading_error_of
from drone_study.training.reward import heading_rate as heading_rate_of

SAME = np.array([0.2, -0.1, 0.3, 0.0])
DEG = math.pi / 180.0
BAND = N.YAW_HUBER_BAND_RAD


def heading_term(psi, k):
    return reward_terms(0.0, SAME, SAME, False, heading_error=psi, heading_rate=0.0, k_yaw=k).heading


def rate_term(rate, k):
    return reward_terms(0.0, SAME, SAME, False, heading_error=0.0, heading_rate=rate, k_yaw=k).heading_rate


@pytest.mark.parametrize("a,want", [(0.0, 0.0), (0.1, 0.1), (-0.1, -0.1), (math.pi, math.pi), (-math.pi, math.pi),
                                    (3 * math.pi, math.pi), (2 * math.pi, 0.0), (0.1 + 2 * math.pi, 0.1), (-0.1 - 4 * math.pi, -0.1),
                                    (350 * DEG, -10 * DEG), (-190 * DEG, 170 * DEG), (1e6, Y.wrap(1e6))])
def test_yw01_wrap_angle_lands_in_minus_pi_to_pi_with_pi_kept(a, want):
    got = wrap_angle(a)
    assert got == pytest.approx(want, abs=1e-9)
    assert -math.pi < got <= math.pi
    assert wrap_angle(a + 1.0) == pytest.approx(Y.wrap(want + 1.0), abs=1e-9)             # and a shifted angle wraps with it


# ---- the Huber function ----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("e", [0.0, 0.01, 0.5 * BAND, BAND, 1.5 * BAND, 10 * DEG, 0.5, math.pi / 2, math.pi])
def test_yw01_huber_is_quadratic_inside_the_band_linear_outside_and_joins_without_a_kink(e):
    got = huber(e)
    assert got == pytest.approx(Y.huber(e), rel=1e-12, abs=1e-15)
    if e <= BAND:
        assert got == pytest.approx(e * e / (2.0 * BAND), rel=1e-12, abs=1e-15)
    else:
        assert got == pytest.approx(e - BAND / 2.0, rel=1e-12)
    assert huber(-e) == pytest.approx(got, rel=1e-12, abs=1e-15)                                   # even
    assert huber(e + 0.3) > got                                                                     # increasing
    assert huber(e, band=2 * BAND) == pytest.approx(Y.huber(e, 2 * BAND), rel=1e-12, abs=1e-15)   # the band is a parameter


def test_yw01_the_huber_slope_vanishes_at_zero_and_is_one_outside_the_band():
    h = 1e-6
    assert (huber(h) - huber(0.0)) / h == pytest.approx(0.0, abs=1e-5)                            # no cusp at zero (4DR-2)
    assert (huber(BAND + h) - huber(BAND - h)) / (2 * h) == pytest.approx(1.0, abs=1e-5)         # continuous slope at the band
    assert (huber(0.5 + h) - huber(0.5 - h)) / (2 * h) == pytest.approx(1.0, abs=1e-6)            # unit slope outside
    assert (huber(0.5 * BAND + h) - huber(0.5 * BAND - h)) / (2 * h) == pytest.approx(0.5, abs=1e-5)   # e / band inside
    assert huber(BAND) == pytest.approx(BAND / 2.0, rel=1e-12) and huber(math.pi) == pytest.approx(math.pi - BAND / 2.0, rel=1e-12)
    assert R.YAW_HUBER_BAND_RAD == pytest.approx(5 * DEG)


# ---- the heading term ----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("k", N.YAW_WEIGHT_LADDER)
@pytest.mark.parametrize("psi", [0.0, 0.5 * BAND, BAND, 0.1, math.pi / 4, math.pi / 2, math.pi, -math.pi / 2, 3 * math.pi / 2,
                                 2 * math.pi, 350 * DEG, -190 * DEG, 45 * DEG, 90 * DEG])
def test_yw01_the_heading_term_is_the_huber_of_the_wrapped_error_over_pi_bounded_by_the_weight(psi, k):
    got = heading_term(psi, k)
    assert got == pytest.approx(Y.heading_term(psi, k), rel=1e-12, abs=1e-15)
    assert got == pytest.approx(-k * Y.huber(Y.wrap(psi)) / math.pi, rel=1e-12, abs=1e-15)
    assert -k * N.HEADING_TERM_BOUND - 1e-15 <= got <= 0.0
    assert heading_term(psi + 2 * math.pi, k) == pytest.approx(got, rel=1e-9, abs=1e-15)      # a full turn changes nothing
    assert heading_term(-psi, k) == pytest.approx(got, rel=1e-12, abs=1e-15)                   # sign-blind
    turned = heading_term(psi + 0.5, k)                                                        # a turned drone pays
    assert turned == pytest.approx(Y.heading_term(psi + 0.5, k), rel=1e-9, abs=1e-15) and turned < 0.0


def test_yw01_the_term_is_zero_on_heading_quadratic_inside_the_band_and_largest_facing_backwards():
    for k in N.YAW_WEIGHT_LADDER:
        assert heading_term(0.0, k) == 0.0 and heading_term(2 * math.pi, k) == pytest.approx(0.0, abs=1e-15)
        assert heading_term(math.pi, k) == pytest.approx(-k * (1.0 - 1.0 / 72.0), rel=1e-12)              # the bound, 71/72 of k
        assert heading_term(-math.pi, k) == pytest.approx(heading_term(math.pi, k), rel=1e-12)
        assert heading_term(1 * DEG, k) == pytest.approx(4 * heading_term(0.5 * DEG, k), rel=1e-9)        # quadratic inside the band
        assert heading_term(20 * DEG, k) - heading_term(10 * DEG, k) == pytest.approx(-k * 10 * DEG / math.pi, rel=1e-9)   # linear outside
        assert heading_term(10 * DEG, k) == pytest.approx(-k * (10 * DEG - BAND / 2.0) / math.pi, rel=1e-12)
        assert heading_term(10 * DEG, k) > -k * 10 * DEG / math.pi                                        # the Huber pays less than the old term
    assert heading_term(0.5, 0.016) == pytest.approx(4 * heading_term(0.5, 0.004), rel=1e-12)             # linear in the weight
    assert heading_term(0.5, 0.064) == pytest.approx(16 * heading_term(0.5, 0.004), rel=1e-12)


# ---- the damping term ----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("k", N.YAW_WEIGHT_LADDER)
@pytest.mark.parametrize("rate", [0.0, 0.05, 0.2094, -0.2094, 0.75, 1.0, -1.5, 2.0, 2.5, -5.0, 10.0, 100.0])
def test_yw01_the_damping_term_is_quadratic_in_the_heading_rate_up_to_the_cap_and_a_quarter_of_the_weight_at_most(rate, k):
    got = rate_term(rate, k)
    assert got == pytest.approx(Y.rate_term(rate, k), rel=1e-12, abs=1e-15)
    r = min(abs(rate), N.YAW_RATE_CAP_RAD_S)
    assert got == pytest.approx(-k * N.YAW_RATE_WEIGHT * r * r / N.YAW_RATE_CAP_RAD_S ** 2, rel=1e-12, abs=1e-15)
    assert -k * N.YAW_RATE_WEIGHT - 1e-15 <= got <= 0.0
    assert rate_term(-rate, k) == pytest.approx(got, rel=1e-12, abs=1e-15)                     # sign-blind
    assert rate_term(abs(rate) + 0.1, k) <= got + 1e-15                                          # more rate pays more, or the same past the cap
    assert rate_term(0.1, k) < 0.0                                                             # the control: a turning drone pays


def test_yw01_the_damping_term_saturates_at_the_cap_and_a_held_heading_pays_nothing():
    for k in N.YAW_WEIGHT_LADDER:
        assert rate_term(0.0, k) == 0.0
        assert rate_term(N.YAW_RATE_CAP_RAD_S, k) == pytest.approx(-k * N.YAW_RATE_WEIGHT, rel=1e-12)
        assert rate_term(7.0, k) == rate_term(N.YAW_RATE_CAP_RAD_S, k) == rate_term(-50.0, k)   # a tumble pays no more than the cap
        assert rate_term(1.0, k) == pytest.approx(4 * rate_term(0.5, k), rel=1e-12)             # quadratic under it
        at_gate = rate_term(math.radians(12.0), k)                                               # the gate's 12 deg/s
        assert at_gate == pytest.approx(-k * 0.25 * (math.radians(12.0) / 2.0) ** 2, rel=1e-12)
        assert -at_gate < 0.003 * k                                                             # under a 350th of the weight: a holder barely pays
    assert rate_term(0.75, 0.064) == pytest.approx(16 * rate_term(0.75, 0.004), rel=1e-12)      # linear in the weight


# ---- the heading and the heading rate of a rotation ----------------------------------------------------------------------------------
@pytest.mark.parametrize("yaw", [0.0, 0.3, -0.7, math.pi / 2, 3.0, -3.1, 170 * DEG])
@pytest.mark.parametrize("pitch,roll", [(0.0, 0.0), (0.3, 0.0), (0.0, -0.4), (0.5, 0.35), (-0.2, 0.6)])
def test_yw01_the_heading_of_a_rotation_is_the_azimuth_of_the_body_x_axis_whatever_the_tilt(yaw, pitch, roll):
    R_ = Y.rot_zyx(yaw, pitch, roll)
    got = heading_error_of(R_)
    assert got == pytest.approx(Y.wrap(yaw), abs=1e-12)
    assert got == pytest.approx(Y.heading_of_rotation(R_), abs=1e-12)
    assert got == pytest.approx(Y.heading_of_quat(Y.quat_from_rotation(R_)), abs=1e-9)        # session 4C's quaternion formula
    x_axis = R_[:, 0]
    assert got == pytest.approx(math.atan2(x_axis[1], x_axis[0]), abs=1e-12)
    assert heading_error_of(Y.rot_z(0.7) @ R_) == pytest.approx(Y.wrap(yaw + 0.7), abs=1e-12)   # turned by 0.7 about the vertical


def test_yw01_a_tilt_alone_is_no_heading_error():
    for pitch, roll in [(0.3, 0.0), (0.0, 0.5), (0.4, -0.4), (-0.6, 0.2)]:
        assert heading_error_of(Y.rot_zyx(0.0, pitch, roll)) == pytest.approx(0.0, abs=1e-12)
        assert heading_error_of(Y.rot_z(0.5) @ Y.rot_zyx(0.0, pitch, roll)) == pytest.approx(0.5, abs=1e-12)   # the control: a turn shows
    assert heading_error_of(np.eye(3)) == 0.0


@pytest.mark.parametrize("yaw,pitch,roll", [(0.0, 0.0, 0.0), (0.3, 0.2, -0.4), (-1.0, 0.5, 0.35), (2.5, -0.3, 0.6), (0.0, 0.48, 0.0)])
@pytest.mark.parametrize("omega", [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.7, 0.0), (0.4, -0.3, 0.9), (2.0, 0.0, -0.5)])
def test_yw01_the_heading_rate_is_the_euler_yaw_rate_of_the_body_rates(yaw, pitch, roll, omega):
    R_ = Y.rot_zyx(yaw, pitch, roll)
    got = heading_rate_of(R_, np.array(omega))
    assert got == pytest.approx(Y.euler_rate(R_, omega), rel=1e-12, abs=1e-15)
    p, q, r = omega
    assert got == pytest.approx((q * math.sin(roll) + r * math.cos(roll)) / math.cos(pitch), rel=1e-12, abs=1e-15)
    assert heading_rate_of(R_, 2 * np.array(omega)) == pytest.approx(2 * got, rel=1e-12, abs=1e-15)    # linear in the rates
    about_the_vertical = R_.T @ np.array([0.0, 0.0, 1.3])                                           # a pure turn about world z at 1.3 rad/s
    assert heading_rate_of(R_, about_the_vertical) == pytest.approx(1.3, abs=1e-9)
    assert heading_rate_of(R_, np.array([0.5, 0.0, 0.0])) == pytest.approx(0.0, abs=1e-12)          # a roll rate alone turns no heading


def test_yw01_a_banked_turn_with_a_held_heading_has_body_z_rate_and_no_heading_rate():
    """The point of 4DR-4 and of damping the heading rate: a drone whose tilt direction rotates (a lap of a circle at constant
    heading) carries a body z rate of about Omega (1 - cos tilt) while its heading rate is zero."""
    n, dt, alpha, omega = 500, 0.01, 0.45, 2.2
    t = np.arange(n) * dt
    Rs = np.array([Y.rot_zyx(0.0, alpha * math.cos(omega * tt), alpha * math.sin(omega * tt)) for tt in t])
    w = Y.body_rates_of(Rs, dt)
    rates = np.array([heading_rate_of(R_, ww) for R_, ww in zip(Rs[2:-2], w[2:-2])])
    assert np.max(np.abs(rates)) < 2e-3                                                              # no heading rate
    body_z = np.sqrt(np.mean(w[2:-2, 2] ** 2))
    assert body_z > 0.15 and body_z == pytest.approx(omega * (1 - math.cos(alpha)), rel=0.3)         # a body z rate all the same, about Omega (1 - cos tilt)
    turning = np.array([Y.rot_zyx(0.8 * tt, alpha * math.cos(omega * tt), alpha * math.sin(omega * tt)) for tt in t])
    w2 = Y.body_rates_of(turning, dt)
    rates2 = np.array([heading_rate_of(R_, ww) for R_, ww in zip(turning[2:-2], w2[2:-2])])
    np.testing.assert_allclose(rates2, 0.8, atol=3e-3)                                               # the control: a real turn reads its rate


def test_yw01_the_reference_heading_is_zero_and_another_reference_shifts_the_error():
    assert R.REFERENCE_YAW == 0.0
    assert heading_error_of(Y.rot_z(0.3)) == pytest.approx(0.3, abs=1e-12)
    assert heading_error_of(Y.rot_z(0.3), reference_yaw=0.3) == pytest.approx(0.0, abs=1e-12)
    assert heading_error_of(Y.rot_z(-3.0), reference_yaw=3.0) == pytest.approx(Y.wrap(-6.0), abs=1e-12)   # wrapped: +0.283
    assert heading_error_of(Y.rot_z(3.0), reference_yaw=-3.0) == pytest.approx(Y.wrap(6.0), abs=1e-12)    # -0.283
    assert heading_error_of(Y.rot_z(math.pi), reference_yaw=0.0) == pytest.approx(math.pi, abs=1e-12)


@pytest.mark.parametrize("bad", [0.0, 0.002, 0.005, 0.02, 0.05, 0.1, 0.128, 1.0, -0.004])
def test_yw01_a_weight_off_the_ladder_is_refused(bad):
    with pytest.raises(ValueError):
        reward_terms(0.1, SAME, SAME, False, heading_error=0.2, heading_rate=0.0, k_yaw=bad)
    for k in N.YAW_WEIGHT_LADDER:                                                             # the control: the ladder is accepted
        assert reward_terms(0.1, SAME, SAME, False, heading_error=0.2, heading_rate=0.0, k_yaw=k).heading < 0.0


def test_yw01_the_heading_arguments_are_required_so_no_caller_gets_a_silent_zero_term():
    k = N.YAW_WEIGHT_LADDER[0]
    with pytest.raises(TypeError):
        reward_terms(0.1, SAME, SAME, False)                                                    # 4A-1's four-argument call
    with pytest.raises(TypeError):
        reward_terms(0.1, SAME, SAME, False, heading_error=0.2, k_yaw=k)                        # the 2026-11-07 call, without the rate
    with pytest.raises(TypeError):
        reward_terms(0.1, SAME, SAME, False, heading_rate=0.2, k_yaw=k)
    with pytest.raises(TypeError):
        reward_terms(0.1, SAME, SAME, False, heading_error=0.2, heading_rate=0.1)
    with pytest.raises(TypeError):
        reward_terms(0.1, SAME, SAME, False, 0.2, 0.1, k)                                        # keyword-only
    assert reward_terms(0.1, SAME, SAME, False, heading_error=0.2, heading_rate=0.1, k_yaw=k).heading < 0.0
