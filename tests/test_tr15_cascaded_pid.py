"""Training rule 15 (spec section 17.2, 'A classical baseline on the same grid' and 'What changes from the stock PID';
section 2's bar reads this controller's RMSE): "Cascaded PID with the firmware's structure and gains (... attitude 6/3/0,
velocity 25/1/0 and z 25/15/0, position 2/0/0 and z 2/0.5/0, position loop at 100 Hz) ... PID keeps the firmware's
gains, with the hover command set to true nominal hover." The four stated changes: "total tilt, not each axis, is
limited to 55 degrees ...; the position loop's correction stays clamped at the firmware's 1.1 m/s, added to the reference
velocity; reference acceleration is fed forward as tilt, atan(a / g) on each axis; and thrust is divided by the cosine of
tilt, within the motors' limit." "Every outer layer runs on the outer stream."

Readings (ruling 4A-17): the tilt whose cosine divides the thrust is the measured one; "total tilt" may be limited as the
angle of the thrust axis or as the size of the two tilt setpoints together; both of the firmware's per-axis tilt limits
(the velocity loop's 22 degree output limit and the 20 degree constraint after it) give way to the 55.

The controller is tested from outside, on views it is handed. The firmware's units (position_controller_pid.c and
platform_defaults_cf2.h at e828ed2): the velocity loop's output is tilt in degrees; the height loop's output times 1,000
is added to the hover command in motor units, thrust going as units squared (3A-3). On a first step the integral terms
add at most 3% (ruling 4A-17). Section 12 names no session for these tests; they are here because the bar reads this
controller (ruling 4A-17).
"""
import math

import numpy as np
import pytest

import spec_numbers as S
import training_numbers as N
from drone_study import pid as P
from drone_study.condition import Condition
from drone_study.pid import CascadedPID
from drone_study.rollout import tracking_rmse
from drone_study.score import score
from drone_study.sensing import OuterReading
from drone_study.stack import Controller, ControllerView, fly

TOL = N.PID_FIRST_STEP_TOL
HOME = np.array([0.3, -0.2, S.HEIGHT_M])
UNITS_HOVER = S.UNITS_MAX / math.sqrt(S.THRUST_TO_WEIGHT)                 # 43,690: thrust as units squared (3A-3)


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def view(*, error=(0.0, 0.0, 0.0), ref_velocity=(0.0, 0.0, 0.0), ref_accel=(0.0, 0.0, 0.0), velocity=None, rotation=None):
    """The drone at HOME, the reference `error` ahead of it, moving at ref_velocity with ref_accel. By default the drone
    moves with the reference and is level."""
    p0, v, a = HOME + np.asarray(error, dtype=float), np.asarray(ref_velocity, dtype=float), np.asarray(ref_accel, dtype=float)
    s = 0.1 * np.arange(5)[:, None]
    points = p0 + v * s + 0.5 * a * s * s                                     # the five reference points: now, and 0.1 to 0.4 s ahead
    reading = OuterReading(position=HOME.copy(), velocity=np.array(v if velocity is None else velocity, dtype=float),
                           rotation=np.eye(3) if rotation is None else np.asarray(rotation, dtype=float), body_rate=np.zeros(3))
    return ControllerView(0.0, reading, points, None)


def first(**kw):
    ctl = CascadedPID()
    ctl.reset()
    return np.asarray(ctl.act(view(**kw)), dtype=float)


def test_tr15_it_is_an_outer_loop_with_no_privilege_and_the_firmwares_mixer():
    ctl = CascadedPID()
    assert isinstance(ctl, Controller) and ctl.privileges == frozenset()      # the noisy, delayed outer stream and nothing else
    assert ctl.mixer(np.ones(4)) is None


def test_tr15_the_gains_are_the_firmwares():
    assert (P.POS_KP, P.POS_Z_KP, P.POS_Z_KI) == (N.PID_POS_KP, N.PID_POS_Z_KP, N.PID_POS_Z_KI) == (2.0, 2.0, 0.5)
    assert (P.VEL_KP, P.VEL_KI, P.VEL_Z_KP, P.VEL_Z_KI) == (25.0, 1.0, 25.0, 15.0)
    assert (P.ATT_KP, P.ATT_KI) == (6.0, 3.0)
    assert (P.VEL_CORRECTION_MAX, P.TILT_LIMIT_DEG, P.THRUST_SCALE) == (1.1, 55.0, 1000.0)


def test_tr15_on_the_reference_and_level_it_commands_true_nominal_hover():
    np.testing.assert_allclose(first(), [S.WEIGHT, 0.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(first(ref_velocity=(1.2, -0.4, 0.0)), [S.WEIGHT, 0.0, 0.0, 0.0], atol=1e-9)   # moving with the reference
    assert first()[0] == pytest.approx(S.T_MAX * (UNITS_HOVER / S.UNITS_MAX) ** 2, rel=1e-9)   # 43,690 units, not the firmware's 36,000


@pytest.mark.parametrize("rotation,tilt_deg", [(rot_x(math.radians(20)), 20), (rot_y(math.radians(-30)), 30),
                                               (rot_x(math.radians(25)) @ rot_y(math.radians(25)), None)])
def test_tr15_thrust_is_divided_by_the_cosine_of_the_tilt(rotation, tilt_deg):
    cos_tilt = rotation[2, 2]
    if tilt_deg is not None:
        assert cos_tilt == pytest.approx(math.cos(math.radians(tilt_deg)), rel=1e-12)
    assert first(rotation=rotation)[0] == pytest.approx(S.WEIGHT / cos_tilt, rel=1e-6)
    assert first()[0] == pytest.approx(S.WEIGHT, rel=1e-9)


def test_tr15_the_tilt_compensation_stays_within_the_motors_limit():
    steep = first(rotation=rot_x(math.radians(70)))                           # weight / cos 70 would be 0.77 N; the motors give 0.60
    assert steep[0] == pytest.approx(S.T_MAX, rel=1e-9)
    assert first(rotation=rot_x(math.radians(50)))[0] == pytest.approx(S.WEIGHT / math.cos(math.radians(50)), rel=1e-6)


@pytest.mark.parametrize("accel", [(3.0, 0.0, 0.0), (-5.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, -2.0, 0.0), (5.0, 5.0, 0.0), (2.5, -4.5, 0.0)])
def test_tr15_reference_acceleration_is_fed_forward_as_tilt_atan_a_over_g_on_each_axis(accel):
    """Level, on the reference, moving with it: the only thing asking for tilt is the reference's acceleration. The tilt
    setpoint shows as the rate command over the attitude gain. Acceleration along +x needs a positive rotation about the
    body y axis (the thrust axis leans to +x); along +y, a negative rotation about x."""
    out = first(ref_accel=accel)
    want_q = N.PID_ATT_KP * math.atan(accel[0] / S.G)
    want_p = -N.PID_ATT_KP * math.atan(accel[1] / S.G)
    assert out[2] == pytest.approx(want_q, rel=TOL, abs=1e-6)
    assert out[1] == pytest.approx(want_p, rel=TOL, abs=1e-6)
    assert abs(out[3]) < 1e-6
    assert np.linalg.norm(out[1:3]) > 1.0                                     # the feedforward is there


@pytest.mark.parametrize("error", [0.05, 0.1, 0.25, -0.2])
def test_tr15_the_position_and_velocity_loops_carry_the_firmwares_gains(error):
    """A position error e gives a velocity correction of 2 e, the velocity loop turns that into 25 degrees of tilt per
    m/s, and the attitude loop into 6 times the tilt in rate."""
    tilt = math.radians(N.PID_VEL_KP * N.PID_POS_KP * error)
    along_x, along_y = first(error=(error, 0.0, 0.0)), first(error=(0.0, error, 0.0))
    assert along_x[2] == pytest.approx(N.PID_ATT_KP * tilt, rel=TOL) and abs(along_x[1]) < 1e-6
    assert along_y[1] == pytest.approx(-N.PID_ATT_KP * tilt, rel=TOL) and abs(along_y[2]) < 1e-6
    assert along_x[0] == pytest.approx(S.WEIGHT, rel=1e-6)                    # a horizontal error asks for no height change


def test_tr15_the_velocity_correction_is_clamped_at_1_1_metres_a_second_and_added_to_the_reference_velocity():
    at_clamp = first(error=(0.55, 0.0, 0.0))                                  # 2 x 0.55 = 1.1 m/s
    assert at_clamp[2] == pytest.approx(N.PID_ATT_KP * math.radians(25.0 * 1.1), rel=TOL)
    for far in (0.6, 1.0, 3.0):
        np.testing.assert_allclose(first(error=(far, 0.0, 0.0)), at_clamp, rtol=1e-9, atol=1e-12)    # no more than 1.1, however far
    assert first(error=(0.5, 0.0, 0.0))[2] < at_clamp[2]                      # and under the clamp it still grows
    np.testing.assert_allclose(first(error=(0.0, -3.0, 0.0))[1], N.PID_ATT_KP * math.radians(25.0 * 1.1), rtol=TOL)
    diagonal = first(error=(3.0, 3.0, 0.0))                                   # each axis has its own 1.1 m/s, as in the firmware
    assert diagonal[2] == pytest.approx(N.PID_ATT_KP * math.radians(25.0 * 1.1), rel=TOL)
    assert diagonal[1] == pytest.approx(-N.PID_ATT_KP * math.radians(25.0 * 1.1), rel=TOL)
    # the reference's own velocity is added on top: a drone at rest under a reference at 0.5 m/s is 0.5 m/s slow
    lag = first(ref_velocity=(0.5, 0.0, 0.0), velocity=(0.0, 0.0, 0.0))
    assert lag[2] == pytest.approx(N.PID_ATT_KP * math.radians(25.0 * 0.5), rel=TOL)
    both = first(error=(3.0, 0.0, 0.0), ref_velocity=(0.5, 0.0, 0.0), velocity=(0.0, 0.0, 0.0))
    assert both[2] == pytest.approx(N.PID_ATT_KP * math.radians(25.0 * 1.6), rel=TOL)       # 1.1 of correction plus the 0.5


def test_tr15_total_tilt_is_limited_to_55_degrees_and_no_single_axis_to_20():
    one_axis = first(error=(3.0, 0.0, 0.0), ref_accel=(5.0, 0.0, 0.0))        # 27.5 degrees of feedback plus atan(5 / g) = 27.0
    want = math.radians(27.5) + math.atan(5.0 / S.G)
    assert math.degrees(want) < 55.0
    assert one_axis[2] == pytest.approx(N.PID_ATT_KP * want, rel=TOL)         # 54.5 degrees on one axis passes whole: not the stock 20
    two_axes = first(error=(3.0, -3.0, 0.0), ref_accel=(5.0, -5.0, 0.0))      # 54.5 on each axis would be 70 degrees of tilt in all
    roll_sp, pitch_sp = two_axes[1] / N.PID_ATT_KP, two_axes[2] / N.PID_ATT_KP
    tilt = math.degrees(math.acos(math.cos(roll_sp) * math.cos(pitch_sp)))    # the angle of the thrust axis the two setpoints give
    # "Total tilt" is held to 55 degrees whether the limit is put on that angle (55.0) or on the size of the two
    # setpoints taken together (52.7 degrees of tilt): either reading passes, and 70 does not.
    assert 52.0 <= tilt <= 55.0 * (1.0 + TOL)
    assert two_axes[1] == pytest.approx(two_axes[2], rel=1e-6) and two_axes[1] > 0      # the direction is kept, only the size is cut


def test_tr15_the_height_correction_is_clamped_at_1_1_metres_a_second_too():
    low = first(error=(0.0, 0.0, -0.8))                                       # 2 x 0.8 would be 1.6 m/s down: 40,000 units under hover
    clamped = S.T_MAX * ((UNITS_HOVER - 27500.0) / S.UNITS_MAX) ** 2          # 1.1 m/s is 27,500 units: 16,190 left
    floor = S.T_MAX * (20000.0 / S.UNITS_MAX) ** 2                            # the firmware's own thrust floor, if it is kept
    assert (1.0 - TOL) * clamped <= low[0] <= (1.0 + TOL) * floor
    assert first(error=(0.0, 0.0, -3.0))[0] == pytest.approx(low[0], rel=1e-9)


def test_tr15_the_attitude_loop_is_six_three_nought():
    """Level setpoint, the drone held rolled 5 degrees: the attitude error is constant. The rate command starts at 6 per
    second of it and, with the integral's 3 per second squared, is 9 per second of it one second later."""
    ctl = CascadedPID(); ctl.reset()
    rolled = view(rotation=rot_x(math.radians(5.0)))
    outs = [np.asarray(ctl.act(rolled), dtype=float) for _ in range(100)]
    error = -math.radians(5.0)
    assert outs[0][1] == pytest.approx(N.PID_ATT_KP * error, rel=TOL)
    assert outs[-1][1] == pytest.approx((N.PID_ATT_KP + N.PID_ATT_KI * 1.0) * error, rel=TOL)
    assert abs(outs[-1][2]) < 0.05 * abs(outs[-1][1])                         # the other axis stays quiet


@pytest.mark.parametrize("error_z", [0.1, 0.05, -0.1])
def test_tr15_the_height_loop_adds_a_thousand_motor_units_per_unit_of_its_output_to_true_hover(error_z):
    """2 per second of position gain, 25 of velocity gain, 1,000 units each: a 0.1 m height error is 5,000 units on the
    43,690 of hover, and thrust goes as units squared."""
    out = first(error=(0.0, 0.0, error_z))
    units = UNITS_HOVER + N.PID_THRUST_SCALE * N.PID_VEL_Z_KP * N.PID_POS_Z_KP * error_z
    want = S.T_MAX * (units / S.UNITS_MAX) ** 2
    assert out[0] - S.WEIGHT == pytest.approx(want - S.WEIGHT, rel=TOL)
    assert (out[0] > S.WEIGHT) == (error_z > 0)
    np.testing.assert_allclose(out[1:], 0.0, atol=1e-6)


def test_tr15_a_large_height_error_asks_for_no_more_than_the_motors_give():
    high = first(error=(0.0, 0.0, 0.3))                                       # 0.6 m/s of correction: 15,000 units over hover
    assert high[0] == pytest.approx(S.T_MAX * ((UNITS_HOVER + 15000.0) / S.UNITS_MAX) ** 2, rel=TOL)
    full = first(error=(0.0, 0.0, 1.0))                                       # the clamp's 1.1 m/s is 27,500 units: past full scale
    assert full[0] == pytest.approx(S.T_MAX, rel=1e-9)
    assert first(error=(0.0, 0.0, 5.0))[0] == pytest.approx(S.T_MAX, rel=1e-9)


def test_tr15_the_height_loops_integrals_carry_the_firmwares_gains():
    """A height error of 5 cm held for a second, the drone not moving. Position z is 2/0.5/0 and velocity z is 25/15/0, each
    integral taking in the current error before its output, as the firmware's pid.c does. After 100 steps that is 4,814
    motor units over hover, of which the two integrals carry 2,314."""
    ctl = CascadedPID(); ctl.reset()
    held = view(error=(0.0, 0.0, 0.05))
    i_pos = i_vel = 0.0
    for _ in range(100):
        out = np.asarray(ctl.act(held), dtype=float)
        i_pos += 0.05 * S.DT_CTRL
        correction = N.PID_POS_Z_KP * 0.05 + N.PID_POS_Z_KI * i_pos
        i_vel += correction * S.DT_CTRL
        raw = N.PID_VEL_Z_KP * correction + N.PID_VEL_Z_KI * i_vel
    assert raw == pytest.approx(4.814, abs=0.01)
    want = S.T_MAX * ((UNITS_HOVER + N.PID_THRUST_SCALE * raw) / S.UNITS_MAX) ** 2
    assert out[0] - S.WEIGHT == pytest.approx(want - S.WEIGHT, rel=TOL)
    no_integrals = S.T_MAX * ((UNITS_HOVER + 2500.0) / S.UNITS_MAX) ** 2      # what the proportional gains alone would ask
    assert out[0] > no_integrals + 0.6 * (want - no_integrals)


def test_tr15_yaw_is_held_at_the_references_zero():
    left = first(rotation=rot_z(math.radians(10)))
    right = first(rotation=rot_z(math.radians(-10)))
    assert left[3] < -0.3 and right[3] > 0.3                                  # turned left, it turns back right
    assert left[3] == pytest.approx(-right[3], rel=1e-6)
    assert abs(left[1]) < 1e-6 and abs(left[2]) < 1e-6 and left[0] == pytest.approx(S.WEIGHT, rel=1e-6)


def test_tr15_the_loops_work_in_the_drones_heading_frame_as_the_firmwares_do():
    """Turned 90 degrees left, an error along world +x lies along the body's -y, so the same lean shows on the other axis."""
    straight = first(error=(0.1, 0.0, 0.0))
    turned = CascadedPID(); turned.reset()
    out = np.asarray(turned.act(view(error=(0.1, 0.0, 0.0), rotation=rot_z(math.radians(90)))), dtype=float)
    assert straight[2] > 0.1
    assert out[1] == pytest.approx(straight[2], rel=1e-6) and abs(out[2]) < 1e-6      # +x in the world is -y in the body: roll positive


def test_tr15_the_integral_terms_build_while_an_error_is_held_and_reset_clears_them():
    ctl = CascadedPID(); ctl.reset()
    held = view(error=(0.1, 0.0, 0.05))
    outs = [np.asarray(ctl.act(held), dtype=float) for _ in range(100)]
    assert outs[-1][2] > 1.05 * outs[0][2] > 0.0                              # attitude 6/3/0 and velocity 25/1/0: the I terms add up
    assert outs[-1][0] > outs[0][0] + 0.002                                   # height 25/15/0: 15 per second on a held error
    assert all(b[2] >= a[2] - 1e-12 for a, b in zip(outs, outs[1:]))
    ctl.reset()
    np.testing.assert_allclose(ctl.act(held), outs[0], rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("name", N.CASE_NAMES)
def test_tr15_it_flies_each_of_the_twelve_cases_in_calm_air(name):
    """Section 17.2: the stock form "would be a strawman" (fig8-faster alone needs 27.7 degrees and 2.2 m/s). With the four
    changes it stays up and inside the radius on every case."""
    record = fly(Condition.calm(name), 20261105, CascadedPID())
    outcome = score(record.trajectory())
    assert outcome.passed, (name, outcome.crash_class, outcome.crash_time)
    assert record.n_steps == S.N_STEPS and tracking_rmse(record) < S.RADIUS_M
