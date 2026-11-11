"""Invariant 5 (spec sections 12, 17.4): a step command gives first-order rotor speed at 72 ms, within 1%; thrust goes
as speed squared; a weak motor scales that motor only, after the mixer.

The units-to-speed mapping the last test pins is 3A-3 (docs/invariant-table.md).
"""
import numpy as np
import pytest

import oracles as O
import spec_numbers as S
from drone_study.actuation import Actuation, MotorBank, omega_to_units, units_to_omega

HOVER = np.full(4, S.OMEGA_HOVER)


def run_step(dt, omega_cmd_scale, seconds=2.5):
    bank = MotorBank(dt, omega0=HOVER.copy())
    n = int(round(seconds / dt))
    cmd = HOVER * omega_cmd_scale
    out = np.empty((n, 4))
    for k in range(n):
        bank.step(cmd)
        out[k] = bank.omega
    t = (np.arange(n) + 1) * dt
    return t, out, HOVER[0], cmd[0]


@pytest.mark.parametrize("dt", (S.DT_PHYS, S.DT_CTRL))
@pytest.mark.parametrize("scale", (1.35, 0.7))          # spin-up and spin-down: the fit is symmetric (section 17.4)
def test_inv5_a_step_command_gives_first_order_rotor_speed_at_72_ms(dt, scale):
    t, w, w0, wc = run_step(dt, scale)
    step = wc - w0
    want = wc + (w0 - wc) * np.exp(-t / S.MOTOR_TAU_S)
    for m in range(4):
        assert np.max(np.abs(w[:, m] - want)) <= S.MOTOR_TOL * abs(step)           # within 1% of the step, throughout
    # the time constant itself: the response reaches 1 - 1/e of the step at t = tau
    frac = float(np.interp(S.MOTOR_TAU_S, t, (w[:, 0] - w0) / step))                  # the response at t = tau
    assert frac == pytest.approx(1 - np.exp(-1), abs=S.MOTOR_TOL)
    y = np.clip(1 - (w[:, 0] - w0) / step, 1e-12, None)
    sel = (t > 0.5 * S.MOTOR_TAU_S) & (t < 3 * S.MOTOR_TAU_S)
    slope = np.polyfit(t[sel], np.log(y[sel]), 1)[0]
    assert -1.0 / slope == pytest.approx(S.MOTOR_TAU_S, rel=S.MOTOR_TOL)             # fitted tau within 1% of 72 ms


@pytest.mark.parametrize("fraction", (0.25, 0.5, 1.0, 1.25, 1.5))
def test_inv5_thrust_goes_as_the_rotor_speed_squared(fraction):
    omega = HOVER * fraction
    bank = MotorBank(S.DT_PHYS, omega0=omega.copy())
    thrust = bank.step(omega)
    np.testing.assert_allclose(thrust, S.KF * omega ** 2, rtol=1e-9, atol=1e-15)
    doubled = MotorBank(S.DT_PHYS, omega0=2 * omega).step(2 * omega)
    np.testing.assert_allclose(doubled, 4 * thrust, rtol=1e-9)
    assert np.all(thrust > 0)


def test_inv5_four_hover_motors_carry_the_weight_and_four_full_motors_give_thrust_to_weight_2_25():
    hover = MotorBank(S.DT_PHYS, omega0=HOVER.copy()).step(HOVER)
    assert hover.sum() == pytest.approx(S.WEIGHT, rel=1e-9)
    full = np.full(4, S.OMEGA_MAX)
    top = MotorBank(S.DT_PHYS, omega0=full.copy()).step(full)
    assert top.sum() == pytest.approx(S.THRUST_TO_WEIGHT * S.WEIGHT, rel=1e-9)


@pytest.mark.parametrize("weak", range(4))
@pytest.mark.parametrize("scale", (0.6, 0.85))
def test_inv5_a_weak_motor_scales_that_motor_only(weak, scale):
    scales = np.ones(4); scales[weak] = scale
    healthy, sick = MotorBank(S.DT_PHYS, omega0=HOVER.copy()), MotorBank(S.DT_PHYS, scales, omega0=HOVER.copy())
    cmd = HOVER * np.array([1.2, 1.1, 1.3, 1.0])
    for k in range(1500):
        a, b = healthy.step(cmd), sick.step(cmd)
        others = [m for m in range(4) if m != weak]
        np.testing.assert_array_equal(b[others], a[others])            # no other motor feels it, at any step
    assert b[weak] == pytest.approx(scale * a[weak], rel=1e-6)          # settled: the weak motor gives `scale` of the thrust
    assert b[weak] < a[weak]


def test_inv5_the_scale_comes_after_the_mixer_which_does_not_know_it():
    healthy, sick = Actuation(S.DT_PHYS), Actuation(S.DT_PHYS, scales=(0.6, 1.0, 1.0, 1.0))
    request = (S.UNITS_MAX / 1.5, 3000.0, -2000.0, 4000.0)
    for _ in range(1500):
        a, b = healthy.step(*request), sick.step(*request)
        np.testing.assert_array_equal(sick.mixer_output, healthy.mixer_output)
    np.testing.assert_allclose(sick.mixer_output, O.mix_capped(*request), rtol=0, atol=1e-9)
    assert b[0] == pytest.approx(0.6 * a[0], rel=1e-6)
    np.testing.assert_allclose(b[1:], a[1:], rtol=1e-12)


def test_inv5_firmware_units_map_to_rotor_speed_as_the_ruling_states():
    """3A-3: 65,535 is the motor's maximum, and the command is linear in rotor speed, so thrust goes as units squared."""
    assert float(units_to_omega(S.UNITS_MAX)) == pytest.approx(S.OMEGA_MAX, rel=1e-12)
    assert float(units_to_omega(0.0)) == 0.0
    assert float(units_to_omega(S.UNITS_MAX / 2)) == pytest.approx(S.OMEGA_MAX / 2, rel=1e-12)
    assert float(units_to_omega(S.UNITS_MAX / 1.5)) == pytest.approx(S.OMEGA_HOVER, rel=1e-12)      # hover at 2/3 of full
    assert float(omega_to_units(S.OMEGA_HOVER)) == pytest.approx(S.UNITS_MAX / 1.5, rel=1e-12)
    bank = Actuation(S.DT_PHYS)
    for _ in range(3000):
        thrust = bank.step(S.UNITS_MAX / 1.5, 0.0, 0.0, 0.0)
    np.testing.assert_allclose(thrust, S.WEIGHT / 4, rtol=1e-6)         # hover units at settled speed carry the weight
