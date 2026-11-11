"""Checks session 3B added beside the frozen tests. These are NOT the frozen tests (`tests/` is untouched); they pin the places
where the build made a reading the frozen tests do not reach, so a later session that changes one sees it.

    uv run pytest tests_3b
"""
import math

import numpy as np
import pytest

from drone_study.cases import CASE_NAMES, case
from drone_study.condition import Condition
from drone_study.controllers import FeasibilityOracle, GeometricController, reference_state
from drone_study.score import PRIMARY_LIMITS, ScoreTracker, Trajectory, score
from drone_study.stack import ScriptedController, fly
from drone_study.vehicle import KF, KM, MASS, T_MAX, WEIGHT


# ---- ruling 3B-1 (A, Taylor, 2026-11-05): the start attitude --------------------------------------------------------------------
def test_the_start_attitude_is_the_one_the_reference_acceleration_needs_and_the_frozen_hover_carries_it():
    """Spec section 2 starts an episode at the attitude the reference's acceleration needs (19 degrees on circle-fast). A hold at
    thrust = weight with zero rates lands from there at 4.8 s, which is why the frozen test
    `test_inv10_evaluation_stops_at_floor_contact_and_runs_to_ten_seconds_otherwise` was changed under the ruling: its hold now
    carries the tilt (thrust whose vertical part is the weight). `start_attitude="level"` stays as a switch and is not the default."""
    from drone_study.score import tilt_degrees

    cond = Condition(case="circle-fast", turb=0.3)
    plain = ScriptedController(lambda v: [WEIGHT, 0.0, 0.0, 0.0])
    tilted = fly(cond, 2, plain)
    assert 15.0 < float(tilt_degrees(tilted.quat[0])) < 25.0
    assert tilted.end_reason == "ground" and tilted.t[-1] < 6.0
    carrying = fly(cond, 2, ScriptedController(lambda v: [WEIGHT / v.reading.rotation[2, 2], 0.0, 0.0, 0.0]))
    assert carrying.end_reason == "time" and carrying.n_steps == 1000
    level = fly(cond, 2, plain, start_attitude="level")
    assert float(tilt_degrees(level.quat[0])) < 5.0 and level.end_reason == "time"


# ---- the incremental scorer is the batch scorer -----------------------------------------------------------------------------
def _wild(view):
    return [1.4 * WEIGHT, 2.0, 0.0, 0.0] if view.t < 0.5 else [T_MAX, 0.0, 0.0, 0.0]


@pytest.mark.parametrize("seed", range(6))
def test_the_tracker_finds_the_crash_score_finds(seed):
    cond = Condition(case=CASE_NAMES[seed], az=45.0 * seed, el=0.0, speed=10.0, phase=0.0)
    for controller in (ScriptedController(_wild), GeometricController()):
        rec = fly(cond, seed, controller)
        want = score(rec.trajectory())
        tracker = ScoreTracker(PRIMARY_LIMITS)
        got = None
        for k in range(rec.n_steps):
            got = tracker.update(rec.t[k], rec.position[k], rec.quat[k], rec.reference_position[k], bool(rec.floor_contact[k]))
            if got is not None:
                break
        if want.passed:
            assert got is None
        else:
            assert got == (want.crash_class, want.crash_time)


def test_a_training_episode_is_the_evaluation_episode_up_to_its_crash():
    cond = Condition(case="circle-fast", turb=0.3)
    ev, tr = fly(cond, 6, ScriptedController(_wild)), fly(cond, 6, ScriptedController(_wild), mode="training")
    n = tr.n_steps
    assert tr.end_reason.startswith("training-crash") and n < ev.n_steps
    for name in ("position", "quat", "body_rate", "rotor_omega", "applied_action", "mixer_output", "air_velocity"):
        np.testing.assert_array_equal(getattr(tr, name), getattr(ev, name)[:n], err_msg=name)


def test_a_flight_is_the_same_twice_in_one_process():
    cond = Condition(case="oval-fast", az=90.0, el=35.0, speed=9.0, phase=0.125, payload=7.5, motor_index=2, motor_scale=0.8, delay=30)
    a, b = fly(cond, 77, GeometricController()), fly(cond, 77, GeometricController())
    for name in ("position", "quat", "body_rate", "rotor_omega", "mixer_output", "applied_action", "air_velocity"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)


# ---- the controllers read the reference as the policy does ------------------------------------------------------------------
@pytest.mark.parametrize("name", CASE_NAMES)
def test_the_five_reference_points_give_the_reference_state_now(name):
    """Ruling 3B-2: the five points are the reference at t, t + 0.1 ... t + 0.4 s. The first is now, so position is exact and the
    degree-4 fit through them reads velocity within 5 mm/s and acceleration within 0.15 m/s^2 (peak 5.1) on every case. Taken
    from t + 0.1 ... t + 0.5 instead, the same fit is off by 3 cm/s and 0.5 m/s^2."""
    c = case(name)
    for t in np.linspace(0.0, 9.0, 31):
        p, v, a = reference_state(c.position(t + 0.1 * np.arange(0, 5)))
        np.testing.assert_allclose(p, c.position(np.array(t)), atol=1e-9)
        np.testing.assert_allclose(v, c.velocity(np.array(t)), atol=5e-3)
        np.testing.assert_allclose(a, c.acceleration(np.array(t)), atol=0.15)


def test_the_oracle_and_the_reference_share_gains_and_controller_state_resets_between_episodes():
    ctrl = GeometricController()
    cond = Condition(case="circle-fast", turb=0.3)
    first = fly(cond, 3, ctrl)
    second = fly(cond, 3, ctrl)                                    # the same instance again: reset() clears the integrator
    np.testing.assert_array_equal(first.position, second.position)
    assert FeasibilityOracle().gains == ctrl.gains


# ---- a code fault is an error row, never a quiet reset ---------------------------------------------------------------------
def test_a_non_finite_action_raises_and_a_blow_up_is_not_silently_reset():
    from drone_study.score import NonFiniteState
    from drone_study.stack import StackEnv

    env = StackEnv(Condition.calm("circle-fast"), 1)
    env.reset()
    env.step([WEIGHT, 0.0, 0.0, 0.0])
    env.step([float("nan"), 0.0, 0.0, 0.0])               # queued; flies in the next step
    with pytest.raises(NonFiniteState):
        env.step([WEIGHT, 0.0, 0.0, 0.0])


# ---- the vehicle is the simulator's -------------------------------------------------------------------------------------------
def test_the_vehicle_constants_are_the_simulators_at_the_pinned_commit():
    from multi_drone_mujoco.envs.base_aviary import DRONE_PARAMS
    from multi_drone_mujoco.utils.enums import DroneModel

    p = DRONE_PARAMS[DroneModel.CF2X]
    assert p["mass"] == MASS and p["thrust2weight_ratio"] == 2.25
    assert KF == pytest.approx(p["kf"] / (2 * math.pi / 60) ** 2, rel=1e-15)
    assert KM == pytest.approx(p["km"] / (2 * math.pi / 60) ** 2, rel=1e-15)
