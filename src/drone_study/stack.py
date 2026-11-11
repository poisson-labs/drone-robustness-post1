"""The episode engine: physics, motors, mixer, rate loop, sensing, tapes, gust, one 100 Hz control step at a time.

Interface 3B inherits:
  StackEnv(condition, seed, *, mode="evaluation", sensing=SensingConfig(), air_model=PRIMARY, mixer=None)
     .tapes                  complete before the first step
     .reset() -> OuterReading
     .step(action) -> StepResult      action = [collective thrust N (0..Tmax), p, q, r rad/s]; one control step is
                                      five 500 Hz substeps; the action takes effect one control step later (10 ms)
     .truth                  TrueState        .time  seconds        .record  EpisodeRecord
     .privileged(allowed)    Privileged with only the allowed fields set (None for an empty set)
  mode "evaluation": the episode ends at 10 s or floor contact and flies on past the tilt and radius thresholds
  (spec section 4). mode "training": it also ends at the first crash class.
  fly(condition, seed, controller, ...) -> EpisodeRecord      run a Controller to the episode's end

How a control step runs. At step k (time t_k = 0.01 k) the controller sees the outer stream's reading and chooses an action.
`step(action)` flies step k with the action chosen at step k - 1 (the hover hold for k = 0: the one-step action latency of
section 17.4), queues the new action for step k + 1, and returns the reading at t_{k+1}. Each of the five 500 Hz substeps
runs, in this order, exactly as the rate-loop test composes them: the on-board gyro reads the true body rate, the rate PID
takes the command and that reading, the mixer and motors turn it into four rotor thrusts, and the physics steps with the
study's air force. Only the rate command and the collective come from the action; nothing from the outer stream reaches the
rate loop.

The record has one row per control step, quantities at the START of step k, plus a final row for the state a step ended in
when that state ended the episode (floor contact, or a training crash). An episode that times out has rows 0 to 999.

A non-finite action or state raises `NonFiniteState` at once (an error row for Prove, never a crash): the model has
MuJoCo's automatic reset disabled, so a blow-up shows as NaN instead of a quiet reset to the start pose.

Additive to the stub: `fly(..., air_model=, start_attitude=)`, `StackEnv(..., start_attitude=)`, `Controller.reset()`,
`StackEnv.reference_ahead()` and friends.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import mujoco
import numpy as np

from drone_study.actuation import Actuation
from drone_study.air import PRIMARY, AirModel, air_wrench_world
from drone_study.cases import case as get_case
from drone_study.condition import Condition
from drone_study.physics import Physics
from drone_study.rate_loop import RateLoop
from drone_study.score import PRIMARY_LIMITS, NonFiniteState, ScoreTracker, Trajectory
from drone_study.sensing import OnboardGyro, OuterReading, OuterStream, SensingConfig, TrueState
from drone_study.streams import N_STEPS, Tapes, build_tapes
from drone_study.vehicle import (DT_CTRL, DT_PHYS, G, KF, SUBSTEPS_PER_STEP, T_MAX, UNITS_MAX, WEIGHT)

# Section 2: "Episodes start on the reference, at its velocity and the attitude its acceleration needs, with section 17.9's
# spread." That is the default, and Taylor's ruling 3B-1 (pick A, 2026-11-05) keeps it. "level" starts flat (plus the same
# spread); it is a switch for checks and is not what the study flies. (An open-loop hold at thrust = weight lands from the
# default start at 4.8 s on circle-fast, so the one frozen test that holds a hover carries the tilt.)
START_ATTITUDE = "acceleration"
HOLD_ACTION = np.array([WEIGHT, 0.0, 0.0, 0.0])      # nothing chosen yet: the hover hold
RATE_LIMIT = 10.0                                     # rad/s, section 2
LOOKAHEAD_S = 0.1                                     # reference points ahead are 100 ms apart
N_AHEAD = 5
_UNITS_PER_ROOT_NEWTON = UNITS_MAX / math.sqrt(T_MAX)


@dataclass(frozen=True)
class Privileged:
    mass: Optional[float] = None
    air_force: Optional[np.ndarray] = None
    state: Optional[TrueState] = None
    motor_scales: Optional[np.ndarray] = None


@dataclass(frozen=True)
class ControllerView:
    t: float
    reading: OuterReading
    reference: np.ndarray            # (5, 3) reference points at 100 ms spacing from now (t, t + 0.1 ... t + 0.4), world frame
    privileged: Optional[Privileged]


class Controller:
    privileges: frozenset = frozenset()

    def mixer(self, motor_scales):
        return None

    def reset(self) -> None:
        """Called once at the start of each episode by `fly`, so a controller shared across episodes (a policy closed over
        by the rollout) starts each from the same state."""

    def act(self, view: ControllerView) -> np.ndarray:
        return np.zeros(4)


class ScriptedController(Controller):
    """Calls fn(view) -> action. Observation-ignoring scripts make an open-loop test policy."""

    def __init__(self, fn, privileges: frozenset = frozenset()):
        self.fn = fn
        self.privileges = privileges

    def act(self, view: ControllerView) -> np.ndarray:
        return np.asarray(self.fn(view), dtype=float)


@dataclass
class EpisodeRecord:
    """One row per control step k (time t_k = 0.01 k), quantities at the START of step k."""
    t: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    quat: np.ndarray
    body_rate: np.ndarray
    rotor_omega: np.ndarray
    air_velocity: np.ndarray       # world frame, gust + turbulence
    applied_action: np.ndarray     # (n, 4) the action in effect during the step
    mixer_output: np.ndarray       # (n, 4) units, last substep of the previous step
    reference_position: np.ndarray
    floor_contact: np.ndarray
    end_reason: str = "time"
    # Additive (ruling 4DB-2): one entry per control step FLOWN, step k at index k (not one per row): True when any of the four
    # mixer commands is at 0 or 65,535 units on any of the step's five 500 Hz substeps. None for a record built by hand.
    motor_limit_substep: Optional[np.ndarray] = None

    @property
    def n_steps(self) -> int:
        return len(self.t)

    def trajectory(self) -> Trajectory:
        return Trajectory(self.t, self.position, self.quat, self.reference_position, self.floor_contact)


@dataclass(frozen=True)
class StepResult:
    reading: OuterReading
    done: bool
    end_reason: Optional[str]


_T_GRID = np.round(np.arange(N_STEPS + 1) * DT_CTRL, 10)
_AHEAD_OFFSETS = LOOKAHEAD_S * np.arange(0, N_AHEAD)
_ROW_FIELDS = ("position", "velocity", "quat", "body_rate", "rotor_omega", "air_velocity", "applied_action",
               "mixer_output", "reference_position")


def _desired_rotation(acc: np.ndarray) -> np.ndarray:
    """Body-to-world rotation with yaw 0 whose thrust axis points along the acceleration the reference needs plus gravity."""
    b3 = acc + np.array([0.0, 0.0, G])
    b3 = b3 / np.linalg.norm(b3)
    b2 = np.cross(b3, [1.0, 0.0, 0.0])
    b2 = b2 / np.linalg.norm(b2)
    b1 = np.cross(b2, b3)
    return np.stack([b1, b2, b3], axis=1)


def _small_rotation(roll: float, pitch: float) -> np.ndarray:
    cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    return Rx @ Ry


class StackEnv:
    def __init__(self, condition: Condition, seed: int, *, mode: str = "evaluation",
                 sensing: SensingConfig = SensingConfig(), air_model: AirModel = PRIMARY, mixer=None,
                 start_attitude: str = START_ATTITUDE):
        if mode not in ("evaluation", "training"):
            raise ValueError(f"mode must be 'evaluation' or 'training', got {mode!r}")
        if start_attitude not in ("acceleration", "level"):
            raise ValueError(f"start_attitude must be 'acceleration' or 'level', got {start_attitude!r}")
        if condition.delay % 10 != 0 or condition.delay < 0:
            raise ValueError(f"delay is whole 10 ms steps, got {condition.delay} ms")
        self.condition, self.seed, self.mode, self.sensing = condition, seed, mode, sensing
        self.start_attitude = start_attitude
        self.tapes: Tapes = build_tapes(condition, seed)
        self.case = get_case(condition.case)
        self.physics = Physics(condition, seed, air_model=air_model, tapes=self.tapes)
        self.motor_scales = condition.motor_scales()
        self.actuation = Actuation(DT_PHYS, self.motor_scales, mixer)
        self.rate_loop = RateLoop(DT_PHYS)
        self.gyro = OnboardGyro(self.tapes.sensor, scale=sensing.gyro_scale, dt=DT_PHYS)
        self.outer = OuterStream(self.tapes.sensor, delay_steps=condition.delay // 10, scale=sensing.outer_scale)
        self._ref_pos, self._ref_vel, self._ref_acc = self.case._state(_T_GRID)
        self._ref_ahead = self.case.position(_T_GRID[:, None] + _AHEAD_OFFSETS[None, :])
        self._ref_ahead[:, 0, :] = self._ref_pos          # the first point is now, bit for bit (ruling 3B-2; 4B's policy reads it)
        n = N_STEPS + 1
        self._rows = {
            "t": _T_GRID,
            "position": np.zeros((n, 3)), "velocity": np.zeros((n, 3)), "quat": np.zeros((n, 4)),
            "body_rate": np.zeros((n, 3)), "rotor_omega": np.zeros((n, 4)), "air_velocity": np.zeros((n, 3)),
            "applied_action": np.zeros((n, 4)), "mixer_output": np.zeros((n, 4)),
            "reference_position": self._ref_pos, "floor_contact": np.zeros(n, dtype=bool),
        }
        self._n_rows = 0
        self._k = 0
        self._motor_limit = np.zeros(N_STEPS, dtype=bool)   # per control step flown: a motor at a limit on any substep (4DB-2)
        self._pending = HOLD_ACTION.copy()
        self._done = False
        self._end_reason = "running"
        self._tracker = ScoreTracker(PRIMARY_LIMITS) if mode == "training" else None
        self._contact = False
        self.time = 0.0

    # ---- views of the state ------------------------------------------------------------------------------------------
    @property
    def truth(self) -> TrueState:
        d = self.physics.data
        return TrueState(np.array(d.qpos[0:3]), np.array(d.qvel[0:3]), self.physics.rotation(), np.array(d.qvel[3:6]))

    @property
    def record(self) -> EpisodeRecord:
        n = self._n_rows
        r = self._rows
        return EpisodeRecord(
            t=r["t"][:n], position=r["position"][:n], velocity=r["velocity"][:n], quat=r["quat"][:n],
            body_rate=r["body_rate"][:n], rotor_omega=r["rotor_omega"][:n], air_velocity=r["air_velocity"][:n],
            applied_action=r["applied_action"][:n], mixer_output=r["mixer_output"][:n],
            reference_position=r["reference_position"][:n], floor_contact=r["floor_contact"][:n],
            end_reason=self._end_reason, motor_limit_substep=self._motor_limit[:self._k].copy(),
        )

    @property
    def step_index(self) -> int:
        return self._k

    def reference_now(self) -> np.ndarray:
        return self._ref_pos[min(self._k, N_STEPS)]

    def reference_ahead(self) -> np.ndarray:
        """(5, 3): the reference at t, t + 0.1 ... t + 0.4 s, world frame. The first point is now, so a controller reads the
        reference's velocity and acceleration from the five accurately (ruling 3B-2)."""
        return self._ref_ahead[min(self._k, N_STEPS)]

    def privileged(self, allowed) -> Optional[Privileged]:
        allowed = frozenset(allowed)
        if not allowed:
            return None
        air_force = None
        if "air_force" in allowed:
            ph = self.physics
            j = min(self._k * SUBSTEPS_PER_STEP, len(ph.air_table) - 1)
            air_force, _ = air_wrench_world(ph.air_table[j], ph.data.qvel[0:3], ph.rotation(), self.actuation.motors.omega,
                                            ph.air_model)
        return Privileged(
            mass=self.physics.mass if "mass" in allowed else None,
            air_force=air_force,
            state=self.truth if "state" in allowed else None,
            motor_scales=self.motor_scales.copy() if "motor_scales" in allowed else None,
        )

    # ---- the episode -------------------------------------------------------------------------------------------------
    def _write_row(self) -> None:
        k, d, r = self._n_rows, self.physics.data, self._rows
        r["position"][k] = d.qpos[0:3]
        r["velocity"][k] = d.qvel[0:3]
        r["quat"][k] = d.qpos[3:7]
        r["body_rate"][k] = d.qvel[3:6]
        r["rotor_omega"][k] = self.actuation.motors.omega
        r["air_velocity"][k] = self.physics.air_table[min(k * SUBSTEPS_PER_STEP, len(self.physics.air_table) - 1)]
        r["applied_action"][k] = self._pending
        r["mixer_output"][k] = self.actuation.mixer_output
        r["floor_contact"][k] = self._contact
        self._n_rows = k + 1

    def reset(self) -> OuterReading:
        ph, start = self.physics, self.tapes.start
        position = self._ref_pos[0] + start.position
        velocity = self._ref_vel[0] + start.velocity
        base = _desired_rotation(self._ref_acc[0]) if self.start_attitude == "acceleration" else np.eye(3)
        R = base @ _small_rotation(float(start.tilt[0]), float(start.tilt[1]))
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, R.reshape(9))
        ph.set_time(0.0)
        ph.set_state(position, quat, velocity, np.zeros(3))
        accel = self._ref_acc[0] if self.start_attitude == "acceleration" else np.zeros(3)
        needed = float(np.linalg.norm(accel + np.array([0.0, 0.0, G]))) / G * WEIGHT
        self.actuation.motors.omega = np.full(4, math.sqrt(needed / (4.0 * KF)))
        self.actuation.mixer_output = np.zeros(4)
        self.rate_loop.reset()
        self._k, self._n_rows = 0, 0
        self._motor_limit[:] = False
        self._pending = HOLD_ACTION.copy()
        self._done, self._end_reason, self._contact = False, "running", ph.floor_contact
        self.time = 0.0
        self._write_row()
        self.outer.push(self.truth)
        return self.outer.read()

    def step(self, action) -> StepResult:
        if self._done:
            raise RuntimeError("the episode has ended; call reset()")
        applied = self._pending
        if not np.all(np.isfinite(applied)):
            raise NonFiniteState(f"non-finite action {applied!r}")
        thrust_n = min(max(float(applied[0]), 0.0), T_MAX)
        units = _UNITS_PER_ROOT_NEWTON * math.sqrt(thrust_n)
        rate_cmd = np.clip(applied[1:4], -RATE_LIMIT, RATE_LIMIT)
        ph, act, gyro, loop = self.physics, self.actuation, self.gyro, self.rate_loop
        qvel = ph.data.qvel
        contact = self._contact
        at_limit = False
        for _ in range(SUBSTEPS_PER_STEP):
            roll, pitch, yaw = loop.step(rate_cmd, gyro.measure(qvel[3:6]))
            ph.substep(act.step(units, roll, pitch, yaw), act.motors.omega)
            if not at_limit:                                # reads the command; changes nothing in the flight (4DB-2)
                u = np.asarray(act.mixer_output)
                at_limit = bool(u.min() <= 0.0 or u.max() >= UNITS_MAX)
            contact = contact or ph.floor_contact
        self._motor_limit[self._k] = at_limit
        if not (np.isfinite(ph.data.qpos).all() and np.isfinite(ph.data.qvel).all()):
            raise NonFiniteState(f"non-finite state at step {self._k + 1}")
        self._contact = contact
        self._pending = np.array(action, dtype=float)
        self._k += 1
        k = self._k
        self.time = float(_T_GRID[k])
        done, reason = False, None
        if k >= N_STEPS and not contact:
            done, reason = True, "time"
        else:
            self._write_row()
            if self._tracker is not None:
                fired = self._tracker.update(self.time, ph.data.qpos[0:3], ph.data.qpos[3:7], self._ref_pos[k], contact)
                if fired is not None:
                    done, reason = True, f"training-crash:{fired[0]}"
            if not done and contact:
                done, reason = True, "ground"
        self.outer.push(self.truth)
        self._done = done
        if done:
            self._end_reason = reason
        return StepResult(self.outer.read(), done, reason)


def fly(condition: Condition, seed: int, controller: Controller, *, mode: str = "evaluation",
        sensing: SensingConfig = SensingConfig(), max_steps: int = N_STEPS, air_model: AirModel = PRIMARY,
        start_attitude: str = START_ATTITUDE) -> EpisodeRecord:
    env = StackEnv(condition, seed, mode=mode, sensing=sensing, air_model=air_model,
                   mixer=controller.mixer(condition.motor_scales()), start_attitude=start_attitude)
    controller.reset()
    reading = env.reset()
    for _ in range(max_steps):
        view = ControllerView(env.time, reading, env.reference_ahead(), env.privileged(controller.privileges))
        result = env.step(controller.act(view))
        reading = result.reading
        if result.done:
            break
    return env.record
