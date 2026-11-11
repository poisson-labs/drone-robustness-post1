"""The MuJoCo rigid body at 500 Hz, with the study's own air force (spec sections 12 inv. 3 and 4, 17.1, 17.3).

The body is the simulator's: `BaseAviary` at the pinned commit generates the CF2X model (mass 0.027 kg, the inertia,
the collision cylinder, the floor plane). The study then turns the simulator's fluid model OFF (density 0, viscosity 0,
`opt.wind` unused: its inertia-box force would add viscous and torque terms S1 cannot switch) and applies its own air force
through `xfrc_applied`, together with the rotor wrench, as ONE assignment per substep. The shipped `_physics` assigns
`xfrc_applied` from the rotor speeds before MuJoCo steps and nothing else touches it, so a wind added anywhere else would be
overwritten or applied late; this is the test (inv. 4) the shipped wrapper fails.

Interface 3B inherits:
  Physics(condition, seed, air_model=PRIMARY, physics_hz=500)
  .model, .data           the mujoco model and data; free joint: qvel[0:3] world linear velocity, qvel[3:6] angular
                          velocity in the BODY frame, qpos[3:7] quaternion w,x,y,z
  .time                   start time of the next substep, seconds; .set_time(t) sets it
  .set_state(...)         position, quaternion (w,x,y,z), world linear velocity, body-frame angular velocity
  .substep(thrust, omega) advance one substep of 1 / physics_hz. thrust: four rotor thrusts in N (weak-motor
                          scale already applied); omega: four rotor speeds in rad/s. The air force is evaluated
                          at the substep's START time, from gust(t) + turbulence tape(index of t), at the start-of-substep
                          velocity and attitude, and applied through xfrc_applied together with the rotor wrench.
Additive keyword (not in the stub): `tapes=` lets the episode engine share the tapes it already built.

A payload is added mass at the centre of mass, inertia left alone (section 3).
"""
from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from drone_study.air import PRIMARY, AirModel, air_wrench_scalars
from drone_study.condition import Condition
from drone_study.gust import Gust
from drone_study.streams import N_SUBSTEPS, Tapes, build_tapes
from drone_study.vehicle import ARM_OVER_ROOT2, HEIGHT_M, KM_OVER_KF, MASS

TAPE_DT = 0.002                       # the tapes are written at 500 Hz
_ROOT = Path(__file__).resolve().parents[2]
_BASE: dict[int, tuple] = {}


def _assets_dir() -> Path:
    env = os.environ.get("MJ_DRONES_CF2_ASSETS")
    if env and (Path(env) / "cf2_0.obj").exists():
        return Path(env)
    carried = _ROOT / "assets" / "cf2"                                    # the meshes this repository carries (MIT, see THIRD_PARTY_NOTICES.md)
    if (carried / "cf2_0.obj").exists():
        return carried
    local = _ROOT / ".work" / "sim-src" / "multi_drone_mujoco" / "assets" / "cf2"
    if (local / "cf2_0.obj").exists():
        return local
    raise FileNotFoundError("the simulator's mesh files are missing: run scripts/fetch_sim_assets.sh (it fetches the pinned commit)")


def _base_model(physics_hz: int):
    """The compiled model for this rate, built once per process through the simulator's own `BaseAviary`."""
    if physics_hz not in _BASE:
        os.environ["MJ_DRONES_CF2_ASSETS"] = str(_assets_dir())
        from multi_drone_mujoco.envs.base_aviary import BaseAviary
        from multi_drone_mujoco.utils.enums import DroneModel
        from multi_drone_mujoco.utils.enums import Physics as SimPhysics

        aviary = BaseAviary(
            drone_model=DroneModel.CF2X, num_drones=1, initial_xyzs=np.array([[0.0, 0.0, HEIGHT_M]]),
            sim_freq=physics_hz, ctrl_freq=physics_hz, physics=SimPhysics.MJC,
        )
        model = aviary.model
        model.opt.density = 0.0
        model.opt.viscosity = 0.0
        model.opt.wind[:] = 0.0
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_AUTORESET    # a blow-up must show as NaN, not a quiet reset
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "drone0")
        _BASE[physics_hz] = (model, body)
    return _BASE[physics_hz]


@dataclass(frozen=True)
class SubstepRecord:
    time: float
    air_velocity: np.ndarray       # world frame, gust + turbulence, m/s
    air_force_world: np.ndarray    # N
    air_moment_world: np.ndarray   # N m, about the centre of mass


class Physics:
    def __init__(self, condition: Condition, seed: int, *, air_model: AirModel = PRIMARY, physics_hz: int = 500,
                 tapes: Tapes | None = None):
        base, self._body = _base_model(physics_hz)
        self.condition = condition
        self.seed = seed
        self.air_model = air_model
        self.dt = 1.0 / physics_hz
        self.model = copy.copy(base)
        self.model.opt.density = 0.0
        self.model.opt.viscosity = 0.0
        self.model.opt.wind[:] = 0.0
        self.mass = MASS + float(condition.payload) / 1000.0
        self.model.body_mass[self._body] = self.mass
        self.data = mujoco.MjData(self.model)
        self.tapes = tapes if tapes is not None else build_tapes(condition, seed)
        self.gust = Gust.from_condition(condition)
        self.time = 0.0
        self._turb = self.tapes.turbulence
        self._has_gust = condition.has_gust
        grid = np.arange(N_SUBSTEPS) * TAPE_DT
        table = np.array(self._turb)
        if self._has_gust:
            table = table + self.gust.air_velocity(grid)
        table.setflags(write=False)
        self.air_table = table                  # air velocity at each 500 Hz tape time: gust + turbulence
        self._xfrc = self.data.xfrc_applied[self._body]
        self.last_substep: SubstepRecord | None = None
        mujoco.mj_forward(self.model, self.data)

    # ---- state -------------------------------------------------------------------------------------------------------
    def set_time(self, t: float) -> None:
        self.time = float(t)

    def set_state(self, position, quat_wxyz, velocity_world, angular_velocity_body) -> None:
        self.data.qpos[0:3] = position
        self.data.qpos[3:7] = quat_wxyz
        self.data.qvel[0:3] = velocity_world
        self.data.qvel[3:6] = angular_velocity_body
        mujoco.mj_forward(self.model, self.data)

    def rotation(self) -> np.ndarray:
        """Body-to-world rotation at the CURRENT quaternion (`data.xmat` is one step stale after a `mj_step`)."""
        r = np.empty(9)
        mujoco.mju_quat2Mat(r, self.data.qpos[3:7])
        return r.reshape(3, 3)

    @property
    def floor_contact(self) -> bool:
        return self.data.ncon > 0

    # ---- the air -----------------------------------------------------------------------------------------------------
    def air_velocity_at(self, t: float) -> np.ndarray:
        """Gust (analytic, at time t) plus the turbulence tape at the nearest 500 Hz index. On a tape time this is the
        precomputed table (the same arithmetic); between tape times the gust is evaluated at t itself."""
        idx = int(round(t / TAPE_DT))
        if 0 <= idx < N_SUBSTEPS and abs(t - idx * TAPE_DT) <= 1e-9:
            return self.air_table[idx]
        v = self._turb[min(max(idx, 0), N_SUBSTEPS - 1)]
        if self._has_gust:
            v = v + self.gust.air_velocity(t)
        return np.array(v)

    # ---- one substep -------------------------------------------------------------------------------------------------
    def substep(self, thrust, omega) -> SubstepRecord:
        t = self.time
        data = self.data
        v_air = self.air_velocity_at(t)
        R = self.rotation()
        v_rel = R.T @ (v_air - data.qvel[0:3])
        fx, fy, fz, mx, my = air_wrench_scalars(float(v_rel[0]), float(v_rel[1]), float(v_rel[2]), float(np.sum(omega)),
                                                self.air_model)
        f_air = R @ np.array([fx, fy, fz])
        m_air = R @ np.array([mx, my, 0.0])
        thr = np.asarray(thrust, dtype=float)
        f0, f1, f2, f3 = thr
        total = f0 + f1 + f2 + f3
        torque_body = np.array([(f0 + f1 - f2 - f3) * ARM_OVER_ROOT2, (-f0 + f1 + f2 - f3) * ARM_OVER_ROOT2,
                                KM_OVER_KF * (-f0 + f1 - f2 + f3)])
        xfrc = self._xfrc
        xfrc[0:3] = R[:, 2] * total + f_air
        xfrc[3:6] = R @ torque_body + m_air
        mujoco.mj_step(self.model, data)
        self.time = t + self.dt
        rec = SubstepRecord(t, np.array(v_air), f_air, m_air)
        self.last_substep = rec
        return rec
