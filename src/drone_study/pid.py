"""The cascaded PID: the firmware's structure and gains, with section 17.2's four stated changes (spec 17.2, R8; 4A-17).

The bar reads it: a nominal policy's calm-air RMSE may be at most 1.25 times this controller's on the same seeds
(section 2). It runs at 100 Hz on the outer stream, with no privilege, and outputs the action a policy does:
[collective thrust N, p, q, r rad/s], into the shared rate loop. Firmware: `position_controller_pid.c`,
`attitude_pid_controller.c`, `pid.c` and `platform_defaults_cf2.h` at `e828ed2`.

  position loop    velocity setpoint = reference velocity + Kp_pos x position error (z: + Ki_pos_z x its integral), the
                   correction clamped at 1.1 m/s on each world axis (firmware: 1 m/s plus 10% overhead). The firmware's
                   position PID output IS the velocity setpoint; adding the reference velocity is stated change 2.
  velocity loop    world frame, as the firmware's: lean in degrees = Kp_vel x velocity error + Ki_vel x its integral on
                   each horizontal axis, plus stated change 3, the feedforward atan(a_ref / g) on each axis; the two
                   leans are turned into the drone's heading frame (yaw) to give the roll and pitch setpoints, as the
                   firmware does. Stated change 1: the total tilt, the angle of the thrust axis the two setpoints give,
                   is limited to 55 degrees (the firmware's two per-axis limits, 22 and 20 degrees, give way to it).
  height           thrust in firmware units = true nominal hover (43,690: thrust as units squared, 3A-3) + 1,000 x
                   (Kp_vz x vertical velocity error + Ki_vz x its integral), floored at the firmware's 20,000 and
                   capped at 65,535; turned into newtons; stated change 4: divided by the cosine of the measured tilt,
                   within the motors' limit.
  attitude loop    rate command = Kp_att x attitude error + Ki_att x its integral, on the Euler (ZYX) roll and pitch
                   errors in radians. Yaw is held at the reference's 0 with the yaw gain the geometric controller uses
                   (2.0 rad/s per rad, `controllers.YAW_KP`; section 17.4: yaw is the weak axis). The spec lists no yaw
                   gains and 4A-17 leaves them unpinned; the firmware's 6/1/0.35, run here at 100 Hz on the delayed
                   outer stream over this yaw rate loop (which settles in 0.87 s), oscillates with a growing swing of
                   about 1.6 s period, and its rate commands eat the collective in the mixer until the drone sinks on
                   full thrust (session 4B's trace, calm fig8-medium). Section 17.2 wants only the outer loop's design
                   to differ between the classical controllers and the policy, so the yaw axis is shared.
  integrals        each takes in the current error before its output, as pid.c does; limits are the firmware's:
                   5000 (pid.c's default) on the position and velocity loops, 20 degrees on roll and pitch.

Interface (4A-19):
  CascadedPID()        a `drone_study.stack.Controller`: privileges empty, the firmware mixer, reset(), act(view)
  the gains, by the firmware's names: POS_KP, POS_Z_KP, POS_Z_KI, VEL_KP, VEL_KI, VEL_Z_KP, VEL_Z_KI, ATT_KP, ATT_KI,
  VEL_CORRECTION_MAX, TILT_LIMIT_DEG, THRUST_SCALE
"""
from __future__ import annotations

import math

import numpy as np

from drone_study.controllers import YAW_KP, reference_state
from drone_study.stack import Controller, ControllerView
from drone_study.vehicle import DT_CTRL, G, T_MAX, UNITS_HOVER, UNITS_MAX

POS_KP = 2.0                       # position x, y: 2/0/0
POS_Z_KP, POS_Z_KI = 2.0, 0.5      # position z: 2/0.5/0
VEL_KP, VEL_KI = 25.0, 1.0         # velocity x, y: 25/1/0, output in degrees of lean
VEL_Z_KP, VEL_Z_KI = 25.0, 15.0    # velocity z: 25/15/0, output in thousands of motor units
ATT_KP, ATT_KI = 6.0, 3.0          # attitude roll, pitch: 6/3/0, rate per radian of error
VEL_CORRECTION_MAX = 1.1           # m/s, each axis (xyVelMax 1.0 x velMaxOverhead 1.10)
TILT_LIMIT_DEG = 55.0
THRUST_SCALE = 1000.0              # motor units per unit of the height loop's output
THRUST_MIN_UNITS = 20000.0         # the firmware's thrust floor
RATE_LIMIT = 10.0                  # rad/s, section 2
I_LIMIT_DEFAULT = 5000.0           # pid.c DEFAULT_PID_INTEGRATION_LIMIT
I_LIMIT_ATT = math.radians(20.0)   # PID_ROLL/PITCH_INTEGRATION_LIMIT, in radians here
_TILT_LIMIT = math.radians(TILT_LIMIT_DEG)
_AXIS_CAP = math.radians(89.0)
_MIN_COS_TILT = 0.1


def _clamp(x: float, limit: float) -> float:
    return min(max(x, -limit), limit)


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _limit_total_tilt(roll: float, pitch: float) -> tuple:
    """Scale the two setpoints together so the thrust axis they give leans at most 55 degrees (ZYX: cos tilt = cos roll cos pitch)."""
    roll, pitch = _clamp(roll, _AXIS_CAP), _clamp(pitch, _AXIS_CAP)
    if math.acos(min(1.0, max(-1.0, math.cos(roll) * math.cos(pitch)))) <= _TILT_LIMIT:
        return roll, pitch
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if math.acos(math.cos(mid * roll) * math.cos(mid * pitch)) > _TILT_LIMIT:
            hi = mid
        else:
            lo = mid
    return lo * roll, lo * pitch


class CascadedPID(Controller):
    privileges = frozenset()

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._i_pos_z = 0.0
        self._i_vel = np.zeros(3)
        self._i_att = np.zeros(2)

    def act(self, view: ControllerView) -> np.ndarray:
        reading = view.reading
        R = np.asarray(reading.rotation, dtype=float)
        position, velocity = np.asarray(reading.position, dtype=float), np.asarray(reading.velocity, dtype=float)
        p_ref, v_ref, a_ref = reference_state(np.asarray(view.reference, dtype=float))
        yaw = math.atan2(R[1, 0], R[0, 0])
        pitch = math.asin(min(1.0, max(-1.0, -R[2, 0])))
        roll = math.atan2(R[2, 1], R[2, 2])

        # position loop (world frame): the correction, clamped per axis, added to the reference velocity
        e = p_ref - position
        self._i_pos_z = _clamp(self._i_pos_z + e[2] * DT_CTRL, I_LIMIT_DEFAULT)
        correction = np.array([_clamp(POS_KP * e[0], VEL_CORRECTION_MAX), _clamp(POS_KP * e[1], VEL_CORRECTION_MAX),
                               _clamp(POS_Z_KP * e[2] + POS_Z_KI * self._i_pos_z, VEL_CORRECTION_MAX)])
        v_sp = v_ref + correction

        # velocity loop (world frame): lean in degrees along world x and y, the height loop's raw output
        ve = v_sp - velocity
        self._i_vel = np.clip(self._i_vel + ve * DT_CTRL, -I_LIMIT_DEFAULT, I_LIMIT_DEFAULT)
        lean_x = VEL_KP * ve[0] + VEL_KI * self._i_vel[0] + math.degrees(math.atan(a_ref[0] / G))
        lean_y = VEL_KP * ve[1] + VEL_KI * self._i_vel[1] + math.degrees(math.atan(a_ref[1] / G))
        raw_z = VEL_Z_KP * ve[2] + VEL_Z_KI * self._i_vel[2]

        # into the heading frame: a lean along body x is pitch, a lean along body y is minus roll
        cy, sy = math.cos(yaw), math.sin(yaw)
        lean_bx = cy * lean_x + sy * lean_y
        lean_by = -sy * lean_x + cy * lean_y
        roll_sp, pitch_sp = _limit_total_tilt(math.radians(-lean_by), math.radians(lean_bx))

        # attitude loop: Euler errors in radians, the integral taking in the current error first
        e_att = np.array([roll_sp - roll, pitch_sp - pitch])
        self._i_att = np.clip(self._i_att + e_att * DT_CTRL, -I_LIMIT_ATT, I_LIMIT_ATT)
        p = ATT_KP * e_att[0] + ATT_KI * self._i_att[0]
        q = ATT_KP * e_att[1] + ATT_KI * self._i_att[1]
        r = YAW_KP * _wrap(0.0 - yaw)

        # thrust: true hover plus the height loop's units, floored and capped, into newtons, over the measured tilt's cosine
        units = min(max(UNITS_HOVER + THRUST_SCALE * raw_z, THRUST_MIN_UNITS), UNITS_MAX)
        thrust = T_MAX * (units / UNITS_MAX) ** 2 / max(float(R[2, 2]), _MIN_COS_TILT)
        thrust = min(max(thrust, 0.0), T_MAX)
        return np.array([thrust, _clamp(p, RATE_LIMIT), _clamp(q, RATE_LIMIT), _clamp(r, RATE_LIMIT)])
