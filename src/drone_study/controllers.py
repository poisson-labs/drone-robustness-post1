"""The geometric controller and its two privileged variants (spec section 17.2; inv. 15).

The geometric controller is Lee et al.'s form with Mellinger and Kumar's integral on position error, run at 100 Hz on the
outer stream and sharing the policy's rate loop and motors: it outputs the same action a policy does, `[collective thrust N,
p, q, r]`. Position, velocity and acceleration of the reference come from the five reference points the policy also sees
(a degree-4 polynomial through them, read at s = 0, the first of them), so the controllers get no more reference than the policy.

  desired force      F = -kp m e_p - kv m e_v - ki m e_i + m (a_ref + g z) - F_air, the feedback part bounded at 0.30 N
                     (about the flat force at 12 m/s), so a steady wind leaves no standing offset
  thrust             T = F . b3  (the measured body z axis), within [0, Tmax]
  desired attitude   b3d along F, heading b1c = world x (reference yaw 0); total tilt held within 55 degrees, just inside
                     the firmware's 60 degree tumble rule
  rate command       omega_cmd = -k_R e_R, e_R = 1/2 (Rd^T R - R^T Rd)^vee, within +-10 rad/s

Gains are not in the spec: session 4A named the grid (ruling 4A-16, pick B: Kp 6 to 30, Kv 3.5 to 14, Ki 1, 2, 4) and session
4B picked the point in calm air by the frozen rule, smallest median RMSE over the 12 calm cases x 8 seeds of the `tuning` salt
among points with no crash: DEFAULT_GAINS = (30, 14, 1), median 4.76 cm, 2026-11-06 (3B's provisional pick was (6, 3.5, 2),
7.3 cm on the look). The one pick is shared by the geometric controller, the wind-only reference and the feasibility oracle,
so the comparison between them is of what they KNOW. It sits on the grid's top edge by design: Kp 30 puts the outer loop's
natural frequency at nine tenths of the attitude loop's 6 rad/s, the cascade's practical ceiling, and Ki 1 is the floor.

`WindOnlyReference`: the true air force fed forward (F_air above) and nothing else: the normal stream, the nominal mass, the
firmware mixer. `FeasibilityOracle`: true mass, true air force, true undelayed noise-free state, and a mixer that knows each
motor's scale.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from drone_study.actuation import ScaleAwareMixer
from drone_study.stack import Controller, ControllerView
from drone_study.vehicle import DT_CTRL, G, MASS, T_MAX

LOOKAHEAD_S = 0.1
ATT_KP = 6.0                       # rad/s per rad of attitude error (roll, pitch)
YAW_KP = 2.0                       # yaw is the slow axis (section 17.4: weak reaction torque)
RATE_LIMIT = 10.0
TILT_LIMIT = math.radians(55.0)
FEEDBACK_FORCE_BOUND_N = 0.30


@dataclass(frozen=True)
class GeometricGains:
    kp: float   # 1/s^2: acceleration per metre of position error
    kv: float   # 1/s: acceleration per m/s of velocity error
    ki: float   # 1/s^3: acceleration per metre-second of integrated position error


DEFAULT_GAINS = GeometricGains(kp=30.0, kv=14.0, ki=1.0)      # session 4B's pick on the ruled grid (scripts/tune_gains.py)

_S = LOOKAHEAD_S * np.arange(0, 5)
_V = np.vander(_S, 5, increasing=True)
_W = np.linalg.inv(_V)                                    # coefficients c_m of the polynomial from the five samples
_W_POS, _W_VEL, _W_ACC = _W[0], _W[1], 2.0 * _W[2]        # value, slope and second derivative at s = 0


def reference_state(points: np.ndarray):
    """(position, velocity, acceleration) of the reference now, from its five points: now, and 0.1 ... 0.4 s ahead."""
    return _W_POS @ points, _W_VEL @ points, _W_ACC @ points


def _vee(M: np.ndarray) -> np.ndarray:
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


class GeometricController(Controller):
    def __init__(self, gains: GeometricGains = DEFAULT_GAINS):
        self.gains = gains
        self._e_i = np.zeros(3)

    def reset(self) -> None:
        self._e_i = np.zeros(3)

    def act(self, view: ControllerView) -> np.ndarray:
        priv = view.privileged
        if priv is not None and priv.state is not None:
            position, velocity, R = priv.state.position, priv.state.velocity, priv.state.rotation
        else:
            position, velocity, R = view.reading.position, view.reading.velocity, view.reading.rotation
        mass = priv.mass if (priv is not None and priv.mass is not None) else MASS
        air = priv.air_force if (priv is not None and priv.air_force is not None) else np.zeros(3)
        p_ref, v_ref, a_ref = reference_state(view.reference)
        g = self.gains
        e_p, e_v = position - p_ref, velocity - v_ref
        bound_i = FEEDBACK_FORCE_BOUND_N / (g.ki * mass) if g.ki > 0 else math.inf
        self._e_i = self._e_i + e_p * DT_CTRL
        n = float(np.linalg.norm(self._e_i))
        if n > bound_i:
            self._e_i *= bound_i / n
        feedback = mass * (-g.kp * e_p - g.kv * e_v - g.ki * self._e_i)
        n = float(np.linalg.norm(feedback))
        if n > FEEDBACK_FORCE_BOUND_N:
            feedback *= FEEDBACK_FORCE_BOUND_N / n
        force = feedback + mass * (a_ref + np.array([0.0, 0.0, G])) - air
        fz = max(force[2], 0.05 * mass * G)                                 # never ask for a downward or empty thrust axis
        horizontal = math.hypot(force[0], force[1])
        limit = fz * math.tan(TILT_LIMIT)
        if horizontal > limit:
            force = np.array([force[0] * limit / horizontal, force[1] * limit / horizontal, fz])
        else:
            force = np.array([force[0], force[1], fz])
        b3d = force / np.linalg.norm(force)
        b2d = np.cross(b3d, [1.0, 0.0, 0.0])
        b2d = b2d / np.linalg.norm(b2d)
        Rd = np.stack([np.cross(b2d, b3d), b2d, b3d], axis=1)
        e_R = 0.5 * _vee(Rd.T @ R - R.T @ Rd)
        thrust = float(np.clip(force @ R[:, 2], 0.0, T_MAX))
        rates = -np.array([ATT_KP, ATT_KP, YAW_KP]) * e_R
        return np.array([thrust, *np.clip(rates, -RATE_LIMIT, RATE_LIMIT)])


class WindOnlyReference(GeometricController):
    """The true air force fed forward and nothing else: the normal stream, the normal mixer."""
    privileges = frozenset({"air_force"})


class FeasibilityOracle(GeometricController):
    """True mass, true air force, undelayed noise-free state, and a mixer that knows each motor's scale."""
    privileges = frozenset({"mass", "air_force", "state", "motor_scales"})

    def mixer(self, motor_scales):
        return ScaleAwareMixer(motor_scales)
