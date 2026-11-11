"""The air force (spec section 17.3): rotor/frame linear drag plus body quadratic drag, in the body frame.

At each substep `v_rel = v_air - v_drone`, turned into the body frame at the current attitude. Two terms:

* Rotor and frame linear drag: `F = K (sum of rotor speeds) v_rel`, with K = (9.1785e-7, 9.1785e-7, 10.311e-7), acting in
  the rotor plane, 5 mm above the centre of mass, so its moment is `[0, 0, 0.005] x F`.
* Body quadratic drag: in the rotor plane `F = 1/2 rho CdA_xy |v_xy| v_xy` (the vector, so the same all round in plan);
  along the body z axis `F = 1/2 rho CdA_z |v_z| v_z`. rho 1.225, Cd 1, CdA_xy = 92 x 29 mm, CdA_z = 92 x 92 mm (O6). It acts
  at the centre of mass in the primary model; S1 tests a centre of pressure 10 mm up.

MuJoCo's own fluid model is off (density and viscosity 0, `opt.wind` unused): `Physics` sets that, and applies this force.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

K_LIN_XY = 9.1785e-7
K_LIN_Z = 10.311e-7
ROTOR_PLANE_OFFSET_M = 0.005
RHO = 1.225
CDA_XY = 0.092 * 0.029
CDA_Z = 0.092 * 0.092


@dataclass(frozen=True)
class AirModel:
    name: str
    linear: bool = True
    quad_scale: float = 1.0
    quad_cop_offset_m: float = 0.0  # centre of pressure of the quadratic term above the centre of mass


PRIMARY = AirModel("primary")
LINEAR_ONLY = AirModel("linear-only", quad_scale=0.0)          # S1
QUAD_DOUBLED = AirModel("quad-doubled", quad_scale=2.0)        # S1
QUAD_COP_10MM = AirModel("quad-cop-10mm", quad_cop_offset_m=0.010)  # S1


def air_wrench_scalars(vx: float, vy: float, vz: float, omega_sum: float, model: AirModel = PRIMARY):
    """(fx, fy, fz, mx, my) in the body frame about the centre of mass, from body-frame relative air velocity (the
    moment has no z part). Plain floats: this is the substep's hot path, and `air_wrench_body` wraps it."""
    fx = fy = fz = 0.0
    if model.linear:
        fx = K_LIN_XY * omega_sum * vx
        fy = K_LIN_XY * omega_sum * vy
        fz = K_LIN_Z * omega_sum * vz
    mx = -ROTOR_PLANE_OFFSET_M * fy
    my = ROTOR_PLANE_OFFSET_M * fx
    q = model.quad_scale
    if q != 0.0:
        k_xy = 0.5 * RHO * CDA_XY * math.hypot(vx, vy) * q
        qx, qy = k_xy * vx, k_xy * vy
        qz = 0.5 * RHO * CDA_Z * abs(vz) * vz * q
        cop = model.quad_cop_offset_m
        if cop != 0.0:
            mx += -cop * qy
            my += cop * qx
        fx += qx
        fy += qy
        fz += qz
    return fx, fy, fz, mx, my


def air_wrench_body(v_rel_body, rotor_omega, model: AirModel = PRIMARY):
    """(force, moment) in the body frame about the centre of mass.

    v_rel_body = v_air - v_drone, in body axes, m/s. rotor_omega: four rotor speeds, rad/s.
    """
    v = np.asarray(v_rel_body, dtype=float)
    fx, fy, fz, mx, my = air_wrench_scalars(float(v[0]), float(v[1]), float(v[2]), float(np.sum(rotor_omega)), model)
    return np.array([fx, fy, fz]), np.array([mx, my, 0.0])


def air_wrench_world(v_air_world, v_drone_world, rot_body_to_world, rotor_omega, model: AirModel = PRIMARY):
    """(force, moment) in the world frame, from world-frame air and drone velocities and the attitude."""
    R = np.asarray(rot_body_to_world, dtype=float)
    v_rel_body = R.T @ (np.asarray(v_air_world, dtype=float) - np.asarray(v_drone_world, dtype=float))
    f, m = air_wrench_body(v_rel_body, rotor_omega, model)
    return R @ f, R @ m
