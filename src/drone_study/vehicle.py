"""The vehicle: the Crazyflie 2.x as the simulator ships it (spec section 2, 17.4), as plain constants.

Every number is the simulator's own (mj-drones-gym at ca170dc, `DRONE_PARAMS[CF2X]`), which keeps the mass, the
inertia and the thrust law (spec 17.4: "the simulator's are kept"). The simulator's `kf` and `km` are per RPM squared
(01-source-checks: "It's N/RPM^2"), so this module converts them to per (rad/s) squared, and rotor speeds anywhere in
the stack are in rad/s. Hover then sums to about 6,064 rad/s over the four rotors, as section 17.3 says.
"""
from __future__ import annotations

import math

MASS = 0.027                       # kg
G = 9.81                           # m/s^2
WEIGHT = MASS * G                  # N
ARM = 0.0397                       # m, motor to centre
IXX, IYY, IZZ = 1.4e-5, 1.4e-5, 2.17e-5
THRUST_TO_WEIGHT = 2.25

_RPM_TO_RADS = 2.0 * math.pi / 60.0
KF = 3.16e-10 / _RPM_TO_RADS ** 2  # N per (rad/s)^2
KM = 7.94e-12 / _RPM_TO_RADS ** 2  # N m per (rad/s)^2
KM_OVER_KF = KM / KF               # m: yaw torque per unit thrust

OMEGA_HOVER = math.sqrt(WEIGHT / (4.0 * KF))                       # rad/s, each rotor
OMEGA_MAX = math.sqrt(THRUST_TO_WEIGHT * WEIGHT / (4.0 * KF))      # rad/s: four of them give thrust-to-weight 2.25
T_MAX = THRUST_TO_WEIGHT * WEIGHT                                  # N, all four rotors

UNITS_MAX = 65535.0                # firmware motor command (3A-3): linear in rotor speed, thrust as units squared
UNITS_HOVER = UNITS_MAX / THRUST_TO_WEIGHT ** 0.5

PHYS_HZ = 500
CTRL_HZ = 100
DT_PHYS = 1.0 / PHYS_HZ
DT_CTRL = 1.0 / CTRL_HZ
SUBSTEPS_PER_STEP = PHYS_HZ // CTRL_HZ
EPISODE_S = 10.0
N_STEPS = 1000
N_SUBSTEPS = 5000
HEIGHT_M = 3.05

MOTOR_TAU_S = 0.072

# The rotor layout of the simulator's `_physics` (CF2X): a thrust-difference mixing for the body torque.
ARM_OVER_ROOT2 = ARM / math.sqrt(2.0)
