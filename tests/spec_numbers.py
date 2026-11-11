"""Every number the tests take from the frozen spec, each with the section it comes from.

The spec is "2026-11-04 Drone Robustness Study - Spec" (frozen 2026-11-05). A number that is not here is either
derived from these by a formula the test writes out, or is a ruling listed in OWED (docs/invariant-table.md).
Nothing here is imported from drone_study: the tests must not borrow their expected values from the code they test.
"""
import math

# ---- section 2: rates, episode ------------------------------------------------------------------------------
PHYS_HZ = 500
CTRL_HZ = 100
DT_PHYS = 1.0 / PHYS_HZ
DT_CTRL = 1.0 / CTRL_HZ
SUBSTEPS_PER_STEP = PHYS_HZ // CTRL_HZ            # 5
EPISODE_S = 10.0
N_STEPS = 1000                                      # "An episode is 1,000 control steps and 5,000 physics substeps"
N_SUBSTEPS = 5000
HEIGHT_M = 3.05                                     # section 2, Task

# ---- vehicle: simulator at ca170dc (section 17.4 keeps the simulator's mass and inertia; 01-source-checks) ------
MASS = 0.027
G = 9.81
THRUST_TO_WEIGHT = 2.25
KF_PER_RPM2 = 3.16e-10                              # N per RPM^2 (01-source-checks: "It's N/RPM^2")
KF = KF_PER_RPM2 * (60.0 / (2.0 * math.pi)) ** 2    # N per (rad/s)^2
KM_PER_RPM2 = 7.94e-12
ARM = 0.0397
IXX, IYY, IZZ = 1.4e-5, 1.4e-5, 2.17e-5
WEIGHT = MASS * G
OMEGA_HOVER = math.sqrt(WEIGHT / (4 * KF))          # rad/s
OMEGA_MAX = math.sqrt(THRUST_TO_WEIGHT * WEIGHT / (4 * KF))
SUM_OMEGA_HOVER = 4 * OMEGA_HOVER                   # section 17.3: "about 6,064 rad/s"
T_MAX = THRUST_TO_WEIGHT * WEIGHT                   # 0.5960 N, section 17.3 table header
LATERAL_LIMIT_N = WEIGHT * math.sqrt(THRUST_TO_WEIGHT ** 2 - 1)   # "0.534 N or 19.8 m/s^2"

# ---- section 17.3: air force ---------------------------------------------------------------------------------
K_LIN_XY = 9.1785e-7
K_LIN_Z = 10.311e-7
ROTOR_PLANE_OFFSET_M = 0.005
RHO = 1.225
CDA_XY = 0.092 * 0.029                              # "2.67e-3 m^2 sideways (92 x 29 mm)"
CDA_Z = 0.092 * 0.092                               # "8.46e-3 m^2 on the top face (92 x 92 mm, O6)"
PARAM_RTOL = 2e-3                                   # the spec prints CdA to 3 figures; 2.67e-3 vs 2.668e-3 is 0.07%
# The table "Scaled to this drone, flat, at hover rotor speed"
TABLE_SPEEDS = (2, 4, 6, 8, 10, 12, 14, 16)
TABLE_LINEAR_N = (0.011, 0.022, 0.033, 0.045, 0.056, 0.067, 0.078, 0.089)
TABLE_PRIMARY_N = (0.018, 0.048, 0.092, 0.149, 0.219, 0.302, 0.398, 0.507)
TABLE_PRIMARY_MS2 = (0.65, 1.79, 3.42, 5.52, 8.11, 11.19, 14.75, 18.79)
TABLE_PRIMARY_SHARE_PCT = (3, 9, 17, 28, 41, 57, 75, 95)
# Half a unit of the last printed digit is the tolerance for a printed table value.
TABLE_TOL_N = 0.0005
TABLE_TOL_MS2 = 0.005
TABLE_TOL_PCT = 0.5

# ---- section 3: the gust and the timing ------------------------------------------------------------------------
GUST_LENGTH_S = 1.0
GUST_LENGTHS_S2 = (0.5, 2.0)                        # S2
FIRST_LAP_AFTER_S = 3.0
SEARCH_PHASES = (0.0, 0.125, 0.25, 0.375)
ZONE_B_PHASES = (0.0, 0.125)
ZONE_C_PHASES = (0.0625, 0.1875, 0.3125, 0.4375)
GUST_SPEEDS_ZONE_B = (2, 3, 4, 5, 6, 7, 8, 10, 12, 14)
EARLIEST_GUST_START_S = 2.86                         # "starts at 2.86 s or later and ends by 9.53 s" (printed to 0.01)
LATEST_GUST_END_S = 9.53
SPEC_ROUNDING_S = 0.005

# ---- section 2: the twelve cases -------------------------------------------------------------------------------
SHAPES = ("fig8", "circle", "oval", "slalom")
TIERS = ("medium", "fast", "faster")
PEAK_SPEED = {"medium": 1.5, "fast": 1.8, "faster": 2.2}
MAX_ACCEL_BOUND = 5.5
LAP_S = {   # printed to 3 decimals
    "fig8": (5.924, 4.937, 4.039), "circle": (4.189, 3.491, 2.856),
    "oval": (6.283, 5.236, 4.284), "slalom": (4.496, 3.359, 2.565),
}
PEAK_ACCEL = {
    "fig8": (2.39, 3.44, 5.14), "circle": (2.25, 3.24, 4.84),
    "oval": (1.50, 2.16, 3.23), "slalom": (1.56, 2.80, 4.80),
}

# ---- section 17.4: actuation and sensing ------------------------------------------------------------------------
MOTOR_TAU_S = 0.072
MOTOR_TOL = 0.01                                    # inv. 5: "within 1%"
GYRO_SIGMA = 0.0039                                 # rad/s, white, at 500 Hz
GYRO_LPF_HZ = 80.0
RATE_GAINS = {"roll": (250.0, 500.0, 2.5), "pitch": (250.0, 500.0, 2.5), "yaw": (120.0, 16.7, 0.0)}
RATE_I_LIMIT = {"roll": 33.3, "pitch": 33.3, "yaw": 166.7}
INT16_MAX = 32767.0                                 # firmware saturateSignedInt16 at e828ed2: +-INT16_MAX
UNITS_MAX = 65535.0
SENSOR_LATENCY_STEPS = 1                            # 10 ms
ACTION_LATENCY_STEPS = 1                            # 10 ms
POS_OFFSET_SIGMA = 0.02
POS_AR_SIGMA = 0.01
VEL_AR_SIGMA = 0.05
ATT_AR_SIGMA = math.radians(1.0)
AR_TAU_S = 0.1
RATE_WHITE_SIGMA = 0.05
DELAY_STEPS = {0: 0, 20: 2, 40: 4, 60: 6}           # section 17.4, "The delay axis"

# ---- section 17.9: the start state ----------------------------------------------------------------------------
START_POS_SIGMA = 0.03
START_VEL_SIGMA = 0.05
START_TILT_SIGMA = math.radians(1.0)

# ---- section 17.3: Dryden ------------------------------------------------------------------------------------
DRYDEN_ALTITUDE_FT = 10.0
DRYDEN_FILTER_SPEED = 2.0
FT = 0.3048
_a = 0.177 + 0.000823 * DRYDEN_ALTITUDE_FT
L_UV_FT = DRYDEN_ALTITUDE_FT / _a ** 1.2
L_UV_M = L_UV_FT * FT                               # 23.05 m ("23.1 m")
L_W_M = DRYDEN_ALTITUDE_FT * FT                     # 3.05 m
SIGMA_U_OVER_SIGMA_W = 1.0 / _a ** 0.4              # 1.96
DRYDEN_LENGTH_SCALES = (1.0, 0.5, 2.0)              # primary, S3

# ---- section 4: crash classes -----------------------------------------------------------------------------------
GROUND_M = 0.05
TILT_RULES = ((60.0, 1.0), (101.5, 0.1))
RADIUS_M = 0.5
S6_TILT_INSTANT = (75.0, 90.0)
S6_RADIUS = (0.3, 0.75, 1.00)

# ---- section 2: observation ------------------------------------------------------------------------------------
N_FRAMES = 16
FRAME_SIZE = 19
N_REF_POINTS = 5
OBS_SIZE = 319
CLIP = 5.0
SCALE_POS = 0.5
SCALE_VEL = 3.0
SCALE_REF = 1.0
SCALE_RATE = 10.0
ACTION_RATE_RANGE = 10.0

# ---- section 3 / 10: held-out file and salts -----------------------------------------------------------------
HELDOUT_SHA256 = "7828ef2d469fc6d832567fe8176a5b6f326dea1a7738e0a38ee64620505eaa9c"
HELDOUT_N = 48
ZONE_B_SALT = "9c3d4a1e317d59fa"

# ---- section 6: determinism ------------------------------------------------------------------------------------
DETERMINISM_IDENTICAL_M = 1e-9
DETERMINISM_FLIP_RATE = 0.005

# ---- test conventions (3A-1, 3A-6, 3A-7) ---------------------------------------------------------------------
MACHINE_ATOL = 1e-12            # "to machine precision" (inv. 1, 8): absolute, scaled by the quantity's size
SIGMA_K = 5.0                   # "within sampling error": five standard errors (inv. 7, 11)
SETTLE_TOL = 0.01               # inv. 6 "settles": within 1% of the step, borrowed from inv. 5's 1%
SETTLE_BY_S = 1.5               # ... by this long after the step, and staying inside the band to the end of the run
