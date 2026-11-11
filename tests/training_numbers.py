"""Public slice (session P1c-repo, 2026-11-10): the numbers of the retrain arms, the box and the feasibility rule are removed with their tests.

Every number the training tests take from the frozen spec, each with the section it comes from, and every number the
spec leaves to session 4A, each marked with the ruling that asks for it (docs/training-invariant-table.md, section 1).

The spec is "2026-11-04 Drone Robustness Study - Spec" (frozen 2026-11-05), read with its log's addenda. Nothing here is
imported from drone_study: the tests must not borrow their expected values from the code they test.
"""
import hashlib
import math

import spec_numbers as S

SIGMA_K = S.SIGMA_K                                  # "within sampling error": five standard errors (3A-6)

# ---- section 5: the arms (this public slice carries the nominal arm alone) --------------------------------------------
ARMS = ("nominal",)
CALM_SHARE = {"nominal": 1.0}
CASE_NAMES = tuple(f"{s}-{t}" for s in S.SHAPES for t in S.TIERS)        # section 2: the twelve

# ---- section 2: the training family ----------------------------------------------------------------------------------
FAMILY_SPEED = (1.5, 2.2)                            # "peak speed uniform over 1.5 to 2.2 m/s"
FAMILY_SCALE = (0.8, 1.2)                            # "geometry scaled uniformly by 0.8 to 1.2"
FAMILY_MAX_ACCEL = 5.5                               # "a draw over 5.5 m/s^2 is drawn again"

# ---- section 2: the reward (three terms named; the form and the numbers are ruling 4A-1) ----------------------------
K_TRACK = 0.1                                        # per (tracking error / TRACK_SCALE_M)
TRACK_SCALE_M = 0.5                                  # the crash radius of section 4, and the observation's position scale
K_SMOOTH = 0.005                                     # per squared change of the action in its +-1 units, summed over four
CRASH_PENALTY = 20.0

# ---- section 2 and 5: PPO (the settings are ruling 4A-2; the fine-tune rate is section 5's) ------------------------
HYPER = {
    "n_envs": 25, "n_steps": 1250, "batch_size": 1250, "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95,
    "clip_range": 0.2, "clip_range_vf": None, "normalize_advantage": True, "ent_coef": 0.0, "vf_coef": 0.5,
    "max_grad_norm": 0.5, "target_kl": None, "use_sde": False, "net_arch_pi": (256, 256), "net_arch_vf": (256, 256),
    "activation": "tanh", "ortho_init": True, "log_std_init": -1.0, "optimizer": "Adam", "adam_eps": 1e-5,
    "nominal_learning_rate": 3e-4, "fine_tune_learning_rate": 1e-4, "device": "cpu", "torch_threads": 1,
}
GAMMA = HYPER["gamma"]
ROLLOUT_STEPS = HYPER["n_envs"] * HYPER["n_steps"]   # 31,250 control steps between updates
HIDDEN = (256, 256)                                  # section 2: "A two-layer MLP, 256 units each, tanh"
FINE_TUNE_LR = 1e-4                                  # section 5: "Learning rate 1e-4"
# Session 3B's measured update cost, CPU-seconds per million steps on one torch thread (gate-3b.md, section 3)
UPDATE_COST_1_THREAD = {64: 162.6, 256: 72.7, 1024: 51.0, 4096: 46.3}
UPDATE_COST_12_THREADS = {64: 848.3, 256: 295.7, 1024: 115.6, 4096: 62.3}

# ---- sections 2 and 5: budgets ---------------------------------------------------------------------------------------
RUNGS = (5_000_000, 10_000_000, 20_000_000, 40_000_000)   # "checkpoints at 5M, 10M, 20M and 40M"
N_PILOTS = 4
B0_PROVISIONAL = 20_000_000

# ---- section 5: reset logging ----------------------------------------------------------------------------------------
RESET_WINDOW_S = 0.5                                 # "the share of its steps within 0.5 s of a reset"
RESET_WINDOW_STEPS = 50                              # the first 50 control steps of an episode (4A-12)

def salt(zone: str) -> str:
    """Section 10 item 6: first 16 hex of sha256("drone-robustness-2026-11-04:<zone>")."""
    return hashlib.sha256(f"drone-robustness-2026-11-04:{zone}".encode()).hexdigest()[:16]


BAR_SALT = "76e7b463820f72ad"                        # section 10 item 6 and section 20.3 point 10
TUNING_SALT = salt("tuning")                         # 4A-16

# ---- section 2: the calm-air bar ---------------------------------------------------------------------------------------
BAR_REPLICATES = 100                                 # "1,200 episodes (12 cases x 100)"
BAR_EPISODES = 1200
BAR_MAX_CRASHES = 5
BAR_MAX_RMSE_M = 0.10
BAR_MAX_RATIO = 1.25
BAR_WILSON_UPPER = 0.01                              # "keeps the Wilson 95% upper end under 1%"
LINEAGES = 16
MAX_FAILURES = 8                                     # "more than 8 failures stop the study for a ruling"

# ---- training seeds (4A-13) ----------------------------------------------------------------------------------------------
FIRST_NOMINAL_SEED = 1
PILOT_SEEDS = (9001, 9002, 9003, 9004)


def run_seed(role: str, nominal: int) -> int:
    return int(hashlib.sha256(f"drone-robustness-2026-11-04:train:{role}:{int(nominal)}".encode()).hexdigest()[:8], 16)


# ---- section 17.2: the cascaded PID and the geometric controller's grid ---------------------------------------------
PID_POS_KP, PID_POS_Z_KP, PID_POS_Z_KI = 2.0, 2.0, 0.5         # "position 2/0/0 and z 2/0.5/0"
PID_VEL_KP, PID_VEL_KI = 25.0, 1.0                              # "velocity 25/1/0"
PID_VEL_Z_KP, PID_VEL_Z_KI = 25.0, 15.0                         # "and z 25/15/0"
PID_ATT_KP, PID_ATT_KI = 6.0, 3.0                               # "attitude 6/3/0"
PID_VEL_CORRECTION_MAX = 1.1                                    # "clamped at the firmware's 1.1 m/s"
PID_TILT_LIMIT_DEG = 55.0                                       # "total tilt ... is limited to 55 degrees"
PID_THRUST_SCALE = 1000.0                                       # firmware position_controller_pid.c at e828ed2: thrustScale
PID_FIRST_STEP_TOL = 0.03                                       # the integral terms' share of a first step's output (4A-17)
GRID_KP = (6.0, 9.0, 13.5, 20.0, 30.0)                          # 4A-16, ruled B: the grid session 4A names
GRID_KV = (3.5, 5.0, 7.0, 10.0, 14.0)
GRID_KI = (1.0, 2.0, 4.0)                                       # never 0: section 17.2's integral is always on
TUNING_REPLICATES = 8

# ---- the yaw addenda of 2026-11-07 and 2026-11-08 (session 4D-A): the heading term, its ladder, the acceptance thresholds ----
YAW_SCALE_RAD = math.pi                              # the heading term is -k_yaw x H(wrapped heading error) / pi
YAW_WEIGHT_LADDER_BASE = (0.004, 0.008, 0.016)       # 4DA-2: smallest first
YAW_WEIGHT_LADDER_EXTENSION = (0.032, 0.064)         # 4DA-3 as amended 2026-11-08: tried in order only when no base weight passes
YAW_WEIGHT_LADDER = YAW_WEIGHT_LADDER_BASE + YAW_WEIGHT_LADDER_EXTENSION
REFERENCE_YAW = 0.0                                  # section 2: "reference yaw 0", every case and every training-family member
YAW_HUBER_BAND_RAD = math.pi / 36.0                  # 4DR-2: 5 degrees; quadratic inside, linear outside
YAW_RATE_WEIGHT = 0.25                               # 4DR-2: the damping term's weight as a share of k_yaw
YAW_RATE_CAP_RAD_S = 2.0                             # 4DR-2: the damping term saturates at 2 rad/s of heading rate
HEADING_TERM_BOUND = 1.0 - YAW_HUBER_BAND_RAD / (2.0 * math.pi)   # H(pi) / pi = 1 - 1/72: the most the heading term pays, per unit k_yaw


def worst_alive_per_step(k):
    """Inside the radius, the wildest action change, facing backwards, the heading rate at or past the cap: 0.18 + k (71/72 + 1/4)."""
    return K_TRACK + K_SMOOTH * 16.0 + k * HEADING_TERM_BOUND + k * YAW_RATE_WEIGHT


def crash_penalty(k):
    """4DR-3: the smallest whole number strictly above the discounted sum for ever of the worst alive step, never under 4A-1's 20."""
    forever = worst_alive_per_step(k) / (1.0 - GAMMA)
    return max(CRASH_PENALTY, float(math.floor(forever) + 1))


CRASH_PENALTIES = {0.004: 20.0, 0.008: 20.0, 0.016: 20.0, 0.032: 22.0, 0.064: 26.0}   # the arithmetic, written out (18.49, 18.99, 19.98, 21.96, 25.91 for ever)
YAW_RATE_RMS_MAX_DEG_S = 12.0                        # 4DA-4: median over the bar's episodes of the yaw-rate RMS (true body z rate). 4DR-4 asked for the
                                                     # heading rate; the re-fly read the geometric controller at 13.5 deg/s of it, so the line stays on the
                                                     # body z rate and the heading rate is reported beside it (ruling 4DA-10)
YAW_DRIFT_MAX_DEG_S = 1.0                            # 4DA-4: median over the episodes of |least-squares slope of the unwrapped heading|
HEADING_45_DEG, HEADING_90_DEG = 45.0, 90.0
PAST_45_MAX = 12                                     # 4DA-4: at most 12 of 1,200 episodes with a largest |heading| past 45 deg (1%)
PAST_90_MAX = 3                                      # 4DA-4: at most 3 of 1,200 past 90 deg (0.25%)
MAX_HEADING_MEDIAN_MAX_DEG = 15.0                    # 4DR-4: the fifth line, median over the episodes of the largest |heading|
YAW_META_FIELDS = ("yaw_rate_rms_deg_s", "heading_rate_rms_deg_s", "yaw_drift_deg_s", "max_abs_yaw_deg")   # 4DA-8: beside rmse in the bar rows' meta
# Session 4C's stored readings (behaviour-table.md, table 3) and the 2026-11-08 re-fly of the same episodes for the heading rate
# (evidence/drone-robustness/4d-yaw/heading-rate-check.md; tests/data/4c-yaw-episodes.json): the controllers hold heading, every
# pi0 of 4B does not. The thresholds must pass the two and fail the sixteen.
PI0_4C_YAW_RATE_RMS_RANGE = (23.2, 62.4)             # body z rate, 4C's table 3 (pi0 4 the lowest, pi0 5 the highest)
CONTROLLER_4C_YAW_RATE_RMS = {"pid": 3.3, "geometric": 5.8}
PI0_4C_HEADING_RATE_RMS_RANGE = (24.4, 62.8)         # Euler yaw rate, the 2026-11-08 re-fly (pi0 6 the lowest, pi0 5 the highest)
CONTROLLER_4C_HEADING_RATE_RMS = {"pid": 4.2, "geometric": 13.5}   # the geometric controller's heading wobbles within 6 degrees but fast
CONTROLLER_4C_MAX_HEADING_MEDIAN = {"pid": 2.3, "geometric": 6.1}
PI0_4C_MAX_HEADING_MEDIAN_RANGE = (13.9, 206.6)      # pi0 6 the lowest: it passes the 15 degree line and fails only the rate line
