"""Independent reference computations for the tests. Closed forms and single-step formulas written from the spec,
never from drone_study. Where a test compares the stack to one of these, the spec section is named in the test."""
import math

import numpy as np

import spec_numbers as S


# ---- cases (section 2; power_output_2b.txt A2 gives the same forms) ---------------------------------------------
def case_omega(shape, tier):
    v = S.PEAK_SPEED[tier]
    return {"fig8": v / math.sqrt(1.0 ** 2 + 4 * 0.5 ** 2), "circle": v / 1.0, "oval": v / 1.5,
            "slalom": math.sqrt(v * v - 1.0) / 0.8}[shape]


def case_state(shape, tier, t):
    """(position, velocity, acceleration), each (..., 3), at z = 3.05 m."""
    t = np.asarray(t, dtype=float)
    w = case_omega(shape, tier)
    u = w * t
    z, one = np.zeros_like(t), np.full_like(t, S.HEIGHT_M)
    if shape == "fig8":
        a, b = 1.0, 0.5
        p = (a * np.sin(u), b * np.sin(2 * u))
        v = (a * w * np.cos(u), 2 * b * w * np.cos(2 * u))
        acc = (-a * w * w * np.sin(u), -4 * b * w * w * np.sin(2 * u))
    elif shape == "circle":
        p = (np.cos(u), np.sin(u)); v = (-w * np.sin(u), w * np.cos(u)); acc = (-w * w * np.cos(u), -w * w * np.sin(u))
    elif shape == "oval":
        a, b = 1.5, 0.75
        p = (a * np.sin(u), b * np.cos(u)); v = (a * w * np.cos(u), -b * w * np.sin(u))
        acc = (-a * w * w * np.sin(u), -b * w * w * np.cos(u))
    else:
        b = 0.8
        p = (1.0 * t, b * np.sin(u)); v = (one * 0 + 1.0, b * w * np.cos(u)); acc = (z, -b * w * w * np.sin(u))
    stack = lambda xy, zz: np.stack([xy[0], xy[1], zz], axis=-1)
    return stack(p, one), stack(v, z), stack(acc, z)


def lap(shape, tier):
    return 2 * math.pi / case_omega(shape, tier)


def gust_peak(shape, tier, phase):
    T = lap(shape, tier)
    return math.ceil(S.FIRST_LAP_AFTER_S / T) * T + phase * T


# ---- the gust (section 3) -----------------------------------------------------------------------------------------
def gust_magnitude(t, speed, peak, length=S.GUST_LENGTH_S):
    t = np.asarray(t, dtype=float)
    start = peak - length / 2
    inside = (t >= start) & (t <= start + length)
    return np.where(inside, 0.5 * speed * (1 - np.cos(2 * np.pi * (t - start) / length)), 0.0)


def gust_direction(travel, az_deg, el_deg):
    """Unit vector of the air VELOCITY. Direction names where the gust comes FROM (0 deg = headwind, azimuth
    anticlockwise seen from above, 90 deg = from the left of travel, elevation up = from above)."""
    t = np.array([travel[0], travel[1], 0.0]); t = t / np.linalg.norm(t)
    left = np.cross([0.0, 0.0, 1.0], t)
    az, el = math.radians(az_deg), math.radians(el_deg)
    source = math.cos(el) * (math.cos(az) * t + math.sin(az) * left) + math.sin(el) * np.array([0.0, 0.0, 1.0])
    return -source


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# ---- the air force (section 17.3) ---------------------------------------------------------------------------------------
def air_wrench_body(v_rel, omega_sum, *, linear=True, quad_scale=1.0, cop=0.0):
    v_rel = np.asarray(v_rel, dtype=float)
    f_lin = np.array([S.K_LIN_XY, S.K_LIN_XY, S.K_LIN_Z]) * omega_sum * v_rel if linear else np.zeros(3)
    m_lin = np.cross([0.0, 0.0, S.ROTOR_PLANE_OFFSET_M], f_lin)
    vxy = v_rel[:2]
    f_quad = np.array([*(0.5 * S.RHO * S.CDA_XY * np.linalg.norm(vxy) * vxy),
                       0.5 * S.RHO * S.CDA_Z * abs(v_rel[2]) * v_rel[2]]) * quad_scale
    m_quad = np.cross([0.0, 0.0, cop], f_quad)
    return f_lin + f_quad, m_lin + m_quad


AIR_MODELS = {   # name -> keyword arguments of air_wrench_body, and the stack's model name
    "primary": dict(linear=True, quad_scale=1.0, cop=0.0),
    "linear-only": dict(linear=True, quad_scale=0.0, cop=0.0),
    "quad-doubled": dict(linear=True, quad_scale=2.0, cop=0.0),
    "quad-cop-10mm": dict(linear=True, quad_scale=1.0, cop=0.010),
}


# ---- the rate loop (section 17.4; firmware pid.c and attitude_pid_controller.c at e828ed2) -------------------------------
class RefPID:
    def __init__(self, kp, ki, kd, ilim, dt=S.DT_PHYS):
        self.kp, self.ki, self.kd, self.ilim, self.dt = kp, ki, kd, ilim, dt
        self.integ = 0.0; self.prev = 0.0

    def update(self, desired, measured):
        err = desired - measured
        p = self.kp * err
        d = self.kd * (-(measured - self.prev)) / self.dt
        self.integ = float(np.clip(self.integ + err * self.dt, -self.ilim, self.ilim))
        i = self.ki * self.integ
        self.prev = measured
        out = float(np.clip(p + i + d, -S.INT16_MAX, S.INT16_MAX))
        return out, p, i, d


class RefRateLoop:
    """Three PIDs in deg/s; inputs and outputs of step() are rad/s and units."""
    def __init__(self, dt=S.DT_PHYS):
        self.pids = [RefPID(*S.RATE_GAINS[a][:3], S.RATE_I_LIMIT[a], dt) for a in ("roll", "pitch", "yaw")]

    def step(self, cmd, meas):
        outs = [pid.update(math.degrees(c), math.degrees(m)) for pid, c, m in zip(self.pids, cmd, meas)]
        return np.array([o[0] for o in outs]), {k: np.array([o[j] for o in outs]) for j, k in ((1, "p"), (2, "i"), (3, "d"))}


# ---- the mixer (section 17.4; simulator motor layout, firmware power_distribution_quadrotor.c) ---------------------------
ROLL_S = np.array([1.0, 1.0, -1.0, -1.0])
PITCH_S = np.array([-1.0, 1.0, 1.0, -1.0])
YAW_S = np.array([-1.0, 1.0, -1.0, 1.0])


def mix_uncapped(thrust, roll, pitch, yaw):
    return thrust + ROLL_S * roll / 2 + PITCH_S * pitch / 2 + YAW_S * yaw


def mix_capped(thrust, roll, pitch, yaw):
    m = mix_uncapped(thrust, roll, pitch, yaw)
    m = m - max(m.max() - S.UNITS_MAX, 0.0)
    return np.maximum(m, 0.0)


def unmix(m):
    m = np.asarray(m, dtype=float)
    return np.array([m.sum() / 4, (ROLL_S * m).sum() / 2, (PITCH_S * m).sum() / 2, (YAW_S * m).sum() / 4])


def body_torques_from_thrusts(f):
    """The simulator's _physics: x and y from thrust differences, z from the reaction torques."""
    f = np.asarray(f, dtype=float)
    k = ARM_OVER_ROOT2 = S.ARM / math.sqrt(2)
    return np.array([(f[0] + f[1] - f[2] - f[3]) * k, (-f[0] + f[1] + f[2] - f[3]) * k,
                     (S.KM_PER_RPM2 / S.KF_PER_RPM2) * (-f[0] + f[1] - f[2] + f[3])])


# ---- Dryden (section 17.3) -------------------------------------------------------------------------------------------
def dryden_autocov(channel, lag_s, sigma_u, scale=1.0, v=S.DRYDEN_FILTER_SPEED):
    sigma = sigma_u / S.SIGMA_U_OVER_SIGMA_W if channel == "w" else sigma_u
    L = (S.L_W_M if channel == "w" else S.L_UV_M) * scale
    tau = L / v
    lag = abs(lag_s)
    if channel == "u":
        return sigma ** 2 * math.exp(-lag / tau)
    return sigma ** 2 * (1 - lag / (2 * tau)) * math.exp(-lag / tau)


# ---- small statistics ---------------------------------------------------------------------------------------------------
def within_se(estimate, truth, se, k=S.SIGMA_K):
    return abs(estimate - truth) <= k * se


def rms_with_se(x, blocks=20):
    """rms of x and a standard error from the spread of block rms values."""
    x = np.asarray(x, dtype=float)
    parts = np.array_split(x, blocks)
    r = np.array([math.sqrt(np.mean(p ** 2)) for p in parts])
    return math.sqrt(np.mean(x ** 2)), float(r.std(ddof=1) / math.sqrt(blocks))
