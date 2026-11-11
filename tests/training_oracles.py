"""Public slice (session P1c-repo, 2026-11-10): the feasibility rule's and the box's oracles are removed with their tests.

Independent reference computations for the training tests: closed forms, brute-force integrals and a plain replay of
the built stack. Written from the spec, never from drone_study.training. The replay uses the stack session 3B built
(`drone_study.stack`, `drone_study.observation`), which the training code sits on and the 3A tests froze."""
import math

import numpy as np

import spec_numbers as S
import training_numbers as N

# ---- the training family (section 2) -----------------------------------------------------------------------------------
_UNIT_SPEED = {"fig8": math.sqrt(1.0 ** 2 + 4 * 0.5 ** 2), "circle": 1.0, "oval": 1.5}


def family_omega(shape, v, k):
    """omega such that the scaled shape's peak reference speed is v: every horizontal coordinate carries the factor k."""
    if shape == "slalom":
        return math.sqrt((v / k) ** 2 - 1.0) / 0.8
    return v / (k * _UNIT_SPEED[shape])


def family_state(shape, v, k, t):
    """(position, velocity, acceleration), each (..., 3): section 2's formula with x and y multiplied by k, z = 3.05 m."""
    t = np.asarray(t, dtype=float)
    w = family_omega(shape, v, k)
    u = w * t
    zero, height = np.zeros_like(t), np.full_like(t, S.HEIGHT_M)
    if shape == "fig8":
        p = (np.sin(u), 0.5 * np.sin(2 * u)); vel = (w * np.cos(u), w * np.cos(2 * u))
        acc = (-w * w * np.sin(u), -2 * w * w * np.sin(2 * u))
    elif shape == "circle":
        p = (np.cos(u), np.sin(u)); vel = (-w * np.sin(u), w * np.cos(u)); acc = (-w * w * np.cos(u), -w * w * np.sin(u))
    elif shape == "oval":
        p = (1.5 * np.sin(u), 0.75 * np.cos(u)); vel = (1.5 * w * np.cos(u), -0.75 * w * np.sin(u))
        acc = (-1.5 * w * w * np.sin(u), -0.75 * w * w * np.cos(u))
    else:
        p = (t, 0.8 * np.sin(u)); vel = (zero + 1.0, 0.8 * w * np.cos(u)); acc = (zero, -0.8 * w * w * np.sin(u))
    stack = lambda xy, z: np.stack([k * xy[0], k * xy[1], z], axis=-1)
    return stack(p, height), stack(vel, zero), stack(acc, zero)


def family_peak_accel(shape, v, k, n=20001):
    """Largest reference acceleration over a lap, by dense sampling."""
    T = 2 * math.pi / family_omega(shape, v, k)
    return float(np.linalg.norm(family_state(shape, v, k, np.linspace(0.0, T, n))[2], axis=-1).max())


def family_peak_accel_closed(shape, v, k):
    return {"fig8": 1.0625 * v * v / k, "circle": v * v / k, "oval": v * v / (1.5 * k), "slalom": (v * v / k - k) / 0.8}[shape]


def family_accept_probability(shape, n=1201):
    """Share of the speed x scale box whose peak acceleration is within the bound, on a fine midpoint grid."""
    vs = N.FAMILY_SPEED[0] + (np.arange(n) + 0.5) / n * (N.FAMILY_SPEED[1] - N.FAMILY_SPEED[0])
    ks = N.FAMILY_SCALE[0] + (np.arange(n) + 0.5) / n * (N.FAMILY_SCALE[1] - N.FAMILY_SCALE[0])
    V, K = np.meshgrid(vs, ks)
    a = {"fig8": 1.0625 * V * V / K, "circle": V * V / K, "oval": V * V / (1.5 * K), "slalom": (V * V / K - K) / 0.8}[shape]
    return float(np.mean(a <= N.FAMILY_MAX_ACCEL))


# ---- statistics --------------------------------------------------------------------------------------------------------
def wilson(k, n, z=1.959963984540054):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def share_se(p, n):
    return math.sqrt(p * (1.0 - p) / n)


def uniform_mean_se(lo, hi, n):
    return (hi - lo) / math.sqrt(12.0 * n)


# ---- the action's units (section 2: thrust 0 to the maximum, rates within +-10 rad/s; the observation maps the same) -----
ACTION_LO = np.array([0.0, -S.ACTION_RATE_RANGE, -S.ACTION_RATE_RANGE, -S.ACTION_RATE_RANGE])
ACTION_HI = np.array([S.T_MAX, S.ACTION_RATE_RANGE, S.ACTION_RATE_RANGE, S.ACTION_RATE_RANGE])
HOLD_PHYSICAL = np.array([S.WEIGHT, 0.0, 0.0, 0.0])           # the stack's hover hold before any action (3B, reading 1)


def to_physical(action):
    a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
    return ACTION_LO + (a + 1.0) * 0.5 * (ACTION_HI - ACTION_LO)


def to_normalised(physical):
    return 2.0 * (np.asarray(physical, dtype=float) - ACTION_LO) / (ACTION_HI - ACTION_LO) - 1.0


HOLD_NORMALISED = to_normalised(HOLD_PHYSICAL)


# ---- a plain replay of one training episode on the built stack ---------------------------------------------------------
class Replay:
    """The stack in training mode, the 3B observer and the fixed scales, driven one action at a time. What the training
    environment must equal, step for step: the observation, the tracking error, the end of the episode."""

    def __init__(self, condition, seed):
        from drone_study.observation import PolicyObserver
        from drone_study.stack import StackEnv
        self.stack = StackEnv(condition, int(seed), mode="training")
        self.observer = PolicyObserver()
        self.reading = self.stack.reset()
        self.last = HOLD_PHYSICAL.copy()
        self.obs = self._observe()
        self.done, self.end_reason, self.steps = False, None, 0

    def _observe(self):
        return self.observer.push(self.reading, self.stack.reference_now(), self.stack.reference_ahead(), self.last).astype(np.float32)

    def view(self):
        from drone_study.stack import ControllerView
        return ControllerView(self.stack.time, self.reading, self.stack.reference_ahead(), None)

    def step(self, action):
        """action in +-1 units. Returns (obs, tracking error m, done, end reason). The action is mapped by the package's own
        `to_physical`, which test_tr07 holds to this module's formula within 1e-12, so the replay does not freeze the order
        of one rounding."""
        from drone_study.training.env import to_physical as package_to_physical
        phys = np.asarray(package_to_physical(action), dtype=float)
        result = self.stack.step(phys)
        self.steps += 1
        self.reading, self.last = result.reading, phys
        self.obs = self._observe()
        self.done, self.end_reason = result.done, result.end_reason
        error = float(np.linalg.norm(self.stack.truth.position - self.stack.reference_now()))
        return self.obs, error, result.done, result.end_reason


def quantised(physical):
    """A physical action as the float32 action-unit vector an environment is handed, so a replay and an environment get
    the same bits."""
    return to_normalised(np.clip(physical, ACTION_LO, ACTION_HI)).astype(np.float32)
