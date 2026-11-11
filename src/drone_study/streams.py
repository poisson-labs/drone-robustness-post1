"""The three seed streams and their tapes (spec sections 6, 17.9, 12 inv. 8).

A tape is an array, complete before the episode's first step, a function of nothing but the seed and the
condition. Three independent named streams: `start`, `turbulence`, `sensor`. Each is a child of one SeedSequence(seed)
(`spawn(3)`, in that order), so no stream's draws depend on how another is used, on the policy, or on how often the tapes
were built. `start` and `sensor` depend on the seed alone; `turbulence` also on the condition's sigma_u (0 is calm air: a
zero tape, while the other two streams still run).

Interface 3B inherits (all arrays float64, time-major, read-only):
  build_tapes(condition, seed) -> Tapes
  Tapes.start       StartState: position (3,) m, velocity (3,) m/s, tilt (2,) rad (roll, pitch): deviations from the
                    reference start, section 17.9 (sigma 3 cm, 5 cm/s, 1 degree)
  Tapes.turbulence  (5000, 3): background air velocity u, v, w along world x, y, z at 500 Hz, m/s
  Tapes.sensor      SensorTape: gyro_white (5000, 3) rad/s, pos_offset (3,) m, pos_ar (1000, 3) m,
                    vel_ar (1000, 3) m/s, att_ar (1000, 3) rad (rotation-vector error), rate_white (1000, 3) rad/s
  Tapes.digests()   {"start": sha256 hex, "turbulence": ..., "sensor": ...} of the exact bytes
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
from scipy import signal

from drone_study.condition import Condition
from drone_study.dryden import dryden_tape

N_SUBSTEPS = 5000
N_STEPS = 1000

START_POS_SIGMA = 0.03
START_VEL_SIGMA = 0.05
START_TILT_SIGMA = math.radians(1.0)

GYRO_SIGMA = 0.0039
POS_OFFSET_SIGMA = 0.02
POS_AR_SIGMA = 0.01
VEL_AR_SIGMA = 0.05
ATT_AR_SIGMA = math.radians(1.0)
AR_TAU_S = 0.1
RATE_WHITE_SIGMA = 0.05
DT_CTRL = 0.01


@dataclass(frozen=True)
class StartState:
    position: np.ndarray
    velocity: np.ndarray
    tilt: np.ndarray


@dataclass(frozen=True)
class SensorTape:
    gyro_white: np.ndarray
    pos_offset: np.ndarray
    pos_ar: np.ndarray
    vel_ar: np.ndarray
    att_ar: np.ndarray
    rate_white: np.ndarray


@dataclass(frozen=True)
class Tapes:
    start: StartState
    turbulence: np.ndarray
    sensor: SensorTape

    def digests(self) -> dict[str, str]:
        def sha(*arrays) -> str:
            h = hashlib.sha256()
            for a in arrays:
                a = np.ascontiguousarray(a, dtype=np.float64)
                h.update(str(a.shape).encode())
                h.update(a.tobytes())
            return h.hexdigest()

        s = self.sensor
        return {
            "start": sha(self.start.position, self.start.velocity, self.start.tilt),
            "turbulence": sha(self.turbulence),
            "sensor": sha(s.gyro_white, s.pos_offset, s.pos_ar, s.vel_ar, s.att_ar, s.rate_white),
        }


def _ar1(rng: np.random.Generator, n: int, sigma: float) -> np.ndarray:
    """(n, 3) AR(1) noise with time constant tau, stationary from its first sample: marginal sigma, correlation
    exp(-lag / tau)."""
    a = math.exp(-DT_CTRL / AR_TAU_S)
    z = rng.standard_normal((n, 3))
    e = sigma * math.sqrt(1.0 - a * a) * z
    e[0] = sigma * z[0]
    return signal.lfilter([1.0], [1.0, -a], e, axis=0)


def _frozen(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float64)
    x.setflags(write=False)
    return x


def build_tapes(condition: Condition, seed: int) -> Tapes:
    seed = int(seed)
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    start_ss, turb_ss, sensor_ss = np.random.SeedSequence(seed).spawn(3)

    rng = np.random.Generator(np.random.PCG64(start_ss))
    start = StartState(
        _frozen(rng.normal(0.0, START_POS_SIGMA, 3)),
        _frozen(rng.normal(0.0, START_VEL_SIGMA, 3)),
        _frozen(rng.normal(0.0, START_TILT_SIGMA, 2)),
    )

    turb_seed = int.from_bytes(turb_ss.generate_state(2, dtype=np.uint64).tobytes(), "little")
    turbulence = _frozen(dryden_tape(turb_seed, N_SUBSTEPS, float(condition.turb)))

    rng = np.random.Generator(np.random.PCG64(sensor_ss))
    gyro_white = rng.normal(0.0, GYRO_SIGMA, (N_SUBSTEPS, 3))
    pos_offset = rng.normal(0.0, POS_OFFSET_SIGMA, 3)
    pos_ar = _ar1(rng, N_STEPS, POS_AR_SIGMA)
    vel_ar = _ar1(rng, N_STEPS, VEL_AR_SIGMA)
    att_ar = _ar1(rng, N_STEPS, ATT_AR_SIGMA)
    rate_white = rng.normal(0.0, RATE_WHITE_SIGMA, (N_STEPS, 3))
    sensor = SensorTape(*(_frozen(x) for x in (gyro_white, pos_offset, pos_ar, vel_ar, att_ar, rate_white)))
    return Tapes(start, turbulence, sensor)
