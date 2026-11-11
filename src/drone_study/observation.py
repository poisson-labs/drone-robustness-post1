"""The policy's observation and its fixed physical scales (spec section 2; inv. 16).

Sixteen frames of (position error, velocity, first two rotation columns, body rate, last action), oldest to
newest, then five reference points: 16 x 19 + 5 x 3 = 319 numbers. All inputs are in the body frame, physical
units. Scales (section 2, "Observation scaling"): position error / 0.5 m, velocity / 3 m/s, reference points /
1 m, body rate / 10 rad/s, rotation entries as they are, last action on its own range mapped to +-1 (thrust
0..Tmax -> -1..1, rates +-10 rad/s -> +-1), everything clipped at +-5. A pure function: no state of any kind.

`PolicyObserver` (additive, not in the stub) is the stateful side the gate and session 4B start from: it turns a
`ControllerView` into the frame history. It holds frames, never statistics.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from drone_study.vehicle import T_MAX

N_FRAMES = 16
N_REFERENCE_POINTS = 5
FRAME_SIZE = 19
OBS_SIZE = N_FRAMES * FRAME_SIZE + N_REFERENCE_POINTS * 3  # 319

SCALE_POS = 0.5
SCALE_VEL = 3.0
SCALE_REF = 1.0
SCALE_RATE = 10.0
ACTION_RATE_RANGE = 10.0
CLIP = 5.0
_ACTION_SCALE = np.array([2.0 / T_MAX, 1.0 / ACTION_RATE_RANGE, 1.0 / ACTION_RATE_RANGE, 1.0 / ACTION_RATE_RANGE])
_ACTION_OFFSET = np.array([-1.0, 0.0, 0.0, 0.0])


@dataclass(frozen=True)
class Frame:
    position_error: np.ndarray   # (3,) m
    velocity: np.ndarray         # (3,) m/s
    rotation_cols: np.ndarray    # (3, 2) first two columns of the body-to-world rotation, expressed as the section says
    body_rate: np.ndarray        # (3,) rad/s
    last_action: np.ndarray      # (4,) thrust N, p, q, r rad/s


def zero_frame() -> Frame:
    return Frame(np.zeros(3), np.zeros(3), np.zeros((3, 2)), np.zeros(3), np.zeros(4))


def frame_row(f: Frame) -> np.ndarray:
    """One frame as its 19 scaled numbers (before the clip)."""
    return np.concatenate([
        np.asarray(f.position_error, dtype=float) / SCALE_POS,
        np.asarray(f.velocity, dtype=float) / SCALE_VEL,
        np.asarray(f.rotation_cols, dtype=float).reshape(6),
        np.asarray(f.body_rate, dtype=float) / SCALE_RATE,
        np.asarray(f.last_action, dtype=float) * _ACTION_SCALE + _ACTION_OFFSET,
    ])


def observe(frames: list[Frame], reference_points: np.ndarray) -> np.ndarray:
    """frames: 16 Frames oldest to newest; reference_points: (5, 3) m. Returns the (319,) scaled observation."""
    if len(frames) != N_FRAMES:
        raise ValueError(f"an observation takes {N_FRAMES} frames, got {len(frames)}")
    refs = np.asarray(reference_points, dtype=float)
    if refs.shape != (N_REFERENCE_POINTS, 3):
        raise ValueError(f"reference_points must be ({N_REFERENCE_POINTS}, 3), got {refs.shape}")
    out = np.concatenate([frame_row(f) for f in frames] + [(refs / SCALE_REF).reshape(-1)])
    return np.clip(out, -CLIP, CLIP)


class PolicyObserver:
    """Frame history for a policy that reads a `ControllerView`: body-frame, from the outer stream only.

    Position error is the reference minus the measured position, rotated into the body frame with the measured attitude; so
    are the velocity and the reference points ahead (each relative to the measured position). Before sixteen frames exist
    the first is repeated. `reset()` clears the history."""

    def __init__(self):
        self._frames: deque[Frame] = deque(maxlen=N_FRAMES)

    def reset(self) -> None:
        self._frames.clear()

    def push(self, reading, reference_now, reference_ahead, last_action) -> np.ndarray:
        R = np.asarray(reading.rotation, dtype=float)
        err_body = R.T @ (np.asarray(reference_now, dtype=float) - reading.position)
        frame = Frame(err_body, R.T @ reading.velocity, R[:, :2], np.asarray(reading.body_rate, dtype=float),
                      np.asarray(last_action, dtype=float))
        if not self._frames:
            self._frames.extend([frame] * N_FRAMES)
        else:
            self._frames.append(frame)
        ahead = (np.asarray(reference_ahead, dtype=float) - reading.position) @ R
        return observe(list(self._frames), ahead)
