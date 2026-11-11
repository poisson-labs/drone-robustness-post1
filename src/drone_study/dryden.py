"""Dryden turbulence tape (spec section 17.3; invariant 11).

The low-altitude forms of MIL-F-8785C (section 3.7.3, Figures 10 and 11; session 3A's addendum), at h = 10 ft, in feet:
`L_u = L_v = h / (0.177 + 0.000823 h)^1.2`, `L_w = h`, `sigma_w = 0.1 W20`, `sigma_u = sigma_v = sigma_w / (0.177 +
0.000823 h)^0.4`. At 10 ft (3.05 m) L_u = 23.05 m and L_w = 3.05 m, and sigma_u / sigma_w = 1.96. The caller gives sigma_u.

Transfer functions, with `tau = L / V` at the FIXED filter speed V = 2.0 m/s (never the drone's airspeed: the tape is a
function of time only, written before the episode and the same for every policy):

* u: first order, `H = K / (1 + tau s)`, autocovariance `sigma^2 exp(-lag / tau)`.
* v, w: second order, `H = K (1 + sqrt(3) tau s) / (1 + tau s)^2`, autocovariance `sigma^2 (1 - lag / 2 tau) exp(-lag / tau)`.

Each channel is simulated by its EXACT discrete update (the state-space form's matrix exponential, with the noise covariance
that update implies), started in the stationary state so the variance at t = 0 is already sigma^2. The second-order
channel's two states are reduced to one output recursion (Cayley-Hamilton), which `scipy.signal.lfilter` runs in C.

The tape is a function of the seed alone (and the arguments). Each channel draws from its own child of `SeedSequence(seed)`,
the initial state first and then the noise, so a longer tape begins with the shorter one.
"""
from __future__ import annotations

import functools
import math

import numpy as np
from scipy import linalg, signal

FT = 0.3048
SQRT3 = math.sqrt(3.0)


def lengths_and_sigma_ratio(altitude_ft: float = 10.0):
    """(L_u = L_v, L_w, sigma_u / sigma_w), with the lengths in metres."""
    a = 0.177 + 0.000823 * altitude_ft
    l_uv = altitude_ft / a ** 1.2 * FT
    l_w = altitude_ft * FT
    return l_uv, l_w, 1.0 / a ** 0.4


@functools.lru_cache(maxsize=64)
def _first_order(tau: float, sigma: float, dt: float):
    a = math.exp(-dt / tau)
    return a, sigma * math.sqrt(1.0 - a * a)


@functools.lru_cache(maxsize=64)
def _second_order(tau: float, sigma: float, dt: float):
    """(c, Phi-trace, Phi-det, g1, g2, chol_noise, chol_state) for H = K (1 + sqrt3 tau s) / (1 + tau s)^2 with output
    variance sigma^2, in controllable canonical form x' = A x + B w, y = c x."""
    A = np.array([[0.0, 1.0], [-1.0 / tau ** 2, -2.0 / tau]])
    B = np.array([[0.0], [1.0]])
    c = np.array([1.0, SQRT3 * tau]) / tau ** 2
    P0 = linalg.solve_continuous_lyapunov(A, -B @ B.T)         # stationary state covariance for unit white noise
    k = sigma / math.sqrt(float(c @ P0 @ c))                   # scale the STATE so the output variance is sigma^2
    P = P0 * k * k
    Phi = linalg.expm(A * dt)
    Q = P - Phi @ P @ Phi.T                                    # the discrete noise covariance this update implies
    Q = 0.5 * (Q + Q.T)
    tr, det = float(np.trace(Phi)), float(np.linalg.det(Phi))
    g1 = c
    g2 = Phi.T @ c - tr * c
    return c, tr, det, g1, g2, Phi, _factor(Q), _factor(P)


def _factor(M: np.ndarray) -> np.ndarray:
    """A square root L with L L^T = M for a symmetric positive semi-definite M (eigen-decomposition, negatives clipped)."""
    w, v = np.linalg.eigh(M)
    return v * np.sqrt(np.clip(w, 0.0, None))


def _channel_first(rng: np.random.Generator, n: int, tau: float, sigma: float, dt: float) -> np.ndarray:
    a, noise_gain = _first_order(tau, sigma, dt)
    z = rng.standard_normal(n)
    e = noise_gain * z
    e[0] = sigma * z[0]                                      # y_0 = sigma z_0: the stationary start
    return signal.lfilter([1.0], [1.0, -a], e)


def _channel_second(rng: np.random.Generator, n: int, tau: float, sigma: float, dt: float) -> np.ndarray:
    c, tr, det, g1, g2, Phi, chol_q, chol_p = _second_order(tau, sigma, dt)
    x0 = chol_p @ rng.standard_normal(2)
    q = rng.standard_normal((max(n - 1, 0), 2)) @ chol_q.T      # rows: q_k, k = 0 .. n-2
    y = np.empty(n)
    y[0] = c @ x0
    if n == 1:
        return y
    y[1] = c @ (Phi @ x0 + q[0])
    if n == 2:
        return y
    e1 = q @ g1
    e2 = q @ g2
    drive = e1[1:] + e2[:-1]                                     # drives y_{k+2}, k = 0 .. n-3
    zi = signal.lfiltic([1.0], [1.0, -tr, det], y=[y[1], y[0]])
    y[2:], _ = signal.lfilter([1.0], [1.0, -tr, det], drive, zi=zi)
    return y


def dryden_tape(seed: int, n: int, sigma_u: float, *, length_scale: float = 1.0, filter_speed: float = 2.0,
                dt: float = 0.002, altitude_ft: float = 10.0) -> np.ndarray:
    """(n, 3) air velocity u, v, w along world x, y, z. Low-altitude MIL-F-8785C forms at `altitude_ft`, all
    lengths multiplied by `length_scale` (S3), at a FIXED filter speed, starting in the stationary state."""
    out = np.zeros((n, 3))
    if sigma_u == 0.0 or n == 0:
        return out
    l_uv, l_w, ratio = lengths_and_sigma_ratio(altitude_ft)
    children = np.random.SeedSequence(int(seed)).spawn(3)
    rng_u, rng_v, rng_w = (np.random.Generator(np.random.PCG64(c)) for c in children)
    tau_uv = l_uv * length_scale / filter_speed
    tau_w = l_w * length_scale / filter_speed
    out[:, 0] = _channel_first(rng_u, n, tau_uv, sigma_u, dt)
    out[:, 1] = _channel_second(rng_v, n, tau_uv, sigma_u, dt)
    out[:, 2] = _channel_second(rng_w, n, tau_w, sigma_u / ratio, dt)
    return out
