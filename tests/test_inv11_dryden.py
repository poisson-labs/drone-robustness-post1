"""Invariant 11 (spec section 17.3): the Dryden tape's spectrum matches the transfer functions at the stated L and sigma
at the fixed filter speed of 2.0 m/s, within sampling error; its variance at t = 0 is already the stationary one.

Method: an ensemble of independent tapes. For a stationary process the covariance between samples a lag apart, taken
ACROSS tapes at a fixed time, is the autocovariance R(lag), the Fourier transform of the spectrum. The u channel's
transfer function is first order, so R_u = sigma^2 exp(-|lag| V/L); the v and w channels' second-order form gives
R = sigma^2 (1 - |lag| V/(2L)) exp(-|lag| V/L) (the forms of MIL-F-8785C section 3.7; power_output_2b.txt A3 uses the
same). Low-altitude lengths at h = 10 ft: L_u = L_v = 23.05 m, L_w = 3.05 m; sigma_u = sigma_v, sigma_w = sigma_u / 1.96.
"""
import functools
import math

import numpy as np
import pytest

import oracles as O
import spec_numbers as S
from drone_study.dryden import dryden_tape

N_TAPES = 2000
N_SAMPLES = S.N_SUBSTEPS               # the whole 10 s episode: the u channel's correlation time is 11.5 s, so short tapes cannot see L_u
CHANNELS = ("u", "v", "w")
LAGS_S = (0.0, 0.05, 0.2, 0.5, 1.0, 2.0, 4.0)
TIMES_S = (0.0, 1.0, 2.0, 3.0, 5.0)    # every (time + lag) stays inside the 10 s tape


@functools.lru_cache(maxsize=1)
def _ensemble(sigma_u, scale):
    x = np.empty((N_TAPES, N_SAMPLES, 3), dtype=np.float32)
    for seed in range(N_TAPES):
        x[seed] = dryden_tape(seed, N_SAMPLES, sigma_u, length_scale=scale)
    x.setflags(write=False)
    return x                            # (tapes, samples, 3)


def ensemble(sigma_u, scale):
    return _ensemble(sigma_u, scale)


SETTINGS = [(0.3, 1.0), (1.5, 1.0), (3.0, 1.0), (1.5, 0.5), (1.5, 2.0)]


@pytest.mark.parametrize("sigma_u,scale", SETTINGS)
def test_inv11_the_tape_matches_the_transfer_functions_at_the_stated_length_and_sigma_from_t_zero(sigma_u, scale):
    """Variance at t = 0 is the stationary one; covariance a lag apart, taken across tapes, is the autocovariance of the
    transfer function at the stated L, for u, v and w, at lags from 0.05 s to 4 s."""
    x = ensemble(sigma_u, scale)
    for c, name in enumerate(CHANNELS):
        sigma = sigma_u / S.SIGMA_U_OVER_SIGMA_W if name == "w" else sigma_u
        for sample in (0, 1, 2):                                   # the stationary variance from the first samples on
            var = float(np.mean(x[:, sample, c].astype(np.float64) ** 2))
            assert abs(var - sigma ** 2) <= S.SIGMA_K * sigma ** 2 * math.sqrt(2.0 / N_TAPES), (name, sample, var, sigma ** 2)
        assert abs(float(x[:, 0, c].mean())) <= S.SIGMA_K * sigma / math.sqrt(N_TAPES)
        r0 = O.dryden_autocov(name, 0.0, sigma_u, scale)
        for lag in LAGS_S:
            want = O.dryden_autocov(name, lag, sigma_u, scale)
            se = math.sqrt((r0 ** 2 + want ** 2) / N_TAPES)
            for t in TIMES_S:
                i, j = int(round(t / S.DT_PHYS)), int(round((t + lag) / S.DT_PHYS))
                got = float(np.mean(x[:, i, c].astype(np.float64) * x[:, j, c]))
                assert abs(got - want) <= S.SIGMA_K * se, (name, "lag", lag, "t", t, got, want)


def test_inv11_the_three_channels_are_independent_and_follow_the_world_axes():
    x = ensemble(1.5, 1.0)
    sig = np.array([1.5, 1.5, 1.5 / S.SIGMA_U_OVER_SIGMA_W])
    for a, b in ((0, 1), (0, 2), (1, 2)):
        cov = float(np.mean(x[:, 700, a].astype(np.float64) * x[:, 700, b]))
        assert abs(cov) <= S.SIGMA_K * sig[a] * sig[b] / math.sqrt(N_TAPES)
    # w is the vertical channel: it is the smaller one (sigma_w = sigma_u / 1.96) and its length is the short one
    assert np.mean(x[:, 700, 2].astype(np.float64) ** 2) < 0.4 * np.mean(x[:, 700, 0].astype(np.float64) ** 2)
    w = O.dryden_autocov("w", 0.5, 1.5)
    assert w < 0.9 * O.dryden_autocov("w", 0.0, 1.5)                          # w decorrelates within a second; u does not
    assert O.dryden_autocov("u", 0.5, 1.5) > 0.95 * O.dryden_autocov("u", 0.0, 1.5)


def test_inv11_the_tape_is_a_function_of_the_seed_alone_at_the_fixed_filter_speed():
    a = dryden_tape(5, 400, 1.5)
    b = dryden_tape(5, 400, 1.5, filter_speed=S.DRYDEN_FILTER_SPEED)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, dryden_tape(5, 400, 1.5))                # same call, same bits
    assert not np.array_equal(a, dryden_tape(6, 400, 1.5))
    assert np.abs(a).max() > 0
    np.testing.assert_array_equal(dryden_tape(5, 200, 1.5), a[:200])          # a longer tape begins with the shorter one
    assert not np.array_equal(dryden_tape(5, 400, 1.5, filter_speed=4.0), a)             # the speed is a real input, fixed at 2.0


@pytest.mark.parametrize("scale", S.DRYDEN_LENGTH_SCALES)
def test_inv11_sigma_is_the_requested_sigma_u_whatever_the_length_scale(scale):
    x = ensemble(1.5, scale)
    var = float(np.mean(x[:, 0, 0].astype(np.float64) ** 2))
    assert var == pytest.approx(1.5 ** 2, rel=S.SIGMA_K * math.sqrt(2.0 / N_TAPES))
