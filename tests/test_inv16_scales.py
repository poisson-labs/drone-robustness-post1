"""Invariant 16 (spec section 2, 'Observation scaling'; sections 5 and 12): the observation scales are section 2's
constants, at every step, in every arm; no running statistic exists in the policy's input path.

The layout of the 319 numbers is not fixed by the spec, so the tests find a field's entries by perturbing it and
seeing which outputs move.
"""
import numpy as np
import pytest

import spec_numbers as S
from drone_study.observation import Frame, N_FRAMES, OBS_SIZE, observe, zero_frame

REFS = np.zeros((S.N_REF_POINTS, 3))
BASE_ACTION = np.array([S.T_MAX / 4, 0.0, 0.0, 0.0])


def frame(**kw):
    f = dict(position_error=np.zeros(3), velocity=np.zeros(3), rotation_cols=np.zeros((3, 2)), body_rate=np.zeros(3),
             last_action=BASE_ACTION.copy())
    f.update(kw)
    return Frame(**f)


def frames_with(index, **kw):
    fs = [frame() for _ in range(N_FRAMES)]
    fs[index] = frame(**kw)
    return fs


def moved(a, b):
    return np.flatnonzero(a != b)


def test_inv16_an_observation_is_319_numbers_from_16_frames_and_5_reference_points():
    out = observe([frame() for _ in range(N_FRAMES)], REFS)
    assert out.shape == (OBS_SIZE,) == (S.OBS_SIZE,)
    assert S.N_FRAMES * S.FRAME_SIZE + S.N_REF_POINTS * 3 == 319
    assert np.all(np.isfinite(out))
    assert np.any(out != 0.0)                      # a quarter thrust maps to -0.5: the vector is not blank


@pytest.mark.parametrize("field,scale,size", [("position_error", S.SCALE_POS, 3), ("velocity", S.SCALE_VEL, 3),
                                              ("body_rate", S.SCALE_RATE, 3)])
@pytest.mark.parametrize("frame_index", (0, 7, 15))
def test_inv16_each_field_is_divided_by_its_fixed_physical_scale(field, scale, size, frame_index):
    base = observe(frames_with(frame_index), REFS)
    value = np.array([0.12, -0.3, 0.2])
    out = observe(frames_with(frame_index, **{field: value}), REFS)
    idx = moved(out, base)
    assert len(idx) == 3
    np.testing.assert_allclose(np.sort(out[idx] - base[idx]), np.sort(value / scale), rtol=1e-12)


def test_inv16_reference_points_are_divided_by_one_metre():
    base = observe([frame() for _ in range(N_FRAMES)], REFS)
    refs = REFS.copy(); refs[2] = [0.4, -0.7, 0.1]
    out = observe([frame() for _ in range(N_FRAMES)], refs)
    idx = moved(out, base)
    assert len(idx) == 3
    np.testing.assert_allclose(np.sort(out[idx] - base[idx]), np.sort(refs[2] / S.SCALE_REF), rtol=1e-12)


def test_inv16_rotation_entries_are_passed_as_they_are():
    base = observe(frames_with(3), REFS)
    cols = np.array([[0.9, 0.1], [-0.2, 0.8], [0.3, -0.4]])
    out = observe(frames_with(3, rotation_cols=cols), REFS)
    idx = moved(out, base)
    assert len(idx) == 6
    np.testing.assert_allclose(np.sort(out[idx] - base[idx]), np.sort(cols.ravel()), rtol=1e-12)


ACTION_RANGES = [(0.0, S.T_MAX), (-S.ACTION_RATE_RANGE, S.ACTION_RATE_RANGE), (-S.ACTION_RATE_RANGE, S.ACTION_RATE_RANGE),
                 (-S.ACTION_RATE_RANGE, S.ACTION_RATE_RANGE)]


@pytest.mark.parametrize("k", range(4), ids=("thrust", "p", "q", "r"))
def test_inv16_the_last_action_is_mapped_from_its_own_range_to_plus_minus_one(k):
    lo, hi = ACTION_RANGES[k]

    def at(value):
        a = BASE_ACTION.copy(); a[k] = value
        return observe(frames_with(5, last_action=a), REFS)

    out_lo, out_hi = at(lo), at(hi)
    idx = moved(out_lo, out_hi)
    assert len(idx) == 1                                       # one entry carries this component
    i = idx[0]
    assert out_lo[i] == pytest.approx(-1.0, abs=1e-12)
    assert out_hi[i] == pytest.approx(1.0, abs=1e-12)
    assert at((lo + hi) / 2)[i] == pytest.approx(0.0, abs=1e-12)
    assert at(lo + 0.25 * (hi - lo))[i] == pytest.approx(-0.5, abs=1e-12)
    assert at(lo + 0.75 * (hi - lo))[i] == pytest.approx(0.5, abs=1e-12)


@pytest.mark.parametrize("field", ["position_error", "velocity", "body_rate"])
def test_inv16_everything_is_clipped_at_plus_minus_five(field):
    out = observe(frames_with(2, **{field: np.array([1e3, -1e3, 1e3])}), REFS)
    assert out.max() <= S.CLIP and out.min() >= -S.CLIP
    assert np.sum(np.abs(out) == S.CLIP) == 3
    huge_ref = REFS.copy(); huge_ref[0] = [50.0, -50.0, 6.0]
    out = observe([frame() for _ in range(N_FRAMES)], huge_ref)
    assert out.max() <= S.CLIP and out.min() >= -S.CLIP and np.sum(np.abs(out) == S.CLIP) == 3


def test_inv16_a_gusts_values_are_not_squashed_by_what_calm_air_looked_like():
    """Calm air would give a running normaliser a position-error scale of about a centimetre. A fixed scale does not."""
    gust = frames_with(15, position_error=np.array([0.2, 0.0, 0.0]))
    fresh = observe(gust, REFS)
    for _ in range(2000):                                                           # a long calm-air history
        observe([frame(position_error=np.random.default_rng(0).normal(0, 0.01, 3)) for _ in range(N_FRAMES)], REFS)
    seasoned = observe(gust, REFS)
    np.testing.assert_array_equal(fresh, seasoned)                                   # no state: the same input, the same bits
    base = observe([frame() for _ in range(N_FRAMES)], REFS)
    assert (fresh - base).max() == pytest.approx(0.2 / S.SCALE_POS, rel=1e-12)       # 0.4, not 20 and not clipped


def test_inv16_the_scaling_is_the_same_at_every_step_and_for_every_caller():
    rng = np.random.default_rng(16)
    fs = [frame(position_error=rng.normal(0, 0.3, 3), velocity=rng.normal(0, 1.5, 3), body_rate=rng.normal(0, 3, 3),
                rotation_cols=rng.normal(0, 0.5, (3, 2)), last_action=np.array([rng.uniform(0, S.T_MAX), *rng.normal(0, 3, 3)]))
          for _ in range(N_FRAMES)]
    refs = rng.normal(0, 0.5, (S.N_REF_POINTS, 3))
    first = observe(fs, refs)
    for _ in range(50):
        observe([frame(velocity=rng.normal(0, 9, 3)) for _ in range(N_FRAMES)], refs * 3)           # unrelated traffic in between
        np.testing.assert_array_equal(observe(fs, refs), first)
    assert np.any(first != 0.0) and np.abs(first).max() <= S.CLIP
