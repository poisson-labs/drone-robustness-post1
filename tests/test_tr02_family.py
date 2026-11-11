"""Training rule 2 (spec section 2, 'Cases', last paragraph; section 17.6): "Training draws from the same shape family:
peak speed uniform over 1.5 to 2.2 m/s, geometry scaled uniformly by 0.8 to 1.2, and a draw over 5.5 m/s^2 is drawn
again. So the twelve are a sample of what it flew."

Readings the spec leaves open (ruling 4A-3): the four shapes are equally likely; one factor, uniform on 0.8 to 1.2,
multiplies every horizontal coordinate of the shape's formula (the slalom's forward run included); the lap rate is
solved from the drawn peak speed; a draw over the bound is redrawn whole, shape included.
"""
import math

import numpy as np
import pytest
from scipy import stats

import spec_numbers as S
import training_numbers as N
import training_oracles as TO
from drone_study import cases
from drone_study.training import family as F
from drone_study.training.family import FamilyDraw, draw_family, family_case, member_of, peak_acceleration

MEMBERS = [(shape, v, k) for shape in S.SHAPES for v, k in ((1.5, 0.8), (1.5, 1.2), (1.83, 1.07), (2.0, 1.0), (2.2, 1.2))]
N_DRAWS = 40_000


@pytest.fixture(scope="module")
def draws():
    rng = np.random.default_rng(20261105)
    return [draw_family(rng) for _ in range(N_DRAWS)]


def test_tr02_the_box_is_the_one_section_2_gives():
    assert tuple(F.SPEED_RANGE) == N.FAMILY_SPEED and tuple(F.SCALE_RANGE) == N.FAMILY_SCALE
    assert F.MAX_ACCELERATION == N.FAMILY_MAX_ACCEL


@pytest.mark.parametrize("shape,v,k", MEMBERS)
def test_tr02_a_member_is_section_2s_formula_scaled_with_its_lap_rate_solved_from_the_peak_speed(shape, v, k):
    ref = family_case(FamilyDraw(shape, v, k))
    w = TO.family_omega(shape, v, k)
    assert ref.omega == pytest.approx(w, rel=1e-12)
    assert ref.period == pytest.approx(2 * math.pi / w, rel=1e-12)
    t = np.linspace(0.0, 10.4, 521)
    pos, vel, acc = TO.family_state(shape, v, k, t)
    np.testing.assert_allclose(ref.position(t), pos, rtol=0, atol=1e-12)
    np.testing.assert_allclose(ref.velocity(t), vel, rtol=0, atol=1e-12)
    np.testing.assert_allclose(ref.acceleration(t), acc, rtol=0, atol=1e-11)
    assert np.all(ref.position(t)[:, 2] == S.HEIGHT_M)                         # 3.05 m, as every case


@pytest.mark.parametrize("shape,v,k", MEMBERS)
def test_tr02_a_members_peak_reference_speed_is_exactly_the_drawn_speed(shape, v, k):
    ref = family_case(FamilyDraw(shape, v, k))
    lap = np.linspace(0.0, 2 * math.pi / TO.family_omega(shape, v, k), 20001)
    speed = np.linalg.norm(ref.velocity(lap), axis=-1)
    assert speed.max() == pytest.approx(v, rel=1e-9)
    assert np.linalg.norm(ref.velocity(np.array(0.0))) == pytest.approx(v, rel=1e-12)      # phase 0 is peak speed, as section 2


@pytest.mark.parametrize("shape,v,k", MEMBERS)
def test_tr02_peak_acceleration_is_the_largest_over_a_lap(shape, v, k):
    got = peak_acceleration(FamilyDraw(shape, v, k))
    assert got == pytest.approx(TO.family_peak_accel(shape, v, k), rel=1e-6)
    assert got == pytest.approx(TO.family_peak_accel_closed(shape, v, k), rel=1e-6)


@pytest.mark.parametrize("name", N.CASE_NAMES)
def test_tr02_the_twelve_cases_are_members_of_the_family(name):
    shape, tier = name.rsplit("-", 1)
    d = member_of(name)
    assert (d.shape, d.peak_speed, d.scale) == (shape, S.PEAK_SPEED[tier], 1.0)
    assert N.FAMILY_SPEED[0] <= d.peak_speed <= N.FAMILY_SPEED[1] and N.FAMILY_SCALE[0] <= d.scale <= N.FAMILY_SCALE[1]
    assert peak_acceleration(d) <= N.FAMILY_MAX_ACCEL
    assert peak_acceleration(d) == pytest.approx(S.PEAK_ACCEL[shape][S.TIERS.index(tier)], abs=0.005)   # section 2's table
    t = np.linspace(0.0, 10.4, 261)
    np.testing.assert_allclose(family_case(d).position(t), cases.case(name).position(t), rtol=0, atol=1e-12)


def test_tr02_a_members_name_is_a_case_name_the_stack_accepts():
    d = FamilyDraw("fig8", 1.7312, 1.05)
    assert d.name and d.name not in N.CASE_NAMES
    assert FamilyDraw("fig8", 1.7312, 1.06).name != d.name and FamilyDraw("oval", 1.7312, 1.05).name != d.name
    t = np.linspace(0.0, 10.4, 131)
    np.testing.assert_allclose(cases.case(d.name).position(t), TO.family_state("fig8", 1.7312, 1.05, t)[0], rtol=0, atol=1e-12)
    assert cases.case(d.name).period == pytest.approx(2 * math.pi / TO.family_omega("fig8", 1.7312, 1.05), rel=1e-12)


def test_tr02_draws_stay_in_the_box_and_reach_its_edges(draws):
    v = np.array([d.peak_speed for d in draws]); k = np.array([d.scale for d in draws])
    assert {d.shape for d in draws} == set(S.SHAPES)
    assert v.min() >= N.FAMILY_SPEED[0] and v.max() <= N.FAMILY_SPEED[1]
    assert k.min() >= N.FAMILY_SCALE[0] and k.max() <= N.FAMILY_SCALE[1]
    assert v.min() < N.FAMILY_SPEED[0] + 0.01 and v.max() > N.FAMILY_SPEED[1] - 0.01
    assert k.min() < N.FAMILY_SCALE[0] + 0.01 and k.max() > N.FAMILY_SCALE[1] - 0.01


def test_tr02_no_draw_is_over_the_acceleration_bound_and_the_bound_bites(draws):
    assert len(draws) == N_DRAWS and all(d.shape in S.SHAPES and d.scale > 0 for d in draws)
    accel = np.array([TO.family_peak_accel_closed(d.shape, d.peak_speed, d.scale) for d in draws])
    assert accel.max() <= N.FAMILY_MAX_ACCEL
    assert accel.max() > N.FAMILY_MAX_ACCEL - 0.05                       # draws right up to the bound survive
    assert TO.family_peak_accel_closed("fig8", 2.2, 0.8) > N.FAMILY_MAX_ACCEL      # and the box does hold members over it
    fast_small = [d for d in draws if d.shape == "fig8" and d.scale < 0.85]
    assert fast_small and max(d.peak_speed for d in fast_small) < 2.2 - 0.05       # so the corner is empty


def test_tr02_where_nothing_is_rejected_speed_and_scale_are_uniform(draws):
    """The oval never passes 5.5 m/s^2 inside the box (3.23 / 0.8 = 4.0), so its speed and scale are plain uniform draws."""
    oval = [d for d in draws if d.shape == "oval"]
    assert len(oval) > 0.2 * N_DRAWS
    v = np.array([d.peak_speed for d in oval]); k = np.array([d.scale for d in oval])
    for x, (lo, hi) in ((v, N.FAMILY_SPEED), (k, N.FAMILY_SCALE)):
        assert abs(x.mean() - (lo + hi) / 2) < N.SIGMA_K * TO.uniform_mean_se(lo, hi, len(x))
        assert stats.kstest(x, "uniform", args=(lo, hi - lo)).pvalue > 1e-6
    assert abs(np.corrcoef(v, k)[0, 1]) < N.SIGMA_K / math.sqrt(len(oval))          # drawn independently


def test_tr02_the_shapes_are_equally_likely_before_the_bound(draws):
    accept = {s: TO.family_accept_probability(s) for s in S.SHAPES}
    assert accept["oval"] == 1.0 and accept["fig8"] < 0.99 and accept["slalom"] < 0.99 and accept["circle"] < 1.0
    total = sum(accept.values())
    for s in S.SHAPES:
        want = accept[s] / total
        got = sum(d.shape == s for d in draws) / N_DRAWS
        assert abs(got - want) < N.SIGMA_K * TO.share_se(want, N_DRAWS), (s, got, want)


def test_tr02_a_draw_over_the_bound_is_redrawn_whole_shape_included():
    """Redrawing the whole draw leaves the oval, which is never rejected, with 25.5% of the accepted draws and the figure
    of eight with 24.5%; redrawing only speed and scale inside a shape would leave each at 25.0%. Half a million draws
    tell the two apart: the gap is about nine standard errors."""
    rng = np.random.default_rng(20261107)
    n = 500_000
    counts = dict.fromkeys(S.SHAPES, 0)
    for _ in range(n):
        shape = draw_family(rng).shape
        assert shape in counts
        counts[shape] += 1
    accept = {s: TO.family_accept_probability(s) for s in S.SHAPES}
    total = sum(accept.values())
    for s in S.SHAPES:
        want = accept[s] / total
        assert abs(counts[s] / n - want) < N.SIGMA_K * TO.share_se(want, n), (s, counts[s] / n, want)
    assert accept["oval"] / total - 0.25 > 8 * TO.share_se(0.25, n)          # so one in four would not pass above


def test_tr02_the_draws_are_a_function_of_the_generator_and_nothing_else():
    r1, r2, r3 = (np.random.default_rng(seed) for seed in (7, 7, 8))
    one, two, other = [draw_family(r1) for _ in range(50)], [draw_family(r2) for _ in range(50)], [draw_family(r3) for _ in range(50)]
    assert one == two
    assert one != other
    assert len(set(one)) == 50
