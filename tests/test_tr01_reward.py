"""Training rule 1 (spec section 2, 'Architecture and training'; section 5, 'The same for every fine-tune'), as the yaw
addenda of 2026-11-07 and 2026-11-08 (session 4D-A) re-freeze it: the reward is tracking error, a heading term, a heading-rate
damping term, a crash penalty and an action-smoothness term, and nothing else, the same in every arm.

Ruling 4A-1 named the first three of the old four; the addenda add the two heading terms and make the penalty a function of the
weight:
  tracking      -0.1 x (distance from the reference / 0.5 m)
  heading       -k_yaw x H(wrapped heading error) / pi, H the Huber function with a 5 degree band (4DR-2)
  heading rate  -k_yaw x 0.25 x min(|Euler yaw rate|, 2 rad/s)^2 / (2 rad/s)^2 (4DR-2)
  smoothness    -0.005 x (squared change of the action in its +-1 units, summed over thrust and the three rates)
  crash         -crash_penalty_for(k_yaw) on the step that ends the episode in a crash class: 20 on the base ladder, 22 at
                0.032 and 26 at 0.064 (4DR-3), the smallest whole number above the most a flight can lose for ever at the
                frozen discount, never under 4A-1's 20
k_yaw on the ladder (0.004, 0.008, 0.016) and its extension (0.032, 0.064).

Re-frozen by 4D-A twice: first for the 2026-11-07 term (-k |wrap e| / pi, penalty 20 at every weight), now for the 2026-11-08
addendum (the Huber form, the damping term, the per-weight penalty, the extension). docs/yaw-invariant-table.md lists what each
test asserted before and after.
"""
import math

import numpy as np
import pytest

import training_numbers as N
import yaw_oracles as Y
from drone_study.training import reward as R
from drone_study.training.reward import RewardTerms, crash_penalty_for, reward_terms

SAME = np.array([0.2, -0.1, 0.3, 0.0])
K = N.YAW_WEIGHT_LADDER[1]                                                 # a ladder weight; the heading tests run over all five


def terms(error, action, previous, crashed, *, psi=0.0, rate=0.0, k=K):
    return reward_terms(error, action, previous, crashed, heading_error=psi, heading_rate=rate, k_yaw=k)


def test_tr01_the_constants_are_the_frozen_ones():
    assert (R.K_TRACK, R.TRACK_SCALE_M, R.K_SMOOTH, R.CRASH_PENALTY) == (N.K_TRACK, N.TRACK_SCALE_M, N.K_SMOOTH, N.CRASH_PENALTY)
    assert R.GAMMA == N.GAMMA == 0.99
    assert R.YAW_SCALE_RAD == N.YAW_SCALE_RAD == math.pi                                       # the addendum's scale
    assert tuple(R.YAW_WEIGHT_LADDER_BASE) == N.YAW_WEIGHT_LADDER_BASE == (0.004, 0.008, 0.016)   # the base ladder, smallest first
    assert tuple(R.YAW_WEIGHT_LADDER_EXTENSION) == N.YAW_WEIGHT_LADDER_EXTENSION == (0.032, 0.064)   # 4DA-3 as amended
    assert tuple(R.YAW_WEIGHT_LADDER) == N.YAW_WEIGHT_LADDER == (0.004, 0.008, 0.016, 0.032, 0.064)
    assert R.REFERENCE_YAW == N.REFERENCE_YAW == 0.0                                           # section 2: reference yaw 0
    assert R.YAW_HUBER_BAND_RAD == pytest.approx(N.YAW_HUBER_BAND_RAD) and math.degrees(R.YAW_HUBER_BAND_RAD) == pytest.approx(5.0)
    assert R.YAW_RATE_WEIGHT == N.YAW_RATE_WEIGHT == 0.25 and R.YAW_RATE_CAP_RAD_S == N.YAW_RATE_CAP_RAD_S == 2.0


@pytest.mark.parametrize("error", [0.05, 0.25, 0.5, 1.0])
def test_tr01_the_tracking_term_is_linear_in_the_distance_from_the_reference(error):
    t = terms(error, SAME, SAME, False)
    assert t.tracking == pytest.approx(-N.K_TRACK * error / N.TRACK_SCALE_M, rel=1e-12)
    assert t.heading == 0.0 and t.heading_rate == 0.0 and t.smoothness == 0.0 and t.crash == 0.0
    assert terms(0.0, SAME, SAME, False).tracking == 0.0
    assert terms(2 * error, SAME, SAME, False).tracking == pytest.approx(2 * t.tracking, rel=1e-12)
    turned = terms(error, SAME, SAME, False, psi=0.5, rate=0.3)              # a heading error and a rate leave the tracking term alone
    assert turned.tracking == t.tracking
    assert turned.heading == pytest.approx(Y.heading_term(0.5, K), rel=1e-12) and turned.heading < 0.0
    assert turned.heading_rate == pytest.approx(Y.rate_term(0.3, K), rel=1e-12) and turned.heading_rate < 0.0


@pytest.mark.parametrize("delta", [np.array([1.0, 0, 0, 0]), np.array([0, 0.5, 0, 0]), np.array([0, 0, -0.5, 0]),
                                   np.array([0, 0, 0, 0.25]), np.array([2.0, 2.0, -2.0, 2.0]), np.array([0.1, -0.2, 0.3, -0.4])])
def test_tr01_the_smoothness_term_is_the_squared_change_of_the_action_summed_over_its_four_parts(delta):
    t = terms(0.0, SAME + delta, SAME, False)
    assert t.smoothness == pytest.approx(-N.K_SMOOTH * float(delta @ delta), rel=1e-12)
    assert t.tracking == 0.0 and t.heading == 0.0 and t.heading_rate == 0.0 and t.crash == 0.0
    assert terms(0.0, SAME - delta, SAME, False).smoothness == pytest.approx(t.smoothness, rel=1e-12)   # sign-blind
    shifted = terms(0.0, SAME + delta - 0.3, SAME - 0.3, False)              # only the change counts, not the level
    assert shifted.smoothness == pytest.approx(t.smoothness, rel=1e-9)
    turned = terms(0.0, SAME + delta, SAME, False, psi=0.3, rate=-0.6)       # the heading terms do not read the action
    assert turned.smoothness == t.smoothness
    assert turned.heading == pytest.approx(Y.heading_term(0.3, K), rel=1e-12)
    assert turned.heading_rate == pytest.approx(Y.rate_term(-0.6, K), rel=1e-12)


@pytest.mark.parametrize("k", N.YAW_WEIGHT_LADDER)
def test_tr01_the_crash_penalty_lands_on_the_crash_step_and_only_there_and_is_the_weights(k):
    crashed = terms(0.6, SAME + 0.1, SAME, True, psi=0.4, rate=0.5, k=k)
    alive = terms(0.6, SAME + 0.1, SAME, False, psi=0.4, rate=0.5, k=k)
    assert crashed.crash == -N.CRASH_PENALTIES[k] == -crash_penalty_for(k) == -N.crash_penalty(k)
    assert alive.crash == 0.0
    assert crashed.tracking == alive.tracking and crashed.smoothness == alive.smoothness     # the other terms still count
    assert crashed.heading == alive.heading == pytest.approx(Y.heading_term(0.4, k), rel=1e-12)   # the heading terms too
    assert crashed.heading_rate == alive.heading_rate == pytest.approx(Y.rate_term(0.5, k), rel=1e-12)
    assert crashed.total == pytest.approx(alive.total - N.CRASH_PENALTIES[k], rel=1e-12)
    assert crashed.crash <= -N.CRASH_PENALTY                                                   # never under 4A-1's 20


def test_tr01_the_crash_penalty_per_weight_is_the_smallest_whole_number_above_the_worst_flight_for_ever():
    """4DR-3: for each weight, the penalty is the smallest whole number strictly above (0.18 + k (1 - band / 2 pi) + 0.25 k) / 0.01,
    and not under 20: 18.49, 18.99 and 19.98 for ever give 20 on the base ladder; 21.96 gives 22 at 0.032; 25.91 gives 26 at 0.064."""
    for k, want in N.CRASH_PENALTIES.items():
        forever = N.worst_alive_per_step(k) / (1.0 - N.GAMMA)
        assert crash_penalty_for(k) == want
        assert crash_penalty_for(k) > forever                                                  # the invariant, strictly
        assert crash_penalty_for(k) == max(N.CRASH_PENALTY, math.floor(forever) + 1)           # and the smallest such whole number
        assert float(crash_penalty_for(k)).is_integer()
        assert R.worst_alive_per_step(k) == pytest.approx(N.worst_alive_per_step(k), rel=1e-12)
    assert (crash_penalty_for(0.004), crash_penalty_for(0.008), crash_penalty_for(0.016)) == (20.0, 20.0, 20.0)
    assert crash_penalty_for(0.032) == 22.0 and crash_penalty_for(0.064) == 26.0
    assert N.worst_alive_per_step(0.016) / (1.0 - N.GAMMA) == pytest.approx(19.9778, abs=1e-4)
    for bad in (0.0, 0.002, 0.02, 0.05, 0.128, 1.0, -0.004):
        with pytest.raises(ValueError):
            crash_penalty_for(bad)


def test_tr01_the_reward_is_the_sum_of_the_five_terms_and_there_is_no_sixth():
    rng = np.random.default_rng(1)
    for _ in range(300):
        e, a, b, crashed = rng.uniform(0, 0.8), rng.uniform(-1, 1, 4), rng.uniform(-1, 1, 4), bool(rng.integers(0, 2))
        psi, rate = rng.uniform(-2 * math.pi, 2 * math.pi), rng.uniform(-4.0, 4.0)
        k = N.YAW_WEIGHT_LADDER[int(rng.integers(0, len(N.YAW_WEIGHT_LADDER)))]
        t = terms(e, a, b, crashed, psi=psi, rate=rate, k=k)
        d = a - b
        want = (-N.K_TRACK * e / N.TRACK_SCALE_M + Y.heading_term(psi, k) + Y.rate_term(rate, k) - N.K_SMOOTH * float(d @ d)
                - (N.CRASH_PENALTIES[k] if crashed else 0.0))
        assert t.total == pytest.approx(want, rel=1e-12, abs=1e-15)
        assert t.total == pytest.approx(t.tracking + t.heading + t.heading_rate + t.smoothness + t.crash, rel=1e-12, abs=1e-15)
        assert t.heading == pytest.approx(Y.heading_term(psi, k), rel=1e-12, abs=1e-15)
        assert t.heading_rate == pytest.approx(Y.rate_term(rate, k), rel=1e-12, abs=1e-15)
    assert isinstance(terms(0.1, SAME, SAME, False), RewardTerms)
    assert set(RewardTerms.__dataclass_fields__) == {"tracking", "heading", "heading_rate", "smoothness", "crash"}
    assert terms(0.1, SAME, SAME, False).total < 0.0                         # a flight off the reference is not free
    assert terms(0.0, SAME, SAME, False, psi=0.2).total < 0.0                # neither is a flight off its heading
    assert terms(0.0, SAME, SAME, False, rate=0.5).total < 0.0               # nor one turning


def test_tr01_flying_on_is_never_worth_less_than_crashing_now():
    """Inside the crash radius the tracking term is at most 0.1 a step, the heading term at most k (1 - 1/72) (facing backwards),
    the damping term at most 0.25 k (at or past the cap), and the largest possible action change costs 0.005 x 4 x 2^2. The
    discounted sum of that for ever is the most a flight can lose by staying up, and the weight's penalty is larger at every
    ladder weight, so ending the episode never pays (4A-1's invariant under the final reward, 4DR-3)."""
    for k in N.YAW_WEIGHT_LADDER:
        worst_alive = terms(N.TRACK_SCALE_M, np.ones(4), -np.ones(4), False, psi=math.pi, rate=10.0, k=k).total
        assert worst_alive == pytest.approx(-N.worst_alive_per_step(k), rel=1e-12)
        assert worst_alive == pytest.approx(-(N.K_TRACK + N.K_SMOOTH * 16.0 + k * N.HEADING_TERM_BOUND + k * N.YAW_RATE_WEIGHT), rel=1e-12)
        assert worst_alive < 0.0
        lost_by_flying_forever = -worst_alive / (1.0 - N.GAMMA)
        assert -terms(0.0, SAME, SAME, True, k=k).crash > lost_by_flying_forever
        assert -terms(0.0, SAME, SAME, True, k=k).crash == N.CRASH_PENALTIES[k]
        # nothing pays more than the worst step: a heading beyond pi wraps back, a rate beyond the cap is clipped
        assert terms(N.TRACK_SCALE_M, np.ones(4), -np.ones(4), False, psi=3 * math.pi, rate=100.0, k=k).total == pytest.approx(worst_alive, rel=1e-12)
    assert N.worst_alive_per_step(0.016) / (1.0 - N.GAMMA) == pytest.approx(19.9778, abs=1e-4)   # the base ladder keeps 4A-1's 20
    assert N.worst_alive_per_step(0.032) / (1.0 - N.GAMMA) == pytest.approx(21.9556, abs=1e-4)
    assert N.worst_alive_per_step(0.064) / (1.0 - N.GAMMA) == pytest.approx(25.9111, abs=1e-4)
