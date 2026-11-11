"""Invariant 15 (spec sections 5, 12, 17.2): the feasibility oracle reads true mass, true air force and undelayed
noise-free state, and its mixer knows each motor's scale, keeping roll and pitch and giving up yaw first (then
collective), so on a constructed payload and weak-motor case, same seed, its largest error is smaller than the
wind-only reference's. The wind-only reference gets the true air force and nothing else.

The two controllers share one set of gains (the spec names none: session 4A picks the grid), so the comparison is of
what they KNOW. Mixer priorities are checked in effective units: a motor at scale s delivers thrust s * T(u), and with
thrust going as units squared (3A-3) that is units sqrt(s) * u.
"""
import numpy as np
import pytest

import oracles as O
import spec_numbers as S
from drone_study.actuation import ScaleAwareMixer, mix
from drone_study.condition import Condition
from drone_study.controllers import DEFAULT_GAINS, FeasibilityOracle, WindOnlyReference
from drone_study.sensing import SensingConfig
from drone_study.stack import Controller, ControllerView, ScriptedController, StackEnv, fly

ORACLE_PRIVILEGES = frozenset({"mass", "air_force", "state", "motor_scales"})
REFERENCE_PRIVILEGES = frozenset({"air_force"})


def test_inv15_the_privileges_are_exactly_the_ones_the_spec_gives():
    assert FeasibilityOracle().privileges == ORACLE_PRIVILEGES
    assert WindOnlyReference().privileges == REFERENCE_PRIVILEGES
    assert FeasibilityOracle().gains == WindOnlyReference().gains == DEFAULT_GAINS


@pytest.mark.parametrize("allowed", [frozenset(), REFERENCE_PRIVILEGES, ORACLE_PRIVILEGES, frozenset({"mass"})])
def test_inv15_the_view_a_controller_gets_carries_only_what_it_was_allowed(allowed):
    cond = Condition(case="circle-fast", az=0.0, el=0.0, speed=8.0, phase=0.0, payload=10.0, motor_index=1, motor_scale=0.7)
    seen = []

    def spy(view):
        seen.append(view.privileged)
        return [S.WEIGHT, 0.0, 0.0, 0.0]

    fly(cond, 3, ScriptedController(spy, privileges=allowed), max_steps=20)
    assert len(seen) == 20
    for p in seen:
        if not allowed:
            assert p is None
            continue
        got = {name for name in ("mass", "air_force", "state", "motor_scales") if getattr(p, name) is not None}
        assert got == set(allowed)
        if "mass" in allowed:
            assert p.mass == pytest.approx(S.MASS + 0.010, rel=1e-12)             # true mass: the airframe plus the 10 g payload
        if "motor_scales" in allowed:
            np.testing.assert_array_equal(p.motor_scales, [1.0, 0.7, 1.0, 1.0])
        if "air_force" in allowed:
            assert np.asarray(p.air_force).shape == (3,) and np.all(np.isfinite(p.air_force))


def test_inv15_the_oracle_reads_nothing_from_the_noisy_delayed_stream_and_the_reference_does():
    cond = Condition(case="circle-fast", turb=0.3, delay=40)
    quiet, loud = SensingConfig(outer_scale=0.0), SensingConfig(outer_scale=50.0)
    o1, o2 = (fly(cond, 4, FeasibilityOracle(), sensing=s, max_steps=200) for s in (quiet, loud))
    np.testing.assert_array_equal(o1.position, o2.position)                          # undelayed, noise-free: the stream is not read
    np.testing.assert_array_equal(o1.rotor_omega, o2.rotor_omega)
    assert np.abs(o1.position - o1.reference_position).max() < 0.5                    # and it really flew the reference
    r1, r2 = (fly(cond, 4, WindOnlyReference(), sensing=s, max_steps=200) for s in (quiet, loud))
    assert not np.array_equal(r1.position, r2.position)                               # the reference does read the normal stream


class Recording(Controller):
    def __init__(self, inner):
        self.inner, self.privileges, self.actions = inner, inner.privileges, []

    def mixer(self, scales):
        return self.inner.mixer(scales)

    def act(self, view):
        a = np.asarray(self.inner.act(view), dtype=float)
        self.actions.append(a)
        return a


def first_action(controller, cond, seed=4):
    rec = Recording(controller)
    fly(cond, seed, rec, max_steps=1)
    return rec.actions[0]


def test_inv15_the_reference_does_not_know_the_mass_or_the_motor_scales_and_the_oracle_does():
    plain = Condition(case="circle-fast", turb=0.3)
    heavy = Condition(case="circle-fast", turb=0.3, payload=15.0)
    weak = Condition(case="circle-fast", turb=0.3, motor_index=0, motor_scale=0.6)
    ref = [first_action(WindOnlyReference(), c) for c in (plain, heavy, weak)]
    np.testing.assert_array_equal(ref[0], ref[1])                                      # same state, same air: same command
    np.testing.assert_array_equal(ref[0], ref[2])
    orc = [first_action(FeasibilityOracle(), c) for c in (plain, heavy, weak)]
    assert orc[1][0] > orc[0][0] * 1.2                                                  # 15 g more mass: about 55% more thrust
    assert orc[1][0] == pytest.approx(orc[0][0] * (S.MASS + 0.015) / S.MASS, rel=0.15)


CONSTRUCTED = {
    "payload 15 g": Condition(case="circle-fast", turb=0.3, payload=15.0),
    "payload 10 g": Condition(case="oval-medium", turb=0.3, payload=10.0),
    "weak motor 0 at 0.6": Condition(case="circle-fast", turb=0.3, motor_index=0, motor_scale=0.6),
    "weak motor 2 at 0.65": Condition(case="slalom-medium", turb=0.3, motor_index=2, motor_scale=0.65),
    "payload 10 g and weak motor 1 at 0.7": Condition(case="oval-fast", turb=0.3, payload=10.0, motor_index=1, motor_scale=0.7),
}


@pytest.mark.parametrize("name", list(CONSTRUCTED))
def test_inv15_on_a_constructed_payload_or_weak_motor_case_the_oracles_largest_error_is_smaller_than_the_references(name):
    cond = CONSTRUCTED[name]

    def worst(controller):
        rec = fly(cond, 7, controller, mode="evaluation")
        return float(np.linalg.norm(rec.position - rec.reference_position, axis=1).max())

    oracle, reference = worst(FeasibilityOracle()), worst(WindOnlyReference())
    assert oracle < reference, (name, oracle, reference)
    assert oracle < S.RADIUS_M                                                          # the oracle flies it
    # the control: the same reference flies the same case when it has nothing to be ignorant of, so the gap is the knowledge
    clean = Condition(case=cond.case, turb=0.3)
    clean_reference = float(np.linalg.norm((r := fly(clean, 7, WindOnlyReference())).position - r.reference_position, axis=1).max())
    assert clean_reference < S.RADIUS_M, clean_reference


# ---- the oracle's mixer ----------------------------------------------------------------------------------------------------
def effective_units(units, scales):
    return np.sqrt(np.asarray(scales, dtype=float)) * np.asarray(units, dtype=float)    # thrust scale s <=> units sqrt(s) (3A-3)


def caps(scales):
    return np.sqrt(np.asarray(scales, dtype=float)) * S.UNITS_MAX


SCALES = (0.6, 1.0, 1.0, 1.0)


def delivered(request, scales=SCALES):
    return effective_units(ScaleAwareMixer(scales).mix(*request), scales)


def test_inv15_a_feasible_request_is_delivered_whole_even_through_a_weak_motor():
    request = (30000.0, 2000.0, -1500.0, 800.0)
    want = O.mix_uncapped(*request)
    assert np.all(want <= caps(SCALES))
    np.testing.assert_allclose(delivered(request), want, rtol=1e-9, atol=1e-6)
    assert not np.allclose(ScaleAwareMixer(SCALES).mix(*request)[0], mix(*request)[0])  # it compensated: motor 0 is commanded harder


def test_inv15_when_yaw_alone_is_infeasible_roll_pitch_and_collective_are_kept_and_yaw_gives_way():
    request = (40000.0, 0.0, 0.0, -24000.0)                                             # motor 0 would need 64000; it has 50763
    got = O.unmix(delivered(request))
    assert got[0] == pytest.approx(request[0], abs=1e-6)
    assert got[1] == pytest.approx(0.0, abs=1e-6) and got[2] == pytest.approx(0.0, abs=1e-6)
    assert request[3] < got[3] < 0.0                                                    # some yaw, less than asked, same sign
    assert np.all(delivered(request) <= caps(SCALES) + 1e-6) and np.all(delivered(request) >= -1e-9)


def test_inv15_when_roll_is_infeasible_even_without_yaw_yaw_is_given_up_entirely_and_collective_gives_way():
    request = (50000.0, 40000.0, 0.0, 6000.0)                                           # motor 0 would need 70000 + yaw
    got = O.unmix(delivered(request))
    assert got[1] == pytest.approx(40000.0, abs=1e-6) and got[2] == pytest.approx(0.0, abs=1e-6)   # roll and pitch kept
    assert got[3] == pytest.approx(0.0, abs=1e-6)                                       # yaw gone
    assert got[0] < request[0] - 1e-3                                                   # collective gave way
    assert got[0] == pytest.approx(caps(SCALES)[0] - 20000.0, abs=1e-6)                 # by exactly as much as the weak motor needs
    assert np.all(delivered(request) <= caps(SCALES) + 1e-6) and np.all(delivered(request) >= -1e-9)


def test_inv15_the_oracles_mixer_follows_the_priority_order_across_random_requests():
    """Where roll and pitch can be delivered at all, they are delivered whole; yaw is cut before collective; nothing
    is below zero or past a motor's capability. (The spec says nothing for a request whose roll and pitch alone do not
    fit, so only the capability limits are asked of it.)"""
    rng = np.random.default_rng(15)
    seen = {"feasible": 0, "yaw cut": 0, "collective cut": 0, "not asked of it": 0}
    for _ in range(600):
        scales = np.where(rng.random(4) < 0.4, rng.uniform(0.6, 1.0, 4), 1.0)
        request = (rng.uniform(0, 65535), rng.uniform(-40000, 40000), rng.uniform(-40000, 40000), rng.uniform(-20000, 20000))
        d = delivered(request, scales)
        cap = caps(scales)
        assert np.all(d >= -1e-6) and np.all(d <= cap + 1e-6)
        rp = O.ROLL_S * request[1] / 2 + O.PITCH_S * request[2] / 2
        lower, upper = np.max(-rp), np.min(cap - rp)        # the collective roll and pitch alone can sit at
        got = O.unmix(d)
        if np.all(O.mix_uncapped(*request) <= cap) and np.all(O.mix_uncapped(*request) >= 0):
            seen["feasible"] += 1
            np.testing.assert_allclose(d, O.mix_uncapped(*request), rtol=1e-9, atol=1e-6)
        elif lower <= request[0] <= upper:                  # roll, pitch and collective fit; yaw is what does not
            seen["yaw cut"] += 1
            assert got[1] == pytest.approx(request[1], abs=1e-5) and got[2] == pytest.approx(request[2], abs=1e-5)
            assert got[0] == pytest.approx(request[0], abs=1e-5)
            assert abs(got[3]) < abs(request[3]) and got[3] * request[3] >= 0
        elif lower <= upper and request[0] > upper:         # even without yaw the collective is too high for roll and pitch
            seen["collective cut"] += 1
            assert got[1] == pytest.approx(request[1], abs=1e-5) and got[2] == pytest.approx(request[2], abs=1e-5)
            assert got[3] == pytest.approx(0.0, abs=1e-5) and got[0] < request[0]
        else:                                               # roll and pitch need MORE collective than asked, or do not fit
            seen["not asked of it"] += 1
    assert all(v > 10 for v in seen.values()), seen                                   # the fuzz reached every branch
