"""Public slice (session P1c-repo, 2026-11-10): the three cases that build a retrain arm's environment on a finding are removed, and the every-arm cases run on the nominal arm, the one this repository carries. No kept case is changed.

Training rule 8 (spec section 4: "In training an episode ends at its first crash class, the same in every arm";
section 5, "Same crash termination"; section 2, "Calm air means calm air ... Motors, latency, sensor noise and
start-state spread are on, because a real drone has them"; section 6 for what a seed does).

The training environment flies the built stack in training mode, hands back the reward of rule 1 (with the heading term and
the heading-rate damping term of the yaw addenda of 2026-11-07 and 2026-11-08, on the true heading and the true Euler yaw rate
after the step, at the weight the environment was built with, and that weight's crash penalty), and says how each episode
ended: `terminated` for a crash class, `truncated` for the 10 s limit.

Re-frozen by 4D-A, twice: `env_for` hands the environment a ladder weight; the three reward tests read the two heading terms
beside the other three; one test is new (the heading terms read the true heading and the true heading rate after the step, and
the same episode at another weight pays in proportion, with that weight's penalty on a crash). The others are unchanged in what
they assert.
"""
import numpy as np
import pytest

import spec_numbers as S
import training_numbers as N
import training_oracles as TO
import yaw_oracles as Y
from drone_study.condition import Condition
from drone_study.controllers import GeometricController
from drone_study.training.env import TrainingEnv
from drone_study.training.reward import RewardTerms
from drone_study.training.sampler import make_sampler

DROP = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)                  # no thrust
SPIN = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)                   # full roll rate at about hover thrust
YAW = np.array([TO.HOLD_NORMALISED[0], 0.0, 0.0, 0.5], dtype=np.float32)   # hover thrust, 5 rad/s of yaw rate
K = N.YAW_WEIGHT_LADDER[1]                                                # the weight these environments train at


def heading_term_of(replay, k=K):
    """The addenda's heading terms on the replay's true state after the step, at k: (heading, heading rate, heading term, rate term)."""
    truth = replay.stack.truth
    psi = Y.heading_of_rotation(truth.rotation)
    rate = Y.euler_rate(truth.rotation, truth.body_rate)
    return psi, rate, Y.heading_term(psi, k), Y.rate_term(rate, k)


def env_for(arm, seed=5, k_yaw=K):
    return TrainingEnv(make_sampler(arm), seed, k_yaw)


def start(env, sources=("calm", "finding", "table")):
    obs, info = env.reset()
    assert info["spec"].source in sources
    return obs, info, TO.Replay(info["spec"].condition, info["episode_seed"])


@pytest.mark.parametrize("arm", ["nominal"])
@pytest.mark.parametrize("action", [DROP, SPIN], ids=["no-thrust", "full-roll-rate"])
def test_tr08_a_training_episode_ends_at_its_first_crash_class_in_every_arm(arm, action):
    env = env_for(arm)
    _, info, replay = start(env)
    assert info["crash_class"] is None and info["t"] == 0.0
    for k in range(S.N_STEPS):
        _, reward, terminated, truncated, info = env.step(action)
        _, _, done, reason = replay.step(action)
        assert terminated == done and not truncated
        if done:
            break
        assert info["crash_class"] is None
    assert terminated and k < 400                                             # it did crash, and the episode stopped there
    assert reason.startswith("training-crash:")
    assert info["crash_class"] == reason.split(":")[1] and info["crash_class"] in ("ground", "tilt", "radius")
    assert info["t"] == pytest.approx(0.01 * (k + 1), abs=1e-9)


def fly_with_controller(env, replay, max_steps=S.N_STEPS, heading=None):
    """Drive the environment with the geometric controller reading a twin of the same episode. Returns the per-step tuples; with
    `heading` a list, appends the twin's (heading, heading rate, heading term, rate term) after each step."""
    ctl = GeometricController(); ctl.reset()
    out = []
    for _ in range(max_steps):
        a = TO.quantised(ctl.act(replay.view()))
        obs, reward, terminated, truncated, info = env.step(a)
        _, error, done, reason = replay.step(a)
        out.append((a, reward, terminated, truncated, info, error, done, reason))
        if heading is not None:
            heading.append(heading_term_of(replay))
        if done:
            break
    return out


def test_tr08_an_episode_that_stays_up_runs_ten_seconds_and_is_cut_off_not_crashed():
    env = env_for("nominal")
    _, _, replay = start(env, ("calm",))
    steps = fly_with_controller(env, replay)
    assert len(steps) == S.N_STEPS == 1000
    a, reward, terminated, truncated, info, error, done, reason = steps[-1]
    assert truncated and not terminated and done and reason == "time"
    assert info["crash_class"] is None and info["reward_terms"].crash == 0.0
    assert all(not s[2] and not s[3] for s in steps[:-1])
    assert info["t"] == pytest.approx(S.EPISODE_S, abs=1e-9)


@pytest.mark.parametrize("arm", ["nominal"])
def test_tr08_each_steps_reward_is_rule_1_on_the_true_distance_the_true_heading_and_the_change_of_action(arm):
    env = env_for(arm)
    assert env.k_yaw == K
    _, _, replay = start(env)
    twin = []
    steps = fly_with_controller(env, replay, max_steps=150, heading=twin)
    assert len(steps) == 150 == len(twin)
    previous = TO.HOLD_NORMALISED                                             # before the first action: the hover hold
    for (a, reward, terminated, truncated, info, error, done, reason), (psi, rate, heading, damping) in zip(steps, twin):
        terms = info["reward_terms"]
        assert isinstance(terms, RewardTerms)
        assert info["tracking_error"] == pytest.approx(error, abs=1e-12) and error > 0.0     # the truth, not the noisy stream
        # psi, rate, heading, damping: the twin's true heading and heading rate after this step, read as it flew
        assert "heading_error" in info and "heading_rate" in info
        assert info["heading_error"] == pytest.approx(Y.wrap(psi), abs=1e-12)
        assert info["heading_rate"] == pytest.approx(rate, rel=1e-9, abs=1e-12)
        d = a.astype(float) - np.asarray(previous, dtype=float)
        assert terms.tracking == pytest.approx(-N.K_TRACK * error / N.TRACK_SCALE_M, rel=1e-9)
        assert terms.heading == pytest.approx(heading, rel=1e-9, abs=1e-15)
        assert terms.heading_rate == pytest.approx(damping, rel=1e-9, abs=1e-15)
        assert terms.smoothness == pytest.approx(-N.K_SMOOTH * float(d @ d), rel=1e-6, abs=1e-12)
        assert terms.crash == 0.0
        assert reward == pytest.approx(terms.tracking + terms.heading + terms.heading_rate + terms.smoothness, rel=1e-9)
        np.testing.assert_allclose(info["action"], a, atol=1e-7)
        previous = a
    assert np.linalg.norm(steps[-1][4]["position"] - steps[-1][4]["reference_position"]) == pytest.approx(steps[-1][5], abs=1e-12)
    assert abs(info["heading_error"]) < 0.2 and abs(info["heading_rate"]) < 1.0            # the geometric controller holds heading


def test_tr08_the_heading_terms_read_the_true_heading_and_heading_rate_after_the_step_at_the_environments_weight():
    """A commanded yaw rate turns the drone; the heading term grows with the wrapped true heading (Huber), the damping term reads the
    true Euler yaw rate, both at the weight handed to the environment, and the same command at another ladder weight pays in
    proportion."""
    def turn(env, replay, k):
        heads, rates, pays, damps, info = [], [], [], [], None
        for _ in range(60):
            _, reward, terminated, truncated, info = env.step(YAW)
            _, error, done, _ = replay.step(YAW)
            psi, rate, heading, damping = heading_term_of(replay, k)
            terms = info["reward_terms"]
            assert "heading_error" in info and "heading_rate" in info
            assert info["heading_error"] == pytest.approx(Y.wrap(psi), abs=1e-12)
            assert info["heading_rate"] == pytest.approx(rate, rel=1e-9, abs=1e-12)
            assert terms.heading == pytest.approx(heading, rel=1e-9, abs=1e-15) and terms.heading <= 0.0
            assert terms.heading_rate == pytest.approx(damping, rel=1e-9, abs=1e-15) and terms.heading_rate <= 0.0
            assert reward == pytest.approx(terms.tracking + terms.heading + terms.heading_rate + terms.smoothness + terms.crash, rel=1e-9)
            heads.append(abs(Y.wrap(psi)))
            rates.append(rate)
            pays.append(terms.heading)
            damps.append(terms.heading_rate)
            if done:
                break
        return np.array(heads), np.array(rates), np.array(pays), np.array(damps), info

    env = env_for("nominal")
    _, _, replay = start(env, ("calm",))
    heads, rates, pays, damps, info = turn(env, replay, K)
    assert len(heads) >= 20 and heads[-1] > 0.3 and pays[-1] < -K * (0.3 - N.YAW_HUBER_BAND_RAD / 2) / np.pi   # it turned and paid for it
    assert np.all(np.diff(heads[5:]) > 0) and np.all(np.diff(pays[5:]) < 0)              # monotone while it turns, under pi
    np.testing.assert_allclose(pays, [Y.heading_term(h, K) for h in heads], rtol=1e-9, atol=1e-15)
    assert np.max(rates) > 2.0 and np.min(damps[10:]) == pytest.approx(-K * N.YAW_RATE_WEIGHT, rel=1e-9)   # 5 rad/s commanded: the cap is reached and the damping saturates
    assert damps[0] > -K * N.YAW_RATE_WEIGHT * 0.5                                        # and the first step, before the rate loop answers, pays less
    other = N.YAW_WEIGHT_LADDER[2]
    env2 = env_for("nominal", k_yaw=other)                                               # the same seed, so the same episodes
    _, _, replay2 = start(env2, ("calm",))
    heads2, rates2, pays2, damps2, _ = turn(env2, replay2, other)
    np.testing.assert_array_equal(heads2, heads)                                          # the weight changes the pay, not the flight
    np.testing.assert_array_equal(rates2, rates)
    np.testing.assert_allclose(pays2, pays * (other / K), rtol=1e-9, atol=1e-15)
    np.testing.assert_allclose(damps2, damps * (other / K), rtol=1e-9, atol=1e-15)
    assert env2.k_yaw == other != env.k_yaw


def test_tr08_calm_air_has_no_wind_and_still_has_noise_latency_and_a_start_spread():
    env = env_for("nominal")
    starts, errors = [], []
    for _ in range(6):
        obs, info, replay = start(env, ("calm",))
        c = info["spec"].condition
        assert c.turb == 0.0 and not c.has_gust and c.payload == 0.0 and c.motor_index is None and c.delay == 0
        assert np.all(replay.stack.tapes.turbulence == 0.0)                   # no wind of any kind
        assert np.any(replay.stack.tapes.sensor.pos_ar != 0.0)                # the sensor noise is on
        starts.append(info["position"] - info["reference_position"])
        errors.append(info["tracking_error"])
        np.testing.assert_allclose(info["position"], replay.stack.truth.position, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(starts[-1]), errors[-1], atol=1e-12)
    spread = np.linalg.norm(np.array(starts), axis=1)
    assert np.all(spread > 0.0) and np.all(spread < 0.3) and len({round(float(x), 9) for x in spread}) == 6   # section 17.9's 3 cm, each its own

