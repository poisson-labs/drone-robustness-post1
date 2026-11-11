"""Public slice (session P1c-repo, 2026-11-10): the three cases that build a sampler of a retrain arm are removed, and the every-arm case runs on the nominal arm, the one this repository carries. No kept case is changed.

Training rule 7 (spec section 2, 'Observation' and 'Observation scaling'; section 5, 'The same for every fine-tune';
section 12 inv. 16): "Fixed physical scales from the first step, the same in every arm ... all clipped at +-5. No
running normaliser anywhere." And section 5: "The observation scales are the fixed physical ones of section 2, because
a running normaliser would hand wide-range arms a calm-air penalty that isn't about learning."

Session 3A froze the scales on `observe()`. These tests hold the training environment to them: what the network is
handed on every step of every arm is the 3B observer's 319 numbers for the stack's own readings, and nothing between
the stack and the network keeps a statistic. The policy's action is in +-1 units, mapped onto thrust 0 to the maximum
and rates within +-10 rad/s (ruling 4A-10), the range the observation's last-action entries are mapped back from.
"""
import numpy as np
import pytest

import spec_numbers as S
import training_numbers as N
import training_oracles as TO
from drone_study.training import env as E
from drone_study.training.env import TrainingEnv, make_envs, to_normalised, to_physical
from drone_study.training.sampler import make_sampler

def sampler_for(arm):
    return make_sampler(arm)


K = N.YAW_WEIGHT_LADDER[1]                                                 # the heading term's weight the environments are built with (4D-A)


def wiggle(n, seed):
    """A gentle open-loop action sequence in +-1 units, as float32: near hover with small rate commands."""
    rng = np.random.default_rng(seed)
    a = np.tile(TO.HOLD_NORMALISED, (n, 1)) + rng.normal(0.0, [0.05, 0.03, 0.03, 0.03], (n, 4))
    return np.clip(a, -1.0, 1.0).astype(np.float32)


def test_tr07_the_spaces_are_the_319_numbers_within_plus_minus_five_and_four_actions_within_plus_minus_one():
    env = TrainingEnv(sampler_for("nominal"), 1, K)
    assert env.observation_space.shape == (S.OBS_SIZE,) == (319,)
    assert env.observation_space.dtype == np.float32
    assert np.all(env.observation_space.low == -S.CLIP) and np.all(env.observation_space.high == S.CLIP)
    assert env.action_space.shape == (4,) and env.action_space.dtype == np.float32
    assert np.all(env.action_space.low == -1.0) and np.all(env.action_space.high == 1.0)


def test_tr07_an_action_maps_from_plus_minus_one_onto_thrust_and_body_rates_and_is_clipped():
    np.testing.assert_allclose(to_physical([-1.0, 0.0, 0.0, 0.0]), [0.0, 0.0, 0.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(to_physical([1.0, 1.0, -1.0, 0.5]), [S.T_MAX, 10.0, -10.0, 5.0], rtol=1e-12)
    np.testing.assert_allclose(to_physical([0.0, 0.25, 0.0, -0.1]), [S.T_MAX / 2, 2.5, 0.0, -1.0], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(to_physical([7.0, -3.0, 1.5, -9.0]), [S.T_MAX, -10.0, 10.0, -10.0], rtol=1e-12)   # section 2's ranges hold
    rng = np.random.default_rng(0)
    for a in rng.uniform(-1, 1, (50, 4)):
        np.testing.assert_allclose(to_normalised(to_physical(a)), a, atol=1e-12)
    np.testing.assert_allclose(E.HOLD_NORMALISED, TO.HOLD_NORMALISED, atol=1e-12)       # hover: thrust = weight, no rates
    np.testing.assert_allclose(to_physical(E.HOLD_NORMALISED), [S.WEIGHT, 0.0, 0.0, 0.0], atol=1e-12)


@pytest.mark.parametrize("arm", ["nominal"])
def test_tr07_the_network_is_handed_the_fixed_scale_observation_of_the_stacks_readings_in_every_arm(arm):
    env = TrainingEnv(sampler_for(arm), 11, K)
    wanted = {"calm"} if N.CALM_SHARE[arm] == 1.0 else ({"finding"} if N.CALM_SHARE[arm] == 0.0 else
                                                       {"calm", "table" if "dr" in arm else "finding"})
    compared = {}
    for episode in range(80):
        obs, info = env.reset()
        spec = info["spec"]
        assert spec.source in wanted
        if compared.get(spec.source, 0) >= 2:
            continue                                                          # two episodes of each kind are enough
        compared[spec.source] = compared.get(spec.source, 0) + 1
        replay = TO.Replay(spec.condition, info["episode_seed"])
        assert obs.dtype == np.float32 and obs.shape == (319,)
        np.testing.assert_array_equal(obs, replay.obs)                        # the first observation, to the bit
        for a in wiggle(60, 100 * episode + 7):
            obs, _, terminated, truncated, info = env.step(a)
            want, _, done, _ = replay.step(a)
            np.testing.assert_array_equal(obs, want)
            assert np.abs(obs).max() <= S.CLIP
            assert (terminated or truncated) == done
            if done:
                break
        if all(compared.get(k, 0) >= 2 for k in wanted):
            break
    assert set(compared) == wanted and all(v >= 2 for v in compared.values())

