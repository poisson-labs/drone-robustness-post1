"""Training rule 17 (spec section 17.9: "the policy's actions (deterministic mean at evaluation, so a crash isn't from
dice in the network)"; section 2, 'Observation' and "No wind sensor and no privileged state, in any arm"; the bar of
section 2 flies each pi0 through `rollout_for`).

A trained network flies as a `Controller`. What it reads when it is evaluated is what it read when it trained: the fixed-
scale observation of the outer stream, with the action it chose the step before as the last action, and what it does is
the network's mean action mapped from +-1 units onto thrust and body rates. Section 12 names no test for this hand-over;
it is here because the bar's reading of a policy rests on it (ruling 4A-18). The network in these tests is untrained.

Needs torch and stable-baselines3: `uv sync --group gate`.
"""
import numpy as np
import pytest
from stable_baselines3 import PPO

import spec_numbers as S
import training_oracles as TO
from drone_study.condition import Condition
from drone_study.stack import Controller, fly
from drone_study.training.env import to_physical
from drone_study.training.hyper import make_ppo, save_checkpoint
from drone_study.training.policy import PolicyController, load_policy
from training_stand import stand_envs

STEPS = 30                                                                # 0.3 s: an untrained network has not left the radius yet
CONDITIONS = [Condition.calm("oval-medium"), Condition(case="fig8-fast", az=90.0, el=0.0, speed=6.0, phase=0.0, delay=40),
              Condition(case="circle-fast", turb=1.5, payload=10.0)]
_MODEL = {}


def network():
    if "m" not in _MODEL:
        model = make_ppo(stand_envs(2), seed=17)
        assert isinstance(model, PPO), "make_ppo built no PPO"
        _MODEL["m"] = model
    return _MODEL["m"]


def twin(model, condition, seed, steps=STEPS, deterministic=True):
    """The same episode flown by hand: the fixed-scale observation in, the network's action out, one step at a time."""
    replay = TO.Replay(condition, seed)
    positions, actions = [replay.stack.truth.position.copy()], []
    for _ in range(steps):
        action, _ = model.predict(replay.obs, deterministic=deterministic)
        actions.append(np.asarray(to_physical(action), dtype=float))
        replay.step(action)
        positions.append(replay.stack.truth.position.copy())
    return np.array(positions), np.array(actions)


def test_tr17_a_policy_is_an_outer_loop_with_no_privilege():
    ctl = PolicyController(network())
    assert isinstance(ctl, Controller) and ctl.privileges == frozenset()
    assert ctl.mixer(np.ones(4)) is None


@pytest.mark.parametrize("condition", CONDITIONS, ids=["calm", "gust-and-delay", "turbulence-and-payload"])
def test_tr17_evaluated_it_reads_what_it_read_in_training_and_takes_the_networks_mean_action(condition):
    model = network()
    record = fly(condition, 41, PolicyController(model), max_steps=STEPS)
    positions, actions = twin(model, condition, 41)
    assert record.n_steps == STEPS + 1
    np.testing.assert_array_equal(record.position, positions)                 # the same flight, to the bit
    np.testing.assert_array_equal(record.applied_action[1:], actions)         # each action takes effect one step later (10 ms)
    np.testing.assert_allclose(record.applied_action[0], [S.WEIGHT, 0.0, 0.0, 0.0], atol=1e-12)
    assert np.ptp(actions[:, 0]) > 0.0 or np.ptp(actions[:, 1]) > 0.0         # the network is steering, not a constant


def test_tr17_the_action_is_the_mean_not_a_sample():
    model = network()
    a = fly(CONDITIONS[0], 43, PolicyController(model), max_steps=STEPS)
    b = fly(CONDITIONS[0], 43, PolicyController(model), max_steps=STEPS)
    np.testing.assert_array_equal(a.position, b.position)                     # no dice in the network
    mean, _ = twin(model, CONDITIONS[0], 43, deterministic=True)
    sampled, _ = twin(model, CONDITIONS[0], 43, deterministic=False)
    np.testing.assert_array_equal(a.position, mean)
    assert not np.array_equal(sampled, mean)                                  # the control: sampling would show


def test_tr17_one_flyer_starts_every_episode_from_an_empty_history():
    model = network()
    ctl = PolicyController(model)
    first = fly(CONDITIONS[1], 44, ctl, max_steps=STEPS)
    fly(CONDITIONS[2], 45, ctl, max_steps=STEPS)                              # another episode in between, on the same object
    again = fly(CONDITIONS[1], 44, ctl, max_steps=STEPS)
    np.testing.assert_array_equal(first.position, again.position)
    np.testing.assert_array_equal(first.position, twin(model, CONDITIONS[1], 44)[0])


def test_tr17_a_saved_checkpoint_flies_as_the_network_it_was_saved_from(tmp_path):
    model = network()
    save_checkpoint(model, tmp_path / "pi0.zip")
    loaded = load_policy(tmp_path / "pi0.zip")
    assert isinstance(loaded, PolicyController)
    a = fly(CONDITIONS[0], 46, loaded, max_steps=STEPS)
    np.testing.assert_array_equal(a.position, twin(model, CONDITIONS[0], 46)[0])
