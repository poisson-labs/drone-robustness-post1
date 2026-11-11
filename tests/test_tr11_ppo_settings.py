"""Training rule 11 (spec section 2, 'Architecture and training': "A two-layer MLP, 256 units each, tanh. PPO from
Stable-Baselines3 with fixed hyperparameters, frozen at session 4A"; 'Observation scaling': "No running normaliser
anywhere"; section 14 and session 3B's gate for what an update costs).

The settings are ruling 4A-2, chosen with session 3B's measured update costs in view (gate-3b.md, section 3): at one
torch thread an update costs 162.6 CPU-seconds per million steps at minibatch 64, 72.7 at 256, 51.0 at 1,024 and 46.3
at 4,096, against about 500 core-seconds to fly the million steps; twelve threads burn 1.3 to 5 times the CPU. So:
one torch thread, 25 environments x 1,250 steps (31,250 a rollout, which divides every budget), minibatch 1,250
(25 a pass), 10 passes.

Needs torch and stable-baselines3: `uv sync --group gate`.
"""
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

import training_numbers as N
from drone_study.training import hyper as H
from drone_study.training.hyper import make_ppo
from training_stand import stand_envs

@pytest.fixture(scope="module")
def model():
    return make_ppo(stand_envs(N.HYPER["n_envs"]), seed=1234)


def test_tr11_the_frozen_settings_are_the_table(model):
    got = {k: (tuple(v) if isinstance(v, (list, tuple)) else v) for k, v in dict(H.HYPERPARAMETERS).items()}
    assert got == N.HYPER
    assert H.NOMINAL_LEARNING_RATE == N.HYPER["nominal_learning_rate"] == 3e-4
    assert H.FINE_TUNE_LEARNING_RATE == N.FINE_TUNE_LR == 1e-4            # section 5


def test_tr11_ppo_is_stable_baselines3s_with_every_frozen_setting(model):
    assert isinstance(model, PPO)
    h = N.HYPER
    assert (model.n_envs, model.n_steps, model.batch_size, model.n_epochs) == (h["n_envs"], h["n_steps"], h["batch_size"], h["n_epochs"])
    assert (model.gamma, model.gae_lambda, model.ent_coef, model.vf_coef, model.max_grad_norm) == \
           (h["gamma"], h["gae_lambda"], h["ent_coef"], h["vf_coef"], h["max_grad_norm"])
    assert model.target_kl is None and model.clip_range_vf is None and model.normalize_advantage is True and model.use_sde is False
    for progress in (1.0, 0.5, 0.0):                                          # constant through the run, no schedule
        assert model.lr_schedule(progress) == pytest.approx(h["nominal_learning_rate"], rel=1e-9)
        assert model.clip_range(progress) == pytest.approx(h["clip_range"], rel=1e-9)
    assert model.seed == 1234 and str(model.device) == "cpu"
    assert model.rollout_buffer.buffer_size * model.n_envs == N.ROLLOUT_STEPS == 31_250


def test_tr11_the_network_is_two_layers_of_256_tanh_for_the_actor_and_for_the_critic(model):
    assert isinstance(model, PPO)
    for net in (model.policy.mlp_extractor.policy_net, model.policy.mlp_extractor.value_net):
        linear = [m for m in net if isinstance(m, torch.nn.Linear)]
        assert [(m.in_features, m.out_features) for m in linear] == [(319, 256), (256, 256)]
        assert [type(m) for m in net if not isinstance(m, torch.nn.Linear)] == [torch.nn.Tanh, torch.nn.Tanh]
    assert model.policy.mlp_extractor.policy_net[0] is not model.policy.mlp_extractor.value_net[0]     # the critic has its own layers
    assert (model.policy.action_net.in_features, model.policy.action_net.out_features) == (256, 4)
    assert (model.policy.value_net.in_features, model.policy.value_net.out_features) == (256, 1)
    assert model.policy.log_std.shape == (4,) and torch.all(model.policy.log_std == N.HYPER["log_std_init"])
    assert model.policy.squash_output is False
    opt = model.policy.optimizer
    assert isinstance(opt, torch.optim.Adam) and opt.defaults["eps"] == N.HYPER["adam_eps"] and opt.defaults["weight_decay"] == 0


def test_tr11_nothing_normalises_what_the_network_reads(model):
    assert isinstance(model, PPO)
    assert model.get_vec_normalize_env() is None and not isinstance(model.get_env(), VecNormalize)
    assert type(model.policy.features_extractor).__name__ == "FlattenExtractor"
    names = [type(m).__name__ for m in model.policy.modules()]
    assert not any("Norm" in n for n in names)                                # no batch, layer or running normaliser in the network
    obs = np.random.default_rng(0).uniform(-5, 5, (3, 319)).astype(np.float32)
    features = model.policy.extract_features(model.policy.obs_to_tensor(obs)[0])
    if isinstance(features, tuple):
        features = features[0]
    np.testing.assert_array_equal(features.detach().numpy(), obs)             # the 319 numbers reach the first layer as they are


def test_tr11_a_rollout_is_31250_steps_and_a_pass_is_25_minibatches(model):
    assert isinstance(model, PPO)
    assert model.n_steps * model.n_envs == 31_250 and 31_250 % model.batch_size == 0
    assert 31_250 // model.batch_size == 25
    assert all(r % 31_250 == 0 and (r // 4) % 31_250 == 0 and (r // 16) % 31_250 == 0 for r in N.RUNGS)


def test_tr11_updates_run_on_one_torch_thread_the_cheap_setting_in_3bs_gate(model):
    assert isinstance(model, PPO)
    assert N.HYPER["torch_threads"] == 1 and torch.get_num_threads() == 1
    for batch in N.UPDATE_COST_1_THREAD:                                      # 3B measured more threads as more CPU at every size
        assert N.UPDATE_COST_12_THREADS[batch] > N.UPDATE_COST_1_THREAD[batch]
    assert 1024 <= model.batch_size <= 4096                                   # in the range where 3B's cost is flat: 51.0 to 46.3
    assert N.UPDATE_COST_1_THREAD[64] > 3 * N.UPDATE_COST_1_THREAD[1024]      # and well away from minibatch 64


def test_tr11_two_builds_on_one_seed_start_from_the_same_weights_and_another_seed_does_not():
    a, b, c = make_ppo(stand_envs(2), seed=7), make_ppo(stand_envs(2), seed=7), make_ppo(stand_envs(2), seed=8)
    assert isinstance(a, PPO)
    sa, sb, sc = a.policy.state_dict(), b.policy.state_dict(), c.policy.state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert any(not torch.equal(sa[k], sc[k]) for k in sa)
