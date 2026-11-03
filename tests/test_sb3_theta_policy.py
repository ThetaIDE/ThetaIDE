"""Real SB3 integration tests for selected PyTorch models."""

import importlib

import pytest
import torch


pytest.importorskip("stable_baselines3")
pytest.importorskip("gymnasium")


def test_blendrl_uses_its_configured_symbolic_observation_builder():
    from src.usr.models.blendrl.agents.blender_agent import BlenderActorCritic

    model = BlenderActorCritic.__new__(BlenderActorCritic)
    torch.nn.Module.__init__(model)
    obs = torch.tensor([[1.0, 2.0, 3.0]])
    model.cfg = {"model": {"symbolic": {"observation_builder": lambda x: x.unsqueeze(1) * 7}}}
    torch.testing.assert_close(model.build_logic_observation(obs), obs.unsqueeze(1) * 7)
    class Actor(torch.nn.Module):
        def get_q_values(self, neural_state, logic_state):
            return logic_state[:, 0]

    model.actor = Actor()
    torch.testing.assert_close(model.get_q_values(obs), obs * 7)
    model.cfg = {"model": {}, "env": {}}
    with pytest.raises(ValueError, match="symbolic observation builder"):
        model.build_logic_observation(obs)


def test_ppo_adapter_accepts_other_environment_and_model_without_symbolic_metadata():
    import gymnasium as gym
    from stable_baselines3 import PPO
    from src.app.core.model_registry import register_model
    from src.app.core.types import ActionResult

    @register_model("test_other_domain_policy")
    class OtherPolicy(torch.nn.Module):
        def __init__(self, n_actions, num_in_features, **kwargs):
            super().__init__()
            self.actor = torch.nn.Linear(num_in_features, n_actions)
            self.critic = torch.nn.Linear(num_in_features, 1)

        def get_action_probs(self, obs):
            return torch.softmax(self.actor(obs), dim=-1)

        def get_action_and_value(self, obs, action=None):
            dist = torch.distributions.Categorical(probs=self.get_action_probs(obs))
            action = dist.sample() if action is None else action
            return ActionResult(action, dist.log_prob(action), dist.entropy(), self.critic(obs))

    adapter = importlib.import_module("src.usr.algorithms.sb3-suite.theta_sb3_policy").ThetaActorCriticPolicy
    env = gym.make("MountainCar-v0")
    model = PPO(adapter, env, n_steps=8, batch_size=4, n_epochs=1,
                policy_kwargs={"theta_config": {"model": {}}, "theta_model": "test_other_domain_policy",
                               "env_name": "other_domain"})
    model.learn(total_timesteps=8)
    assert isinstance(model.policy.theta_model_module, OtherPolicy)
    env.close()


@pytest.mark.parametrize("actor_mode", ["hybrid", "logic", "neural"])
def test_ppo_trains_and_reloads_actual_blendrl(tmp_path, actor_mode):
    import gymnasium as gym
    from stable_baselines3 import PPO
    from src.usr.models.blendrl.agents.blender_agent import BlenderActorCritic
    from src.usr.models.dnn.dnn_model import DNNModel

    policy = importlib.import_module("src.usr.algorithms.sb3-suite.theta_sb3_policy").ThetaActorCriticPolicy
    cfg = {"model": {"architecture": "blendrl", "actor_mode": actor_mode, "blend_function": "softmax",
                     "blender": {"mode": "neural", "model": {"name": "basics/dnn", "hidden_sizes": [16, 16]}},
                     "neural": {"type": "dnn", "hidden_sizes": [32, 32]},
                     "symbolic": {"type": "nsfr", "rules": "variant2"}}, "agent": {},
           "env": {"symbolic_observation_builder": "in.envs.cartpole.valuation:build_symbolic_observation"}}
    kwargs = {"theta_config": cfg, "theta_model": "blendrl", "env_name": "cartpole",
              "action_meanings": ["left", "right"]}
    env = gym.make("CartPole-v1")
    model = PPO(policy, env, n_steps=8, batch_size=4, n_epochs=1, seed=4, policy_kwargs=kwargs)
    selected = model.policy.theta_model_module
    assert isinstance(selected, BlenderActorCritic)
    assert selected.module_types == ["logic", "neural"]
    neural = selected.policy_modules[1]
    assert isinstance(neural, DNNModel)
    assert isinstance(selected.blender, DNNModel)
    assert neural.network[0].out_features == 32
    assert selected.blender.network[0].out_features == 16
    before = {name: value.detach().clone() for name, value in selected.named_parameters()}
    model.learn(total_timesteps=16)
    after = dict(selected.named_parameters())
    if actor_mode == "hybrid":
        assert any(not torch.equal(before[name], after[name]) for name in before if name.startswith("policy_modules.1."))
        assert any(not torch.equal(before[name], after[name]) for name in before if name.startswith("blender."))
    obs = torch.tensor([[0., 0., -0.1, 0.]], dtype=torch.float32)
    probs_before = selected.get_action_probs(obs).detach()
    checkpoint = tmp_path / "blendrl.zip"
    model.save(checkpoint)
    restored = PPO.load(checkpoint, env=env)
    assert isinstance(restored.policy.theta_model_module, BlenderActorCritic)
    torch.testing.assert_close(restored.policy.theta_model_module.get_action_probs(obs), probs_before)
    env.close()


def test_missing_model_methods_report_compatibility_error():
    import gymnasium as gym
    from stable_baselines3 import PPO
    from src.app.core.model_registry import register_model

    @register_model("test_incomplete_sb3_model")
    class Incomplete(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

    policy = importlib.import_module("src.usr.algorithms.sb3-suite.theta_sb3_policy").ThetaActorCriticPolicy
    env = gym.make("CartPole-v1")
    with pytest.raises(ValueError, match="implement get_action_probs"):
        PPO(policy, env, policy_kwargs={"theta_config": {"model": {}}, "theta_model": "test_incomplete_sb3_model",
                                       "env_name": "cartpole", "action_meanings": ["left", "right"]})
    from stable_baselines3 import DQN
    dqn_policy = importlib.import_module("src.usr.algorithms.sb3-suite.theta_sb3_policy").ThetaDQNPolicy
    with pytest.raises(ValueError, match="implement get_q_values"):
        DQN(dqn_policy, env, policy_kwargs={"theta_config": {"model": {}}, "theta_model": "test_incomplete_sb3_model",
                                           "env_name": "cartpole", "action_meanings": ["left", "right"]})
    env.close()


def test_model_can_reject_an_engine_specific_configuration():
    from src.usr.models.blendrl.agents.blender_agent import BlenderActorCritic

    model = BlenderActorCritic.__new__(BlenderActorCritic)
    torch.nn.Module.__init__(model)
    model.cfg = {"model": {}}
    model.blend_function = "gumbel_softmax"
    with pytest.raises(ValueError, match="requires blend_function=softmax"):
        model.validate_engine_compatibility("sb3", "on_policy")
    model.validate_engine_compatibility("another_engine", "on_policy")


def test_dqn_uses_selected_model_q_values_and_reloads(tmp_path):
    import gymnasium as gym
    from stable_baselines3 import DQN
    from src.usr.models.mlp.mlp_model import MLPQNetwork

    policy = importlib.import_module("src.usr.algorithms.sb3-suite.theta_sb3_policy").ThetaDQNPolicy
    cfg = {"model": {"architecture": "basics/mlp", "hidden_dims": [11, 7], "activation": "relu"}}
    env = gym.make("CartPole-v1")
    model = DQN(policy, env, buffer_size=40, learning_starts=0, batch_size=4, train_freq=1,
                seed=3, policy_kwargs={"theta_config": cfg, "theta_model": "basics/mlp", "env_name": "cartpole",
                                       "action_meanings": ["left", "right"]})
    selected = model.policy.q_net.theta_model_module
    assert isinstance(selected, MLPQNetwork)
    assert [selected.network[0].out_features, selected.network[2].out_features] == [11, 7]
    obs = torch.tensor([[0., 0., -0.1, 0.]], dtype=torch.float32)
    torch.testing.assert_close(model.policy.q_net(obs), selected.get_q_values(obs))
    before = {name: value.detach().clone() for name, value in selected.named_parameters()}
    model.learn(total_timesteps=12)
    assert any(not torch.equal(before[name], value) for name, value in selected.named_parameters())
    values = selected.get_q_values(obs).detach()
    checkpoint = tmp_path / "dqn.zip"
    model.save(checkpoint)
    restored = DQN.load(checkpoint, env=env)
    assert isinstance(restored.policy.q_net.theta_model_module, MLPQNetwork)
    torch.testing.assert_close(restored.policy.q_net(obs), values)
    env.close()


def test_dqn_uses_blendrl_to_build_symbolic_observations():
    import gymnasium as gym
    from stable_baselines3 import DQN
    from src.usr.models.blendrl.agents.blender_agent import BlenderActorCritic

    adapter = importlib.import_module("src.usr.algorithms.sb3-suite.theta_sb3_policy").ThetaDQNPolicy
    cfg = {"model": {"architecture": "blendrl", "actor_mode": "hybrid", "blend_function": "softmax",
                     "blender": {"mode": "neural", "model": {"name": "basics/dnn", "hidden_sizes": [16, 16]}},
                     "neural": {"type": "dnn", "hidden_sizes": [32, 32]},
                     "symbolic": {"type": "nsfr", "rules": "variant2"}},
           "env": {"symbolic_observation_builder": "in.envs.cartpole.valuation:build_symbolic_observation"}}
    env = gym.make("CartPole-v1")
    model = DQN(adapter, env, buffer_size=40, learning_starts=0, batch_size=4, train_freq=1,
                policy_kwargs={"theta_config": cfg, "theta_model": "blendrl", "env_name": "cartpole",
                               "action_meanings": ["left", "right"]})
    selected = model.policy.q_net.theta_model_module
    assert isinstance(selected, BlenderActorCritic)
    obs = torch.tensor([[0., 0., -0.1, 0.]], dtype=torch.float32)
    torch.testing.assert_close(model.policy.q_net(obs), selected.get_q_values(obs))
    model.learn(total_timesteps=8)
    env.close()

@pytest.mark.parametrize('algorithm_name,policy_name', [('ThetaPPO', 'ThetaActorCriticPolicy'), ('ThetaDQN', 'ThetaDQNPolicy')])
def test_model_optimizer_groups_follow_sb3_schedule(algorithm_name, policy_name):
    import gymnasium as gym
    from src.app.core.model_registry import register_model
    from src.app.core.types import ActionResult

    @register_model('test_grouped_model')
    class Grouped(torch.nn.Module):
        def __init__(self, n_actions, num_in_features, **kwargs):
            super().__init__()
            self.actor = torch.nn.Linear(num_in_features, n_actions)
            self.critic = torch.nn.Linear(num_in_features, 1)

        def configure_optimizer_groups(self, base_lr, weight_decay=0, cfg=None):
            return [{'params': self.actor.parameters(), 'lr': base_lr * 2},
                    {'params': self.critic.parameters(), 'lr': base_lr * .5}]

        def get_action_probs(self, obs):
            return self.actor(obs).softmax(-1)

        def get_action_and_value(self, obs, action=None):
            d = torch.distributions.Categorical(probs=self.get_action_probs(obs))
            action = d.sample() if action is None else action
            return ActionResult(action, d.log_prob(action), d.entropy(), self.critic(obs))

        def get_q_values(self, obs):
            return self.actor(obs) + self.critic(obs)

    module = importlib.import_module('src.usr.algorithms.sb3-suite.theta_sb3_policy')
    env = gym.make('CartPole-v1')
    kwargs = {'theta_config': {'model': {}}, 'theta_model': 'test_grouped_model', 'env_name': 'cartpole'}
    algo = getattr(module, algorithm_name)
    extra = {'n_steps': 8, 'batch_size': 4, 'n_epochs': 1} if algorithm_name == 'ThetaPPO' else {'buffer_size': 40, 'learning_starts': 0, 'batch_size': 4}
    model = algo(getattr(module, policy_name), env, learning_rate=lambda progress: .001 * (1 + progress), policy_kwargs=kwargs, **extra)
    assert [g['lr'] for g in model.policy.optimizer.param_groups] == pytest.approx([.004, .001])
    model._current_progress_remaining = .5
    model._logger = __import__('stable_baselines3').common.logger.configure(folder=None, format_strings=[])
    model._update_learning_rate(model.policy.optimizer)
    assert [g['lr'] for g in model.policy.optimizer.param_groups] == pytest.approx([.003, .00075])
    model.learn(total_timesteps=8)
    assert model.policy.optimizer.param_groups[0]['lr'] == pytest.approx(4 * model.policy.optimizer.param_groups[1]['lr'])
    env.close()


def test_blendrl_optimizer_rates_allow_null_model_defaults_and_agent_fallback():
    from src.usr.models.blendrl.agents.blender_agent import BlenderActorCritic
    model = BlenderActorCritic.__new__(BlenderActorCritic)
    torch.nn.Module.__init__(model)
    model.cfg = {'model': {'logic_lr': None, 'blender_lr': .003}, 'agent': {'logic_lr': .002}}
    model.module_types = ['logic', 'neural']
    model.actor = torch.nn.Module()
    model.actor.policy_modules = torch.nn.ModuleList([torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)])
    model.actor.blender = torch.nn.Linear(2, 2)
    assert [g['lr'] for g in model.configure_optimizer_groups(.001)] == pytest.approx([.002, .001, .003])


def test_blendrl_routing_monitor_reports_experts_and_keeps_gradients_usable():
    import gymnasium as gym
    module = importlib.import_module('src.usr.algorithms.sb3-suite.theta_sb3_policy')
    cfg = {'model': {'architecture': 'blendrl', 'actor_mode': 'hybrid', 'blend_function': 'softmax',
                     'logic_lr': .002, 'blender_lr': .003,
                     'blender': {'mode': 'neural', 'model': {'name': 'basics/dnn', 'hidden_sizes': [16, 16]}},
                     'neural': {'type': 'dnn', 'hidden_sizes': [32, 32]},
                     'symbolic': {'type': 'nsfr', 'rules': 'variant2'}},
           'env': {'symbolic_observation_builder': 'in.envs.cartpole.valuation:build_symbolic_observation'}}
    env = gym.make('CartPole-v1')
    model = module.ThetaPPO(module.ThetaActorCriticPolicy, env, learning_rate=.001, n_steps=8, batch_size=4, n_epochs=1,
                            policy_kwargs={'theta_config': cfg, 'theta_model': 'blendrl', 'env_name': 'cartpole',
                                           'action_meanings': ['left', 'right']})
    selected = model.policy.theta_model_module
    metrics, samples = selected.collect_monitoring(torch.tensor([[0., 0., -.1, 0.], [0., 0., .1, 0.]]))
    assert samples["weights"].shape == (2, 2)
    assert 'blend/logic/weight_mean' in metrics
    assert 0 <= metrics['blend/logic_0/coverage'] <= 1
    assert 'blend/agreement/logic_0__neural_1/top_action' in metrics
    assert not selected.actor._capture_experts
    model.learn(total_timesteps=8)
    env.close()
