"""The editable SB3 adapter (sb3-dev) drives selected PyTorch models through SB3.

Models are fixture packages in a private workspace. BlendRL and the Hub models inside these
adapters are covered by theta-plugins tests/test_native_integration.py.
"""

import json
import shutil
from pathlib import Path

import pytest
import torch


pytest.importorskip("stable_baselines3")
pytest.importorskip("gymnasium")

MODELS = """import torch
from src.app.api.models import ActionResult


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


class Incomplete(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))


class Grouped(OtherPolicy):
    def configure_optimizer_groups(self, base_lr, weight_decay=0, cfg=None):
        return [{'params': self.actor.parameters(), 'lr': base_lr * 2},
                {'params': self.critic.parameters(), 'lr': base_lr * .5}]

    def get_action_probs(self, obs):
        return self.actor(obs).softmax(-1)

    def get_q_values(self, obs):
        return self.actor(obs) + self.critic(obs)
"""
CAPABILITIES = {"test_other_domain_policy": "OtherPolicy", "test_incomplete_sb3_model": "Incomplete",
                "test_grouped_model": "Grouped"}


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A private workspace with the tracked sb3-dev adapter and a fixture model package."""
    from src.app.core.manifest_schema import validate_manifest_data
    from src.app.workspace import WorkspacePaths
    paths = WorkspacePaths(tmp_path)
    project = Path(__file__).resolve().parents[1]
    shutil.copytree(project / "src/usr/algorithms/sb3-dev", paths.user("algorithm") / "sb3-dev",
                    ignore=shutil.ignore_patterns("__pycache__"))
    models = paths.user("model") / "test-models"
    models.mkdir(parents=True)
    (models / "models.py").write_text(MODELS)
    manifest = {"id": "test-models", "name": "Test models", "kind": "model", "version": "1.0.0",
                "storage_version": 3, "api_version": 1,
                "capabilities": [{"id": key, "kind": "model", "entry_point": f"models.py:{name}"}
                                 for key, name in CAPABILITIES.items()],
                "payload": {"implementations": ["models.py"], "resources": [], "configs": {}},
                "dependencies": {"components": [], "pip": []}}
    assert validate_manifest_data(manifest) == (True, None)
    (models / ".theta_component.json").write_text(json.dumps(manifest))
    monkeypatch.setenv("THETA_WORKSPACE", str(tmp_path))
    monkeypatch.delenv("THETA_DATA_DIR", raising=False)
    return paths


def policy_module(paths):
    from src.app.core.provider_catalog import get_catalog
    from src.app.core.provider_loader import load_module
    return load_module(get_catalog(paths).package("sb3-dev"), "theta_sb3_policy.py", paths=paths)


def test_ppo_adapter_accepts_other_environment_and_model_without_symbolic_metadata(workspace):
    import gymnasium as gym
    from stable_baselines3 import PPO

    adapter = policy_module(workspace).ThetaActorCriticPolicy
    env = gym.make("MountainCar-v0")
    model = PPO(adapter, env, n_steps=8, batch_size=4, n_epochs=1,
                policy_kwargs={"theta_config": {"model": {}}, "theta_model": "test_other_domain_policy",
                               "env_name": "other_domain"})
    model.learn(total_timesteps=8)
    assert type(model.policy.theta_model_module).__name__ == "OtherPolicy"
    env.close()


def test_missing_model_methods_report_compatibility_error(workspace):
    import gymnasium as gym
    from stable_baselines3 import DQN, PPO

    module = policy_module(workspace)
    env = gym.make("CartPole-v1")
    kwargs = {"theta_config": {"model": {}}, "theta_model": "test_incomplete_sb3_model",
              "env_name": "cartpole", "action_meanings": ["left", "right"]}
    with pytest.raises(ValueError, match="implement get_action_probs"):
        PPO(module.ThetaActorCriticPolicy, env, policy_kwargs=kwargs)
    with pytest.raises(ValueError, match="implement get_q_values"):
        DQN(module.ThetaDQNPolicy, env, policy_kwargs=kwargs)
    env.close()


@pytest.mark.parametrize('algorithm_name,policy_name', [('ThetaPPO', 'ThetaActorCriticPolicy'), ('ThetaDQN', 'ThetaDQNPolicy')])
def test_model_optimizer_groups_follow_sb3_schedule(workspace, algorithm_name, policy_name):
    import gymnasium as gym
    module = policy_module(workspace)
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
