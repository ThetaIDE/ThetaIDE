"""Regression coverage for partial rules and gate/value interactions."""
import importlib

import pytest
import torch

pytest.importorskip("stable_baselines3")


def build_policy():
    import gymnasium as gym
    module = importlib.import_module("src.usr.algorithms.sb3-suite.theta_sb3_policy")
    cfg = {"model": {"actor_mode": "hybrid", "blend_function": "softmax", "logic_confidence_floor": .05,
                     "neural": {"type": "dnn", "hidden_sizes": [32, 32], "activation": "tanh",
                                "dueling": False, "shared_features": False},
                     "symbolic": {"type": "nsfr", "rules": "variant2"},
                     "blender": {"mode": "neural", "model": {"name": "basics/dnn", "hidden_sizes": [8]}}},
           "env": {"symbolic_observation_builder": "in.envs.cartpole.valuation:build_symbolic_observation"}}
    env = gym.make("CartPole-v1")
    algo = module.ThetaPPO(module.ThetaActorCriticPolicy, env, seed=4, n_steps=16, batch_size=8,
                           n_epochs=2, policy_kwargs={"theta_config": cfg, "theta_model": "blendrl",
                                                     "env_name": "cartpole", "action_meanings": ["left", "right"]})
    return algo, env


def test_actual_partial_rules_abstain_when_unmatched_and_report_coverage():
    algo, env = build_policy()
    try:
        model = algo.policy.theta_model_module
        obs = torch.tensor([[0., 0., -.1, 0.], [0., 0., .1, 0.], [0., 0., 0., 0.]])
        logic = model.policy_modules[0]
        mapped = model.actor._map_logic_output(logic(model.build_logic_observation(obs)), logic)
        assert mapped[0, 0] > .99
        torch.testing.assert_close(mapped[1:], torch.full((2, 2), .5))
        metrics, _ = model.collect_monitoring(obs)
        assert metrics["blend/logic_0/coverage"] == pytest.approx(1 / 3)
        # Neural expert options must not be silently discarded by the composite.
        assert isinstance(model.policy_modules[1].network[1], torch.nn.Tanh)
        algo.learn(total_timesteps=32)
    finally:
        env.close()


def test_weak_rule_confidence_remains_differentiable_above_the_noise_floor():
    algo, env = build_policy()
    try:
        model = algo.policy.theta_model_module
        score = torch.tensor([[.3], [.9]], requires_grad=True)
        probs = model.actor._map_logic_output(score, model.policy_modules[0])
        assert .5 < probs[0, 0] < probs[1, 0] < 1
        probs[:, 0].sum().backward()
        assert torch.all(score.grad > 0)
        # Other expert plugins already return action probabilities.
        distribution = torch.tensor([[.7, .3]])
        torch.testing.assert_close(model.actor._map_logic_output(distribution, torch.nn.Identity()), distribution)
        # Missing calibration in historical checkpoint configs retains behavior.
        model.actor.logic_confidence_floor = None
        torch.testing.assert_close(model.actor._map_logic_output(torch.tensor([[.02]]), model.policy_modules[0]),
                                   torch.tensor([[1., 0.]]))
    finally:
        env.close()


def test_missing_symbolic_critic_does_not_reward_gate_collapse():
    algo, env = build_policy()
    try:
        model = algo.policy.theta_model_module
        assert model.logic_critic is None
        obs = torch.tensor([[0., 0., -.1, 0.], [0., 0., .1, 0.]])
        logits = torch.tensor([[5., -5.], [-5., 5.]], requires_grad=True)
        value = model.get_value(obs, blending_weights=logits.softmax(-1))
        torch.testing.assert_close(value, model.get_neural_value(obs))
        value.sum().backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits), atol=1e-7, rtol=0)
    finally:
        env.close()


def test_independent_value_features_keep_value_loss_out_of_policy_trunk(tmp_path):
    algo, env = build_policy()
    try:
        model = algo.policy.theta_model_module
        neural = model.policy_modules[1]
        obs = torch.tensor([[.01, .02, -.1, .01], [.02, .01, .1, -.01]])
        neural.get_value(obs).square().sum().backward()
        assert all(p.grad is None for p in neural.network.parameters())
        assert any(p.grad is not None and torch.any(p.grad != 0) for p in neural.value_network.parameters())
        algo.learn(total_timesteps=32)
        checkpoint = tmp_path / "independent.zip"
        algo.save(checkpoint)
        restored = type(algo).load(checkpoint)
        torch.testing.assert_close(restored.policy.theta_model_module.get_action_probs(obs), model.get_action_probs(obs))
        torch.testing.assert_close(restored.policy.theta_model_module.get_value(obs), model.get_value(obs))
    finally:
        env.close()


def test_symbolic_gate_names_follow_expert_order():
    from src.usr.models.blendrl.agents.blender_agent import BlenderActor

    class Gate(torch.nn.Module):
        def get_prednames(self):
            return ["neural_agent", "logic_agent"]

    for kinds in [["logic", "neural"], ["neural", "logic"]]:
        actor = BlenderActor(None, [torch.nn.Identity(), torch.nn.Identity()], kinds,
                             Gate(), "hybrid", "logic", "softmax")
        assert actor.blender_id_to_pred_indices == {i: [0 if k == "neural" else 1]
                                                    for i, k in enumerate(kinds)}
