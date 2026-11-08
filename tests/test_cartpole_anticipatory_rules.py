"""Behavioral checks for the local anticipatory CartPole experiment assets."""

import importlib
from pathlib import Path

import pytest
import torch
import yaml

pytest.importorskip("stable_baselines3")
ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "in/config/experiment/Belndrl Bench/Cartpole Anticipatory Test.yaml"
pytestmark = pytest.mark.skipif(not RECIPE.exists(), reason="Local anticipatory experiment is not installed")


def valuation():
    from src.app.services.rulesets import resolve_ruleset
    resource = resolve_ruleset(ROOT, "cartpole", "anticipatory_v1")
    spec = importlib.util.spec_from_file_location("anticipatory_valuation", resource.valuation_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_policy():
    import gymnasium as gym

    recipe = yaml.safe_load(RECIPE.read_text())
    method = recipe["methods"]["BlendRL"]
    cfg = {"model": method["model"], "agent": method["agent"], "env": recipe["env"]}
    adapter = importlib.import_module("src.usr.algorithms.sb3-suite.theta_sb3_policy")
    env = gym.make("CartPole-v1")
    algo = adapter.ThetaPPO(
        adapter.ThetaActorCriticPolicy, env, seed=recipe["seed"],
        learning_rate=method["agent"]["learning_rate"], n_steps=32, batch_size=16, n_epochs=2,
        policy_kwargs={"theta_config": cfg, "theta_model": "blendrl",
                       "env_name": "cartpole", "action_meanings": ["left", "right"]},
    )
    return algo, env


def test_recovering_pole_overrides_current_lean_and_is_symmetric():
    vm = valuation()
    states = torch.tensor([[0., 0., .03, -.8], [0., 0., -.03, .8]])
    assert vm.short_fall_left(states)[0] > .98
    assert vm.long_fall_left(states)[0] > .98
    assert vm.short_fall_right(states)[1] > .98
    assert vm.long_fall_right(states)[1] > .98
    for left, right in [(vm.short_fall_left, vm.short_fall_right),
                        (vm.long_fall_left, vm.long_fall_right)]:
        torch.testing.assert_close(left(states), right(-states))
    # Near a reversal, the candidate horizons can disagree for learning.
    recovering = torch.tensor([[0., 0., .05, -.2]])
    assert vm.short_fall_right(recovering).item() > .98
    assert vm.long_fall_left(recovering).item() > .98


def test_exact_balance_abstains_and_observation_shape_is_checked():
    vm = valuation()
    state = torch.zeros(3, 4)
    torch.testing.assert_close(vm.build_symbolic_observation(state)[:, 0], state)
    for fn in [vm.short_fall_left, vm.short_fall_right, vm.long_fall_left, vm.long_fall_right]:
        assert torch.all(fn(state) < .02)
    with pytest.raises(ValueError, match="four state values"):
        vm.build_symbolic_observation(torch.zeros(3, 3))


def test_real_policy_loads_four_rules_and_ppo_updates_all_experts():
    algo, env = build_policy()
    try:
        model = algo.policy.theta_model_module
        logic = model.policy_modules[0]
        assert len(logic.clauses) == 4
        assert logic.im.W.shape == (2, 4)
        assert model.module_types == ["logic", "neural"]
        obs = torch.tensor([[0., 0., .03, -.8], [0., 0., -.03, .8],
                            [0., 0., .05, -.2], [0., 0., -.05, .2]])
        mapped = model.actor._map_logic_output(logic(model.build_logic_observation(obs)), logic)
        assert mapped[0, 0] > mapped[0, 1]
        assert mapped[1, 1] > mapped[1, 0]
        balanced = model.actor._map_logic_output(
            logic(model.build_logic_observation(torch.zeros(1, 4))), logic)
        torch.testing.assert_close(balanced, torch.full((1, 2), .5))
        parts = [logic, model.policy_modules[1], model.blender]
        initial = [[p.detach().clone() for p in part.parameters()] for part in parts]
        algo.learn(total_timesteps=64)
        for part, before in zip(parts, initial):
            assert any(not torch.equal(p, old) for p, old in zip(part.parameters(), before))
    finally:
        env.close()


def test_compiled_rule_only_controller_balances_held_out_starts():
    algo, env = build_policy()
    try:
        # Exercise the same 2x4 trainable reasoner used by PPO, holding a
        # uniform clause mixture fixed for this deterministic reference.
        logic = algo.policy.theta_model_module.policy_modules[0]
        with torch.no_grad():
            logic.im.W.zero_()
        vm = valuation()
        for seed in [91001, 91017, 91039]:
            obs, _ = env.reset(seed=seed)
            reward = 0
            while True:
                with torch.no_grad():
                    scores = logic(vm.build_symbolic_observation(torch.as_tensor(obs).unsqueeze(0)))
                action = logic.prednames.index("right") == scores.argmax(-1).item()
                obs, r, terminated, truncated, _ = env.step(int(action))
                reward += r
                if terminated or truncated:
                    break
            assert reward == 500
    finally:
        env.close()


def test_recipe_resolves_to_one_fixed_training_run():
    from src.app.pipeline.compose import compose_experiment, validate_composed
    from src.app.pipeline.config import parse_methods_dict
    from src.app.pipeline.variability import make_plan

    composed = compose_experiment("Belndrl Bench/Cartpole Anticipatory Test")
    validate_composed(composed)
    assert not composed.is_sweep
    methods = parse_methods_dict(composed.cfg)
    assert list(methods) == ["BlendRL"]
    assert not methods["BlendRL"].get("tune")
    plan = make_plan(composed.cfg, {"methods": methods, "is_interactive": True})
    assert plan["training_seeds"] == [1234679]
    assert composed.cfg.env.env_id == "CartPole-v1"
    assert composed.cfg.env.symbolic_name == "cartpole"
