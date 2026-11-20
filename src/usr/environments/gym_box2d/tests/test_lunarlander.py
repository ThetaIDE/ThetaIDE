"""Provider protocol and native-unit rule checks; run from the workspace root."""
import importlib.util
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[5]


def load(path):
    spec = importlib.util.spec_from_file_location("lunar_test_module", ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vector_protocol_and_seed():
    provider = load("src/usr/environments/gym_box2d/env_vectorized.py")
    env = provider.VectorizedEnv(n_envs=2)
    try:
        first = env.reset(seed=842)
        torch.testing.assert_close(first, env.reset(seed=842), rtol=0, atol=0)
        assert first.shape == (2, 8)
        assert int(env.n_actions) == 4
        assert env.get_action_meanings() == ["noop", "fire_left", "fire_main", "fire_right"]
        obs, reward, terminated, truncated, info = env.step(torch.tensor([0, 2]))
        assert obs.shape == (2, 8)
        assert reward.shape == terminated.shape == truncated.shape == (2,)
        assert torch.isfinite(obs).all()
    finally:
        env.close()
    with pytest.raises(ValueError, match="positive integer"):
        provider.VectorizedEnv(n_envs=0)


def test_full_controller_matches_gymnasium_heuristic():
    from gymnasium.envs.box2d.lunar_lander import heuristic
    full = load("src/usr/environments/gym_box2d/resources/rules/lunarlander/full_controller/valuation.py")
    rng = np.random.default_rng(843)
    states = rng.normal(size=(1000, 8)).astype(np.float32)
    states[:, 6:] = rng.integers(0, 2, size=(1000, 2))
    env = gym.make("LunarLander-v3")
    try:
        expected = np.array([heuristic(env, state) for state in states])
        np.testing.assert_array_equal(full.controller_action(torch.tensor(states)).numpy(), expected)
        batch = torch.tensor(states)
        side = full.attitude_priority(batch) > .5
        clauses = torch.stack([
            side & (full.attitude_settled(batch) > .5),
            side & (full.rotate_left(batch) > .5),
            (full.main_needed(batch) > .5) & (full.descent_priority(batch) > .5),
            side & (full.rotate_right(batch) > .5)], dim=1)
        assert torch.all(clauses.sum(dim=1) == 1)
        np.testing.assert_array_equal(clauses.int().argmax(dim=1).numpy(), expected)
    finally:
        env.close()


def test_full_rules_retain_horizontal_position_and_velocity_feedback():
    full = load("src/usr/environments/gym_box2d/resources/rules/lunarlander/full_controller/valuation.py")
    state = torch.tensor([[0., 1., 0., 0., 0., 0., 0., 0.]])
    assert full.controller_action(state).item() == 0
    for feature in [0, 2]:
        shifted = state.clone()
        shifted[0, feature] = 1.
        assert full.controller_action(shifted).item() == 1
        shifted[0, feature] = -1.
        assert full.controller_action(shifted).item() == 3


def test_partial_rules_ignore_horizontal_position_and_map_engine_actions():
    partial = load("src/usr/environments/gym_box2d/resources/rules/lunarlander/attitude_descent/valuation.py")
    states = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., -0.4, 0., 0., 0.],
                           [0., 0., 0., -1., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0.4, 0., 0., 0.]])
    assert partial.controller_action(states).tolist() == [0, 1, 2, 3]
    moved = states.clone()
    moved[:, 0] = 2.
    moved[:, 2] = -3.
    torch.testing.assert_close(partial.controller_action(states), partial.controller_action(moved))
    truths = torch.stack([partial.idle(states), partial.correct_left(states),
                          partial.thrust(states), partial.correct_right(states)], dim=1)
    assert (truths > 0.5).sum(dim=1).tolist() == [1, 1, 1, 1]
