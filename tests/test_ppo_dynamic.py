from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from src.app.api.dynamic import Boundary, MutableParameter, Replay


def test_replay_validates_parameters_and_alignment():
    parameters = {"agent.learning_rate": MutableParameter(0, exclusive_minimum=True)}
    change = {"step": 8, "parameters": {"agent.learning_rate": 0.01}}
    replay = Replay(OmegaConf.create([change]), parameters, rollout=8, start=0, stop=16)
    assert replay.update(Boundary(0, {}, {})) == {}
    assert replay.update(Boundary(8, {}, {})) == change["parameters"]
    for invalid in [
        {"step": 9, "parameters": change["parameters"]},
        {"step": 8, "parameters": {"agent.learning_rate": 0}},
        {"step": 8, "parameters": {"agent.gamma": 0.9}},
    ]:
        with pytest.raises(ValueError):
            Replay([invalid], parameters, rollout=8, start=0, stop=16)


def test_native_ppo_transfer_preserves_progress_and_applies_recipient_update(tmp_path):
    sb3 = pytest.importorskip("stable_baselines3")
    torch = pytest.importorskip("torch")
    gym = pytest.importorskip("gymnasium")
    from src.usr.algorithms.sb3_dev.ppo_dynamic import Session

    cfg = OmegaConf.create({
        "agent": {"policy": "MlpPolicy"}, "env": {"name": "CartPole-v1"},
        "total_timesteps": 8, "theta_dynamic": {"api_version": 1},
    })
    context = SimpleNamespace(check_cancelled=lambda: None)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    donor_env, recipient_env = gym.make("CartPole-v1"), gym.make("CartPole-v1")
    try:
        donor = sb3.PPO("MlpPolicy", donor_env, n_steps=8, batch_size=4, n_epochs=1, seed=11)
        donor_session = Session(donor, cfg, tmp_path, 11, context)
        donor.learn(8)
        checkpoint = donor_session.checkpoint()

        recipient = sb3.PPO("MlpPolicy", recipient_env, n_steps=8, batch_size=4, n_epochs=1, seed=13)
        cfg.theta_dynamic = {"api_version": 1, "start": 8, "source": str(checkpoint),
                             "schedule": [{"step": 8, "parameters": {
                                 "agent.learning_rate": 0.002, "agent.ent_coef": 0.01}}]}
        session = Session(recipient, cfg, tmp_path, 13, context)
        assert recipient.num_timesteps == donor.num_timesteps == 8
        assert recipient._n_updates == donor._n_updates
        for left, right in zip(recipient.policy.parameters(), donor.policy.parameters(), strict=True):
            torch.testing.assert_close(left, right)
        session.boundary()
        recipient.learn(8, reset_num_timesteps=False)
        assert recipient.num_timesteps == 16
        assert recipient.ent_coef == 0.01
        assert recipient.lr_schedule(0.0) == 0.002
        assert all(group["lr"] == 0.002 for group in recipient.policy.optimizer.param_groups)

        cfg.env.name = "incompatible"
        with pytest.raises(ValueError, match="Incompatible or incomplete"):
            Session(recipient, cfg, tmp_path, 13, context)
    finally:
        donor_env.close()
        recipient_env.close()
        torch.set_num_threads(threads)


def test_local_tuners_use_runner_contract_without_provider_names(monkeypatch):
    from src.app.pipeline import local_budget_tuning
    from src.usr.tuners.dehb.component import Tuner as DEHBTuner
    from src.usr.tuners.pbt.component import Tuner as PBTTuner

    profile = {
        "unit": "transitions", "quantum": 4, "budget_path": "example.budget",
        "metrics": ("eval/custom",), "protected_paths": ("agent.rollout_size",),
        "transfer_format": "example.training/v1",
        "mutable_parameters": {"model.noise": MutableParameter(-1, maximum=1)},
    }
    monkeypatch.setattr(local_budget_tuning, "runner_tuning_profile", lambda cfg, *, paths: profile)
    cfg = OmegaConf.create({
        "agent": {"name": "independent/algorithm"},
        "tuning": {"metric": "eval/custom"},
        "theta_hpo_space": {"model.noise": {"type": "float", "low": -0.5, "high": 0.5}},
    })
    DEHBTuner().validate_runner(cfg, paths=None)
    PBTTuner().validate_runner(cfg, paths=None)

    cfg.theta_hpo_space["model.noise"].high = 2
    with pytest.raises(ValueError, match="above the allowed range"):
        PBTTuner().validate_runner(cfg, paths=None)

    cfg.theta_hpo_space["agent.rollout_size"] = {"type": "int", "low": 4, "high": 8}
    with pytest.raises(ValueError, match="change the runner's budget"):
        DEHBTuner().validate_runner(cfg, paths=None)
