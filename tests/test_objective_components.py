"""Objectives score full seed panels, preserve failures, and execute edited source."""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.app.objectives.api import (
    ObjectiveResult, RunResult, TrialResults, create_objective, load_objective, validate_objective,
)
from src.app.objectives.runner import run_objective


def run(seed, points, status="completed"):
    return RunResult(seed, status, events=[{"training_step": x, "metric": "eval/reward", "episode_outcomes": [y]} for x, y in points])


def test_auc_rewards_speed_and_uses_worst_seeds():
    component = load_objective("learning_curve")
    params = component.parameters({"curve": {"budget": 100}})
    fast = run(1, [(0, 0), (20, 500), (100, 500)])
    slow = run(2, [(0, 0), (100, 500)])
    result = component.evaluate(TrialResults([fast, slow], {}), params)
    assert result.diagnostics["per_run_auc"] == pytest.approx([0.9, 0.5])
    assert result.value == pytest.approx(0.5)
    assert component.evaluate(TrialResults([fast, RunResult(3, "failed")], {}), params).value == 0


def test_exponential_auc_and_variance_penalty():
    component = load_objective("learning_curve")
    params = component.parameters({"curve": {"budget": 100, "half_life": 100}, "aggregation": {"method": "mean_minus_std", "std_penalty": 1}})
    result = component.evaluate(TrialResults([run(1, [(0, 0), (100, 500)])], {}), params)
    decay = math.log(2) / 100
    expected = (1 - math.exp(-decay*100) * (1 + decay*100)) / (100 * decay * (1-math.exp(-decay*100)))
    assert result.value == pytest.approx(expected)
    params["curve"]["half_life"] = 0
    result = component.evaluate(TrialResults([run(1, [(0, 0), (100, 500)]), run(2, [(0, 500), (100, 500)])], {}), params)
    assert result.value == pytest.approx(0.75 - math.sqrt(0.125))


def test_curves_must_cover_same_budget_and_clip_overshoot():
    component = load_objective("learning_curve")
    params = component.parameters({"curve": {"budget": 100}})
    with pytest.raises(ValueError, match="through budget"):
        component.evaluate(TrialResults([run(1, [(20, 0), (100, 500)])], {}), params)
    result = component.evaluate(TrialResults([run(1, [(0, 0), (200, 500)])], {}), params)
    assert result.value == pytest.approx(0.25)


def test_late_window_prefers_sustained_learning_and_interpolates_boundaries():
    component = load_objective("learning_curve")
    params = component.parameters({"curve": {"budget": 100, "start_fraction": .75},
                                   "aggregation": {"method": "mean_lower_tail", "fraction": .5,
                                                   "tail_weight": .25}})
    early_collapse = run(1, [(0, 500), (50, 500), (75, 0), (100, 0)])
    late_success = run(2, [(0, 0), (50, 0), (75, 500), (100, 500)])
    result = component.evaluate(TrialResults([early_collapse, late_success], {}), params)
    assert result.diagnostics["per_run_auc"] == [0, 1]
    assert result.value == pytest.approx(.375)
    assert result.diagnostics["completed_window_minima"] == [0, 500]
    result = component.evaluate(TrialResults([run(3, [(0, 0), (200, 1000)])], {}), params)
    assert result.value == pytest.approx(.875)
    params["curve"]["half_life"] = 25
    result = component.evaluate(TrialResults([late_success], {}), params)
    assert result.value == pytest.approx(1)
    assert component.evaluate(TrialResults([RunResult(4, "failed"), late_success], {}), params).value == pytest.approx(.375)
    with pytest.raises(ValueError, match="start_fraction"):
        component.parameters({"curve": {"start_fraction": 1}})


def test_workspace_objectives_reload_and_schema_is_optional(tmp_path):
    component = create_objective("mine", workspace=tmp_path)
    component.source.write_text("def evaluate(results, params):\n    return 12\n")
    assert component.evaluate(TrialResults([], {}), {}).value == 12
    component.source.write_text("def evaluate(results, params):\n    return 13\n")
    assert component.evaluate(TrialResults([], {}), {}).value == 13
    manifest = component.source.with_suffix(".json")
    metadata = json.loads(manifest.read_text())
    metadata.pop("schema")
    metadata.pop("defaults")
    manifest.write_text(json.dumps(metadata))
    assert load_objective("mine", tmp_path).parameters({"anything": {"nested": 1}}) == {"anything": {"nested": 1}}
    with pytest.raises(ValueError, match="already exists"):
        create_objective("mine", workspace=tmp_path)


def test_workspace_objective_discovery_does_not_depend_on_launch_directory(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import src.app.workspace
    from src.app.objectives.api import available_objectives

    workspace = tmp_path / "workspace"
    component = create_objective("custom", workspace=workspace)
    launch_directory = tmp_path / "launcher"
    launch_directory.mkdir()
    monkeypatch.chdir(launch_directory)
    monkeypatch.setenv("THETA_WORKSPACE", str(workspace))
    assert "custom" in available_objectives()
    assert load_objective("custom").source == component.source
    assert "custom" not in available_objectives(launch_directory)


@pytest.mark.parametrize("source", ["float('nan')", "float('inf')", "[1, 2]"])
def test_invalid_user_scores_fail(tmp_path, source):
    component = create_objective("mine", workspace=tmp_path)
    component.source.write_text(f"def evaluate(results, params):\n    return {source}\n")
    with pytest.raises((ValueError, TypeError)):
        component.evaluate(TrialResults([], {}), {})


def config():
    return OmegaConf.create({"group": "g", "experiment_id": "e", "paradigm": "online_rl", "seed": 1,
                             "total_timesteps": 100, "eval_episodes": 2, "agent": {"name": "m"},
                             "objective": {"component": "learning_curve"},
                             "theta_objective_runner": {"script": "src/app/train.py"},
                             "variability": {"training_seeds": [1, 2], "parallel_runs": 2, "evaluation_seed": 10}})


def test_validation_requires_data_and_rejects_pruning():
    cfg = config()
    cfg.eval_episodes = 0
    with pytest.raises(ValueError, match="eval_episodes"):
        validate_objective(cfg)
    cfg.eval_episodes = 2
    cfg.tuning = {"component": "optuna", "options": {"optuna": {"pruner": "median"}}}
    from component_fixtures import optional_workspace
    with pytest.raises(ValueError, match="pruning disabled"):
        validate_objective(cfg, paths=optional_workspace("optuna"))
    cfg.tuning = None
    cfg.objective.params = {"aggregation": {"fraction": 0}}
    with pytest.raises(ValueError, match="fraction"):
        validate_objective(cfg)


def test_sampling_preserves_failed_runs_and_isolates_seed_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []
    def execute(child, directory):
        calls.append(child)
        assert child.objective is None and child.theta_tuning is None
        assert child.agent.seed == child.seed
        if child.seed == 2:
            raise RuntimeError("did not learn")
        events = [{"training_step": x, "metric": "eval/reward", "episode_outcomes": [500]} for x in [0, 100]]
        (directory / "evaluation_events.jsonl").write_text("\n".join(map(json.dumps, events)))
    assert run_objective(config(), execute=execute) == 0
    paths = list(tmp_path.glob("results/logs/g/e/m/version_execution_*/objective.json"))
    manifest = json.loads(paths[0].read_text())
    assert [run["status"] for run in manifest["runs"]] == ["completed", "failed"]
    assert len({child.variability_version for child in calls}) == 2
    assert manifest["result"]["diagnostics"]["per_run_auc"] == [1, 0]


@pytest.mark.parametrize("sweep", [False, True])
def test_real_subprocess_samples_return_one_score(tmp_path, monkeypatch, sweep):
    root = Path(__file__).resolve().parents[1]
    script = tmp_path / "fake_train.py"
    script.write_text('''import json
from pathlib import Path
def main(cfg):
    path = Path("results/logs") / cfg.group / cfg.experiment_id / cfg.agent.name / cfg.variability_version
    path.mkdir(parents=True, exist_ok=True)
    events = [{"training_step": x, "metric": "eval/reward", "episode_outcomes": [cfg.seed * 100]} for x in [0, 100]]
    (path / "evaluation_events.jsonl").write_text("\\n".join(map(json.dumps, events)))
    return 1.0
''')
    cfg = config()
    cfg.theta_objective_runner.script = str(script)
    cfg.variability.evaluation_seed = None
    cfg.hydra = {"searchpath": [f"file://{root / 'in/config'}"]}
    OmegaConf.save(cfg, tmp_path / "test.yaml")
    env = {**os.environ, "PYTHONPATH": str(root), "HYDRA_FULL_ERROR": "1"}
    command = [sys.executable, "-m", "src.app.objectives.runner", "--config-path", str(tmp_path), "--config-name", "test"]
    storage = f"sqlite:///{tmp_path / 'study.db'}"
    if sweep:
        command.extend(["--multirun", "hydra/sweeper=optuna_online", "hydra.sweeper.n_trials=2",
                        f"hydra.sweeper.storage={storage}", "hydra.sweeper.study_name=objectives", "+x=choice(1,2)"])
    result = subprocess.run(command,
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    manifest_path = next(tmp_path.glob("results/logs/g/e/m/version_*/objective.json"))
    manifest = json.loads(manifest_path.read_text())
    assert manifest["result"]["value"] == pytest.approx(0.2)
    assert all(run["status"] == "completed" for run in manifest["runs"])
    if sweep:
        import optuna
        study = optuna.load_study(study_name="objectives", storage=storage)
        assert [trial.value for trial in study.trials] == pytest.approx([0.2, 0.2])
        assert study.user_attrs["theta_objective_metric"] == "objective/learning_curve"
        from component_fixtures import optional_module
        _optuna_utils = optional_module('optuna', 'utils.py')
        promote_best_trial_checkpoint = _optuna_utils.promote_best_trial_checkpoint
        monkeypatch.chdir(tmp_path)
        promote_best_trial_checkpoint("g", "e", "m", storage, "objectives")
        summary = json.loads(Path("results/checkpoints/g/e/m/best_trial_summary.json").read_text())
        assert summary["checkpoint"] is None
        assert len(summary["samples"]) == 2


def test_local_dispatch_routes_objectives_and_carries_sampling(monkeypatch):
    from src.app.pipeline import local_runner
    calls = []
    monkeypatch.setattr(local_runner, "run_experiment", lambda args, **kwargs: calls.append((args, kwargs)))
    cfg = config()
    cfg.experiment_name = "test"
    local_runner._run_methods_sequential(cfg, {"methods": {"m": {"agent": "ppo", "model": "dnn"}},
        "sanitized_extra_args": [], "storage_url": None, "is_sweep": False})
    args, kwargs = calls[0]
    assert kwargs["script_entrypoint"] == "src/app/pipeline/repeated_task.py"
    assert kwargs["coordinator"] is True
    assert any(arg.startswith("++theta_execution.script=") for arg in args)
    assert any(arg.startswith("++variability=") and "[1,2]" in arg for arg in args)
