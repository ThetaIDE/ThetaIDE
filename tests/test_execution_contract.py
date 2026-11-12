"""Exercise the registered runner through the real coordinator and worker process."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
from omegaconf import OmegaConf

from src.app.pipeline.exceptions import ConfigurationError
from src.app.core.execution import reject_selectors, execution_backend
from src.app.pipeline.engine import resolve_engine, DEFAULT_ENGINE_SCRIPT


@pytest.mark.parametrize("key", ["entrypoint", "runner", "script", "engine", "venv", "venv_python"])
def test_removed_selectors_fail(key):
    with pytest.raises((ValueError, ConfigurationError), match="Removed execution fields"):
        resolve_engine({key: "obsolete"})


def test_domain_fields_are_allowed():
    reject_selectors({"parameters": {"engine": "solver", "script": "feature"}}, "method")


def test_conflicts_report_every_field():
    with pytest.raises((ValueError, ConfigurationError), match=r"method.entrypoint, method.runner"):
        resolve_engine({"entrypoint": "a", "runner": "b"})


def test_dev_backend_and_python_are_explicit():
    assert resolve_engine({"agent": "sb3-dev/ppo", "python": "/custom/python"}) == (DEFAULT_ENGINE_SCRIPT, "/custom/python")
    assert execution_backend({"agent": "sb3-dev/ppo"})[0] == "sb3"


def test_old_selected_hub_profile_fails():
    with pytest.raises((ValueError, ConfigurationError), match="Removed execution fields|future compatible release"):
        resolve_engine({"agent": "sb3/ppo"})


def sentinel_workspace(tmp_path, runner="Sentinel"):
    root = Path(__file__).resolve().parents[1]
    config = tmp_path / "in/config"
    shutil.copytree(root / "src/app/config_defaults", config)
    (config / "paradigms").mkdir(exist_ok=True)
    (config / "paradigms/probe.yaml").write_text(
        f"type: base\nrunner: {runner}\nenvironment_source: live_environment\n"
        "method: {algorithm: forbidden, model: required}\nallowed_agents: []\n")
    (config / "env").mkdir(exist_ok=True)
    (config / "env/probe.yaml").write_text("name: probe\n")
    (config / "model").mkdir(exist_ok=True)
    (config / "model/probe.yaml").write_text("architecture: probe\nlr: 0.001\n")
    (config / "experiment").mkdir(exist_ok=True)
    (config / "experiment/probe.yaml").write_text(
        "# @package _global_\ndefaults: [override /env: probe]\nparadigm: probe\ngroup: gate\nexperiment_id: sentinel\n"
        "no_plot: true\nmethods:\n  one:\n    model: probe\n    python: " + sys.executable + "\n")
    package = tmp_path / "src/usr/packages/probe"
    package.mkdir(parents=True)
    (package / ".theta_component.json").write_text(json.dumps({
        "id": "probe", "name": "Probe", "description": "Execution test", "author": "Theta", "version": "1.0.0", "storage_version": 2, "api_version": 1, "kind": "suite",
        "payload": {"implementations": ["probe.py"], "resources": [], "configs": {}},
        "capabilities": [
            {"id": "Sentinel", "kind": "component", "entry_point": "probe.py:Sentinel"},
            {"id": "Alternate", "kind": "component", "entry_point": "probe.py:Alternate"},
            {"id": "probe", "kind": "model", "entry_point": "probe.py:Sentinel"},
            {"id": "probe", "kind": "env", "entry_point": "probe.py:Sentinel"},
            {"id": "old", "kind": "algorithm", "entry_point": "probe.py:Sentinel"}]}))
    (package / "probe.py").write_text(
        "from src.app.core.execution import RunResult\n"
        "class Sentinel:\n execution_contract=1\n def run(self,cfg,context):\n"
        "  with (context.workspace.root/'calls.txt').open('a') as f: f.write('called\\n')\n"
        "  (context.artifacts/'best_model.ckpt').write_text('sentinel')\n"
        "  return RunResult(metrics={'probe': 7}, metric_definitions={'probe': {'direction':'maximize','meaning':'sentinel'}}, objective=7)\n"
        "class Alternate:\n execution_contract=1\n def run(self,cfg,context):\n"
        "  with (context.workspace.root/'calls.txt').open('a') as f: f.write('alternate\\n')\n"
        "  return RunResult(metrics={'probe': 42}, objective=42)\n")
    return root


@pytest.mark.parametrize("runner", ["Sentinel", "Alternate", "MissingRunner"])
def test_pipeline_invokes_registered_runner_once(tmp_path, runner):
    root = sentinel_workspace(tmp_path, runner)
    env = {**os.environ, "THETA_WORKSPACE": str(tmp_path), "PYTHONPATH": str(root), "QT_QPA_PLATFORM": "offscreen"}
    env.pop("THETA_PROVIDER_SNAPSHOT", None)
    env.pop("THETA_DATA_DIR", None)
    result = subprocess.run([sys.executable, str(root / 'run_pipeline.py'), 'probe'], cwd=tmp_path,
                            env=env, text=True, capture_output=True, timeout=90)
    if runner == "MissingRunner":
        assert result.returncode != 0
        assert not (tmp_path / 'calls.txt').exists()
        return
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / 'calls.txt').read_text() == ('alternate\n' if runner == 'Alternate' else 'called\n')
    records = list((tmp_path / 'results/checkpoints').rglob('execution.json'))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record['status'] == 'completed'
    assert record['result']['metrics'] == {'probe': 42 if runner == 'Alternate' else 7}
    assert record['runner'] == runner


def test_pipeline_repeat_uses_worker_for_each_seed(tmp_path):
    root = sentinel_workspace(tmp_path)
    environment = {**os.environ, 'THETA_WORKSPACE': str(tmp_path), 'PYTHONPATH': str(root)}
    environment.pop('THETA_DATA_DIR', None)
    environment.pop('THETA_PROVIDER_SNAPSHOT', None)
    result = subprocess.run([sys.executable, str(root / 'run_pipeline.py'), 'probe',
                             '++variability.training_runs=2'], cwd=tmp_path, env=environment,
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / 'calls.txt').read_text() == 'called\ncalled\n'
    records = list((tmp_path / 'results/checkpoints').rglob('execution.json'))
    assert len(records) == 2
    assert len({json.loads(p.read_text())['run_id'] for p in records}) == 2


def test_migration_is_explicit_and_conflicts_need_resolution():
    from src.app.pipeline.migrate_execution import migrate, ManualResolution
    old = {'methods': {'m': {'agent': 'sb3/ppo', 'entrypoint': 'package://sb3-suite/sb3_runner.py'}}}
    new = migrate(old)
    assert new['methods']['m'] == {'agent': 'sb3-dev/ppo', 'backend': 'sb3'}
    assert 'entrypoint' in old['methods']['m']
    with pytest.raises(ManualResolution):
        migrate({'entrypoint': 'custom.py'})
    with pytest.raises(ManualResolution):
        migrate({'entrypoint': 'a.py', 'runner': 'b.py'})


def test_configured_evaluator_runs_once():
    from types import SimpleNamespace
    from unittest.mock import Mock
    from src.app.core.lightning_builder import ConfiguredEvaluatorCallback
    evaluator = SimpleNamespace(evaluate=Mock(return_value={'heldout/score': 0.5}))
    log = Mock()
    trainer = SimpleNamespace(logger=log, datamodule=SimpleNamespace(val_dataloader=lambda: 'validation'), global_step=10, current_epoch=0)
    callback = ConfiguredEvaluatorCallback(evaluator, {'paradigm': 'supervised'})
    callback.on_validation_start(trainer, 'model')
    callback.on_validation_end(trainer, 'model')
    callback.on_fit_end(trainer, 'model')
    evaluator.evaluate.assert_called_once_with('model', 'validation', {'paradigm': 'supervised'})
    log.log_metrics.assert_called_once_with({'heldout/score': 0.5}, step=10)


def test_pipeline_sweep_invokes_worker_once_per_trial(tmp_path):
    root = sentinel_workspace(tmp_path)
    experiment = tmp_path / 'in/config/experiment/probe.yaml'
    with experiment.open('a') as handle:
        handle.write('    tune:\n      model.lr: {type: choice, values: [0.001, 0.002]}\n')
    environment = {**os.environ, 'THETA_WORKSPACE': str(tmp_path), 'PYTHONPATH': str(root)}
    environment.pop('THETA_DATA_DIR', None)
    environment.pop('THETA_PROVIDER_SNAPSHOT', None)
    result = subprocess.run([sys.executable, str(root / 'run_pipeline.py'), 'probe',
                             '++tuning.component=optuna', '++tuning.n_trials=2',
                             '++tuning.metric=probe', '++tuning.direction=maximize'],
                            cwd=tmp_path, env=environment, text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / 'calls.txt').read_text() == 'called\ncalled\n'
    records = list((tmp_path / 'results/checkpoints').rglob('execution.json'))
    assert len(records) == 2
    assert all(json.loads(p.read_text())['status'] == 'completed' for p in records)


@pytest.mark.parametrize('outcome', ['failed', 'cancelled'])
def test_worker_finalizes_exception_and_cancellation(tmp_path, monkeypatch, outcome):
    from types import SimpleNamespace
    from src.app.pipeline.provider_worker import execute_method
    from src.app.services.cancellation import TrainingCancelled
    from src.app.core import paradigm_loader
    closed = []
    class Runner:
        def run(self, cfg, context):
            context.on_cleanup(lambda: closed.append(True))
            if outcome == 'cancelled':
                raise TrainingCancelled()
            raise RuntimeError('backend failure')
    monkeypatch.setenv('THETA_WORKSPACE', str(tmp_path))
    monkeypatch.delenv('THETA_PROVIDER_SNAPSHOT', raising=False)
    monkeypatch.setattr(paradigm_loader, 'load_paradigm_definition',
                        lambda _: SimpleNamespace(runner_cls=Runner, raw={'runner': 'test'}))
    with pytest.raises(TrainingCancelled if outcome == 'cancelled' else RuntimeError):
        execute_method(OmegaConf.create({'paradigm': 'test', 'model': {'name': 'one'}}))
    record = json.loads(next(tmp_path.rglob('execution.json')).read_text())
    assert record['status'] == outcome
    assert closed == [True]


def test_selected_obsolete_provider_names_owner_and_remedy(tmp_path, monkeypatch):
    sentinel_workspace(tmp_path)
    monkeypatch.setenv('THETA_WORKSPACE', str(tmp_path))
    with pytest.raises(ValueError, match=r"probe@1.0.0.*execution_contract=None.*migrated"):
        execution_backend({'agent': 'old'})
