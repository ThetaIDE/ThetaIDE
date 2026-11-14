"""Independent tuner vocabulary and runner metadata govern pruning preflight."""
import builtins
import json
from pathlib import Path
import shutil

import pytest

from src.app.core.manifest_schema import validate_manifest_data
from src.app.core.provider_catalog import get_catalog
from src.app.objectives.api import validate_objective
from src.app.pipeline.tuning_features import TuningFeatures, validate_tuning_compatibility
from src.app.pipeline.validation import _validate_tuning
from src.app.pipeline.exceptions import ConfigurationError
from src.app.workspace import WorkspacePaths


def provider(tmp_path, *, features=None, intermediate=True, query="return cfg.get('stop_early', False)"):
    root = tmp_path / 'src/usr/packages/grid'
    shutil.copytree(Path(__file__).parent / 'fixtures/portable_tuner', root)
    source = root / 'component.py'
    if query is not None:
        source.write_text(source.read_text() + '\n    def pruning_enabled(self, cfg):\n        ' + query + '\n')
    manifest = root / '.theta_component.json'
    data = json.loads(manifest.read_text())
    data['capabilities'][0]['tuning_features'] = features if features is not None else {
        'objective_modes': ['metric', 'component'], 'pruning': True}
    manifest.write_text(json.dumps(data))
    runner = tmp_path / 'src/usr/packages/runner'
    runner.mkdir()
    data = {'id': 'runner', 'name': 'Runner', 'kind': 'algorithm', 'version': '1.0.0',
            'storage_version': 2, 'api_version': 1,
            'capabilities': [{'id': 'unfamiliar', 'kind': 'algorithm', 'entry_point': 'run.py:execute',
                              'execution_contract': 1, 'backend': 'new-engine',
                              'execution_features': {'tuning': True, 'winner_promotion': True,
                                                     'intermediate_values': intermediate}}],
            'payload': {'implementations': ['run.py'], 'resources': [], 'configs': {}}}
    (runner / '.theta_component.json').write_text(json.dumps(data))
    (runner / 'run.py').write_text('raise AssertionError("metadata must not execute runner")')
    return WorkspacePaths(tmp_path, application=tmp_path / 'empty-app')


def config(**extra):
    return {'tuning': {'component': 'fixture-grid', 'metric': 'score', 'direction': 'maximize'},
            'methods': {'one': {'agent': 'unfamiliar', 'tune': {'config.x': {'type': 'int', 'low': 1, 'high': 2}}}},
            **extra}


def test_independent_pruning_preflight_without_optuna(tmp_path, monkeypatch):
    paths = provider(tmp_path)
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in ('optuna', 'sqlalchemy'):
            raise AssertionError('independent tuning imported ' + name)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    monkeypatch.chdir(tmp_path.parent)
    assert validate_tuning_compatibility(config(stop_early=True), paths=paths)
    _validate_tuning(config(stop_early=True), paths=paths)
    assert get_catalog(paths).resolve('tuner', 'fixture-grid')


def test_runner_intermediates_are_required_only_for_active_pruning(tmp_path):
    paths = provider(tmp_path, intermediate=False)
    _validate_tuning(config(), paths=paths)
    with pytest.raises(ConfigurationError, match="Method 'one'.*intermediate-value"):
        _validate_tuning(config(stop_early=True), paths=paths)


def test_worker_rechecks_pruning_before_importing_runner(tmp_path):
    from src.app.core.execution import RunContext, execute_backend
    paths = provider(tmp_path, intermediate=False)
    cfg = {**config(stop_early=True), 'agent': 'unfamiliar'}
    with pytest.raises(ValueError, match='intermediate-value'):
        execute_backend(cfg, RunContext('trial', paths, 1, tmp_path))


@pytest.mark.parametrize('extra,message', [
    ({'objective': {'component': 'unused'}}, 'custom objectives require pruning disabled'),
    ({'variability': {}}, 'repeated tuning requires pruning disabled'),
])
def test_unsupported_pruning_combinations_fail(tmp_path, extra, message):
    paths = provider(tmp_path)
    with pytest.raises(ValueError, match=message):
        validate_tuning_compatibility(config(stop_early=True, **extra), paths=paths)


def test_provider_can_declare_objective_and_repeated_pruning(tmp_path):
    paths = provider(tmp_path, features={'objective_modes': ['component'], 'pruning': True,
                                        'pruning_objectives': True, 'pruning_repeated_runs': True})
    assert validate_tuning_compatibility(config(stop_early=True, objective={}, variability={}), paths=paths)
    with pytest.raises(ValueError, match='does not support metric objectives'):
        validate_tuning_compatibility(config(), paths=paths)


def test_objective_validation_uses_selected_workspace_before_loading_source(tmp_path):
    paths = provider(tmp_path)
    with pytest.raises(ValueError, match='custom objectives require pruning disabled'):
        validate_objective(config(stop_early=True, objective={'component': 'unused'}), paths=paths)


def test_independent_objective_pruning_validation_from_unrelated_cwd(tmp_path, monkeypatch):
    paths = provider(tmp_path, features={'objective_modes': ['component'], 'pruning': True,
                                        'pruning_objectives': True})
    root = tmp_path / 'src/usr/packages/objective'
    root.mkdir()
    (root / 'score.py').write_text('raise AssertionError("validation executed objective")')
    (root / 'score.json').write_text(json.dumps({'api_version': 1, 'source': 'score.py'}))
    (root / '.theta_component.json').write_text(json.dumps({
        'id': 'score', 'name': 'Score', 'version': '1.0.0', 'kind': 'objective',
        'storage_version': 2, 'api_version': 1,
        'capabilities': [{'id': 'score', 'kind': 'objective', 'entry_point': 'score.json'}],
        'payload': {'implementations': ['score.py'], 'resources': ['score.json'], 'configs': {}}}))
    monkeypatch.chdir(tmp_path.parent)
    component, params = validate_objective(config(stop_early=True, objective={'component': 'score'}), paths=paths)
    assert component.source == root / 'score.py'
    _validate_tuning(config(stop_early=True, objective={'component': 'score'}), paths=paths)


def test_pruning_declaration_requires_query_but_legacy_metric_tuners_work(tmp_path):
    paths = provider(tmp_path / 'declared', query=None)
    with pytest.raises(ValueError, match='missing pruning_enabled'):
        validate_tuning_compatibility(config(), paths=paths)
    paths = provider(tmp_path / 'legacy', features={}, query=None)
    _validate_tuning(config(), paths=paths)


def test_disabled_tuner_is_rejected_before_query_execution(tmp_path):
    from src.app.core.contributions import ContributionRegistry
    paths = provider(tmp_path, query='raise AssertionError("disabled query ran")')
    ContributionRegistry(paths).set_enabled('tuner', 'fixture-grid', False)
    with pytest.raises(ConfigurationError, match='disabled'):
        validate_tuning_compatibility(config(), paths=paths)


@pytest.mark.parametrize('query,message', [('return 1', 'must return a boolean'),
    ('return True', 'does not declare pruning support')])
def test_invalid_provider_queries_fail(tmp_path, query, message):
    paths = provider(tmp_path, features={}, query=query)
    with pytest.raises(ValueError, match=message):
        validate_tuning_compatibility(config(), paths=paths)


@pytest.mark.parametrize('features', [{'pruning': 'yes'}, {'objective_modes': 'metric'},
    {'objective_modes': ['unknown']}, {'pruning_objectives': True}, {'unknown': True}])
def test_invalid_features_fail_runtime_and_manifest_validation(tmp_path, features):
    with pytest.raises(ValueError):
        TuningFeatures.from_metadata({'tuning_features': features})
    paths = provider(tmp_path, features=features)
    data = json.loads((paths.root / 'src/usr/packages/grid/.theta_component.json').read_text())
    assert not validate_manifest_data(data)[0]


def test_plain_objectives_do_not_load_tuners(tmp_path, monkeypatch):
    monkeypatch.setattr('src.app.pipeline.tuners.get_tuner', lambda *a, **k: pytest.fail('loaded tuner'))
    assert not validate_tuning_compatibility({'objective': {}}, paths=WorkspacePaths(tmp_path))


@pytest.mark.parametrize('cfg,active', [
    ({'tuning': {'options': {'optuna': {'pruner': 'median'}}}}, True),
    ({'tuning': {'pruner': 'median'}}, True),
    ({'hydra': {'sweeper': {'pruner': 'median'}}}, True),
    ({'theta_tuning': {'pruner': 'none'}, 'tuning': {'pruner': 'median'}}, False),
    ({}, False),
])
def test_optuna_owns_legacy_and_runtime_pruning_options(cfg, active, tmp_path):
    from component_fixtures import install_optional
    from src.app.core.provider_loader import load_module
    paths = WorkspacePaths(tmp_path / 'workspace')
    install_optional(paths, 'optuna', tmp_path / 'archives')
    component = load_module(get_catalog(paths).package('optuna'), 'component.py', paths=paths)
    assert component.OptunaTuner().pruning_enabled(cfg) is active
