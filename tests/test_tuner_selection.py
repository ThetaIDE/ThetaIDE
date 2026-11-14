"""Tuning never guesses a provider outside the bounded historical receipt reader."""
import copy
import json
from pathlib import Path
import shutil

import pytest
from omegaconf import OmegaConf

from src.app.pipeline.exceptions import ConfigurationError
from src.app.pipeline.tuners import get_tuner, tuner_name
from src.app.pipeline.tuner_execution import TunerExecutionPlan, prepare_plan, record_config, record_plan, record_overrides
from src.app.workspace import WorkspacePaths


@pytest.mark.parametrize('tuning', [None, {}, [], 'optuna', {'component': None}, {'component': ''},
    {'component': '  '}, {'component': 3}, {'component': False}, {'component': ['optuna']},
    {'component': ' optuna'}, {'component': 'optuna '}])
def test_missing_and_malformed_selectors_fail_before_provider_resolution(monkeypatch, tuning):
    monkeypatch.setattr('src.app.core.provider_loader.resolve_provider', lambda *a, **k: pytest.fail('guessed a provider'))
    cfg = {'tuning': tuning}
    with pytest.raises(ConfigurationError, match='tuning.component'):
        tuner_name(cfg)
    with pytest.raises(ConfigurationError, match='tuning.component'):
        get_tuner(cfg)


@pytest.mark.parametrize('name', ['', ' ', 0, False, []])
def test_explicit_setup_selector_is_validated(name):
    with pytest.raises(ConfigurationError, match='tuning.component'):
        get_tuner(name=name)


def install(root):
    shutil.copytree(Path(__file__).parent / 'fixtures/portable_tuner', root / 'src/usr/tuners/grid')
    return WorkspacePaths(root, application=root / 'empty-app')


def test_one_installed_tuner_is_not_an_implicit_default(tmp_path):
    paths = install(tmp_path)
    with pytest.raises(ConfigurationError, match='tuning.component'):
        get_tuner({'tuning': {}}, paths=paths)
    assert get_tuner({'tuning': {'component': 'fixture-grid'}}, paths=paths).api_version == 1
    assert get_tuner(name='fixture-grid', paths=paths).api_version == 1


def test_omegaconf_selector_and_explicit_portable_execution(tmp_path):
    paths = install(tmp_path)
    cfg = OmegaConf.create({'tuning': {'component': 'fixture-grid'}})
    assert tuner_name(cfg) == 'fixture-grid'
    assert prepare_plan(cfg, 'a' * 32, paths=paths).component == 'fixture-grid'


def test_explicit_alias_and_case_resolve_to_canonical_portable_identity(tmp_path):
    paths = install(tmp_path)
    manifest = paths.root / 'src/usr/tuners/grid/.theta_component.json'
    data = json.loads(manifest.read_text())
    data['capabilities'][0]['aliases'] = ['grid-alias']
    manifest.write_text(json.dumps(data))
    cfg = {'tuning': {'component': 'GRID-ALIAS'}}
    assert get_tuner(cfg, paths=paths).api_version == 1
    assert prepare_plan(cfg, 'a' * 32, paths=paths).component == 'fixture-grid'
    assert cfg['tuning']['component'] == 'GRID-ALIAS'


def test_command_compilation_uses_explicit_workspace_from_unrelated_cwd(tmp_path, monkeypatch):
    from src.app.pipeline.commands import build_method_overrides
    paths = install(tmp_path / 'workspace')
    # Retained core paradigm definitions come from the application, not the
    # independently selected research workspace.
    paths = WorkspacePaths(paths.root)
    monkeypatch.chdir(tmp_path)
    cfg = {'paradigm': 'supervised', 'tuning': {'component': 'fixture-grid'}}
    result = build_method_overrides('one', {'model': 'example', 'lr': .1}, cfg=cfg, workspace=paths)
    assert '++lr=0.1' in result or any('0.1' in arg for arg in result)


def test_unselected_sweep_rejected_by_validation_remote_and_worker(tmp_path):
    from src.app.pipeline.validation import _validate_tuning
    from src.app.core.execution import RunContext, execute_backend
    cfg = {'tuning': {'metric': 'score', 'direction': 'maximize'}}
    paths = install(tmp_path)
    for check in (lambda: _validate_tuning(cfg, paths=paths),
                  lambda: prepare_plan(cfg, 'a' * 32, paths=paths),
                  lambda: execute_backend(cfg, RunContext('trial', paths, 1, tmp_path))):
        with pytest.raises(ConfigurationError, match='tuning.component'):
            check()


@pytest.mark.parametrize('tuning', [{}, {'component': None}, {'component': ''}])
def test_old_receipt_config_is_adapted_without_mutation(tuning):
    record = {'schema_version': 1, 'execution_id': 'a' * 32, 'config': {'tuning': tuning}}
    before = copy.deepcopy(record)
    cfg = record_config(record)
    assert cfg['tuning']['component'] == 'optuna'
    assert record == before
    assert record_plan(record).component == tuner_name(cfg)


@pytest.mark.parametrize('marker', [None, TunerExecutionPlan('optuna', 'a' * 32).record()])
def test_new_receipts_never_inherit_historical_selection(marker):
    record = {'execution_id': 'a' * 32, 'tuner_execution': marker, 'config': {'tuning': {}}}
    cfg = record_config(record)
    assert cfg['tuning'] == {}
    with pytest.raises(ConfigurationError, match='tuning.component'):
        tuner_name(cfg)


def test_fixed_and_other_tuner_receipts_keep_their_original_meaning():
    for cfg in ({}, {'tuning': None}, {'tuning': {'component': 'another'}}):
        record = {'execution_id': 'a' * 32, 'config': cfg}
        assert record_config(record) == cfg
        assert record_config(record) is not cfg
    with pytest.raises(ValueError, match='legacy site receipt schema'):
        record_config({'schema_version': 99, 'execution_id': 'a' * 32, 'config': {'tuning': {}}})


@pytest.mark.parametrize('override', ['tuning.component=optuna', '++tuning.component="OPTUNA"'])
def test_consistent_historical_overrides_are_normalized_in_a_copy(override):
    record = {'execution_id': 'a' * 32, 'config': {'tuning': {}}, 'overrides': ['seed=3', override]}
    before = copy.deepcopy(record)
    assert record_overrides(record) == ['seed=3', '++tuning.component="optuna"']
    assert record == before


@pytest.mark.parametrize('override', ['tuning.component=another', '++tuning.component=42'])
def test_conflicting_historical_selection_is_rejected(override):
    record = {'execution_id': 'a' * 32, 'config': {'tuning': {}}, 'overrides': [override]}
    with pytest.raises(ConfigurationError, match='conflicting tuner selections'):
        record_overrides(record)


def test_historical_continuation_identity_matches_explicit_recipe(tmp_path):
    from src.app.sites.continuation import compatibility
    record = {'execution_id': 'a' * 32, 'config': {'tuning': {
        'metric': 'score', 'direction': 'maximize', 'n_trials': 2}}}
    current = copy.deepcopy(record_config(record))
    current['tuning']['n_trials'] = 10
    from component_fixtures import install_optional
    paths = WorkspacePaths(tmp_path / 'workspace')
    install_optional(paths, 'optuna', tmp_path / 'archives')
    assert compatibility(record_config(record), paths=paths) == compatibility(current, paths=paths)
    current['tuning']['metric'] = 'different-score'
    assert compatibility(record_config(record), paths=paths) != compatibility(current, paths=paths)


def test_tuner_telemetry_does_not_infer_an_owner(tmp_path):
    from src.app.core.provider_catalog import get_catalog
    from src.app.telemetry.monitors import available_monitors
    root = tmp_path / 'src/usr/tuners/provider'
    root.mkdir(parents=True)
    (root / 'provider.py').write_text('raise AssertionError("metadata inspection imported a tuner")')
    manifest = {'id': 'provider', 'name': 'Provider', 'kind': 'tuner', 'version': '1.0.0',
        'storage_version': 2, 'api_version': 1,
        'capabilities': [{'id': 'optuna', 'kind': 'tuner', 'entry_point': 'provider.py:Tuner',
                          'monitoring': {'provides': [{'name': 'tuner/probe', 'label': 'Probe'}]}}],
        'payload': {'implementations': ['provider.py'], 'resources': [], 'configs': {}}}
    (root / '.theta_component.json').write_text(json.dumps(manifest))
    catalog = get_catalog(WorkspacePaths(tmp_path, application=tmp_path / 'empty-app'))
    assert not available_monitors({'tuning': {}}, catalog=catalog, include_shared=False)
    assert [m.name for m in available_monitors({'tuning': {'component': 'optuna'}}, catalog=catalog, include_shared=False)] == ['tuner/probe']
