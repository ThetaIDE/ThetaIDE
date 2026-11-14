"""Independent providers exercise contracts without privileged backend names."""
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from src.app.core.execution import RunContext, execution_backend, execute_backend
from src.app.core.execution_features import ExecutionFeatures, LIGHTNING_FEATURES
from src.app.core.provider_catalog import get_catalog
from src.app.pipeline.engine import runner_supports_episode_outcomes
from src.app.services.resource_budget import GIB, ResourceBudget, workload_estimate, uses_accelerator
from src.app.workspace import WorkspacePaths


def provider(tmp_path, *, features=None, hints=None, hooks=False):
    root = tmp_path / 'src/usr/packages/independent'
    root.mkdir(parents=True)
    capability = {'id': 'probe/learn', 'kind': 'algorithm', 'entry_point': 'adapter.py:execute',
                  'execution_contract': 1, 'backend': 'unfamiliar-engine',
                  'execution_features': features if features is not None else asdict(LIGHTNING_FEATURES),
                  'resource_hints': hints if hints is not None else {'base_memory_gib': .4, 'cpu': .5, 'accelerator': False}}
    if hooks:
        capability.update(execution_preflight='hooks.py:validate', resource_hint_hook='hooks.py:resources')
    manifest = {'id': 'independent', 'name': 'Independent', 'version': '1.0.0', 'storage_version': 2,
                'api_version': 1, 'kind': 'suite', 'capabilities': [capability],
                'payload': {'implementations': ['adapter.py', 'hooks.py'], 'resources': [], 'configs': {}}}
    (root / '.theta_component.json').write_text(json.dumps(manifest))
    (root / 'adapter.py').write_text(
        'from src.app.core.execution import RunResult\n'
        'def execute(cfg,context):\n return RunResult(metrics={"independent": 3})\n')
    (root / 'hooks.py').write_text(
        'def validate(cfg, *, paths):\n'
        ' if cfg.get("invalid"): raise ValueError("independent constraint")\n'
        ' assert paths.root.name == cfg["expected_workspace"]\n'
        'def resources(cfg, *, paths):\n return {"cpu": 2.5}\n')
    return WorkspacePaths(tmp_path)


def test_independent_dispatch_and_selected_preflight(tmp_path, monkeypatch):
    paths = provider(tmp_path, hooks=True)
    monkeypatch.chdir(tmp_path.parent)
    cfg = {'agent': 'probe/learn', 'expected_workspace': tmp_path.name}
    assert execution_backend(cfg, paths=paths)[0] == 'unfamiliar-engine'
    assert execute_backend(cfg, RunContext('one', paths, 1, tmp_path)).metrics == {'independent': 3}
    with pytest.raises(ValueError, match='independent constraint'):
        execution_backend({**cfg, 'invalid': True}, paths=paths)
    # Metadata browsing must not import either the adapter or its hooks.
    (paths.root / 'src/usr/packages/independent/hooks.py').write_text('raise AssertionError("catalog executed code")')
    assert get_catalog(paths).resolve('algorithm', 'probe/learn')


@pytest.mark.parametrize('extra,message', [
    ({'evaluator': 'external'}, 'native evaluation'),
    ({'resume_from': 'saved.native'}, 'exact training resume'),
    ({'recover': True}, 'exact training resume'),
    ({'data_module': 'custom'}, 'custom data_module'),
    ({'dataset_path': 'transitions.npz'}, 'dataset_path input'),
    ({'tuning': {'component': 'independent', 'n_trials': 2}}, 'tuning/winner-promotion'),
])
def test_conservative_features_reject_unsupported_operations(tmp_path, extra, message):
    paths = provider(tmp_path, features={})
    with pytest.raises(ValueError, match=message):
        execution_backend({'agent': 'probe/learn', **extra}, paths=paths)


def test_declared_features_allow_operations_without_backend_allowlist(tmp_path):
    paths = provider(tmp_path, features={**asdict(LIGHTNING_FEATURES), 'episode_outcomes': True})
    cfg = {'agent': 'probe/learn', 'evaluator': 'custom', 'resume_from': 'saved.native',
           'data_module': 'custom', 'tuning': {'component': 'another', 'n_trials': 2}}
    assert execution_backend(cfg, paths=paths)[0] == 'unfamiliar-engine'
    assert runner_supports_episode_outcomes(cfg, paths=paths)


def test_resource_hints_apply_in_selected_workspace_and_budget_still_limits(tmp_path):
    paths = provider(tmp_path, hooks=True)
    cfg = {'agent': 'probe/learn'}
    assert not uses_accelerator(cfg, paths=paths)
    estimate = workload_estimate(cfg, paths=paths)
    assert .4 * GIB < estimate['memory'] < .5 * GIB
    assert estimate['cpu'] == 2.5
    reason = ResourceBudget().admission_reason(cores=2, total=8*GIB, available=8*GIB,
        active_memory=[], next_memory=estimate['memory'], next_cpu=estimate['cpu'])
    assert 'CPU' in reason
    assert not uses_accelerator({'trainer': {'accelerator': 'cpu'}, 'agent': 'missing'}, paths=paths)


@pytest.mark.parametrize('hints', [{'cpu': 0}, {'cpu': float('nan')}, {'base_memory_gib': -1},
                                  {'accelerator': 'false'}, {'unknown': 1}])
def test_invalid_resource_hints_fail(tmp_path, hints):
    paths = provider(tmp_path, hints=hints)
    with pytest.raises(ValueError, match='resource hints|resource_hints'):
        workload_estimate({'agent': 'probe/learn'}, paths=paths)


def test_workspace_estimates_and_nested_device_choices_are_independent(tmp_path):
    first = provider(tmp_path / 'first', hints={'accelerator': False})
    second = provider(tmp_path / 'second', hints={'accelerator': True})
    cfg = {'agent': 'probe/learn'}
    assert not uses_accelerator(cfg, paths=first)
    assert uses_accelerator(cfg, paths=second)
    assert workload_estimate(cfg, paths=first)['key'] != workload_estimate(cfg, paths=second)['key']
    assert not uses_accelerator({'trainer': {'accelerator': 'cpu'}, 'methods': {'x': cfg}}, paths=second)
    assert not uses_accelerator({'methods': {'x': {**cfg, 'trainer': {'accelerator': 'cpu'}}}}, paths=second)


@pytest.mark.parametrize('features', [{'tuning': 'yes'}, {'input_types': 'dataset'},
                                     {'resume_modes': ['magic']}, {'evaluation': 'magic'}, {'typo': True}])
def test_invalid_features_fail(features):
    with pytest.raises(ValueError):
        ExecutionFeatures.from_metadata({'execution_features': features})


@pytest.mark.parametrize('agent,extra,message', [
    ('sb3-dev/ppo', {'evaluator': 'custom'}, 'native evaluation'),
    ('sb3-dev/ppo', {'resume_from': 'x.zip'}, 'exact training resume'),
    ('d3rlpy-dev/bc', {'data_module': 'custom'}, 'custom data_module'),
    ('d3rlpy-dev/bc', {'tuning': {'component': 'optuna', 'n_trials': 2}}, 'tuning/winner-promotion'),
])
def test_migrated_adapters_preserve_rejections(agent, extra, message):
    with pytest.raises(ValueError, match=message):
        execution_backend({'agent': agent, **extra})


@pytest.mark.parametrize('reference,declared,message', [
    ('hooks.py:validate', False, 'outside declared payload'),
    ('missing.py:validate', True, 'Missing or escaping payload'),
    ('../outside.py:validate', True, 'Unsafe package path'),
])
def test_hook_payload_validation_is_metadata_only(tmp_path, reference, declared, message):
    from src.app.components.archive import validate_native_package
    paths = provider(tmp_path, hooks=True)
    root = paths.root / 'src/usr/packages/independent'
    (root / 'hooks.py').write_text('raise AssertionError("validation executed provider code")')
    manifest = json.loads((root / '.theta_component.json').read_text())
    manifest['capabilities'][0]['execution_preflight'] = reference
    if not declared:
        manifest['payload']['implementations'].remove('hooks.py')
    with pytest.raises(ValueError, match=message):
        validate_native_package(manifest, root)


@pytest.mark.parametrize('result', ['None', '{"cpu": 0}', '{"cpu": float("nan")}',
                                    '{"accelerator": "no"}', '{"unknown": 1}'])
def test_invalid_hook_resource_hints_fail_at_planning(tmp_path, result):
    paths = provider(tmp_path, hooks=True)
    root = paths.root / 'src/usr/packages/independent'
    (root / 'hooks.py').write_text(f'def resources(cfg, *, paths):\n return {result}\n')
    with pytest.raises(ValueError, match='resource_hint|resource hints'):
        workload_estimate({'agent': 'probe/learn'}, paths=paths)


def test_sb3_resource_hook_matches_native_policy_and_explicit_cpu():
    cfg = {'agent': 'sb3-dev/ppo', 'model': {'sb3_native_policy': True}}
    assert uses_accelerator(cfg)
    assert not uses_accelerator({**cfg, 'trainer': {'accelerator': 'cpu'}})
