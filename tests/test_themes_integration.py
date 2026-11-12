"""AL catalog visibility and real bootstrap execution in a clean workspace."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from frontend.config_model import ConfigTree
from src.app.core.manifest_schema import validate_manifest_file
from src.app.core.provider_catalog import get_catalog
from src.app.pipeline.compose import compose_experiment, validate_composed
from src.app.workspace import WorkspacePaths

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src/usr/packages/themes-dev'


def historical_test_workspace(root):
    """Tests explicitly request the historical demo; never migrate user configs."""
    import yaml
    config = root / 'in/config'
    agent = config / 'agent/themes-dev/em_edm.yaml'
    profile = yaml.safe_load(agent.read_text())
    profile['implementation'] = 'demo_reconstruction'
    agent.write_text(yaml.safe_dump(profile))
    for recipe in (config / 'experiment/AL').rglob('*.yaml'):
        doc = yaml.safe_load(recipe.read_text())
        for method in (doc.get('methods') or {}).values():
            model = method.get('model')
            if model == 'themes-dev/ticc':
                method['model'] = 'themes-dev/bundle'
            elif isinstance(model, dict) and model.get('name') == 'themes-dev/ticc':
                model = dict(model)
                model.pop('name')
                method['model'] = {'name':'themes-dev/bundle', 'stage':model}
        recipe.write_text('# @package _global_\n'+yaml.safe_dump(doc))


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    shutil.copytree(ROOT / 'src/app/config_defaults', tmp_path / 'in/config')
    provider = tmp_path / 'src/usr/packages/themes-dev'
    shutil.copytree(SOURCE, provider, ignore=shutil.ignore_patterns('__pycache__', '*.egg-info', 'build'))
    manifest = json.loads((provider / '.theta_component.json').read_text())
    for dest, source in manifest['payload']['configs'].items():
        path = tmp_path / dest
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(provider / source, path)
    historical_test_workspace(tmp_path)
    monkeypatch.setenv('THETA_WORKSPACE', str(tmp_path))
    monkeypatch.delenv('THETA_DATA_DIR', raising=False)
    monkeypatch.delenv('THETA_PROVIDER_SNAPSHOT', raising=False)
    return WorkspacePaths(tmp_path)


def launch(paths, experiment='themes_synthetic_smoke', overrides=()):
    result = subprocess.run([sys.executable, str(ROOT / 'run_pipeline.py'), 'AL/codex/' + experiment, *overrides],
        cwd=paths.root, env={**os.environ, 'THETA_WORKSPACE': str(paths.root),
                            'PYTHONPATH': str(ROOT), 'MPLCONFIGDIR': os.environ.get('MPLCONFIGDIR', str(paths.root / '.mpl')),
                            'LOKY_MAX_CPU_COUNT': '2'}, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_panel_lists_al_group_and_compatible_profiles(workspace):
    tree = ConfigTree(workspace.config)
    assert tree.group('AL').paradigm == 'apprenticeship'
    assert {e.name for e in tree.experiments_in('AL')} == {
        'AL/codex/themes_synthetic', 'AL/codex/themes_synthetic_smoke',
        'AL/codex/themes_mimic_q10_120', 'AL/codex/themes_mimic_cleantrain_q10_116',
        'AL/codex/themes_paper_synthetic','AL/codex/themes_mimic_q10_120_port',
        'AL/themes_mimic_q10_120'}
    assert 'themes-dev/em_edm' in {a.name for a in tree.agents_for('apprenticeship')}
    assert 'themes-dev/bundle' in {m.name for m in tree.models_for('apprenticeship')}
    for experiment in tree.experiments_in('AL'):
        if 'synthetic' not in experiment.name:
            continue  # External CSV recipes are covered with generated CSVs in loader tests.
        validate_composed(compose_experiment(experiment.name, workspace=workspace))


def test_provider_manifest_and_all_payloads_exist():
    valid, error = validate_manifest_file(SOURCE / '.theta_component.json')
    assert valid, error
    manifest = json.loads((SOURCE / '.theta_component.json').read_text())
    from src.app.components.archive import validate_native_package
    validate_native_package(manifest, SOURCE)
    for key in ('implementations', 'resources'):
        assert all((SOURCE / name).exists() for name in manifest['payload'][key])
    assert all((SOURCE / name).is_file() for name in manifest['payload']['configs'].values())


def test_real_bootstrap_exports_and_fresh_reload(workspace):
    result = launch(workspace)
    assert result.stdout.count('[Apprenticeship] evaluate') == 1
    records = list(workspace.results.rglob('execution.json'))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record['status'] == 'completed'
    assert record['runner'] == 'themes-dev/runner'
    assert record['metadata']['backend'] == 'numerical'
    assert record['metadata']['algorithm']['id'] == 'themes-dev/em_edm'
    assert record['result']['progress']['phases'] == ['load', 'stages', 'segments', 'policies', 'evaluate', 'export', 'complete']
    assert np.isfinite(record['result']['metrics']['eval/mixture_nll'])
    bundle = records[0].parent / 'bundle'
    assert (bundle / 'overview.png').stat().st_size > 1000
    assert list(workspace.results.rglob('metrics.csv'))
    metrics = pd.read_csv(next(workspace.results.rglob('metrics.csv')))
    assert {'ticc/changed_fraction', 'em/nll', 'edm/policy_0/cross_entropy'} <= set(metrics)
    assert metrics['step'].is_monotonic_increasing
    snapshots = list(workspace.results.rglob('snapshot_*.json'))
    assert len(snapshots) == 2
    from frontend.monitor.visual_data import VisualSnapshot
    for path in snapshots:
        meta = json.loads(path.read_text())
        assert len(meta['visuals']) == 3
        assert meta['probe_set_sha256']
        for name in meta['visuals']:
            snapshot = VisualSnapshot({'path': path, 'name': name})
            assert snapshot.visual['kind'] in ('matrix', 'distribution')
            snapshot.close()
    probes = next(workspace.results.rglob('probes.npz'))
    with np.load(probes, allow_pickle=False) as bank, np.load(bundle/'grouped_segments.npz', allow_pickle=False) as data:
        train = np.repeat(data['visit_split']=='train', np.diff(data['visit_offsets'])) & data['policy_scoring_mask']
        assert train[bank['sample_ids']].all()
    reload = subprocess.run([sys.executable, '-m', 'themes_al', 'verify', '--output', str(bundle)],
        cwd=workspace.root, env={**os.environ, 'PYTHONPATH': str(workspace.root / 'src/usr/packages/themes-dev')},
        text=True, capture_output=True, timeout=45)
    assert reload.returncode == 0, reload.stdout + reload.stderr
    assert json.loads(reload.stdout)['source_lineage_checked']


@pytest.mark.parametrize('override,message', [
    ('++resume_from=/tmp/model.pt', 'exact training resume is unsupported'),
    ('++evaluator=missing/provider', 'Unresolved apprenticeship evaluator'),
    ('++dataset_path=/tmp/missing-demonstrations.npz', 'dataset|Dataset|demonstration'),
    ('++methods.ticc_em_edm.model=missing/model', 'missing|Unresolved|Unknown'),
    ('++methods.ticc_em_edm.agent={name:themes-dev/em_edm,updates:0}', 'minimum|greater than|positive'),
    ('++data_module=TabularNpzDataModule', 'trajectories|apprenticeship'),
])
def test_invalid_inputs_fail_before_worker_launch(workspace, override, message):
    from src.app.pipeline.exceptions import ConfigurationError
    with pytest.raises((ValueError, ConfigurationError), match=message):
        validate_composed(compose_experiment('AL/codex/themes_synthetic_smoke', [override], workspace=workspace))
    assert not list(workspace.results.rglob('execution.json'))


def test_standalone_and_theta_have_equal_scientific_outputs(workspace):
    launch(workspace)
    record = next(workspace.results.rglob('execution.json'))
    bundle = record.parent / 'bundle'
    standalone = workspace.root / 'standalone'
    result = subprocess.run([sys.executable, '-m', 'themes_al', 'run',
        '--dataset', str(workspace.root / 'src/usr/packages/themes-dev/resources/synthetic.npz'),
        '--output', str(standalone), '--patients', '0', '--stages', '2', '--groups', '2',
        '--min-group-segments', '1', '--ticc-iterations', '2', '--em-iterations', '2',
        '--updates', '3', '--sgld-steps', '1', '--seed', '42', '--no-plots'],
        cwd=workspace.root, env={**os.environ, 'PYTHONPATH': str(workspace.root / 'src/usr/packages/themes-dev'),
                                'LOKY_MAX_CPU_COUNT': '2'}, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    with np.load(bundle / 'grouped_segments.npz') as integrated, np.load(standalone / 'grouped_segments.npz') as direct:
        for key in ('physiological_stages', 'segment_start', 'segment_stop', 'policy_observations', 'segment_policy_probabilities'):
            np.testing.assert_allclose(integrated[key], direct[key], rtol=1e-6, atol=1e-8)


def test_selected_stage_model_substitution_changes_execution(workspace):
    provider = workspace.root / 'src/usr/packages/themes-dev'
    # A selected provider with different behavior must actually run.
    path = provider / 'alternate.py'
    path.write_text('from .themes_al.core import MultiSeriesTICC\n'
                    'class SingleStage(MultiSeriesTICC):\n'
                    ' def __init__(self, clusters, *args): super().__init__(1, *args)\n')
    manifest_path = provider / '.theta_component.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['capabilities'].append({'id': 'themes-dev/single', 'kind': 'model',
        'entry_point': 'alternate.py:SingleStage', 'backend': 'numerical', 'paradigms': ['apprenticeship']})
    manifest['payload']['implementations'].append('alternate.py')
    manifest_path.write_text(json.dumps(manifest))
    profile = workspace.config / 'model/themes-dev/single.yaml'
    profile.write_text('architecture: themes-dev/single\nstages: 1\nticc_iterations: 1\n')
    launch(workspace, overrides=['++methods.ticc_em_edm.model={name:themes-dev/bundle,stage:{architecture:themes-dev/single,stages:1,ticc_iterations:1}}'])
    record = json.loads(next(workspace.results.rglob('execution.json')).read_text())
    assert record['metadata']['model'] == 'themes-dev/bundle'
    assert record['result']['metrics']['model/stages'] == 1
    assert record['result']['metrics']['data/segments'] == 40


def test_gui_launch_preparation_uses_the_same_recipe(workspace, monkeypatch):
    from src.app.pipeline import compose
    from src.app.services import config_service
    monkeypatch.setattr(compose, 'CONFIG_DIR', workspace.config)
    monkeypatch.setattr(config_service, 'CONFIG_DIR', workspace.config)
    monkeypatch.setattr(config_service, 'LOGS_DIR', workspace.results / 'logs')
    launch = config_service.prepare_launch('AL/codex/themes_synthetic_smoke')
    assert launch['group'] == 'AL'
    assert launch['experiment_id'] == 'codex/themes_synthetic_smoke'
    assert launch['local']
    assert launch['agents'] == ['ticc_em_edm']


def test_al_repeats_have_independent_models_and_logs(workspace):
    launch(workspace, overrides=['++variability.training_runs=2'])
    records = list(workspace.results.rglob('execution.json'))
    assert len(records) == 2
    states = [json.loads(p.read_text()) for p in records]
    assert all(s['status'] == 'completed' for s in states)
    assert len({p.parent for p in records}) == 2
    assert len(list(workspace.results.rglob('metrics.csv'))) == 2
    seeds = [json.loads((p.parent / 'bundle/manifest.json').read_text())['parameters']['seed'] for p in records]
    assert len(set(seeds)) == 2


def test_disabled_rich_monitors_skip_probes_and_snapshots(workspace):
    launch(workspace, overrides=['++telemetry.monitors.disabled=[themes/]'])
    assert not list(workspace.results.rglob('snapshot_*.json'))
    assert not list(workspace.results.rglob('probes.npz'))
    state=json.loads(next(workspace.results.rglob('execution.json')).read_text())
    assert state['result']['metrics']['eval/mixture_nll']==pytest.approx(0.6907346119934862)


def test_snapshot_cap_reserves_final_trained_model(workspace):
    launch(workspace, overrides=['++methods.ticc_em_edm.agent={name:themes-dev/em_edm,monitor_max_snapshots:1}'])
    snapshots=list(workspace.results.rglob('snapshot_*.json'))
    assert len(snapshots)==1
    meta=json.loads(snapshots[0].read_text())
    assert meta['details']['round']==2
    assert meta['optimizer_updates']==9


def test_monitor_catalog_is_headless_and_declares_numerical_outputs(workspace):
    code="""import sys,json
from src.app.telemetry.monitors import available_monitors
cfg={'paradigm':'apprenticeship','methods':{'m':{'agent':'themes-dev/em_edm','model':'themes-dev/ticc'}}}
outputs=available_monitors(cfg)
names={o.name for o in outputs}
assert {'ticc/changed_fraction','em/nll','edm/policy_0/cross_entropy','themes/feature_ablation'}<=names
assert 'torch' not in sys.modules and 'PyQt6' not in sys.modules
print('headless catalog passed')
"""
    result=subprocess.run([sys.executable,'-c',code],cwd=workspace.root,
        env={**os.environ,'THETA_WORKSPACE':str(workspace.root),'PYTHONPATH':str(ROOT)},
        text=True,capture_output=True,timeout=45)
    assert result.returncode==0,result.stdout+result.stderr
