"""Verify built artifacts with an isolated interpreter from an unrelated directory.

Build: python -c 'from setuptools.build_meta import build_sdist, build_wheel; build_sdist("/tmp/dist"); build_wheel("/tmp/dist")'
Install: python -m venv /tmp/theta-installed; /tmp/theta-installed/bin/python -m pip install /tmp/dist/*.whl
Run this script with --python /tmp/theta-installed/bin/python --wheel ... --sdist ... --output ...
No development import path or system-site-packages is used for the probe.

With --hub-root (a Theta Hub checkout) and the environment installed with the ``gui`` extra, it
also runs the installed entry points from a fresh external workspace: ``theta component
install`` installs SB3, ``theta`` trains CartPole there, the ``theta-ide`` desktop opens on that
workspace, and the desktop's job manager launches the experiment. Installing the components
pip-installs their requirements into the environment.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import time
import zipfile

PROBE = r'''
import importlib.metadata as metadata
import json, pathlib, sys
assert sys.prefix != sys.base_prefix
assert not any('site-packages' in p and not pathlib.Path(p).resolve().is_relative_to(pathlib.Path(sys.prefix).resolve()) for p in sys.path)
from src.app.workspace import WorkspacePaths
from src.app.workspace_inventory import inventory
from src.app.core.provider_catalog import get_catalog
from src.app.services.config_service import config_schema
from frontend.config_model import ConfigTree
paths = WorkspacePaths(pathlib.Path.cwd() / 'workspace')
paths.root.mkdir()
assert pathlib.Path(paths.application).is_relative_to(pathlib.Path(sys.prefix).resolve()), paths.application
before = inventory(paths)
assert config_schema(workspace=paths)['base_experiment'] is None
assert ConfigTree(paths.config, workspace=paths) is not None
assert inventory(paths) == before
assert not list(paths.root.iterdir()), 'read-only discovery mutated clean workspace'
assert not any(name.startswith(('PyQt6', 'optuna', 'sqlalchemy')) for name in sys.modules)
# Exercise installed GUI metadata, coordinator and real worker discovery in two independent workspaces.
import os, subprocess
from src.app.pipeline.compose import compose_experiment
agreement = []
for width, name in enumerate(('one', 'two'), 31):
    selected = WorkspacePaths(pathlib.Path.cwd() / name)
    provider = selected.user('package') / 'distribution-probe'
    (provider / 'configs/model').mkdir(parents=True)
    (provider / 'provider.py').write_text("raise AssertionError('metadata imported provider code')\n")
    (provider / 'configs/model/probe.yaml').write_text(f'architecture: probe\nwidth: {width}\n')
    (provider / '.theta_component.json').write_text(json.dumps({
        'id': 'distribution-probe', 'name': 'Probe', 'kind': 'model', 'version': '1.0.0',
        'storage_version': 3, 'api_version': 1,
        'capabilities': [{'id': 'probe', 'kind': 'model', 'entry_point': 'provider.py:Model'}],
        'payload': {'implementations': ['provider.py'], 'resources': [],
                    'configs': {'model/probe.yaml': 'configs/model/probe.yaml'}}}))
    (selected.config / 'experiment').mkdir(parents=True)
    (selected.config / 'experiment/probe.yaml').write_text('# @package _global_\ndefaults: [override /model: probe]\n')
    before_selected = inventory(selected)
    assert ConfigTree(selected.config, workspace=selected).models['probe'].parameters['width'] == width
    assert compose_experiment('probe', workspace=selected).cfg.model.width == width
    worker = subprocess.run([sys.executable, '-I', '-m', 'src.app.train', '--cfg', 'job', '+experiment=probe'],
                            env={**os.environ, **selected.process_environment()}, capture_output=True, text=True)
    assert worker.returncode == 0, worker.stderr
    assert f'width: {width}' in worker.stdout
    assert inventory(selected) == before_selected
    assert not (selected.root / 'in').exists(), 'a v3 workspace never creates in/'
    agreement.append({'workspace': name, 'owner': 'distribution-probe', 'width': width, 'worker_exit': worker.returncode})
# Check native checkpoint import identity after workspace relocation in a fresh installed process.
import pickle, zipfile
from src.app.components.installer import HubInstaller
from src.app.components.models import HubComponent, ReleaseInfo
from src.app.core.provider_loader import resolve_provider
checkpoint_meta = {'id': 'checkpoint-probe', 'name': 'Checkpoint probe', 'kind': 'model', 'version': '1.0.0',
 'storage_version': 3, 'api_version': 1,
 'capabilities': [{'id': 'checkpoint-probe', 'kind': 'model', 'entry_point': 'model.py:Model'}],
 'payload': {'implementations': ['model.py'], 'resources': [], 'configs': {}}}
archive = pathlib.Path.cwd() / 'checkpoint.zip'
with zipfile.ZipFile(archive, 'w') as bundle:
    bundle.writestr('.theta_component.json', json.dumps(checkpoint_meta))
    bundle.writestr('model.py', 'class Model: value = 7\n')
installer = HubInstaller(pathlib.Path.cwd() / 'checkpoint-before', pathlib.Path.cwd() / 'checkpoint-before/.thetaide')
component = HubComponent.from_dict(checkpoint_meta)
component.releases = {'1.0.0': ReleaseInfo('1.0.0', archive.as_uri(), '')}
installer.install(component, install_dependencies=False)
model = resolve_provider('model', 'checkpoint-probe', installer.paths)
(installer.paths.root / 'saved.pkl').write_bytes(pickle.dumps(model()))
moved = pathlib.Path.cwd() / 'checkpoint-after'
installer.paths.root.rename(moved)
code = "from src.app.core.provider_catalog import ProviderCatalog; import pickle,sys,pathlib; ProviderCatalog(); assert pickle.loads(pathlib.Path(sys.argv[1]).read_bytes()).value == 7"
reload = subprocess.run([sys.executable, '-I', '-c', code, str(moved / 'saved.pkl')],
                        env={**os.environ, 'THETA_WORKSPACE': str(moved)}, capture_output=True, text=True)
assert reload.returncode == 0, reload.stderr
print(json.dumps({'application': str(paths.application), 'prefix': sys.prefix,
 'sys_path': sys.path, 'packages': [{'id': p.id, 'origin': p.origin, 'version': p.version} for p in get_catalog(paths).packages],
 'inventory': before, 'dependencies': sorted((d.metadata['Name'], d.version) for d in metadata.distributions()),
 'headless_imports': True, 'clean_workspace_unchanged': True, 'two_workspace_agreement': agreement, 'relocated_checkpoint_reload': True}, default=str))
'''

RECIPE = """# @package _global_
defaults:
  - override /env: gym_classic_control/cartpole
  - _self_
paradigm: online_rl
group: installed
experiment_id: cli_smoke
seed: 42
total_timesteps: 64
eval_episodes: 1
no_plot: true
monitors: {}
methods:
  cpu:
    agent: {name: sb3/ppo, n_steps: 16, batch_size: 8, n_epochs: 1}
    model: {name: basics/dnn, hidden_sizes: [16, 16]}
"""

DESKTOP_LAUNCH = r'''
import json, pathlib, sys
from PyQt6.QtCore import QCoreApplication, QTimer
from frontend.runner_manager import JobManager
app = QCoreApplication([])
manager = JobManager(project_root=pathlib.Path.cwd())
job = manager.launch('installed/smoke', ['experiment_id=desktop_smoke'])
finished = {}
def changed(record):
    if record['job_id'] == job['job_id'] and record['status'] in ('completed', 'failed', 'error', 'cancelled'):
        finished.update(record); app.quit()
manager.job_changed.connect(changed)
QTimer.singleShot(600_000, app.quit)
app.exec()
print(json.dumps({'status': finished.get('status'), 'experiment': job['experiment'], 'group': job['group'],
                  'experiment_id': job['experiment_id']}))
'''


def hub_archive(hub, component_id):
    for path in sorted((hub / 'registry').rglob('*.json')):
        entry = json.loads(path.read_text())
        if isinstance(entry, dict) and entry.get('id') == component_id and entry.get('storage_version') == 3:
            return hub / 'packages' / Path(entry['releases'][entry['version']]['url']).name
    raise RuntimeError(f'No storage-v3 release of {component_id} in {hub}')


def installed_launch(python, hub, directory, environment):
    """The installed commands against an external workspace that is their working directory."""
    bin_dir = python.absolute().parent
    workspace = Path(directory) / 'external'
    workspace.mkdir()

    def run(*args, timeout=900, env=None):
        result = subprocess.run([str(a) for a in args], cwd=workspace, env=env or environment,
                                capture_output=True, text=True, timeout=timeout)
        if result.returncode:
            raise RuntimeError(f'{args[0]} {args[1:4]} failed:\n{result.stdout[-4000:]}\n{result.stderr[-4000:]}')
        return result

    run(bin_dir / 'theta', '--help')
    for component in ('dnn', 'gym_classic_control', 'sb3-suite'):
        run(bin_dir / 'theta', 'component', 'install', hub_archive(hub, component))
    recipe = workspace / 'src/usr/config/experiment/installed/smoke.yaml'
    recipe.parent.mkdir(parents=True)
    recipe.write_text(RECIPE)
    started = time.monotonic()
    run(bin_dir / 'theta', 'installed/smoke')
    results = workspace / 'src/usr/results'
    cli_metrics = sorted(str(p.relative_to(workspace)) for p in (results / 'logs/installed/cli_smoke').rglob('metrics.csv'))
    assert cli_metrics, 'the CLI run wrote no metrics under src/usr/results'
    cli_seconds = round(time.monotonic() - started, 1)
    desktop_env = {**environment, 'QT_QPA_PLATFORM': 'offscreen', 'THETAIDE_DIRECT': '1'}
    run(bin_dir / 'theta-ide', '--workspace', workspace, '--startup-diagnostics', Path(directory) / 'desktop',
        '--startup-latency-only', '--diagnostics-exit-after', '8', env=desktop_env, timeout=300)
    launch = json.loads(run(python.absolute(), '-I', '-c', DESKTOP_LAUNCH, env=desktop_env).stdout.strip().splitlines()[-1])
    assert launch['status'] == 'completed', launch
    desktop_metrics = sorted(str(p.relative_to(workspace)) for p in (results / 'logs/installed/desktop_smoke').rglob('metrics.csv'))
    assert desktop_metrics, 'the desktop launch wrote no metrics under src/usr/results'
    assert not any(workspace.glob('results')) and not (workspace / 'in').exists()
    return {'entry_points': ['theta', 'theta-ide'], 'components': ['dnn', 'gym_classic_control', 'sb3-suite'],
            'cli_run': {'metrics': cli_metrics, 'seconds': cli_seconds}, 'desktop_opened': True,
            'desktop_launch': {**launch, 'metrics': desktop_metrics}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('python', 'wheel', 'sdist', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--hub-root', type=Path, help='Theta Hub checkout for the installed launch checks')
    args = parser.parse_args()
    with zipfile.ZipFile(args.wheel) as archive:
        wheel_files = sorted(archive.namelist())
    with tarfile.open(args.sdist) as archive:
        sdist_files = sorted('/'.join(p.split('/')[1:]) for p in archive.getnames())
    # The authoring workspace (src/usr, results included) and workspace state are never distributed.
    rooted = ('src/usr/', '.thetaide/')
    anywhere = ('src/plugins/components/installed/', 'src/plugins/app/installed/', 'node_modules/', '.venv/', 'optional_components/')
    for files in (wheel_files, sdist_files):
        assert not any(name.startswith(rooted) or any(part in name for part in anywhere) for name in files)
        assert 'src/plugins/inventory.json' in files
        assert 'src/app/config_defaults/config.yaml' in files
        assert 'frontend/icons/play.svg' in files or any(n.startswith('frontend/icons/') for n in files)
        # The distribution ships no user payload.
        assert not [n for n in files if n.startswith('src/usr/') and Path(n).suffix]
    environment = {k: v for k, v in os.environ.items() if not k.startswith(('PYTHON', 'THETA_'))}
    with tempfile.TemporaryDirectory(prefix='theta-installed-probe-') as directory:
        run = subprocess.run([str(args.python.absolute()), '-I', '-c', PROBE], cwd=directory,
                             env=environment, capture_output=True, text=True)
        if run.returncode:
            raise RuntimeError(run.stderr)
        report = json.loads(run.stdout)
        if args.hub_root:
            report['installed_launch'] = installed_launch(args.python, args.hub_root.resolve(), directory, environment)
    report['artifacts'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.wheel, args.sdist)}
    report['contents'] = {'wheel': wheel_files, 'sdist': sdist_files}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('inventory', 'contents', 'dependencies', 'sys_path')}, indent=2))


if __name__ == '__main__':
    main()
