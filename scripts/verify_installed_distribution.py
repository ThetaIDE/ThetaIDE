"""Verify built artifacts with an isolated interpreter from an unrelated directory.

Build: python -c 'from setuptools.build_meta import build_sdist, build_wheel; build_sdist("/tmp/dist"); build_wheel("/tmp/dist")'
Install: python -m venv /tmp/theta-installed; /tmp/theta-installed/bin/python -m pip install /tmp/dist/*.whl
Run this script with --python /tmp/theta-installed/bin/python --wheel ... --sdist ... --output ...
No development import path or system-site-packages is used for the probe.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
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
        'storage_version': 2, 'api_version': 1,
        'capabilities': [{'id': 'probe', 'kind': 'model', 'entry_point': 'provider.py:Model'}],
        'payload': {'implementations': ['provider.py'], 'resources': [],
                    'configs': {'in/config/model/probe.yaml': 'configs/model/probe.yaml'}}}))
    (selected.user_config / 'experiment').mkdir(parents=True)
    (selected.user_config / 'experiment/probe.yaml').write_text('# @package _global_\ndefaults: [override /model: probe]\n')
    before_selected = inventory(selected)
    assert ConfigTree(selected.config, workspace=selected).models['probe'].parameters['width'] == width
    assert compose_experiment('probe', workspace=selected).cfg.model.width == width
    worker = subprocess.run([sys.executable, '-I', '-m', 'src.app.train', '--cfg', 'job', '+experiment=probe'],
                            env={**os.environ, **selected.process_environment()}, capture_output=True, text=True)
    assert worker.returncode == 0, worker.stderr
    assert f'width: {width}' in worker.stdout
    assert inventory(selected) == before_selected
    agreement.append({'workspace': name, 'owner': 'distribution-probe', 'width': width, 'worker_exit': worker.returncode})
# Check native checkpoint import identity after workspace relocation in a fresh installed process.
import pickle, zipfile
from src.app.components.installer import HubInstaller
from src.app.components.models import HubComponent, ReleaseInfo
from src.app.core.provider_loader import resolve_provider
checkpoint_meta = {'id': 'checkpoint-probe', 'name': 'Checkpoint probe', 'kind': 'model', 'version': '1.0.0',
 'storage_version': 2, 'api_version': 1,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('python', 'wheel', 'sdist', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    with zipfile.ZipFile(args.wheel) as archive:
        wheel_files = sorted(archive.namelist())
    with tarfile.open(args.sdist) as archive:
        sdist_files = sorted('/'.join(p.split('/')[1:]) for p in archive.getnames())
    forbidden = ('src/plugins/components/installed/', 'src/plugins/app/installed/',
                 'in/datasets/', 'results/', '.thetaide/', 'node_modules/', '.venv/', 'optional_components/')
    for files in (wheel_files, sdist_files):
        assert not any(any(name.startswith(part) if part in ('results/', '.thetaide/', 'in/datasets/') else part in name for part in forbidden) for name in files)
        assert 'src/plugins/inventory.json' in files
        assert 'src/app/config_defaults/config.yaml' in files
        assert 'frontend/icons/play.svg' in files or any(n.startswith('frontend/icons/') for n in files)
        # Fixed saved-target shims are the only allowed usr payload.
        allowed = {'src/usr/__init__.py', 'src/usr/algorithms/__init__.py',
                   'src/usr/algorithms/agent_registry.py', 'src/usr/algorithms/base_agent.py',
                   'src/usr/models/__init__.py', 'src/usr/models/sequential_spec/__init__.py',
                   'src/usr/models/sequential_spec/sequential_spec.py'}
        assert not [n for n in files if n.startswith('src/usr/') and Path(n).suffix and n not in allowed]
    environment = {k: v for k, v in os.environ.items() if not k.startswith(('PYTHON', 'THETA_'))}
    with tempfile.TemporaryDirectory(prefix='theta-installed-probe-') as directory:
        run = subprocess.run([str(args.python.absolute()), '-I', '-c', PROBE], cwd=directory,
                             env=environment, capture_output=True, text=True)
        if run.returncode:
            raise RuntimeError(run.stderr)
    report = json.loads(run.stdout)
    report['artifacts'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.wheel, args.sdist)}
    report['contents'] = {'wheel': wheel_files, 'sdist': sdist_files}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('inventory', 'contents', 'dependencies', 'sys_path')}, indent=2))


if __name__ == '__main__':
    main()
