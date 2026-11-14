"""User overrides survive real package transactions, including interrupted saves."""
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from src.app.components.configuration import ConfigurationFiles
from src.app.components.installer import HubInstaller
from src.app.components.models import HubComponent, ReleaseInfo
from src.app.components.transactions import FileTransaction, workspace_lock
from src.app.workspace import WorkspacePaths


def release(tmp_path, version, width, *, name='boundary-net'):
    root = tmp_path / (name + version)
    (root / 'configs/model').mkdir(parents=True)
    (root / 'model.py').write_text('class Model: pass\n')
    (root / 'configs/model/boundary.yaml').write_text(f'architecture: boundary-net\nwidth: {width}\n')
    meta = {'id': name, 'name': name, 'kind': 'model', 'version': version,
            'storage_version': 2, 'api_version': 1,
            'capabilities': [{'id': 'boundary-net', 'kind': 'model', 'entry_point': 'model.py:Model'}],
            'payload': {'implementations': ['model.py'], 'resources': [],
                        'configs': {'in/config/model/boundary.yaml': 'configs/model/boundary.yaml'}}}
    (root / '.theta_component.json').write_text(json.dumps(meta))
    archive = tmp_path / (name + version + '.zip')
    with ZipFile(archive, 'w') as stream:
        for path in root.rglob('*'):
            if path.is_file(): stream.write(path, path.relative_to(root))
    component = HubComponent.from_dict(meta)
    component.releases = {version: ReleaseInfo(version, archive.as_uri(), '')}
    return component


def test_override_survives_update_failed_replacement_conflict_and_uninstall(tmp_path, monkeypatch):
    paths = WorkspacePaths(tmp_path / 'workspace')
    installer = HubInstaller(paths.root, paths.data_dir)
    installer.install(release(tmp_path, '1.0.0', 17), install_dependencies=False)
    files = ConfigurationFiles(paths)
    override = files.create_override('model/boundary')
    files.configure(override, ['width=9'])
    override.chmod(0o640)
    before = override.read_bytes()
    provenance = (paths.data_dir / 'config_overrides.json').read_bytes()
    installer.install(release(tmp_path, '2.0.0', 22), install_dependencies=False)
    assert override.read_bytes() == before and override.stat().st_mode & 0o777 == 0o640
    assert paths.config_resolver().resolve('model/boundary').file == override
    assert (paths.installed_components / 'boundary-net/configs/model/boundary.yaml').read_text().endswith('width: 22\n')
    def fail(*args): raise OSError('interrupted package publication')
    monkeypatch.setattr(installer, '_failure_point', fail)
    with pytest.raises(OSError, match='publication'):
        installer.install(release(tmp_path, '3.0.0', 33), install_dependencies=False)
    assert override.read_bytes() == before
    assert json.loads((paths.installed_components / 'boundary-net/.theta_component.json').read_text())['version'] == '2.0.0'
    with pytest.raises(ValueError, match='collision'):
        installer.install(release(tmp_path, '1.0.0', 40, name='conflicting'), install_dependencies=False)
    monkeypatch.undo()
    from src.app.core.provider_catalog import get_catalog
    from src.app.components.lifecycle import uninstall
    package = get_catalog(paths).package('boundary-net')
    assert uninstall(installer, {**package.manifest, 'target_dir': str(package.root)})
    assert override.read_bytes() == before
    assert (paths.data_dir / 'config_overrides.json').read_bytes() == provenance


def test_next_lifecycle_recovers_interrupted_override_transaction(tmp_path):
    paths = WorkspacePaths(tmp_path / 'workspace')
    installer = HubInstaller(paths.root, paths.data_dir)
    installer.install(release(tmp_path, '1.0.0', 17), install_dependencies=False)
    override = ConfigurationFiles(paths).create_override('model/boundary')
    override.write_text('architecture: boundary-net\nwidth: 8\n')
    override.chmod(0o640)
    before = override.read_bytes()
    record = paths.data_dir / 'config_overrides.json'
    provenance = record.read_bytes()
    # Leave an active journal as a killed process would, without __exit__.
    transaction = FileTransaction(paths, 'config_override', 'model/boundary')
    transaction.protect([override, record])
    override.write_text('incomplete: true\n')
    record.write_text('{}')
    with workspace_lock(paths): pass
    assert override.read_bytes() == before
    assert override.stat().st_mode & 0o777 == 0o640
    assert record.read_bytes() == provenance
    assert not transaction.directory.exists()


def test_development_inspection_and_read_use_effective_provider_profiles(tmp_path):
    from src.app.components.development import ComponentDevelopment
    paths = WorkspacePaths(tmp_path / 'workspace')
    installer = HubInstaller(paths.root, paths.data_dir)
    installer.install(release(tmp_path, '1.0.0', 17), install_dependencies=False)
    files = ConfigurationFiles(paths)
    override = files.create_override('model/boundary')
    files.configure(override, ['width=9'])
    development = ComponentDevelopment(paths)
    assert development.inspect('boundary-net')['profiles'][0]['path'] == str(override)
    assert development.validate_profile(override)['components'][0]['id'] == 'boundary-net'
    assert files.read('model/boundary')[2]['width'] == 9
