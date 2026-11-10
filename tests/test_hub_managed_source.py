import json
import stat
from pathlib import Path

from hub_test_support import native_component

from frontend.hub.client import HubClient
from src.app.components.installer import HubInstaller
from src.app.services.local_components import LocalComponents


def test_hub_update_replaces_modified_source_and_reinstall_uses_release(tmp_path):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    component = native_component(tmp_path, files={"model.py": 'VALUE = "published"\n'},
        capabilities=[{"id": "example", "kind": "model", "entry_point": "model.py:VALUE"}])
    installer = HubInstaller(workspace, workspace / '.thetaide')
    assert installer.install(component)
    source = workspace / 'src/plugins/components/installed/example/model.py'
    assert not stat.S_IMODE(source.stat().st_mode) & 0o222
    source.chmod(0o644)
    source.write_text('VALUE = "old local edit"\n')
    manifest = source.parent / ".theta_component.json"
    assert installer.install(component)
    assert source.read_text() == 'VALUE = "published"\n'
    assert any(p.read_text() == 'VALUE = "old local edit"\n' for p in installer.paths.data_dir.glob('retained_packages/example/*/model.py'))
    assert not stat.S_IMODE(source.stat().st_mode) & 0o222
    service = LocalComponents(workspace)
    service.snapshot(manifest, source='hub')
    cached = service.state()['records']['example']['package']
    assert installer.uninstall(component)
    assert not source.parent.exists()
    assert 'example' not in service.state()['records']
    assert not Path(cached).exists()
    assert installer.install(component)
    assert source.read_text() == 'VALUE = "published"\n'


def test_old_hub_snapshots_do_not_override_remote_catalog(tmp_path):
    service = LocalComponents(tmp_path)
    package = service.root / 'packages/example'
    package.mkdir(parents=True)
    (package / '.theta_component.json').write_text(json.dumps(dict(id='example', name='Old', kind='model', version='0.1.0')))
    service._write({'records': {'example': {'source': 'hub', 'package': str(package)}}, 'hidden': []})
    client = HubClient(tmp_path, tmp_path / '.thetaide')
    assert not package.exists()
    assert client.components == []
    assert service.state()['records'] == {}
