"""Declared cross-package exports and requester dependency validation."""
import json

import pytest

from src.app.core.manifest_schema import validate_manifest_data
from src.app.core.provider_catalog import get_catalog
from src.app.core.provider_loader import load_module, provider_export
from src.app.workspace import WorkspacePaths


def package(paths, package_id, files, *, version='1.0.0', exports=None, components=None):
    root = paths.user('package') / package_id
    root.mkdir(parents=True)
    for name, source in files.items():
        (root / name).write_text(source)
    manifest = {'id': package_id, 'name': package_id, 'version': version, 'kind': 'package',
                'storage_version': 3, 'api_version': 1,
                'capabilities': [{'id': package_id, 'kind': 'component', 'entry_point': f'{next(iter(files))}:VALUE'}],
                'payload': {'implementations': list(files), 'resources': [], 'configs': {}},
                'dependencies': {'components': components or [], 'pip': []}}
    if exports is not None:
        manifest['exports'] = exports
    (root / '.theta_component.json').write_text(json.dumps(manifest))
    assert validate_manifest_data(manifest) == (True, None)
    return root


@pytest.fixture
def workspace(tmp_path):
    paths = WorkspacePaths(tmp_path, application=tmp_path / 'empty-app')
    package(paths, 'lattice-lib', {'helpers.py': 'VALUE = 1\ndef double(x):\n    return 2 * x\nPRIVATE = "hidden"\n'},
            version='1.4.0', exports={'double': 'helpers.py:double'})
    return paths


def consumer(paths, components, call="provider_export('lattice-lib', 'double')"):
    package(paths, 'consumer', {'use.py': 'from src.app.api.packages import configured_export, provider_export\n'
                                          f'VALUE = {call}\n'}, components=components)
    return load_module(get_catalog(paths).package('consumer'), 'use.py', paths=paths)


@pytest.mark.parametrize('declaration', ['lattice-lib', {'id': 'lattice-lib'}, {'id': 'lattice-lib', 'version': '>=1.2,<2'}])
def test_declared_dependency_loads_a_declared_export(workspace, declaration):
    assert consumer(workspace, [declaration]).VALUE(4) == 8


def test_undeclared_dependency_is_rejected_with_the_requester_named(workspace):
    with pytest.raises(ImportError, match="'consumer' uses lattice-lib:double but does not declare"):
        consumer(workspace, [])


def test_incompatible_version_constraint_is_rejected(workspace):
    with pytest.raises(ImportError, match=r'requires lattice-lib >=2.*1\.4\.0 is installed'):
        consumer(workspace, [{'id': 'lattice-lib', 'version': '>=2'}])


def test_only_declared_exports_are_public(workspace):
    with pytest.raises(ImportError, match=r"does not export 'PRIVATE' \(declared exports: double\)"):
        consumer(workspace, ['lattice-lib'], "provider_export('lattice-lib', 'PRIVATE')")


def test_missing_dependency_package_is_actionable(workspace):
    with pytest.raises(ImportError, match="'absent-lib' is not installed"):
        consumer(workspace, ['absent-lib'], "provider_export('absent-lib', 'double')")


def test_host_code_loads_declared_exports_without_a_requester(workspace):
    assert provider_export('lattice-lib', 'double', paths=workspace)(3) == 6


def test_exports_must_name_payload_implementations():
    manifest = {'id': 'bad', 'name': 'bad', 'version': '1.0.0', 'kind': 'package', 'storage_version': 3,
                'api_version': 1, 'capabilities': [],
                'payload': {'implementations': ['a.py'], 'resources': [], 'configs': {}}}
    for exports, message in [({'x': 'b.py:f'}, 'declared payload implementation'), ({'x': 'a.py'}, "file.py:attribute"),
                             ('a.py:f', 'exports must map')]:
        valid, error = validate_manifest_data({**manifest, 'exports': exports})
        assert not valid and message in error


def test_a_configured_export_needs_no_dependency_declaration(workspace):
    assert consumer(workspace, [], "configured_export('export://lattice-lib/double')").VALUE(5) == 10


def test_a_configured_export_is_still_limited_to_declared_exports(workspace):
    with pytest.raises(ImportError, match=r"does not export 'PRIVATE'"):
        consumer(workspace, [], "configured_export('export://lattice-lib/PRIVATE')")


@pytest.mark.parametrize('reference', ['lattice-lib/double', 'export://lattice-lib', 'export:///double', 'package://x/a.py:f'])
def test_configured_exports_must_use_the_export_scheme(workspace, reference):
    from src.app.api.packages import configured_export
    with pytest.raises(ValueError, match='export://<package>/<name>'):
        configured_export(reference, paths=workspace)


def test_selected_capability_resolves_and_loads_a_configured_selector(workspace):
    from src.app.api.packages import selected_capability
    selected = selected_capability('component', 'LATTICE-LIB', paths=workspace)
    assert (selected.package_id, selected.id, selected.kind) == ('lattice-lib', 'lattice-lib', 'component')
    assert selected.package_version == '1.4.0'
    from src.app.core.provider_catalog import provider_hash
    digest = selected.package_sha256
    assert len(digest) == 64 and digest == provider_hash(get_catalog(workspace).package('lattice-lib'))
    assert selected.load() == 1
    assert selected_capability('component', 'absent', paths=workspace) is None
