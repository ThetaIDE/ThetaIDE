"""Storage boundaries, metadata discovery, transactions, migration and worker integration."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile, ZipInfo

import pytest

from src.app.components.installer import HubInstaller
from src.app.components.models import HubComponent, ReleaseInfo
from src.app.components.transactions import FileTransaction, provider_lease, workspace_lock
from src.app.core.provider_catalog import ProviderCatalog, DuplicateProvider, content_hash, file_hash
from src.app.core.provider_loader import resolve_provider
from src.app.workspace import PROJECT_ROOT, WorkspacePaths


def archive(tmp_path, package_id="toy-suite", kind="model", version="1.0.0", files=None, **extra):
    meta = {"id": package_id, "name": package_id, "kind": kind, "version": version, "storage_version": 2, "api_version": 1,
            "capabilities": [{"id": "toy", "kind": kind, "entry_point": "model.py:Toy"}],
            "payload": {"implementations": ["model.py", "helper.py"], "resources": [], "configs": {"in/config/model/toy.yaml": "profiles/toy.yaml"}}, **extra}
    target = tmp_path / (package_id + "-" + version + ".zip")
    payload = {"model.py": "from .helper import VALUE\nclass Toy:\n    value=VALUE\n", "helper.py": "VALUE=1\n", "profiles/toy.yaml": "architecture: toy\nvalue: 1\n", **(files or {})}
    with ZipFile(target, "w") as z:
        z.writestr(".theta_component.json", json.dumps(meta))
        for relative, text in payload.items(): z.writestr(relative, text)
    return HubComponent(id=package_id, name=package_id, kind=kind, version=version,
                        releases={version: ReleaseInfo(version, target.as_uri(), "")})


def test_metadata_headless_relative_imports_and_workspace_identity(tmp_path, monkeypatch):
    for name, value in (("a", 1), ("b", 2)):
        workspace = tmp_path / name
        installer = HubInstaller(workspace, tmp_path / (name + "-custom-state"))
        component = archive(tmp_path, files={"helper.py": f"VALUE={value}\n", "marker.py": "raise AssertionError('discovery executed code')"})
        installer.install(component, install_dependencies=False)
        paths = installer.paths
        catalog = ProviderCatalog(paths)
        package, cap = catalog.resolve("model", "toy")
        assert package.origin == "local_archive" and not package.writable
        assert not paths.research.exists() and not (workspace / "frontend").exists()
        monkeypatch.setenv("THETA_WORKSPACE", str(workspace))
        cls = resolve_provider("model", "toy", paths)
        assert cls.value == value
        if name == "a": first = cls
        else: assert first is not cls
    result = subprocess.run([sys.executable, "-c", "from src.app.core.provider_catalog import ProviderCatalog; import sys; ProviderCatalog(); assert not any(x.startswith(('PyQt6','optuna','stable_baselines3','torch')) for x in sys.modules)"], cwd=PROJECT_ROOT, env={**os.environ, "THETA_WORKSPACE": str(tmp_path / 'b')}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("boundary", ["config", "package", "record"])
def test_update_failure_restores_package_configs_and_modes(tmp_path, monkeypatch, boundary):
    installer = HubInstaller(tmp_path / "ws", tmp_path / "state")
    component = archive(tmp_path)
    installer.install(component, install_dependencies=False)
    original = content_hash(installer.resolve_target_dir(component))
    config = installer.paths.config / "model/toy.yaml"
    config.write_text("architecture: toy\nvalue: 99\n")
    before = config.read_bytes()
    update = archive(tmp_path, version="2.0.0", files={"helper.py": "VALUE=7\n", "profiles/toy.yaml": "architecture: toy\nvalue: 2\nnew: true\n"})
    def fail(current, identity):
        if current == boundary: raise RuntimeError("injected")
    monkeypatch.setattr(installer, "_failure_point", fail)
    with pytest.raises(RuntimeError, match="injected"): installer.install(update, install_dependencies=False)
    assert content_hash(installer.resolve_target_dir(component)) == original
    assert config.read_bytes() == before
    assert not list((installer.paths.root / ".thetaide/transactions").glob("*/journal.json"))


def test_success_update_invalidation_and_uninstall_edit_retention(tmp_path):
    installer = HubInstaller(tmp_path / "ws", tmp_path / "state")
    component = archive(tmp_path)
    installer.install(component, install_dependencies=False)
    first = resolve_provider("model", "toy", installer.paths)
    installer.install(archive(tmp_path, version="2.0.0", files={"helper.py": "VALUE=2\n"}), install_dependencies=False)
    second = resolve_provider("model", "toy", installer.paths)
    assert first is not second and second.value == 2
    profile = installer.paths.config / "model/toy.yaml"
    profile.write_text("architecture: toy\nvalue: 99\n")
    installer.uninstall(component)
    assert ProviderCatalog(installer.paths).resolve("model", "toy") is None
    assert any(p.read_text().endswith("value: 99\n") for p in installer._retained_config_dir(component.id).rglob("*.yaml"))


def test_active_jobs_and_concurrent_operations_defer_changes(tmp_path):
    installer = HubInstaller(tmp_path / "ws", tmp_path / "state")
    component = archive(tmp_path)
    with provider_lease(installer.paths):
        with pytest.raises(RuntimeError, match="in use"): installer.install(component, install_dependencies=False)
    with workspace_lock(installer.paths):
        with pytest.raises(RuntimeError, match="Another workspace"): installer.install(component, install_dependencies=False)
    installer.install(component, install_dependencies=False)


@pytest.mark.parametrize("payload", [{"src/app/evil.py": "x=1"}, {"frontend/evil.py": "x=1"}, {"../escape": "bad"}, {"in/datasets/x.csv": "data"}, {"in/rules/x": "legacy"}])
def test_archive_escape_and_protected_payload_rejected(tmp_path, payload):
    installer = HubInstaller(tmp_path / "ws", tmp_path / "state")
    with pytest.raises(ValueError): installer.install(archive(tmp_path, files=payload), install_dependencies=False)
    assert not installer.paths.research.exists()
    assert not installer.paths.installed_components.exists()


def test_bundled_and_cross_kind_dependency_ownership(tmp_path):
    installer = HubInstaller(tmp_path / "ws", tmp_path / "state")
    component = archive(tmp_path, package_id="my-panel", kind="plugin", capabilities=[],
                        dependencies={"components": [{"id": "sequential_spec", "version": ">=1.0.0"}]},
                        files={"plugin.json": json.dumps({"id": "my-panel", "entry_point": "Panel", "core": True}),
                               "__init__.py": "class Panel: pass\n"})
    installer.install(component, install_dependencies=False)
    assert installer.resolve_target_dir(component).parent == installer.paths.installed_app
    assert ProviderCatalog(installer.paths).package("sequential_spec").origin == "bundled"
    with pytest.raises(ValueError, match="bundled"):
        installer.install(archive(tmp_path, package_id="sequential_spec"), install_dependencies=False)


def test_duplicate_owners_and_explicit_user_override(tmp_path):
    paths = WorkspacePaths(tmp_path)
    for relative in ("src/usr/models/a", "src/usr/models/b"):
        root = tmp_path / relative; root.mkdir(parents=True)
        (root / ".theta_component.json").write_text(json.dumps({"id": root.name, "name":root.name, "kind":"model", "version": "1.0.0", "storage_version":2, "api_version":1, "payload":{"implementations":["model.py"],"resources":[],"configs":{}}, "capabilities": [{"id": "same", "kind": "model", "entry_point": "model.py:Model"}]}))
        (root / "model.py").write_text("class Model: pass")
    with pytest.raises(DuplicateProvider, match="a@.*b@"):
        ProviderCatalog(paths).resolve("model", "same")
    paths.data_dir.mkdir()
    (paths.data_dir / "provider_overrides.json").write_text(json.dumps({"model:same": "b"}))
    assert ProviderCatalog(paths).resolve("model", "same")[0].id == "b"


def legacy_workspace(tmp_path):
    paths = WorkspacePaths(tmp_path / "ws", tmp_path / "custom-state")
    source = paths.research / "models/toy"
    source.mkdir(parents=True)
    code = source / "model.py"
    code.write_text("from src.app.core.model_registry import register_model\n@register_model('toy')\nclass Toy: pass\n")
    (source / "__init__.py").write_text("from .model import Toy\n")
    meta = {"id": "toy", "name": "Toy", "kind": "model", "version": "1.2.3", "installed_from": "https://example.test/toy.zip",
            "source_hashes": {"model.py": file_hash(code)}, "target_path": "src/usr/models/toy", "config_paths": []}
    (source / ".theta_component.json").write_text(json.dumps(meta))
    return paths, source


def test_legacy_sources_are_diagnosed_without_discovery_or_mutation(tmp_path):
    paths, source = legacy_workspace(tmp_path)
    before = content_hash(source)
    catalog = ProviderCatalog(paths)
    assert catalog.package('toy') is None
    assert any('storage_version' in message or 'native contract' in message for message in catalog.diagnostics)
    assert content_hash(source) == before and not paths.installed_components.exists()


def test_abandoned_transaction_restores_config(tmp_path):
    paths = WorkspacePaths(tmp_path)
    target = paths.root / 'in/config/edited.yaml'
    target.parent.mkdir(parents=True); target.write_text('value: before')
    transaction = FileTransaction(paths, 'install', 'toy')
    transaction.protect([target]); target.write_text('value: interrupted')
    with workspace_lock(paths): pass
    assert target.read_text() == 'value: before'


def test_legacy_import_contract_identity_and_paradigms():
    from src.app.core import agent_registry as canonical
    from src.usr.algorithms import agent_registry as legacy
    assert canonical is legacy and canonical.AGENT_REGISTRY is legacy.AGENT_REGISTRY
    old = importlib.import_module("src.usr.models.sequential_spec.sequential_spec")
    new = importlib.import_module("src.plugins.components.bundled.sequential_spec.sequential_spec")
    assert old is new
    for name in ("online_rl", "offline_rl", "supervised", "unsupervised"):
        assert (PROJECT_ROOT / "in/config/paradigms" / (name + ".yaml")).is_file()
        assert (PROJECT_ROOT / "src/app/core/paradigm_impls/base" / (name + ".py")).is_file()


def test_fresh_worker_resolves_native_script_and_records_snapshot(tmp_path):
    installer = HubInstaller(tmp_path / "ws", tmp_path / "state")
    component = archive(tmp_path, payload={"implementations":["model.py","helper.py","run.py"],"resources":[],"configs":{"in/config/model/toy.yaml":"profiles/toy.yaml"}},
                        files={"run.py": "from .helper import VALUE\nfrom pathlib import Path\nPath('worker.txt').write_text(str(VALUE))\n"})
    installer.install(component, install_dependencies=False)
    env = {**os.environ, **installer.paths.process_environment(), "PYTHONPATH": str(PROJECT_ROOT)}
    result = subprocess.run([sys.executable, "-m", "src.app.pipeline.resource_worker", "package://toy-suite/run.py"],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (installer.paths.root / "worker.txt").read_text() == "1"


def test_managed_export_run_editable_fork_and_uninstall(tmp_path, monkeypatch):
    from src.app.components.development import ComponentDevelopment
    from src.app.services.component_exporter import ComponentExporter, ComponentExportOptions
    installer = HubInstaller(tmp_path / "ws", tmp_path / "state")
    component = archive(tmp_path)
    installer.install(component, install_dependencies=False)
    before = content_hash(installer.resolve_target_dir(component))
    monkeypatch.setenv('THETA_WORKSPACE', str(installer.paths.root))
    copy = ComponentDevelopment(installer.paths).copy('toy-suite', 'toy_fork')
    assert copy.is_file()
    fork = ProviderCatalog(installer.paths).package('toy_fork')
    assert fork.writable and fork.root.is_relative_to(installer.paths.research)
    assert resolve_provider('model', 'toy_fork', installer.paths).value == 1
    assert content_hash(installer.resolve_target_dir(component)) == before
    output = tmp_path / 'roundtrip.zip'
    result = ComponentExporter(installer.paths.root).export(copy, ComponentExportOptions(output_path=output, version='0.1.0'))
    assert result.success, result.error
    other = HubInstaller(tmp_path / 'other', tmp_path / 'other-state')
    imported = HubComponent(id='toy_fork', name='toy_fork', kind='model', version='0.1.0',
        releases={'0.1.0':ReleaseInfo('0.1.0',output.as_uri(),'')})
    other.install(imported, install_dependencies=False)
    assert resolve_provider('model', 'toy_fork', other.paths).value == 1
    installer.uninstall(component)
    assert fork.root.is_dir() and copy.is_file()


@pytest.mark.parametrize('field,value', [('storage_version', 999), ('api_version', 99)])
def test_future_contract_fails_before_writes(tmp_path, field, value):
    installer = HubInstaller(tmp_path / 'ws', tmp_path / 'state')
    with pytest.raises(ValueError, match='Unsupported'):
        installer.install(archive(tmp_path, **{field:value}), install_dependencies=False)
    assert not installer.paths.installed_components.exists()


def test_distribution_boundaries_and_bundled_assets():
    import tomllib
    settings = tomllib.loads((PROJECT_ROOT/'pyproject.toml').read_text())
    includes = settings['tool']['setuptools']['packages']['find']['include']
    assert 'src.*' not in includes and 'frontend.*' in includes
    assert (PROJECT_ROOT/'setup.py').is_file()
    for relative in ('text_editor/web/vendor/editor.bundle.js', 'text_editor/web/vendor/THIRD_PARTY_LICENSES.txt', 'documentation/diagrams/architecture_overview.svg'):
        assert (WorkspacePaths().bundled_app/relative).is_file()


def test_core_import_boundaries_and_installer_policy():
    allowed = {'src/app/core/import_aliases.py', 'src/app/pipeline/optuna_sweeper.py', 'src/app/pipeline/optuna_utils.py', 'src/app/pipeline/tuners_optuna.py'}
    import ast
    for path in (PROJECT_ROOT/'src/app').rglob('*.py'):
        if any(part in ('config_defaults','__pycache__') for part in path.parts): continue
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node,ast.ImportFrom) and node.module and node.module.startswith('src.plugins.'):
                assert relative in allowed, f'Framework imports provider directly: {relative}:{node.lineno}'
    paths=WorkspacePaths(PROJECT_ROOT)
    for relative in ('src/app/x','frontend/x','src/plugins/components/bundled/x','src/usr/models/x','in/datasets/x'):
        with pytest.raises(ValueError): paths.require_managed(PROJECT_ROOT/relative)


def test_checkpoint_import_identity_survives_fresh_process_and_relocation(tmp_path, monkeypatch):
    from src.app.core.provider_loader import module_prefix
    import pickle, shutil
    installer=HubInstaller(tmp_path/'before',tmp_path/'state')
    installer.install(archive(tmp_path), install_dependencies=False)
    monkeypatch.setenv('THETA_WORKSPACE',str(installer.paths.root))
    cls=resolve_provider('model','toy',installer.paths)
    (installer.paths.root/'saved.pkl').write_bytes(pickle.dumps(cls()))
    moved=tmp_path/'after'
    shutil.move(installer.paths.root,moved)
    code="from src.app.core.provider_catalog import ProviderCatalog; import pickle; from pathlib import Path; x=pickle.loads(Path('saved.pkl').read_bytes()); assert x.value==1"
    result=subprocess.run([sys.executable,'-c',code],cwd=moved,env={**os.environ,'THETA_WORKSPACE':str(moved),'PYTHONPATH':str(PROJECT_ROOT)},capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_all_four_framework_dispatch_without_algorithm_plugins(tmp_path):
    code = '''import importlib.abc, sys
class Absent(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'PyQt6','optuna','stable_baselines3'}: raise ImportError('optional dependency absent')
sys.meta_path.insert(0,Absent())
from src.app.core.paradigm_loader import load_paradigm_definition
from src.app.pipeline.validation import list_paradigms
for name in ('online_rl','offline_rl','supervised','unsupervised'):
    assert name in list_paradigms()
    paradigm=load_paradigm_definition(name)
    assert paradigm.data_module_cls is not None
'''
    result=subprocess.run([sys.executable,'-c',code],cwd=tmp_path,env={**os.environ,'PYTHONPATH':str(PROJECT_ROOT),'THETA_WORKSPACE':str(tmp_path)},capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_loose_objective_files_do_not_implicitly_override_bundled_provider(tmp_path):
    paths = WorkspacePaths(tmp_path)
    root = paths.user('objective'); root.mkdir(parents=True)
    (root/'learning_curve.json').write_text('{"id":"learning_curve"}')
    assert ProviderCatalog(paths).resolve('objective','learning_curve')[0].origin == 'bundled'


def test_editor_defers_authored_source_save_while_job_active(tmp_path, monkeypatch):
    from src.plugins.app.bundled.text_editor.documents import Document
    paths=WorkspacePaths(tmp_path)
    source=paths.user('model')/'mine/model.py';source.parent.mkdir(parents=True);source.write_text('VALUE=1\n')
    monkeypatch.setenv('THETA_WORKSPACE',str(tmp_path))
    document=Document.open(source);document.text='VALUE=2\n'
    with provider_lease(paths):
        with pytest.raises(RuntimeError,match='in use'):document.save()
    assert source.read_text()=='VALUE=1\n'
    document.save();assert source.read_text()=='VALUE=2\n'


def test_legacy_environment_and_plot_roots_are_ignored(tmp_path):
    paths = WorkspacePaths(tmp_path)
    for root in (paths.root/'in/envs/authored', paths.root/'plot/old'):
        root.mkdir(parents=True)
        (root/'implementation.py').write_text("raise AssertionError('legacy code executed')")
    assert all(p.origin == 'bundled' for p in ProviderCatalog(paths).packages)


def test_custom_workspace_legacy_imports_cannot_read_application_user_source(tmp_path):
    code="""import importlib
from src.app.core.provider_catalog import ProviderCatalog
try: importlib.import_module('src.usr.models.dnn')
except ModuleNotFoundError: pass
else: raise AssertionError('Application user source leaked into selected workspace')
"""
    result=subprocess.run([sys.executable,'-c',code],cwd=tmp_path,env={**os.environ,'PYTHONPATH':str(PROJECT_ROOT),'THETA_WORKSPACE':str(tmp_path)},capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_registry_discovery_never_executes_new_format_provider(tmp_path, monkeypatch):
    from src.app.core.model_registry import auto_discover_models
    installer=HubInstaller(tmp_path/'workspace',tmp_path/'state')
    component=archive(tmp_path,files={'model.py':"raise RuntimeError('discovery executed provider')\n"})
    installer.install(component,install_dependencies=False)
    monkeypatch.setenv('THETA_WORKSPACE',str(installer.paths.root))
    auto_discover_models()
    with pytest.raises(RuntimeError,match='discovery executed provider'):
        resolve_provider('model','toy',installer.paths)


def test_run_snapshot_verifies_active_config_rules_and_override_identity(tmp_path):
    from src.app.pipeline.provider_worker import provider_snapshot,verify_snapshot
    installer=HubInstaller(tmp_path/'workspace',tmp_path/'state')
    installer.install(archive(tmp_path),install_dependencies=False)
    paths=installer.paths;snapshot=provider_snapshot(paths)
    overrides=paths.data_dir/'provider_overrides.json';overrides.parent.mkdir(parents=True,exist_ok=True)
    overrides.write_text(json.dumps({'model:toy':'another_owner'}))
    with pytest.raises(RuntimeError,match='override'):verify_snapshot(snapshot,paths)
    overrides.write_text('{}')
    rule=paths.rules/'env/test/clauses.txt';rule.parent.mkdir(parents=True);rule.write_text('rule(X).\n')
    with pytest.raises(RuntimeError,match='rules'):verify_snapshot(snapshot,paths)
    rule.unlink()
    (paths.config/'model/toy.yaml').write_text('architecture: toy\nvalue: 9\n')
    with pytest.raises(RuntimeError,match='config'):verify_snapshot(snapshot,paths)


def test_fresh_remote_install_has_same_provider_payload_identity(tmp_path):
    from src.app.pipeline.provider_worker import provider_snapshot,verify_snapshot
    from src.app.core.provider_catalog import provider_hash
    component=archive(tmp_path)
    first=HubInstaller(tmp_path/'first',tmp_path/'first-state');second=HubInstaller(tmp_path/'second',tmp_path/'second-state')
    first.install(component,install_dependencies=False);second.install(component,install_dependencies=False)
    a=ProviderCatalog(first.paths).package(component.id);b=ProviderCatalog(second.paths).package(component.id)
    assert a.manifest['target_dir']!=b.manifest['target_dir']
    assert provider_hash(a)==provider_hash(b)
    verify_snapshot(provider_snapshot(first.paths),second.paths)


def test_cli_install_update_inspect_and_uninstall_roundtrip(tmp_path):
    component=archive(tmp_path)
    archive_path=tmp_path/(component.id+'-'+component.version+'.zip')
    workspace=tmp_path/'workspace';state=tmp_path/'custom-state'
    prefix=[sys.executable,'-m','src.app.components','--root',str(workspace),'--data-dir',str(state)]
    def run(*args):
        result=subprocess.run([*prefix,*map(str,args)],cwd=PROJECT_ROOT,capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        return json.loads(result.stdout)
    installed=run('install',archive_path,'--no-dependencies')
    assert installed['origin']=='local_archive' and 'src/plugins/components/installed' in installed['root']
    assert run('inspect',component.id)['read_only']
    run('update',archive_path,'--no-dependencies')
    assert run('uninstall',component.id)['removed']
    assert not (workspace/'src/usr').exists()


def test_workspace_switch_keeps_base_paradigms_and_selected_overrides(tmp_path, monkeypatch):
    from src.app.pipeline.validation import list_paradigms, load_paradigm
    for label in ('first', 'second'):
        workspace = tmp_path / label
        directory = workspace / 'in/config/paradigms'
        directory.mkdir(parents=True)
        (directory / 'online_rl.yaml').write_text('marker: ' + label + '\n')
        monkeypatch.setenv('THETA_WORKSPACE', str(workspace))
        assert {'online_rl', 'offline_rl', 'supervised', 'unsupervised'} <= set(list_paradigms())
        assert load_paradigm('online_rl')['marker'] == label
