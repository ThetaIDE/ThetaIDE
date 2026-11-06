"""Terminal development works with the same services and with GUI imports forbidden."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src.app.components.configuration import ConfigurationFiles
from src.app.components.development import ComponentDevelopment
from src.app.components.installer import HubInstaller
from src.app.workspace import WorkspacePaths

REPOSITORY = Path(__file__).resolve().parents[1]
HEADLESS = '''import importlib.abc, runpy, sys
class NoGui(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('frontend', 'PyQt6', 'PySide6'):
            raise RuntimeError('GUI import forbidden: ' + fullname)
sys.meta_path.insert(0, NoGui())
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name='__main__')
'''


def cli(root, *arguments):
    process = subprocess.run([sys.executable, "-c", HEADLESS, str(REPOSITORY / "run_pipeline.py"),
                              "component", "--root", str(root), *arguments],
                             cwd=root, capture_output=True, text=True,
                             env={**os.environ, "MPLCONFIGDIR": str(root / ".test-cache/matplotlib"),
                                  "XDG_CACHE_HOME": str(root / ".test-cache")})
    try:
        data = json.loads(process.stdout if process.returncode == 0 else process.stderr.splitlines()[-1])
    except ValueError:
        pytest.fail(f"Command failed ({process.returncode}):\n{process.stdout}\n{process.stderr}")
    return process, data


@pytest.fixture
def installed_model(tmp_path):
    root = tmp_path / "research checkout with spaces"
    source = root / "src/usr/models/original"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("from .model import Example\n")
    (source / "model.py").write_text('''from src.app.core.model_registry import register_model
@register_model('original')
class Example:
    def __init__(self, **kwargs): pass
    def forward(self): return 11
''')
    profile = root / "in/config/model/original/default.yaml"
    profile.parent.mkdir(parents=True)
    profile.write_text("# @package model\narchitecture: original\nwidth: 16\n")
    meta = {"id": "original", "kind": "model", "name": "Original", "version": "1.0.0",
            "target_path": "src/usr/models/original", "installed_from": "https://example.invalid/release.zip",
            "config_path": "in/config/model/original/default.yaml",
            "config_paths": ["in/config/model/original/default.yaml"],
            "parameter_schema": {"x-theta-validate": True, "type": "object",
                                 "properties": {"width": {"type": "integer", "minimum": 1}}}}
    (source / ".theta_component.json").write_text(json.dumps(meta))
    HubInstaller(root, root / ".thetaide").protect_installed_components()
    return root, source, profile


def run_model(root, component):
    code = f'''import src.usr.models
src.usr.models.__path__.insert(0, {str(root / 'src/usr/models')!r})
import src.app.core.model_registry as registry
registry.__file__ = {str(root / 'src/app/core/model_registry.py')!r}
print(registry.build_model({component!r}).forward())
'''
    return subprocess.check_output([sys.executable, "-c", code], cwd=REPOSITORY, text=True).strip()


def test_terminal_copy_edit_configure_validate_and_fresh_execution(installed_model):
    root, source, profile = installed_model
    baseline = {p: p.read_bytes() for p in (source / "model.py", source / ".theta_component.json", profile)}
    process, info = cli(root, "inspect", "original")
    assert process.returncode == 0, info
    assert info["origin"] == "hub" and info["read_only"]
    process, result = cli(root, "copy", "original", "my_model", "--name", "My model")
    assert process.returncode == 0, result
    implementation = root / "src/usr/models/my_model/model.py"
    implementation.write_text(implementation.read_text().replace("return 11", "return 29"))
    process, result = cli(root, "config", "set", "model/my_model/default", "width=32")
    assert process.returncode == 0, result
    process, result = cli(root, "config", "validate", "model/my_model/default")
    assert process.returncode == 0, result
    assert result["components"][0]["id"] == "my_model"
    assert result["components"][0]["source"] == str(implementation.parent)
    assert run_model(root, "original") == "11"
    assert run_model(root, "my_model") == "29"
    assert all(p.read_bytes() == contents for p, contents in baseline.items())
    # Subsequent processes inspect the active implementation, not a stale retained snapshot.
    _, updated = cli(root, "inspect", "my_model")
    assert updated["origin"] == "local" and not updated["read_only"]
    assert updated["parent"]["id"] == "original"
    assert updated["source_digest"] != info["source_digest"]


def test_shared_config_copy_protection_conflicts_and_schema(installed_model):
    root, source, profile = installed_model
    configs = ConfigurationFiles(WorkspacePaths(root))
    with pytest.raises(PermissionError, match="read-only"):
        configs.configure(profile, ["width=99"])
    process, error = cli(root, "config", "set", "model/original/default", "width=99")
    assert process.returncode == 2 and error["type"] == "PermissionError"
    copy = configs.copy(profile, "model/custom/default")
    configs.configure(copy, ["width=33"])
    assert copy.read_text().startswith("# @package model")
    assert yaml.safe_load(copy.read_text())["architecture"] == "original"
    with pytest.raises(FileExistsError):
        configs.copy(profile, copy)
    with pytest.raises(ValueError, match="inside in/config"):
        configs.copy(profile, "../../outside")
    (root / "in/config/escape").symlink_to(root.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="inside in/config"):
        configs.copy(profile, "escape/outside")
    configs.configure(copy, ["width=0"])
    process, error = cli(root, "config", "validate", "model/custom/default")
    assert process.returncode == 2 and error["type"] == "ValidationError"
    assert (source / ".theta_component.json").stat().st_mode & 0o222 == 0


def test_ambiguous_identity_and_unsupported_copy_fail_without_writes(installed_model):
    root, source, _ = installed_model
    other = root / "in/envs/other"
    other.mkdir(parents=True)
    manifest = other / ".theta_component.json"
    manifest.write_text(json.dumps({"id": "original", "name": "Collision", "kind": "env", "version": "1.0.0"}))
    service = ComponentDevelopment(root)
    with pytest.raises(ValueError, match="Ambiguous"):
        service.copy("original", "fork")
    assert not (root / "src/usr/models/fork").exists()
    manifest.write_text(json.dumps({"id": "other", "name": "Other", "kind": "env", "version": "1.0.0"}))
    with pytest.raises(ValueError, match="unsupported"):
        service.copy("other", "fork")
    assert not (root / ".thetaide/components/packages/fork").exists()


def test_context_does_not_write_into_hub_payload(installed_model):
    root, source, _ = installed_model
    process, result = cli(root, "context", "original")
    assert process.returncode == 0, result
    context = Path(result["context"])
    assert context.is_relative_to(root / ".thetaide/components/context")
    assert "Hub originals are read-only" in context.read_text()
    assert not (source / "COMPONENT_CONTEXT.md").exists()


def test_config_copy_failure_leaves_no_partial_file(installed_model, monkeypatch):
    root, _, profile = installed_model
    configs = ConfigurationFiles(WorkspacePaths(root))
    target = root / "in/config/model/broken/default.yaml"
    def failed_publish(*args):
        raise OSError("simulated disk failure")
    monkeypatch.setattr(os, "link", failed_publish)
    with pytest.raises(OSError, match="simulated"):
        configs.copy(profile, target)
    assert not target.exists()
    assert list(target.parent.iterdir()) == []


def test_nonmodel_profile_uses_its_declared_validation_schema(installed_model):
    root, _, _ = installed_model
    source = root / "src/usr/data/example"
    source.mkdir(parents=True)
    profile = root / "in/config/data/example/default.yaml"
    profile.parent.mkdir(parents=True)
    profile.write_text("batch_size: 0\n")
    (source / ".theta_component.json").write_text(json.dumps({
        "id": "example", "name": "Example", "kind": "datamodule", "version": "1.0.0",
        "config_path": "in/config/data/example/default.yaml",
        "parameter_schema": {"x-theta-validate": True, "type": "object",
                             "properties": {"batch_size": {"type": "integer", "minimum": 1}}},
    }))
    process, error = cli(root, "config", "validate", "data/example/default")
    assert process.returncode == 2 and error["type"] == "ValidationError"


def test_path_policy_and_compatibility_modules_share_class_identity(tmp_path):
    from frontend.hub.installer import HubInstaller as LegacyInstaller
    from frontend.hub.models import HubComponent as LegacyComponent
    from src.app.components.models import HubComponent

    assert LegacyInstaller is HubInstaller
    assert LegacyComponent is HubComponent
    paths = WorkspacePaths(tmp_path)
    assert paths.config == tmp_path / "in/config"
    assert paths.research == tmp_path / "src/usr"
    assert paths.config_file("in/config/model/a.yaml") == paths.config_file("model/a")


def test_experiment_preflight_reports_effective_method_source(installed_model, fixture_workspace):
    root, _, _ = installed_model
    shutil.copytree(fixture_workspace / "in/config", root / "in/config", dirs_exist_ok=True)
    # The UI fixture stores legacy defaults as a mapping; actual Hydra method
    # composition needs a normal runnable agent config.
    (root / "in/config/agent/ppo.yaml").write_text("algorithm: ppo\ncompatible_paradigms: [online_rl]\nlr: 0.0003\n")
    process, copied = cli(root, "copy", "original", "my_model")
    assert process.returncode == 0, copied
    arguments = ("validate-experiment", "thetaide/cartpole_ppo_reference", "++methods.ppo.model=my_model/default")
    process, result = cli(root, *arguments, "++methods.ppo.model_params.width=48")
    assert process.returncode == 0, result
    model = next(c for c in result["resolved_components"] if c["kind"] == "model")
    assert model["components"][0]["id"] == "my_model"
    assert model["effective_parameters"]["width"] == 48
    # An invalid constructor parameter is checked after method override composition.
    process, error = cli(root, *arguments, "++methods.ppo.model_params.width=0")
    assert process.returncode == 2, error
    assert "minimum" in error["error"]
    # A runtime architecture override must be reported, never hidden by the profile path.
    process, result = cli(root, *arguments, "++model.architecture=original")
    assert process.returncode == 0, result
    model = next(c for c in result["resolved_components"] if c["kind"] == "model")
    assert model["components"][0]["id"] == "original"
