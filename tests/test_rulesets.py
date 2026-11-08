"""Rules use one identity and resource path across GUI, runtime and lifecycle."""
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import pytest

from src.app.services.rulesets import (
    available_rulesets,
    resolve_ruleset, rules_folder_targets, symbolic_environment,
)


def resource(root, env="cartpole", name="default", *, legacy=False):
    path = root / ("in/rules" if legacy else "src/usr/rules") / env / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "clauses.txt").write_text("left(X):-leaning_left(X).\n")
    return path


def test_catalog_resolves_profiles_overrides_and_qualified_selectors(tmp_path):
    folder = resource(tmp_path)
    resource(tmp_path, "mountaincar")
    resource(tmp_path, name="anticipatory_v1")
    profile = tmp_path / "in/config/env/gym_classic_control/cartpole.yaml"
    profile.parent.mkdir(parents=True)
    profile.write_text("symbolic_name: cartpole\n")
    from frontend.config_model import ConfigTree
    tree = ConfigTree(tmp_path / "in/config")
    assert tree.ruleset_names("gym_classic_control/cartpole") == ["anticipatory_v1", "default"]
    assert tree.ruleset_names() == ["cartpole/anticipatory_v1", "cartpole/default", "mountaincar/default"]
    assert resolve_ruleset(tmp_path, None, "cartpole/default").directory == folder
    assert symbolic_environment({"name": "gym_classic_control/cartpole", "symbolic_name": "mountaincar"}, tmp_path) == "mountaincar"
    with pytest.raises(ValueError, match="belongs to"):
        resolve_ruleset(tmp_path, "cartpole", "mountaincar/default")
    with pytest.raises(ValueError, match="Select an environment"):
        resolve_ruleset(tmp_path, None, "default")


def test_editor_opens_exact_folder_or_catalog_without_basename_search(tmp_path):
    folder = resource(tmp_path)
    resource(tmp_path, "mountaincar")
    context = {"workspace_root": str(tmp_path), "config": {"env": "cartpole"}}
    assert rules_folder_targets(value="default", context=context)[0]["path"] == str(folder)
    context["config"] = {}
    assert rules_folder_targets(value="default", context=context)[0]["path"] == str(tmp_path / "src/usr/rules")
    from frontend.parameter_actions import opening_targets
    targets = opening_targets({"type": "open-in-editor", "domain": "ruleset"}, value="cartpole/default", context=context, workspace=tmp_path)
    assert targets == [("Open rules folder", folder)]


def test_ruleset_owns_valuation_and_rejects_path_escape(tmp_path):
    folder = resource(tmp_path)
    vm = folder / "valuation.py"
    vm.write_text("def leaning_left(agent): return agent\n")
    meta = folder / "ruleset.yaml"
    meta.write_text("format_version: 1\nvaluation: valuation.py\n")
    assert resolve_ruleset(tmp_path, "cartpole", "default").valuation_path == vm
    meta.write_text("valuation: ../../escape.py\n")
    with pytest.raises(ValueError, match="inside"):
        resolve_ruleset(tmp_path, "cartpole", "default").valuation_path
    for name in ["../default", "/tmp/evil", "default/../../evil"]:
        with pytest.raises(ValueError):
            resolve_ruleset(tmp_path, "cartpole", name)


@pytest.mark.parametrize("legacy_kind", ["directory", "alias", "broken_alias"])
def test_legacy_locations_are_errors_even_with_valid_canonical_rules(tmp_path, legacy_kind):
    resource(tmp_path)
    old = tmp_path / "in/rules"
    old.parent.mkdir()
    if legacy_kind == "directory":
        resource(tmp_path, legacy=True)
    else:
        old.symlink_to(tmp_path / ("src/usr/rules" if legacy_kind == "alias" else "missing"), target_is_directory=True)
    with pytest.raises(ValueError, match="Unsupported legacy rules location"):
        available_rulesets(tmp_path, "cartpole")
    with pytest.raises(ValueError, match="Unsupported legacy rules location"):
        rules_folder_targets(value="default", context={"workspace_root": str(tmp_path), "config": {}})
    assert old.exists() or old.is_symlink()
    assert (tmp_path / "src/usr/rules/cartpole/default/clauses.txt").exists()


@pytest.mark.parametrize("legacy_reference", ["asset", "metadata"])
def test_legacy_package_is_rejected_before_existing_installation_changes(tmp_path, legacy_reference):
    from src.app.components.installer import HubInstaller
    from src.app.components.models import HubComponent, ReleaseInfo
    installer = HubInstaller(tmp_path, tmp_path / ".thetaide")
    target = tmp_path / "src/usr/models/example"
    target.mkdir(parents=True)
    marker = target / "model.py"
    marker.write_text("# existing model\n")
    marker.chmod(0o444)
    manifest = {"id": "example", "name": "Example", "kind": "model", "version": "1.0.0", "installed_from": "https://example.test/old.zip"}
    (target / ".theta_component.json").write_text(json.dumps(manifest))
    archive = tmp_path / "legacy.zip"
    with ZipFile(archive, "w") as z:
        incoming = {**manifest, "version": "2.0.0"}
        if legacy_reference == "metadata":
            incoming["asset_paths"] = ["in/rules/cartpole/default"]
        z.writestr(".theta_component.json", json.dumps(incoming))
        z.writestr("model.py", "# replacement model\n")
        if legacy_reference == "asset":
            z.writestr("in/rules/cartpole/default/clauses.txt", "rules")
    component = HubComponent(id="example", name="Example", kind="model", version="2.0.0",
        releases={"2.0.0": ReleaseInfo(tag="v2.0.0", url=archive.as_uri(), sha256="")})
    with pytest.raises(ValueError, match="Unsupported"):
        installer.install(component, install_dependencies=False)
    assert marker.read_text() == "# existing model\n"
    assert not marker.stat().st_mode & 0o222
    assert json.loads((target / ".theta_component.json").read_text())["version"] == "1.0.0"
    assert not (tmp_path / "in/rules").exists()
    assert not (tmp_path / "src/usr/rules").exists()


def test_canonical_package_lifecycle_preserves_shared_and_user_rules(tmp_path):
    from src.app.components.installer import HubInstaller
    from src.app.components.models import HubComponent, ReleaseInfo
    installer = HubInstaller(tmp_path, tmp_path / ".thetaide")
    for name in ["nsfr", "neumann"]:
        archive = tmp_path / f"{name}.zip"
        with ZipFile(archive, "w") as z:
            z.writestr(".theta_component.json", json.dumps({"id":name,"name":name,"kind":"model","version":"1.0.0","storage_version":2,"api_version":1,"capabilities":[],"rules_root":"resources/rules","payload":{"implementations":["common.py"],"resources":["resources/rules"],"configs":{f"in/config/model/{name}.yaml":f"configs/model/{name}.yaml"}}}))
            z.writestr("resources/rules/cartpole/default/clauses.txt", "left(X):-leaning_left(X).\n")
            z.writestr(f"configs/model/{name}.yaml", "rules: default\n")
            z.writestr("common.py", "# source\n")
        component = HubComponent(id=name, name=name, kind="model", version="1.0.0",
            releases={"1.0.0": ReleaseInfo(tag="v1.0.0", url=archive.as_uri(), sha256="")})
        assert installer.install(component)
    from src.app.services.rulesets import resolve_ruleset
    shared = resolve_ruleset(tmp_path, "cartpole", "default").directory / "clauses.txt"
    user = resource(tmp_path, name="my_rules")
    assert shared.is_file() and not (tmp_path / "in/rules").exists()
    assert shared.is_relative_to(installer.paths.installed_components)
    assert not (tmp_path / "src/usr/rules/cartpole/default").exists()
    owner = installer.find_component_for_path(shared)
    assert owner is not None
    target = installer.paths.installed_components / "nsfr"
    meta = json.loads((target / ".theta_component.json").read_text())
    assert not meta["installed_files"]
    assert installer.uninstall_by_metadata({**meta, "target_dir": target})
    assert resolve_ruleset(tmp_path, "cartpole", "default").directory.is_dir() and user.is_dir()
    assert installer.uninstall(component)
    assert not any(r.name == "default" for r in available_rulesets(tmp_path, "cartpole")) and user.is_dir()
    assert not (tmp_path / "in/rules").exists()


def test_authored_nsfr_package_loads_ruleset_predicates_in_fresh_process(tmp_path):
    source = Path(__file__).resolve().parents[2] / "theta-plugins/models/nsfr"
    if not (source / "resources/rules/cartpole/anticipatory_v1/ruleset.yaml").exists():
        pytest.skip("Authoritative NSFR source with the new resource format is not available")
    from src.app.components.installer import HubInstaller
    from src.app.components.models import HubComponent, ReleaseInfo
    archive = tmp_path / "nsfr.zip"
    with ZipFile(archive, "w") as z:
        for p in source.rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts and p.name != ".DS_Store":
                z.write(p, p.relative_to(source))
    installer = HubInstaller(tmp_path, tmp_path / ".thetaide")
    version = json.loads((source / ".theta_component.json").read_text())["version"]
    component = HubComponent(id="nsfr", name="NSFR", kind="model", version=version,
        releases={version: ReleaseInfo(tag=f"v{version}", url=archive.as_uri(), sha256="")})
    assert installer.install(component)
    project = Path(__file__).resolve().parents[1]
    code = f'''import sys
sys.path.insert(0, {str(project)!r})
sys.path.insert(0, {str(tmp_path / "src/usr/models")!r})
import torch
from src.app.core.provider_loader import provider_object
get_nsfr_model = provider_object("nsfr", "common.py:get_nsfr_model")
from src.app.services.rulesets import resolve_ruleset
m = get_nsfr_model("cartpole", "anticipatory_v1", "cpu", train=True)
x = torch.tensor([[[0., 0., .03, -.8], [0., 0., .03, -.8]]])
y = m(x)
assert y[0, m.prednames.index("left")] > y[0, m.prednames.index("right")]
assert len(m.clauses) == 4 and tuple(m.im.W.shape) == (2, 4)
assert m.fc.vm.val_fns["short_fall_left"].__code__.co_filename.endswith("anticipatory_v1/valuation.py")
fixed = get_nsfr_model("cartpole", "anticipatory_v1", "cpu", train=False)
assert tuple(fixed(x).shape) == (1, 2)
print("ruleset valuation loaded from installed canonical resource")
'''
    import os
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env={**os.environ, **installer.paths.process_environment()}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "canonical resource" in result.stdout


def test_experiment_export_keeps_ruleset_owned_valuations_and_symbolic_environment(tmp_path):
    from src.app.services.experiment_exporter import ExperimentExporter, ExportOptions
    folder = resource(tmp_path, name="anticipatory_v1")
    (folder / "ruleset.yaml").write_text("valuation: valuation.py\n")
    (folder / "valuation.py").write_text("# user-owned predicates\n")
    config = tmp_path / "in/config"
    profile = config / "env/gym/cartpole.yaml"
    profile.parent.mkdir(parents=True)
    profile.write_text("symbolic_name: cartpole\nenv_id: CartPole-v1\n")
    environment = tmp_path / "src/usr/environments/cartpole"
    environment.mkdir(parents=True)
    (environment / "valuation.py").write_text("# observation builder\n")
    (environment / 'env_vectorized.py').write_text('class Env: pass\n')
    (environment / '.theta_component.json').write_text(json.dumps({'id':'cartpole','name':'CartPole','kind':'env','version':'1.0.0','storage_version':2,'api_version':1,
        'capabilities':[{'id':'cartpole','aliases':['gym/cartpole'],'kind':'env','entry_point':'env_vectorized.py:Env','resource_root':'.'}],
        'payload':{'implementations':['env_vectorized.py'],'resources':['valuation.py'],'configs':{}}}))
    recipe = config / "experiment/demo/check.yaml"
    recipe.parent.mkdir(parents=True)
    recipe.write_text("defaults:\n- override /env: gym/cartpole\nenv: {num_envs: 1}\n")
    result = ExperimentExporter(tmp_path).export("demo/check", ExportOptions(output_path=tmp_path / "experiment.zip"))
    assert result.success, result.error
    with ZipFile(result.archive_path) as z:
        assert "resources/rules/cartpole/anticipatory_v1/valuation.py" in z.namelist()
        assert "resources/rules/cartpole/anticipatory_v1/ruleset.yaml" in z.namelist()
        assert "cartpole" in [d["id"] for d in json.loads(z.read(".theta_component.json"))["dependencies"]["components"]]
        assert not any(name.startswith("providers/") for name in z.namelist())
        assert not any(name.startswith("in/rules/") for name in z.namelist())
