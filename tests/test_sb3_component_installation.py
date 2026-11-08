"""Install real component source archives into a clean workspace and train."""

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import zipfile

import pytest


@pytest.mark.skipif(importlib.util.find_spec("stable_baselines3") is None, reason="SB3 is optional")
def test_fresh_plugin_install_runs_selected_blendrl(tmp_path, monkeypatch):
    from frontend.hub.installer import HubInstaller
    from frontend.hub.models import AuthorInfo, HubComponent, ReleaseInfo

    project = Path(__file__).resolve().parents[1]
    sources = project.parent / "theta-plugins"
    remote_hub = os.environ.get("THETA_TEST_REMOTE_HUB")
    if not remote_hub and not (sources / "methods/sb3/theta_sb3_policy.py").is_file():
        pytest.skip("Sibling theta-plugins sources are not present")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    shutil.copytree(project / "src/app/config_defaults", workspace / "in/config")
    # These are fixture configs, not installed/protected components in the new
    # workspace. Do not inherit the source checkout's read-only file modes.
    for path in (workspace / "in/config").rglob("*"):
        if path.is_file():
            path.chmod(path.stat().st_mode | 0o200)
    archives = tmp_path / "archives"
    archives.mkdir()
    components = [
        ("sb3-suite", "method", "methods/sb3", "src/usr/algorithms/sb3-suite"),
        ("dnn", "model", "models/dnn", "src/usr/models/dnn"),
        ("nature_cnn", "model", "models/nature_cnn", "src/usr/models/nature_cnn"),
        ("mlp", "model", "models/mlp", "src/usr/models/mlp"),
        ("nsfr", "model", "models/nsfr", "src/usr/models/nsfr"),
        ("neumann", "model", "models/neumann", "src/usr/models/neumann"),
        ("blendrl", "model", "models/blendrl", "src/usr/models/blendrl"),
        ("gym_classic_control", "env", "envs/gym_classic_control", "in/envs/gym_classic_control"),
    ]
    monkeypatch.setattr("frontend.hub.installer.check_missing_dependencies", lambda _: [])
    catalog = {}
    hub_dir = Path(os.environ["THETA_TEST_LOCAL_HUB"]) if os.environ.get("THETA_TEST_LOCAL_HUB") else None
    remote_entries = {}
    if remote_hub:
        import json
        with urllib.request.urlopen(remote_hub, timeout=20) as response:
            remote_entries = {entry["id"]: entry for entry in json.load(response)["components"]}
    candidate_dir = str(hub_dir / "packages") if hub_dir else os.environ.get("THETA_PLUGIN_CANDIDATE_DIR")
    candidate_names = {"sb3-suite": "sb3_suite-1.0.6.zip", "blendrl": "blendrl-1.1.0.zip",
                       "mlp": "mlp-1.0.1.zip", "nsfr": "nsfr-1.0.3.zip", "neumann": "neumann-1.0.3.zip",
                       "gym_classic_control": "gym_classic_control-1.0.3.zip"}
    for name, kind, relative, target in components:
        source = sources / relative
        archive = Path(candidate_dir) / candidate_names[name] if candidate_dir and name in candidate_names else archives / f"{name}.zip"
        if not remote_hub and not (candidate_dir and name in candidate_names):
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as out:
                for path in source.rglob("*"):
                    if path.is_file() and "__pycache__" not in path.parts and (not path.name.startswith(".") or path.name == ".theta_component.json"):
                        out.write(path, path.relative_to(source))
        if remote_hub:
            catalog[name] = HubComponent.from_dict(remote_entries[name])
        elif hub_dir:
            import json
            entry = json.loads((hub_dir / "registry" / (kind + "s") / f"{name}.json").read_text())
            catalog[name] = HubComponent.from_dict(entry)
            version = catalog[name].version
            release = catalog[name].releases[version]
            release.url = (hub_dir / "packages" / release.url.rsplit("/", 1)[1]).as_uri()
        else:
            import json
            with zipfile.ZipFile(archive) as native_archive:
                metadata=json.loads(native_archive.read('.theta_component.json'))
            catalog[name]=HubComponent.from_dict(metadata)
            catalog[name].releases={metadata['version']:ReleaseInfo(metadata['version'],archive.as_uri(),'')}
    if remote_hub:
        catalog["thetalib"] = HubComponent.from_dict(remote_entries["thetalib"])
    elif hub_dir:
        import json
        provider = HubComponent.from_dict(json.loads((hub_dir / "registry/plugins/thetalib.json").read_text()))
        release = provider.releases[provider.version]
        release.url = (hub_dir / "packages" / release.url.rsplit("/", 1)[1]).as_uri()
        catalog[provider.id] = provider
    installer = HubInstaller(workspace, tmp_path / "state", lookup_component_fn=catalog.get)
    import json
    installer.paths.data_dir.mkdir(parents=True, exist_ok=True)
    for name, _, _, _ in components:
        assert installer.install(catalog[name])
    if hub_dir:
        assert not (tmp_path / "state/plugins/thetalib").exists()

    experiment = workspace / "in/config/experiment/package_smoke.yaml"
    experiment.parent.mkdir(parents=True, exist_ok=True)
    experiment.write_text("# @package _global_\nparadigm: online_rl\ngroup: g\nexperiment_id: e\n"
                          "total_timesteps: 16\neval_episodes: 1\n", encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(project), "THETA_WORKSPACE": str(workspace), "THETA_DATA_DIR": str(installer.paths.data_dir), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    # Exercise normal profile dispatch rather than bypassing it with a fixed URI.
    from src.app.pipeline.engine import resolve_engine
    monkeypatch.setenv("THETA_WORKSPACE", str(workspace))
    monkeypatch.setenv("THETA_DATA_DIR", str(installer.paths.data_dir))
    runner, _ = resolve_engine({"agent": "sb3/ppo"})
    assert runner == "package://sb3-suite/sb3_runner.py"
    command = [sys.executable, "-m", "src.app.pipeline.provider_worker", str(runner), "+experiment=package_smoke", "agent=sb3/ppo", "model=blendrl",
               "env=gym_classic_control/cartpole", "++agent.name=m", "agent.n_steps=8", "agent.batch_size=4",
               "agent.n_epochs=1", "++model.architecture=blendrl", "++model.symbolic.rules=variant2",
               "++model.neural.hidden_sizes=[32,32]", "model.logic_lr=0.0002", "model.blender_lr=0.0001",
               "++model.neural.shared_features=false", "++model.neural.activation=tanh",
               "model.monitoring_interval=1", "model.monitoring_max_samples=16"]
    result = subprocess.run(command, cwd=workspace, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    import pandas as pd
    metrics = pd.read_csv(workspace / "results/logs/g/e/m/version_0/metrics.csv")
    assert "blend/train/entropy_normalized" in metrics.columns
    assert (workspace / "results/logs/g/e/m/version_0/model_monitoring_samples.npz").is_file()
    assert (installer.paths.installed_components / "blendrl/plot/blendrl/routing.py").is_file()
    assert not (workspace / "src/usr").exists()
    ppo_command = [arg for arg in command if arg not in ("++model.symbolic.rules=variant2", "++model.neural.hidden_sizes=[32,32]", "++model.neural.shared_features=false", "++model.neural.activation=tanh", "model.logic_lr=0.0002", "model.blender_lr=0.0001", "model.monitoring_interval=1", "model.monitoring_max_samples=16")]
    ppo_command = ["model=basics/mlp" if arg == "model=blendrl" else
                   "++model.architecture=basics/mlp" if arg == "++model.architecture=blendrl" else
                   "++agent.name=ppo" if arg == "++agent.name=m" else arg for arg in ppo_command]
    result = subprocess.run(ppo_command, cwd=workspace, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    check = ("from stable_baselines3 import PPO; "
             "m=PPO.load('results/checkpoints/g/e/m/best_model.zip'); "
             "assert type(m.policy.theta_model_module).__name__ == 'BlenderActorCritic'; "
             "assert m.policy.theta_model_module.policy_modules[1].network[0].out_features == 32; "
             "assert m.policy.theta_model_module.policy_modules[1].value_network is not None; "
             "assert type(m.policy.theta_model_module.policy_modules[1].network[1]).__name__ == 'Tanh'; "
             "rates=[g['lr'] for g in m.policy.optimizer.param_groups]; "
             "assert any(abs(x-0.0002)<1e-10 for x in rates) and any(abs(x-0.0001)<1e-10 for x in rates)")
    result = subprocess.run([sys.executable, "-c", check], cwd=workspace, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
