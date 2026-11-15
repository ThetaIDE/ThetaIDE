"""Unit tests for standard benchmark suites, gym_classic_control, and suite meta-packages."""
import json
from pathlib import Path
import shutil
import tempfile
from zipfile import ZipFile

import pytest
from omegaconf import OmegaConf

from src.app.pipeline.engine import resolve_engine
from src.app.core.env_vectorized import VectorizedBaseEnv
from src.app.components.installer import HubInstaller
from src.app.components.models import AuthorInfo, HubComponent, ReleaseInfo
from component_fixtures import LocalHub


def test_benchmark_engine_resolution():
    """Runner selection is data driven by each method profile's entrypoint."""
    for path in (
        "src/usr/algorithms/sb3-suite/sb3_runner.py",
        "src/usr/algorithms/d3rlpy-suite/d3rlpy_runner.py",
    ):
        from src.app.pipeline.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError, match="Removed execution fields"):
            resolve_engine({"entrypoint": path})


def test_gym_classic_control_environments(tmp_path):
    """The released gym_classic_control environments instantiate and step cleanly."""
    from src.app.core.provider_catalog import get_catalog
    from src.app.core.provider_loader import load_module
    installer = HubInstaller(workspace_dir=tmp_path / "workspace", data_dir=tmp_path / "data")
    LocalHub().install(installer, "gym_classic_control")
    package = get_catalog(installer.paths).package("gym_classic_control")
    VectorizedEnv = load_module(package, "env_vectorized.py", paths=installer.paths).VectorizedEnv

    # CartPole
    cp_env = VectorizedEnv(name="cartpole", n_envs=2)
    obs = cp_env.reset()
    assert obs.shape == (2, 4)
    assert int(cp_env.n_actions) == 2
    step_res = cp_env.step([0, 1])
    assert step_res[0].shape == (2, 4)
    cp_env.close()

    # MountainCar
    mc_env = VectorizedEnv(name="mountaincar", n_envs=1)
    obs = mc_env.reset()
    assert obs.shape == (1, 2)
    assert int(mc_env.n_actions) == 3
    mc_env.close()


def test_suite_installation_and_modular_uninstallation(tmp_path):
    """A suite installs independently owned packages whose profiles resolve from the packages."""
    from src.app.core.provider_catalog import get_catalog
    installer = HubInstaller(workspace_dir=tmp_path / "workspace", data_dir=tmp_path / "data")
    hub = LocalHub()
    suite = hub.install(installer, "basic_suite")
    paths = installer.paths
    children = ("thetalib", "dnn", "nature_cnn", "dueling_resnet", "sb3-suite", "d3rlpy-suite", "atari-env", "gym_classic_control")
    catalog = get_catalog(paths)
    for package_id in ("basic_suite", *children):
        package = catalog.package(package_id)
        assert package is not None and package.managed, package_id
        assert package.root == paths.installation(package_id, package.manifest["kind"])

    # Profiles resolve from their owning package; installation writes no workspace configuration.
    resolver = paths.config_resolver()
    for selector, owner in (("model/basics/dnn.yaml", "dnn"), ("model/basics/nature_cnn.yaml", "nature_cnn"),
                            ("agent/sb3/ppo.yaml", "sb3-suite"), ("agent/d3rlpy/cql.yaml", "d3rlpy-suite"),
                            ("env/gym_classic_control/cartpole.yaml", "gym_classic_control")):
        assert resolver.resolve(selector).package.id == owner, selector
    assert not paths.config.exists() or not any(paths.config.rglob("*.yaml"))

    # A package another installed package requires cannot be removed first.
    with pytest.raises(ValueError, match="dnn is required by basic_suite"):
        installer.uninstall(hub.component("dnn"))

    # Uninstalling the suite leaves its independently owned dependencies installed.
    assert installer.uninstall(suite)
    catalog = get_catalog(paths)
    assert catalog.package("basic_suite") is None
    assert all(catalog.package(package_id) is not None for package_id in children)


def test_missing_component_run_guard(tmp_path):
    """Verify that an experiment with missing/uninstalled components cannot be run and displays a clear missing component indicator."""
    from unittest.mock import MagicMock
    from frontend.app import Window

    # Mock Window minimally without launching a full Qt event loop
    win = MagicMock()
    win.config_tree = MagicMock()
    win.active = None
    win.compose_valid = True
    win.current_experiment = "test_group/test_exp"
    win.compose_error = ""

    mock_btn = MagicMock()
    win.start_button = mock_btn
    win.queue_button = MagicMock()
    win.stop_button = MagicMock()
    win.builder_status = MagicMock()
    win.state_label = None

    # Simulate an experiment referencing an uninstalled agent and model
    mock_viewer = MagicMock()
    mock_viewer._get_explicit_env.return_value = "cartpole"
    mock_viewer._get_base_env.return_value = None
    mock_viewer.raw_data = {
        "paradigm": "online_rl",
        "env": "cartpole",
        "methods": {
            "my_ppo": {
                "agent": "sb3/ppo",
                "model": "basics/dnn",
            }
        }
    }
    win.config_viewer = mock_viewer

    # Tree with missing sb3/ppo and basics/dnn
    mock_tree = MagicMock()
    mock_tree.environments = {"cartpole": MagicMock()}
    mock_tree.agents = {}
    mock_tree.models = {}
    from src.app.workspace import WorkspacePaths
    mock_tree.workspace = WorkspacePaths(tmp_path)
    mock_tree.config_root = mock_tree.workspace.config
    win.config_tree.tree_model = mock_tree
    win.workspace_paths = mock_tree.workspace

    # Test check_missing_components
    missing = Window.check_missing_components(win)
    assert any("algorithm: sb3/ppo" in m or "agent: sb3/ppo" in m for m in missing)
    assert any("model: basics/dnn" in m for m in missing)

    # Test update_launch_state disables button and sets missing component tooltip & text
    Window.update_launch_state(win)
    mock_btn.setEnabled.assert_called_with(False)
    assert "Missing component" in mock_btn.setToolTip.call_args[0][0]
    assert "Missing component" in mock_btn.setText.call_args[0][0]

    # Named model mappings must be checked by their value, not the key "name". A package
    # declaring basics/dnn makes the model available; a profile alone would not.
    shutil.copytree(Path(__file__).parent / "fixtures/workspace_usr/models/test-models",
                    mock_tree.workspace.user("model") / "test-models")
    mock_viewer.raw_data["methods"]["my_ppo"]["model"] = {
        "name": "basics/dnn", "hidden_sizes": [64, 64]
    }
    assert not any(m.startswith("model:") for m in Window.check_missing_components(win))


def test_component_run_guard_uses_composed_inherited_environment(tmp_path, monkeypatch):
    """A child recipe must be checked after its _base.yaml is composed."""
    from unittest.mock import MagicMock
    from frontend.app import Window

    win = MagicMock()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("THETA_WORKSPACE", str(tmp_path))
    monkeypatch.delenv("THETA_DATA_DIR", raising=False)
    config_root = tmp_path / "src/usr/config"
    for group in ("agent", "model", "env"):
        (config_root / group).mkdir(parents=True)
    (config_root / "agent" / "fixture_agent.yaml").write_text("algorithm: fixture_agent\n", encoding="utf-8")
    (config_root / "model" / "fixture_model.yaml").write_text("architecture: fixture_model\n", encoding="utf-8")
    (config_root / "env" / "fixture_env.yaml").write_text("env_id: CartPole-v1\n", encoding="utf-8")
    # Profiles alone do not establish provider availability. Declare native
    # providers without relying on imports or the developer's installed Hub.
    provider = tmp_path / "src/usr/models/fixture"
    provider.mkdir(parents=True)
    (provider / "provider.py").write_text("raise AssertionError('UI executed provider')\n")
    (provider / ".theta_component.json").write_text(json.dumps({
        "id": "fixture", "name": "Fixture", "version": "1.0.0", "kind": "model",
        "storage_version": 3, "api_version": 1,
        "capabilities": [{"id": name, "kind": kind, "entry_point": "provider.py:Provider"}
                         for kind, name in [("env", "fixture_env"), ("algorithm", "fixture_agent"),
                                            ("model", "fixture_model")]],
        "payload": {"implementations": ["provider.py"], "resources": [], "configs": {}},
    }))

    win.current_experiment = "fixtures/online_smoke"
    win.compose_error = ""
    viewer = MagicMock()
    viewer._get_explicit_env.return_value = None
    viewer._get_base_env.return_value = "fixture_env"
    viewer.raw_data = {
        "methods": {"smoke": {"agent": "fixture_agent", "model": "fixture_model"}},
    }
    win.config_viewer = viewer

    tree = MagicMock()
    tree.config_root = tmp_path
    tree.environments = {"fixture_env": MagicMock()}
    tree.agents = {"fixture_agent": MagicMock()}
    tree.models = {"fixture_model": MagicMock()}
    win.config_tree = MagicMock(tree_model=tree)
    win.composed_config = {
        "paradigm": "online_rl",
        "env": {"name": "fixture_env", "env_id": "CartPole-v1"},
        "methods": {"smoke": {"agent": "fixture_agent", "model": "fixture_model"}},
    }

    assert Window.check_missing_components(win) == []


def test_dl_basics_suite_and_mnist_package_install_and_uninstall(tmp_path):
    """dl_basics_suite installs mlp, transformer, lstm and autoencoder; mnist installs and uninstalls on its own."""
    from src.app.core.provider_catalog import get_catalog
    installer = HubInstaller(workspace_dir=tmp_path / "workspace", data_dir=tmp_path / "data")
    hub = LocalHub()
    suite = hub.install(installer, "dl_basics_suite")
    resolver = installer.paths.config_resolver()
    for model in ("mlp", "transformer", "lstm", "autoencoder"):
        assert get_catalog(installer.paths).package(model) is not None, model
        assert resolver.resolve(f"model/basics/{model}.yaml").package.id == model
    mnist = hub.install(installer, "mnist")
    assert installer.paths.config_resolver().resolve("env/mnist/mnist.yaml").package.id == "mnist"
    assert installer.uninstall(mnist)
    assert get_catalog(installer.paths).package("mnist") is None
    assert installer.uninstall(suite)
    assert installer.uninstall(hub.component("mlp"))
    catalog = get_catalog(installer.paths)
    assert catalog.package("mlp") is None
    assert catalog.package("transformer") is not None and catalog.package("lstm") is not None


def test_nsfr_and_neumann_packages(tmp_path):
    """nsfr and neumann install as independent packages; neumann loads NSFR helpers through declared exports."""
    from src.app.core.provider_catalog import get_catalog
    from src.app.core.provider_loader import provider_export
    installer = HubInstaller(workspace_dir=tmp_path / "workspace", data_dir=tmp_path / "data")
    hub = LocalHub()
    for package_id in ("nsfr", "neumann"):
        hub.install(installer, package_id)
        package = get_catalog(installer.paths).package(package_id)
        assert package is not None and (package.root / f"{package_id}.py").is_file()
        assert installer.paths.config_resolver().resolve(f"model/{package_id}.yaml").package.id == package_id
    assert callable(provider_export("nsfr", "get_nsfr_model", paths=installer.paths))
    assert callable(provider_export("neumann", "get_neumann_model", paths=installer.paths))

