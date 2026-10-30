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
from frontend.hub.installer import HubInstaller
from frontend.hub.models import AuthorInfo, HubComponent, ReleaseInfo


def test_benchmark_engine_resolution():
    """Runner selection is data driven by each method profile's entrypoint."""
    for path in (
        "src/usr/algorithms/sb3-suite/sb3_runner.py",
        "src/usr/algorithms/d3rlpy-suite/d3rlpy_runner.py",
    ):
        script, _ = resolve_engine({"entrypoint": path})
        assert script == path


def test_gym_classic_control_environments():
    """Verify gym_classic_control vectorized environments instantiate and step cleanly."""
    from src.app.core.utils import load_module
    mod_path = Path("in/envs/gym_classic_control/env_vectorized.py")
    if not mod_path.exists():
        pytest.skip("gym_classic_control plugin is uninstalled; skipping local import")
    mod = load_module(str(mod_path))
    VectorizedEnv = mod.VectorizedEnv

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


def _is_accessible_file(path: Path) -> bool:
    try:
        if not path.is_file():
            return False
        with open(path, "rb") as f:
            f.read(1)
        return True
    except (OSError, PermissionError):
        return False


def test_suite_installation_and_modular_uninstallation(tmp_path):
    """Verify that installing a benchmark suite installs child components and child components remain independently manageable."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    installer = HubInstaller(workspace_dir=workspace, data_dir=data_dir)

    # Locate the packaged basic_suite from theta-hub
    hub_packages = Path(__file__).resolve().parents[2] / "theta-hub" / "packages"
    suite_zip = hub_packages / "basic_suite-1.0.0.zip"
    if not _is_accessible_file(suite_zip):
        pytest.skip(f"Packaged suite zip not accessible at {suite_zip}")
    with ZipFile(suite_zip) as archive:
        payload = [name for name in archive.namelist() if name != ".theta_component.json"]
    if not payload:
        pytest.skip("Packaged basic_suite contains metadata only; build its component payload to test installation")

    suite_comp = HubComponent(
        id="basic_suite",
        name="ThetaIDE Basics",
        kind="suite",
        version="1.0.0",
        description="Comprehensive RL benchmark suite",
        author=AuthorInfo(name="Theta-IDE Team"),
        releases={
            "1.0.0": ReleaseInfo(
                tag="v1.0.0",
                url=suite_zip.as_uri(),
                sha256="",
            )
        },
    )

    # Install the suite
    installed = installer.install(suite_comp)
    assert installed is True

    # 1. Verify suite tracking directory was created
    suite_target = workspace / ".thetaide" / "suites" / "basic_suite"
    assert suite_target.is_dir()
    meta_path = suite_target / ".theta_component.json"
    assert meta_path.is_file()
    meta_data = json.loads(meta_path.read_text(encoding="utf-8"))
    assert "installed_components" in meta_data
    assert "dnn" in meta_data["installed_components"]
    assert "sb3-suite" in meta_data["installed_components"]
    assert "d3rlpy-suite" in meta_data["installed_components"]
    assert "atari-env" in meta_data["installed_components"]
    assert "gym_classic_control" in meta_data["installed_components"]

    # 2. Verify constituent child components were installed into their respective target dirs
    assert (workspace / "src" / "usr" / "models" / "dnn").is_dir()
    assert (workspace / "src" / "usr" / "models" / "dueling_resnet").is_dir()
    assert (workspace / "src" / "usr" / "models" / "nature_cnn").is_dir()
    assert (workspace / "src" / "usr" / "algorithms" / "sb3").is_dir() or (workspace / "src" / "usr" / "methods" / "sb3").is_dir()
    assert (workspace / "src" / "usr" / "algorithms" / "d3rlpy").is_dir() or (workspace / "src" / "usr" / "methods" / "d3rlpy").is_dir()
    assert (workspace / "in" / "envs" / "atari").is_dir()
    assert (workspace / "in" / "envs" / "gym_classic_control").is_dir()

    # 3. Verify configs were deployed into basics/ subfolder
    assert (workspace / "in" / "config" / "model" / "basics" / "dnn.yaml").is_file()
    assert (workspace / "in" / "config" / "model" / "basics" / "nature_cnn.yaml").is_file()
    assert (workspace / "in" / "config" / "agent" / "sb3" / "ppo.yaml").is_file()
    assert (workspace / "in" / "config" / "agent" / "d3rlpy" / "cql.yaml").is_file()
    assert (workspace / "in" / "config" / "env" / "gym_classic_control" / "cartpole.yaml").is_file()

    # 4. Modular uninstallation of a single child component (dnn)
    dnn_meta = json.loads((workspace / "src" / "usr" / "models" / "dnn" / ".theta_component.json").read_text(encoding="utf-8"))
    dnn_meta["target_dir"] = str(workspace / "src" / "usr" / "models" / "dnn")
    uninstalled_dnn = installer.uninstall_by_metadata(dnn_meta)
    assert uninstalled_dnn is True
    assert not (workspace / "src" / "usr" / "models" / "dnn").exists()
    assert not (workspace / "in" / "config" / "model" / "basics" / "dnn.yaml").exists()
    assert (workspace / "in" / "config" / "model").is_dir()

    # Remaining child components are still intact!
    assert (workspace / "src" / "usr" / "models" / "dueling_resnet").is_dir()
    assert (workspace / "src" / "usr" / "algorithms" / "sb3").is_dir() or (workspace / "src" / "usr" / "methods" / "sb3").is_dir()
    assert (workspace / "src" / "usr" / "algorithms" / "d3rlpy").is_dir() or (workspace / "src" / "usr" / "methods" / "d3rlpy").is_dir()
    assert (workspace / "in" / "envs" / "atari").is_dir()

    # 5. Full suite uninstallation protects category directories (agent/, model/, env/)
    uninstalled_suite = installer.uninstall(suite_comp)
    assert uninstalled_suite is True
    assert (workspace / "in" / "config" / "agent").is_dir()
    assert (workspace / "in" / "config" / "model").is_dir()
    assert (workspace / "in" / "config" / "env").is_dir()
    assert not (workspace / "in" / "config" / "agent" / "sb3").exists()
    assert not (workspace / "in" / "config" / "model" / "basics").exists()


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
    mock_tree.config_root = tmp_path
    win.config_tree.tree_model = mock_tree

    # Test check_missing_components
    missing = Window.check_missing_components(win)
    assert any("algorithm: sb3/ppo" in m or "agent: sb3/ppo" in m for m in missing)
    assert any("model: basics/dnn" in m for m in missing)

    # Test update_launch_state disables button and sets missing component tooltip & text
    Window.update_launch_state(win)
    mock_btn.setEnabled.assert_called_with(False)
    assert "Missing component" in mock_btn.setToolTip.call_args[0][0]
    assert "Missing component" in mock_btn.setText.call_args[0][0]

    # Named model mappings must be checked by their value, not the key "name".
    model_dir = tmp_path / "model" / "basics"
    model_dir.mkdir(parents=True)
    (model_dir / "dnn.yaml").write_text("architecture: dnn\n")
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
    config_root = tmp_path / "in" / "config"
    for group in ("agent", "model", "env"):
        (config_root / group).mkdir(parents=True)
    (config_root / "agent" / "fixture_agent.yaml").write_text("algorithm: fixture_agent\n", encoding="utf-8")
    (config_root / "model" / "fixture_model.yaml").write_text("architecture: fixture_model\n", encoding="utf-8")
    (config_root / "env" / "fixture_env.yaml").write_text("env_id: CartPole-v1\n", encoding="utf-8")

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
    """Verify that installing dl_basics_suite installs mlp, transformer, lstm, and autoencoder, and mnist installs cleanly."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    installer = HubInstaller(workspace_dir=workspace, data_dir=data_dir)

    hub_packages = Path(__file__).resolve().parents[2] / "theta-hub" / "packages"
    suite_zip = hub_packages / "dl_basics_suite-1.0.0.zip"
    if not _is_accessible_file(suite_zip):
        pytest.skip(f"Packaged suite zip not accessible at {suite_zip}")

    suite_comp = HubComponent(
        id="dl_basics_suite",
        name="Deep Learning Basics Suite",
        kind="suite",
        version="1.0.0",
        description="Deep learning models suite",
        author=AuthorInfo(name="Theta-IDE Team"),
        releases={"1.0.0": ReleaseInfo(tag="v1.0.0", url=suite_zip.as_uri(), sha256="")},
    )

    # 1. Install dl_basics_suite
    installed = installer.install(suite_comp)
    assert installed is True

    # Constituent models installed
    assert (workspace / "src" / "usr" / "models" / "mlp").is_dir()
    assert (workspace / "src" / "usr" / "models" / "transformer").is_dir()
    assert (workspace / "src" / "usr" / "models" / "lstm").is_dir()
    assert (workspace / "src" / "usr" / "models" / "autoencoder").is_dir()

    # Configs deployed
    assert (workspace / "in" / "config" / "model" / "basics" / "mlp.yaml").is_file()
    assert (workspace / "in" / "config" / "model" / "basics" / "transformer.yaml").is_file()
    assert (workspace / "in" / "config" / "model" / "basics" / "lstm.yaml").is_file()
    assert (workspace / "in" / "config" / "model" / "basics" / "autoencoder.yaml").is_file()

    # 2. Install mnist plugin
    mnist_zip = hub_packages / "mnist-1.0.0.zip"
    assert mnist_zip.is_file()
    mnist_comp = HubComponent(
        id="mnist",
        name="MNIST Handwritten Digits",
        kind="env",
        version="1.0.0",
        description="MNIST benchmark",
        author=AuthorInfo(name="Theta-IDE Team"),
        releases={"1.0.0": ReleaseInfo(tag="v1.0.0", url=mnist_zip.as_uri(), sha256="")},
    )
    installed_mnist = installer.install(mnist_comp)
    assert installed_mnist is True
    assert (workspace / "in" / "envs" / "mnist").is_dir()
    assert (workspace / "in" / "config" / "env" / "mnist" / "mnist.yaml").is_file()

    # 3. Modular uninstallation of mlp
    mlp_meta = json.loads((workspace / "src" / "usr" / "models" / "mlp" / ".theta_component.json").read_text(encoding="utf-8"))
    mlp_meta["target_dir"] = str(workspace / "src" / "usr" / "models" / "mlp")
    uninstalled_mlp = installer.uninstall_by_metadata(mlp_meta)
    assert uninstalled_mlp is True
    assert not (workspace / "src" / "usr" / "models" / "mlp").exists()
    assert not (workspace / "in" / "config" / "model" / "basics" / "mlp.yaml").exists()

    # Remaining models intact
    assert (workspace / "src" / "usr" / "models" / "transformer").is_dir()
    assert (workspace / "src" / "usr" / "models" / "lstm").is_dir()


def test_nsfr_and_neumann_packages(tmp_path):
    """Verify that nsfr and neumann can be installed as independent Hub packages with their rules/grammars."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    installer = HubInstaller(workspace_dir=workspace, data_dir=data_dir)
    hub_packages = Path(__file__).resolve().parents[2] / "theta-hub" / "packages"

    for comp_id in ("nsfr", "neumann"):
        pkg_zip = hub_packages / f"{comp_id}-1.0.0.zip"
        if not _is_accessible_file(pkg_zip):
            pytest.skip(f"Packaged component zip not accessible at {pkg_zip}")
        comp = HubComponent(
            id=comp_id,
            name=comp_id.upper(),
            kind="model",
            version="1.0.0",
            description=f"{comp_id} reasoner",
            author=AuthorInfo(name="Hikaru Shindo"),
            releases={"1.0.0": ReleaseInfo(tag="v1.0.0", url=pkg_zip.as_uri(), sha256="")},
        )
        installed = installer.install(comp)
        assert installed is True
        target = workspace / "src" / "usr" / "models" / comp_id
        assert target.is_dir()
        assert (target / f"{comp_id}.py").is_file()
        assert (workspace / "in" / "config" / "model" / f"{comp_id}.yaml").is_file()

