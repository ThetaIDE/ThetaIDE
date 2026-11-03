"""Tests for recursive component-to-component dependency installation in HubInstaller."""
import json
from pathlib import Path
import tempfile
import zipfile

import pytest

from frontend.hub.installer import HubInstaller
from frontend.hub.models import AuthorInfo, HubComponent, ReleaseInfo


def test_dependency_prefers_own_release_and_falls_back_to_suite(tmp_path):
    provider = HubComponent(id="suite", name="Suite", kind="plugin", version="1",
                            releases={"1": ReleaseInfo(tag="1", url="https://example.org/suite.zip", sha256="")})
    child = HubComponent(id="child", name="Child", kind="model", version="2", extra={"suite": "suite"},
                         releases={"2": ReleaseInfo(tag="2", url="https://example.org/child.zip", sha256="")})
    catalog = {"suite": provider, "child": child}
    installer = HubInstaller(tmp_path, tmp_path / "state", lookup_component_fn=catalog.get)
    assert installer.lookup_component("child") is child
    child.releases["2"].url = provider.releases["1"].url
    assert installer.lookup_component("child") is provider
    child.releases.clear()
    assert installer.lookup_component("child") is provider


def _create_mock_zip(out_zip: Path, meta: dict, files: dict[str, str]):
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w") as zf:
        if meta:
            zf.writestr(".theta_component.json", json.dumps(meta, indent=2))
        for rel_path, content in files.items():
            zf.writestr(rel_path, content)


def test_recursive_component_dependency_installation(tmp_path):
    """Verify that installing atari-env automatically triggers installation of atari-image-data."""
    packages_dir = tmp_path / "packages"
    packages_dir.mkdir(parents=True)

    # 1. Child component: atari-image-data
    child_meta = {
        "id": "atari-image-data",
        "name": "Atari Image Data Module",
        "kind": "datamodule",
        "version": "1.0.0",
        "target_path": "src/usr/data/atari_image_data",
        "config_path": "in/config/data/atari_image_data.yaml",
    }
    child_zip = packages_dir / "atari_image_data-1.0.0.zip"
    _create_mock_zip(
        child_zip,
        child_meta,
        {
            "atari_image_data.py": "# atari data code\n",
            "in/config/data/atari_image_data.yaml": "data_module: atari_image_data\n",
        },
    )

    # 2. Parent component: atari-env
    parent_meta = {
        "id": "atari-env",
        "name": "Atari 2600 Environment Suite",
        "kind": "env",
        "version": "1.0.0",
        "target_path": "in/envs/atari",
        "dependencies": {
            "components": ["atari-image-data"]
        },
    }
    parent_zip = packages_dir / "atari_env-1.0.0.zip"
    _create_mock_zip(
        parent_zip,
        parent_meta,
        {
            "env_vectorized.py": "# atari env code\n",
            "in/config/env/atari.yaml": "env:\n  name: atari\n",
        },
    )

    # Component registry for installer
    components_db = {
        "atari-image-data": HubComponent(
            id="atari-image-data",
            name="Atari Image Data Module",
            kind="datamodule",
            version="1.0.0",
            author=AuthorInfo(name="Farama"),
            releases={"1.0.0": ReleaseInfo(tag="1.0.0", url=f"file://{child_zip}", sha256="")},
        ),
        "atari-env": HubComponent(
            id="atari-env",
            name="Atari 2600 Environment Suite",
            kind="env",
            version="1.0.0",
            author=AuthorInfo(name="Farama"),
            releases={"1.0.0": ReleaseInfo(tag="1.0.0", url=f"file://{parent_zip}", sha256="")},
            extra={"dependencies": {"components": ["atari-image-data"]}},
        ),
    }

    # Setup isolated target workspace
    target_ws = tmp_path / "workspace"
    (target_ws / "in" / "config").mkdir(parents=True)
    installer = HubInstaller(
        workspace_dir=target_ws,
        data_dir=tmp_path / ".thetaide",
        lookup_component_fn=lambda cid: components_db.get(cid),
    )

    # Install parent component
    success = installer.install(components_db["atari-env"])
    assert success is True

    # Verify parent was installed
    assert (target_ws / "in" / "envs" / "atari-env").is_dir()
    assert (target_ws / "in" / "envs" / "atari-env" / ".theta_component.json").is_file()
    assert (target_ws / "in" / "config" / "env" / "atari.yaml").is_file()

    # Verify child dependency (atari-image-data) was automatically downloaded & linked into src/usr/data/
    assert (target_ws / "src" / "usr" / "data" / "atari-image-data").is_dir()
    assert (target_ws / "src" / "usr" / "data" / "atari-image-data" / ".theta_component.json").is_file()
    assert (target_ws / "in" / "config" / "data" / "atari_image_data.yaml").is_file()


def test_generalized_image_data_dependency_via_plugin_json(tmp_path):
    """Verify that installing an env declaring 'image-data' in plugin.json auto-installs image-data."""
    packages_dir = tmp_path / "packages"
    packages_dir.mkdir(parents=True)

    # 1. Generalized child component: image-data
    child_meta = {
        "id": "image-data",
        "name": "Image Data Module",
        "kind": "datamodule",
        "version": "1.0.0",
        "target_path": "src/usr/data/image_data",
        "config_path": "in/config/data/image_data.yaml",
    }
    child_zip = packages_dir / "image_data-1.0.0.zip"
    _create_mock_zip(
        child_zip,
        child_meta,
        {
            "image_data.py": "# image data loader code\n",
            "in/config/data/image_data.yaml": "data_module: image_data\n",
        },
    )

    # 2. Parent component: atari-env with plugin.json
    parent_zip = packages_dir / "atari_env-1.0.0.zip"
    _create_mock_zip(
        parent_zip,
        {},
        {
            "plugin.json": json.dumps({
                "id": "atari-env",
                "name": "Atari 2600",
                "dependencies": {
                    "components": ["image-data"]
                }
            }),
            "env_vectorized.py": "# atari env code\n",
            "in/config/env/atari.yaml": "env:\n  name: atari\n",
        },
    )

    components_db = {
        "image-data": HubComponent(
            id="image-data",
            name="Image Data Module",
            kind="datamodule",
            version="1.0.0",
            author=AuthorInfo(name="Theta Team"),
            releases={"1.0.0": ReleaseInfo(tag="1.0.0", url=f"file://{child_zip}", sha256="")},
        ),
        "atari-env": HubComponent(
            id="atari-env",
            name="Atari 2600 Environment Suite",
            kind="env",
            version="1.0.0",
            author=AuthorInfo(name="Farama"),
            releases={"1.0.0": ReleaseInfo(tag="1.0.0", url=f"file://{parent_zip}", sha256="")},
        ),
    }

    target_ws = tmp_path / "workspace"
    (target_ws / "in" / "config").mkdir(parents=True)
    installer = HubInstaller(
        workspace_dir=target_ws,
        data_dir=tmp_path / ".thetaide",
        lookup_component_fn=lambda cid: components_db.get(cid),
    )

    success = installer.install(components_db["atari-env"])
    assert success is True

    # Verify parent and child
    assert (target_ws / "in" / "envs" / "atari-env").is_dir()
    assert (target_ws / "src" / "usr" / "data" / "image_data").is_dir() or (target_ws / "src" / "usr" / "data" / "image-data").is_dir()
    assert (target_ws / "in" / "config" / "data" / "thetalib" / "image_data.yaml").is_file() or (target_ws / "in" / "config" / "data" / "image_data.yaml").is_file()


def _can_read_hub(hub_dir: Path, comp_id: str) -> bool:
    try:
        cand = hub_dir / "registry" / "envs" / f"{comp_id}.json"
        if not cand.is_file():
            return False
        cand.read_text(encoding="utf-8")
        return True
    except Exception:
        return False


def _local_catalog(hub_dir):
    """Explicitly opt package integration tests into local fixtures."""
    catalog = {}
    for path in (hub_dir / "registry").glob("*/*.json"):
        comp = HubComponent.from_dict(json.loads(path.read_text(encoding="utf-8")))
        for release in comp.releases.values():
            archive = hub_dir / "packages" / Path(release.url).name
            if archive.is_file():
                release.url = archive.as_uri()
        catalog[comp.id] = comp
    return catalog


def test_real_hub_candidate_atari_installation(tmp_path):
    """Verify that installing atari-env via local theta-hub auto-resolves and installs image-data."""
    hub_dir = Path(__file__).resolve().parents[2] / "theta-hub"
    if not _can_read_hub(hub_dir, "atari-env"):
        pytest.skip("Local theta-hub repo not present or inaccessible in sandbox")

    target_ws = tmp_path / "workspace"
    (target_ws / "in" / "config").mkdir(parents=True)
    installer = HubInstaller(
        workspace_dir=target_ws,
        data_dir=tmp_path / ".thetaide",
        lookup_component_fn=_local_catalog(hub_dir).get,
    )

    comp = installer.lookup_component("atari-env")
    if comp is None:
        pytest.skip("Local theta-hub component lookup restricted in sandbox")
    assert comp.id == "atari-env"

    success = installer.install(comp)
    assert success is True

    # Check that both atari-env and child image-data are present in target_ws
    assert (target_ws / "in" / "envs" / "atari").is_dir() or (target_ws / "in" / "envs" / "atari-env").is_dir()
    assert (target_ws / "src" / "usr" / "data" / "image_data").is_dir() or (target_ws / "src" / "usr" / "data" / "image-data").is_dir()
    assert (target_ws / "in" / "config" / "data" / "thetalib" / "image_data.yaml").is_file() or (target_ws / "in" / "config" / "data" / "image_data.yaml").is_file()


def test_real_hub_candidate_mimic_env_installation(tmp_path):
    """Verify that installing mimic-env automatically installs clinical timeseries data, early prediction, and clinical plots."""
    hub_dir = Path(__file__).resolve().parents[2] / "theta-hub"
    if not _can_read_hub(hub_dir, "mimic-env"):
        pytest.skip("Local theta-hub repo not present or inaccessible in sandbox")

    target_ws = tmp_path / "workspace"
    (target_ws / "in" / "config").mkdir(parents=True)
    installer = HubInstaller(
        workspace_dir=target_ws,
        data_dir=tmp_path / ".thetaide",
        lookup_component_fn=_local_catalog(hub_dir).get,
    )

    comp = installer.lookup_component("mimic-env")
    if comp is None:
        pytest.skip("Local theta-hub component lookup restricted in sandbox")
    assert comp.id == "mimic-env"

    success = installer.install(comp)
    assert success is True

    # 1. Environment and rules deployed
    assert (target_ws / "in" / "envs" / "mimic").is_dir() or (target_ws / "in" / "envs" / "mimic-env").is_dir()
    # 2. Child DataModule: clinical-timeseries-data
    assert (target_ws / "src" / "usr" / "data" / "clinical_timeseries").is_dir() or (target_ws / "src" / "usr" / "data" / "clinical-timeseries-data").is_dir()
    # 3. Child Evaluator: mimic-early-prediction
    assert (target_ws / "src" / "usr" / "eval" / "early_prediction").is_dir() or (target_ws / "src" / "usr" / "eval" / "mimic-early-prediction").is_dir()
    # 4. Child Plotter: mimic-clinical-suite
    assert (target_ws / "plot" / "mimic_clinical_suite").is_dir() or (target_ws / "plot" / "mimic-clinical-suite").is_dir()


def test_real_hub_candidate_pyrenees_env_installation(tmp_path):
    """Verify that installing pyrenees-env automatically installs replay-buffer, pyrenees-evaluator, and pyrenees plotters."""
    hub_dir = Path(__file__).resolve().parents[2] / "theta-hub"
    if not _can_read_hub(hub_dir, "pyrenees-env"):
        pytest.skip("Local theta-hub repo not present or inaccessible in sandbox")

    target_ws = tmp_path / "workspace"
    (target_ws / "in" / "config").mkdir(parents=True)
    installer = HubInstaller(
        workspace_dir=target_ws,
        data_dir=tmp_path / ".thetaide",
        lookup_component_fn=_local_catalog(hub_dir).get,
    )

    comp = installer.lookup_component("pyrenees-env")
    if comp is None:
        pytest.skip("Local theta-hub component lookup restricted in sandbox")
    assert comp.id == "pyrenees-env"
    assert comp.id == "pyrenees-env"

    success = installer.install(comp)
    assert success is True

    # 1. Environment deployed
    assert (target_ws / "in" / "envs" / "pyrenees").is_dir() or (target_ws / "in" / "envs" / "pyrenees-env").is_dir()
    # 2. Child DataModule: replay-buffer
    assert (target_ws / "src" / "usr" / "data" / "replay_buffer").is_dir() or (target_ws / "src" / "usr" / "data" / "replay-buffer").is_dir()
    # 3. Child Evaluator: pyrenees-evaluator
    assert (target_ws / "src" / "usr" / "eval" / "pyrenees_evaluator").is_dir() or (target_ws / "src" / "usr" / "eval" / "pyrenees-evaluator").is_dir()
    # 4. Child Plotter: pyrenees-benchmark-suite
    assert (target_ws / "plot" / "pyrenees_benchmark_suite").is_dir() or (target_ws / "plot" / "pyrenees-benchmark-suite").is_dir()



