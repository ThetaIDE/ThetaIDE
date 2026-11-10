"""Unit and integration tests for Theta HubClient, HubInstaller, and models."""
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile
from unittest.mock import patch
from io import BytesIO

from hub_test_support import native_component

os.environ["QT_QPA_PLATFORM"] = "offscreen"

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    from frontend.hub import HubClient, HubComponent, HubComponentCard, HubDialog, ReleaseInfo
    from frontend.theme import theme_color
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestHubClientAndInstaller(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="theta_test_hub_"))
        self.workspace_dir = self.temp_dir / "workspace"
        self.data_dir = self.temp_dir / "data"
        self.workspace_dir.mkdir(parents=True)
        self.data_dir.mkdir(parents=True)

        self.client = HubClient(
            workspace_dir=self.workspace_dir,
            data_dir=self.data_dir,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_hub_component_serialization(self):
        data = {
            "id": "test-method",
            "name": "Test RL Method",
            "kind": "method",
            "version": "1.0.0",
            "description": "A test reinforcement learning algorithm.",
            "author": {"name": "Alice", "github": "alice"},
            "repository": "https://github.com/alice/test-method",
            "tags": ["rl", "ppo"],
            "releases": {
                "1.0.0": {
                    "tag": "v1.0.0",
                    "url": "https://example.com/test.zip",
                    "sha256": "abc12345"
                }
            }
        }
        comp = HubComponent.from_dict(data)
        self.assertEqual(comp.id, "test-method")
        self.assertEqual(comp.kind, "method")
        self.assertEqual(comp.author.name, "Alice")
        self.assertEqual(comp.author.github, "alice")
        self.assertIn("1.0.0", comp.releases)
        self.assertEqual(comp.latest_release.tag, "v1.0.0")

        # Roundtrip to dict
        roundtrip = comp.to_dict()
        self.assertEqual(roundtrip["id"], "test-method")
        self.assertEqual(roundtrip["releases"]["1.0.0"]["sha256"], "abc12345")

    def test_package_metadata_survives_serialization_and_resolves_provided_ids(self):
        suite = HubComponent.from_dict({
            "id": "conformance-suite",
            "name": "Conformance Suite",
            "kind": "suite",
            "version": "1.0.0",
            "provides": ["conformance-probe"],
            "aliases": ["legacy-conformance"],
        })
        self.client.components = [suite]

        assert suite.to_dict()["provides"] == ["conformance-probe"]
        assert self.client.get_component("conformance-probe") is suite
        assert self.client.get_component("legacy_conformance") is suite

    def test_remote_catalog_and_archive_are_not_shadowed_by_sibling_hub(self):
        sibling = self.workspace_dir.parent / "theta-hub"
        (sibling / "dist").mkdir(parents=True)
        (sibling / "dist/index.json").write_text('{"components": []}')
        (sibling / "packages").mkdir()
        (sibling / "packages/component.zip").write_bytes(b"stale")
        registry = {"components": [{"id": "remote", "name": "Remote", "kind": "model", "version": "1"}]}

        class Response(BytesIO):
            headers = {}

        def fetch(request, timeout):
            if request.full_url.endswith("index.json"):
                return Response(json.dumps(registry).encode())
            return Response(b"remote archive")

        with patch("urllib.request.urlopen", side_effect=fetch) as network:
            assert self.client.fetch_index_sync()[0].id == "remote"
            path = self.client.installer.download_and_verify("https://example.org/component.zip")
            try:
                assert path.read_bytes() == b"remote archive"
            finally:
                path.unlink()
            assert network.call_count == 2

    def test_search_and_filtering(self):
        c1 = HubComponent.from_dict({
            "id": "cql-continuous", "name": "CQL Continuous", "kind": "method",
            "version": "1.0.0", "description": "Offline RL method", "tags": ["rl", "offline"]
        })
        c2 = HubComponent.from_dict({
            "id": "markdown-notes", "name": "Markdown Notes", "kind": "plugin",
            "version": "0.1.0", "description": "Markdown notebook", "tags": ["notes"]
        })
        c3 = HubComponent.from_dict({
            "id": "neumann-fast", "name": "Neumann Reasoner", "kind": "model",
            "version": "0.2.0", "description": "NeSy model", "tags": ["symbolic"]
        })

        self.client.components = [c1, c2, c3]

        # Kind filter
        plugins = self.client.search(kind="plugin")
        self.assertEqual(len(plugins), 1)
        self.assertEqual(plugins[0].id, "markdown-notes")

        # Query filter
        methods = self.client.search(query="cql")
        self.assertEqual(len(methods), 1)
        self.assertEqual(methods[0].id, "cql-continuous")

        # Tag filter
        notes = self.client.search(tag="notes")
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0].id, "markdown-notes")

    def test_target_directory_resolution_by_kind(self):
        for kind in ("plugin", "method", "model", "env"):
            component = HubComponent(id="example", name="Example", kind=kind, version="1.0.0")
            family = "app" if kind == "plugin" else "components"
            expected = self.workspace_dir / "src/plugins" / family / "installed/example"
            self.assertEqual(self.client.installer.resolve_target_dir(component), expected.resolve())
            self.assertFalse(self.client.installer.resolve_target_dir(component).is_relative_to(self.data_dir.resolve()))

    def test_download_sha256_verify_and_install(self):
        comp = native_component(
            self.temp_dir, "sample-rl-agent", kind="method",
            files={"main.py": "class Agent: pass\n"},
            capabilities=[{"id": "sample-rl-agent", "kind": "algorithm", "entry_point": "main.py:Agent"}],
            configs={"in/config/agent/sample-rl-agent.yaml": "learning_rate: 0.001\n"},
        )
        installer = self.client.installer
        self.assertFalse(installer.check_installed(comp)[0])
        self.assertTrue(installer.install(comp))
        dest = installer.resolve_target_dir(comp)
        deployed_cfg = self.workspace_dir / "in/config/agent/sample-rl-agent.yaml"
        self.assertTrue((dest / "main.py").is_file())
        self.assertTrue((dest / "configs/agent/sample-rl-agent.yaml").is_file())
        self.assertEqual(deployed_cfg.read_text(), "learning_rate: 0.001\n")
        self.assertEqual((dest / "main.py").stat().st_mode & 0o222, 0)
        self.assertEqual((dest / ".theta_component.json").stat().st_mode & 0o222, 0)
        self.assertTrue(installer.is_installed_file(deployed_cfg))
        self.assertTrue(installer.install(comp))
        self.assertEqual((dest / "main.py").stat().st_mode & 0o222, 0)
        self.assertEqual(installer.check_installed(comp), (True, "1.0.0"))
        self.assertTrue(installer.uninstall(comp))
        self.assertFalse(dest.exists())
        self.assertFalse(deployed_cfg.exists())
        self.assertFalse(installer.check_installed(comp)[0])

    def test_checksum_mismatch_fails_safely(self):
        zip_path = self.temp_dir / "corrupted_package.zip"
        with ZipFile(zip_path, "w") as zf:
            zf.writestr("test.txt", "hello")

        comp = HubComponent(
            id="corrupted-comp",
            name="Corrupted",
            kind="plugin",
            version="1.0.0",
            releases={
                "1.0.0": ReleaseInfo(
                    tag="v1.0.0",
                    url=f"file://{zip_path.resolve()}",
                    sha256="ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",  # Wrong hash!
                )
            }
        )

        installer = self.client.installer
        with self.assertRaises(ValueError) as ctx:
            installer.install(comp, "1.0.0")
        self.assertIn("Checksum verification failed", str(ctx.exception))

        dest = installer.resolve_target_dir(comp)
        self.assertFalse(dest.exists())

    def test_hub_dialog_ui_initialization(self):
        """HubDialog builds UI and filter pills cleanly."""
        dialog = HubDialog(self.client, mode="components")
        self.assertEqual(dialog.windowTitle(), "Theta Community Hub")
        self.assertIn("method", dialog.pills)
        dialog.close()

        plugins_dialog = HubDialog(self.client, mode="plugins")
        self.assertEqual(plugins_dialog.windowTitle(), "Theta Community Plugins")
        self.assertIn("plugin", plugins_dialog.pills)
        plugins_dialog.close()

    def test_hub_component_card_button_colors_and_state(self):
        """Cards use the theme install color and destructive uninstall color."""
        comp = HubComponent(
            id="test-card-comp",
            name="Card Comp",
            kind="method",
            version="1.0.0",
            is_installed=False,
        )
        card = HubComponentCard(comp, self.client)
        # Uninstalled components use the theme primary color
        self.assertEqual(card.btn_action.text(), "Install")
        self.assertIn(theme_color("primary"), card.btn_action.styleSheet())  # Install follows the theme
        self.assertEqual(card.btn_action.objectName(), "actionInstall")

        # Simulate install completion
        comp.is_installed = True
        comp.installed_version = "1.0.0"
        card.update_action_state()

        # Installed component has red Uninstall button
        self.assertEqual(card.btn_action.text(), "Uninstall")
        self.assertIn("#cc241d", card.btn_action.styleSheet())
        self.assertEqual(card.btn_action.objectName(), "actionUninstall")

        # Update available has update styling
        comp.version = "2.0.0"
        self.assertTrue(comp.has_update)
        card.update_action_state()
        self.assertIn("Update to v2.0.0", card.btn_action.text())
        self.assertIn(theme_color("secondary"), card.btn_action.styleSheet())  # Update follows the theme
        self.assertFalse(card.btn_uninstall_update.isHidden())
        self.assertEqual(card.btn_uninstall_update.text(), "Uninstall")
        self.assertIn("#cc241d", card.btn_uninstall_update.styleSheet())
        requested = []
        card._prompt_and_uninstall = lambda: requested.append(True)
        card.btn_uninstall_update.click()
        self.assertEqual(requested, [True])

        # Simulate uninstall completion
        comp.is_installed = False
        comp.installed_version = None
        card.update_action_state()
        self.assertEqual(card.btn_action.text(), "Install")
        self.assertIn(theme_color("primary"), card.btn_action.styleSheet())  # Install follows the theme
        self.assertTrue(card.btn_uninstall_update.isHidden())

    def test_suite_expands_to_individual_member_actions(self):
        suite = HubComponent(id="suite", name="Suite", kind="suite", version="1",
                             is_installed=True, installed_version="1",
                             extra={"dependencies": {"components": ["child"]}})
        child = HubComponent(id="child", name="Child", kind="model", version="1")
        self.client.components = [suite, child]
        card = HubComponentCard(suite, self.client)
        self.assertTrue(card.btn_action.isHidden())
        self.assertTrue(card.btn_uninstall_update.isHidden())
        self.assertTrue(card.members.isHidden())
        card.btn_members.click()
        self.assertFalse(card.members.isHidden())
        self.assertEqual(card.child_rows["child"][1].text(), "Install")
        requested = []
        self.client.install_async = lambda component: requested.append(component.id)
        card.child_rows["child"][1].click()
        self.assertEqual(requested, ["child"])
        child.is_installed = True
        card.update_action_state()
        self.assertEqual(card.child_rows["child"][1].text(), "Uninstall")
        suite.version = "2"
        card.update_action_state()
        self.assertEqual(card.btn_action.text(), "Update to v2")
        self.assertTrue(card.btn_uninstall_update.isHidden())

    def test_bulk_member_actions_follow_state_and_run_sequentially(self):
        from unittest.mock import patch
        from PyQt6.QtWidgets import QDialog

        suite = HubComponent(id="suite", name="Suite", kind="suite", version="1",
                             is_installed=True, installed_version="1",
                             extra={"dependencies": {"components": ["a", "b"]}})
        a = HubComponent(id="a", name="A", kind="model", version="1")
        b = HubComponent(id="b", name="B", kind="model", version="1")
        self.client.components = [suite, a, b]
        card = HubComponentCard(suite, self.client)
        self.assertFalse(card.btn_install_all.isHidden())
        self.assertTrue(card.btn_uninstall_all.isHidden())
        requested = []
        self.client.install_async = lambda child: requested.append(child.id)
        card.btn_install_all.click()
        self.assertEqual(requested, ["a"])
        a.is_installed = True
        self.client.installFinished.emit("a", True, "")
        self.assertEqual(requested, ["a", "b"])
        b.is_installed = True
        self.client.installFinished.emit("b", True, "")
        self.assertTrue(card.btn_install_all.isHidden())
        self.assertFalse(card.btn_uninstall_all.isHidden())
        self.client.uninstall_async = lambda child: requested.append(child.id)
        with patch("frontend.hub.dialog.UninstallComponentDialog.exec", return_value=QDialog.DialogCode.Accepted):
            card.btn_uninstall_all.click()
        self.assertEqual(requested[-1], "a")
        a.is_installed = False
        self.client.uninstallFinished.emit("a", True, "")
        self.assertEqual(requested[-1], "b")
        # Failure stops the queue and restores the controls.
        self.client.uninstallFinished.emit("b", False, "failed")
        self.assertTrue(card.btn_install_all.isEnabled())
        self.assertFalse(card.btn_install_all.isHidden())
        self.assertFalse(card.btn_uninstall_all.isHidden())

    def test_experiment_members_visible_only_after_install(self):
        experiment = HubComponent(id="experiment", name="Experiment", kind="experiment", version="1",
                                  extra={"dependencies": {"components": ["suite"]}})
        suite = HubComponent(id="suite", name="Suite", kind="suite", version="1",
                             extra={"dependencies": {"components": ["model"]}})
        model = HubComponent(id="model", name="Model", kind="model", version="1")
        self.client.components = [experiment, suite, model]
        card = HubComponentCard(experiment, self.client)
        self.assertTrue(card.btn_members.isHidden())
        experiment.is_installed = True
        experiment.installed_version = "1"
        card.update_action_state()
        self.assertFalse(card.btn_members.isHidden())
        self.assertEqual(card.btn_action.text(), "Uninstall")
        card.btn_members.click()
        self.assertIn("model", card.child_rows)
        self.assertEqual(card.child_rows["model"][1].text(), "Install")
        experiment.is_installed = False
        card.update_action_state()
        self.assertTrue(card.btn_members.isHidden())
        self.assertTrue(card.members.isHidden())

    def test_suite_dependencies_remain_browsable_as_independent_packages(self):
        suite = HubComponent(id="example-suite", name="Suite", kind="suite", version="1.0.0",
                             extra={"dependencies": {"components": ["example-model", "example-data"]}})
        model = HubComponent(id="example-model", name="Model", kind="model", version="1.0.0")
        data = HubComponent(id="example-data", name="Data", kind="datamodule", version="1.0.0")
        plugin = HubComponent(id="example-plugin", name="Plugin", kind="plugin", version="1.0.0")
        self.client.components = [suite, model, data, plugin]
        self.assertEqual({c.id for c in self.client.search()}, {suite.id, model.id, data.id, plugin.id})
        plugin_dialog = HubDialog(self.client, mode="plugins")
        self.assertTrue(plugin_dialog._matches_mode(plugin))
        for component in (suite, model, data):
            self.assertFalse(plugin_dialog._matches_mode(component))
        plugin_dialog.close()
        component_dialog = HubDialog(self.client, mode="components")
        for component in self.client.components:
            self.assertTrue(component_dialog._matches_mode(component))
        component_dialog.close()

    def test_suite_dependencies_have_independent_installations(self):
        model = native_component(self.temp_dir, "suite-model", configs={
            "in/config/model/suite-model.yaml": "architecture: suite-model\n",
        })
        data = native_component(self.temp_dir, "suite-data", kind="datamodule", configs={
            "in/config/data/suite-data.yaml": "data_module: suite-data\n",
        })
        suite = native_component(self.temp_dir, "example-suite", kind="suite", dependencies=[model.id, data.id])
        self.client.components = [suite, model, data]
        installer = self.client.installer
        self.assertTrue(installer.install(suite))
        for component in (suite, model, data):
            self.assertEqual(installer.check_installed(component), (True, component.version))
            self.assertTrue(installer.resolve_target_dir(component).is_dir())
        with self.assertRaisesRegex(ValueError, "required by example-suite"):
            installer.uninstall(model)
        self.assertTrue(installer.uninstall(suite))
        self.assertFalse(installer.resolve_target_dir(suite).exists())
        # Removing a suite never removes another package's implementation or profiles.
        for component in (model, data):
            self.assertTrue(installer.resolve_target_dir(component).is_dir())
            self.assertTrue(installer.has_configs(component))
            self.assertTrue(installer.uninstall(component))
            self.assertFalse(installer.resolve_target_dir(component).exists())
            self.assertFalse(installer.has_configs(component))


