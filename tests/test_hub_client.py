"""Unit and integration tests for Theta HubClient, HubInstaller, and models."""
import hashlib
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

os.environ["QT_QPA_PLATFORM"] = "offscreen"

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    from frontend.hub import AuthorInfo, HubClient, HubComponent, HubComponentCard, HubDialog, HubInstaller, ReleaseInfo
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

    def test_installer_resolves_an_internal_dependency_to_its_provider(self):
        suite = HubComponent.from_dict({
            "id": "conformance-suite", "name": "Conformance Suite", "kind": "suite", "version": "1.0.0",
        })
        internal = HubComponent.from_dict({
            "id": "conformance-probe", "name": "Conformance Probe", "kind": "model", "version": "1.0.0",
            "suite": "conformance-suite", "hidden": True,
        })
        self.client.components = [suite, internal]

        assert self.client.installer.lookup_component("conformance-probe") is suite

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
        installer = self.client.installer

        plugin_comp = HubComponent(id="my_plugin", name="P", kind="plugin", version="1.0")
        method_comp = HubComponent(id="my_method", name="M", kind="method", version="1.0")
        model_comp = HubComponent(id="my_model", name="Mod", kind="model", version="1.0")
        env_comp = HubComponent(id="my_env", name="E", kind="env", version="1.0")

        self.assertEqual(installer.resolve_target_dir(plugin_comp), self.data_dir / "plugins" / "my_plugin")
        self.assertEqual(installer.resolve_target_dir(method_comp), self.workspace_dir / "src" / "usr" / "algorithms" / "my_method")
        self.assertEqual(installer.resolve_target_dir(model_comp), self.workspace_dir / "src" / "usr" / "models" / "my_model")
        self.assertEqual(installer.resolve_target_dir(env_comp), self.workspace_dir / "in" / "envs" / "my_env")

    def test_download_sha256_verify_and_install(self):
        # 1. Create a dummy package zip
        pkg_src = self.temp_dir / "pkg_src"
        pkg_src.mkdir()
        (pkg_src / "main.py").write_text("# Test component payload", encoding="utf-8")
        (pkg_src / "config.yaml").write_text("learning_rate: 0.001", encoding="utf-8")

        zip_path = self.temp_dir / "sample_package.zip"
        with ZipFile(zip_path, "w") as zf:
            zf.write(pkg_src / "main.py", arcname="main.py")
            zf.write(pkg_src / "config.yaml", arcname="config.yaml")

        # Compute valid SHA-256
        valid_sha = hashlib.sha256(zip_path.read_bytes()).hexdigest()

        comp = HubComponent(
            id="sample-rl-agent",
            name="Sample RL Agent",
            kind="method",
            version="1.0.0",
            releases={
                "1.0.0": ReleaseInfo(
                    tag="v1.0.0",
                    url=f"file://{zip_path.resolve()}",
                    sha256=valid_sha,
                )
            }
        )

        installer = self.client.installer
        is_installed, ver = installer.check_installed(comp)
        self.assertFalse(is_installed)

        # 2. Test successful install
        success = installer.install(comp, "1.0.0")
        self.assertTrue(success)

        # Verify files on disk
        dest = self.workspace_dir / "src" / "usr" / "algorithms" / "sample-rl-agent"
        deployed_cfg = self.workspace_dir / "in" / "config" / "agent" / "sample-rl-agent.yaml"
        self.assertTrue((dest / "main.py").exists())
        self.assertTrue((dest / "config.yaml").exists())
        self.assertTrue((dest / ".theta_component.json").exists())
        self.assertTrue(deployed_cfg.exists())
        self.assertEqual((dest / "main.py").stat().st_mode & 0o222, 0)
        self.assertEqual(deployed_cfg.stat().st_mode & 0o222, 0)
        self.assertEqual((dest / ".theta_component.json").stat().st_mode & 0o222, 0)

        # Updates can replace protected files and leave the result protected.
        self.assertTrue(installer.install(comp, "1.0.0"))
        self.assertEqual((dest / "main.py").stat().st_mode & 0o222, 0)
        self.assertEqual(deployed_cfg.stat().st_mode & 0o222, 0)

        is_installed, ver = installer.check_installed(comp)
        self.assertTrue(is_installed)
        self.assertEqual(ver, "1.0.0")

        # 3. Test uninstall
        uninstalled = installer.uninstall(comp)
        self.assertTrue(uninstalled)
        self.assertFalse(dest.exists())
        self.assertFalse(deployed_cfg.exists())
        is_installed, ver = installer.check_installed(comp)
        self.assertFalse(is_installed)

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

        dest = self.data_dir / "plugins" / "corrupted-comp"
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
        """HubComponentCard displays green Install and red Uninstall buttons and transitions cleanly."""
        comp = HubComponent(
            id="test-card-comp",
            name="Card Comp",
            kind="method",
            version="1.0.0",
            is_installed=False,
        )
        card = HubComponentCard(comp, self.client)
        # Uninstalled component has green Install button
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

    def test_suite_filtering_and_dialog_display(self):
        """Package metadata hides supplied components and exposes the installable suite."""
        thetalib = HubComponent.from_dict({
            "id": "thetalib",
            "name": "ThetaLib Core Foundation Suite",
            "kind": "suite",
            "version": "1.0.0",
            "description": "Core foundation",
            "provides": ["standard-figures", "image-data", "replay-buffer", "tabular-data"],
        })
        std_fig = HubComponent.from_dict({
            "id": "standard-figures",
            "name": "Standard Figures",
            "kind": "plotter",
            "version": "1.0.0",
            "suite": "thetalib",
            "hidden": True,
        })
        img_data = HubComponent.from_dict({
            "id": "image-data",
            "name": "Image Data",
            "kind": "datamodule",
            "version": "1.0.0",
            "suite": "thetalib",
            "hidden": True,
        })
        plugin_comp = HubComponent.from_dict({
            "id": "obsidian-notes",
            "name": "Obsidian Notes",
            "kind": "plugin",
            "version": "0.1.0",
        })

        self.client.components = [thetalib, std_fig, img_data, plugin_comp]

        # 1. Search with include_internal=False (default)
        results = self.client.search()
        result_ids = [c.id for c in results]
        self.assertIn("thetalib", result_ids)
        self.assertIn("obsidian-notes", result_ids)
        self.assertNotIn("standard-figures", result_ids)
        self.assertNotIn("image-data", result_ids)

        # 2. Search with include_internal=True
        all_results = self.client.search(include_internal=True)
        all_ids = [c.id for c in all_results]
        self.assertIn("standard-figures", all_ids)
        self.assertIn("image-data", all_ids)

        # 3. HubDialog filtering for plugins mode
        plugin_dialog = HubDialog(self.client, mode="plugins")
        self.assertFalse(plugin_dialog._matches_mode(thetalib))  # suites are installable components, not plugins
        self.assertTrue(plugin_dialog._matches_mode(plugin_comp))
        self.assertFalse(plugin_dialog._matches_mode(std_fig))
        self.assertFalse(plugin_dialog._matches_mode(img_data))
        plugin_dialog.close()

        # 4. HubDialog filtering for components mode
        comp_dialog = HubDialog(self.client, mode="components")
        self.assertTrue(comp_dialog._matches_mode(thetalib))
        self.assertTrue(comp_dialog._matches_mode(plugin_comp))
        self.assertFalse(comp_dialog._matches_mode(std_fig))
        self.assertFalse(comp_dialog._matches_mode(img_data))
        comp_dialog.close()

    def test_suite_installation_and_clean_uninstall(self):
        """Verify a suite installs root-relative files and supplied-component status is metadata-driven."""
        zip_path = self.temp_dir / "thetalib-1.0.0.zip"
        with ZipFile(zip_path, "w") as zf:
            zf.writestr(".theta_component.json", json.dumps({
                "id": "thetalib",
                "name": "ThetaLib Core Foundation Suite",
                "kind": "suite",
                "version": "1.0.0",
                "provides": ["standard-figures", "image-data", "replay-buffer", "tabular-data"],
                "installed_dirs": [
                    "plot/standard_figures",
                    "src/usr/data/image_data",
                ],
                "config_paths": [
                    "in/config/plot/thetalib/convergence.yaml",
                    "in/config/data/thetalib/image_data.yaml",
                ]
            }))
            zf.writestr("plot/standard_figures/__init__.py", "")
            zf.writestr("plot/standard_figures/convergence.py", "# Convergence plotter")
            zf.writestr("src/usr/data/image_data/__init__.py", "")
            zf.writestr("src/usr/data/image_data/image_data.py", "# Image data module")
            zf.writestr("in/config/plot/thetalib/convergence.yaml", "title: Convergence\n")
            zf.writestr("in/config/data/thetalib/image_data.yaml", "batch_size: 64\n")

        sha256_hash = hashlib.sha256(zip_path.read_bytes()).hexdigest()

        thetalib_comp = HubComponent(
            id="thetalib",
            name="ThetaLib Core Foundation Suite",
            kind="suite",
            version="1.0.0",
            extra={
                "provides": ["standard-figures", "image-data", "replay-buffer", "tabular-data"],
                "installed_dirs": ["plot/standard_figures", "src/usr/data/image_data"],
                "config_paths": ["in/config/plot/thetalib/convergence.yaml", "in/config/data/thetalib/image_data.yaml"],
            },
            releases={
                "1.0.0": ReleaseInfo(
                    tag="v1.0.0",
                    url=f"file://{zip_path.resolve()}",
                    sha256=sha256_hash,
                )
            }
        )

        installer = self.client.installer
        is_installed, _ = installer.check_installed(thetalib_comp)
        self.assertFalse(is_installed)

        # Install thetalib
        success = installer.install(thetalib_comp, "1.0.0")
        self.assertTrue(success)

        # Verify files deployed
        conv_py = self.workspace_dir / "plot" / "standard_figures" / "convergence.py"
        img_py = self.workspace_dir / "src" / "usr" / "data" / "image_data" / "image_data.py"
        conv_cfg = self.workspace_dir / "in" / "config" / "plot" / "thetalib" / "convergence.yaml"
        img_cfg = self.workspace_dir / "in" / "config" / "data" / "thetalib" / "image_data.yaml"

        self.assertTrue(conv_py.exists())
        self.assertTrue(img_py.exists())
        self.assertTrue(conv_cfg.exists())
        self.assertTrue(img_cfg.exists())

        # Check installed status
        is_installed, ver = installer.check_installed(thetalib_comp)
        self.assertTrue(is_installed)
        self.assertEqual(ver, "1.0.0")

        # Constituent component lookup and check_installed
        self.client.components = [thetalib_comp]
        std_fig_comp = HubComponent.from_dict({
            "id": "standard-figures", "name": "Standard Figures", "kind": "plotter",
            "version": "1.0.0", "suite": "thetalib", "hidden": True,
        })
        sub_installed, sub_ver = installer.check_installed(std_fig_comp)
        self.assertTrue(sub_installed)
        self.assertEqual(sub_ver, "1.0.0")

        # Uninstall thetalib
        uninstalled = installer.uninstall(thetalib_comp)
        self.assertTrue(uninstalled)

        # Verify all files cleaned up
        self.assertFalse(conv_py.exists())
        self.assertFalse(img_py.exists())
        self.assertFalse(conv_cfg.exists())
        self.assertFalse(img_cfg.exists())

        # Verify empty dirs pruned
        self.assertFalse((self.workspace_dir / "plot" / "standard_figures").exists())
        self.assertFalse((self.workspace_dir / "src" / "usr" / "data" / "image_data").exists())
        self.assertFalse((self.workspace_dir / "in" / "config" / "plot" / "thetalib").exists())
        self.assertFalse((self.workspace_dir / "in" / "config" / "data" / "thetalib").exists())

        is_installed, _ = installer.check_installed(thetalib_comp)
        self.assertFalse(is_installed)


