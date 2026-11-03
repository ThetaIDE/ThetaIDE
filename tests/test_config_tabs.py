"""Unit tests for ConfigTabManager multi-instance tab architecture in Theta-IDE."""
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtWidgets import QApplication, QMessageBox

app = QApplication.instance() or QApplication(sys.argv[:1])

from frontend.config_tab_manager import ConfigTabManager, PreviewTabBar
from frontend.config_viewer import ConfigViewer


class TestConfigTabManager(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.exp_dir = self.root / "experiment" / "demo"
        self.exp_dir.mkdir(parents=True)

        self.f1 = self.exp_dir / "exp1.yaml"
        self.f1.write_text("experiment_id: exp1\nseed: 42\nbudget: {total_timesteps: 1000}\n", encoding="utf-8")

        self.f2 = self.exp_dir / "exp2.yaml"
        self.f2.write_text("experiment_id: exp2\nseed: 7\nbudget: {total_timesteps: 2000}\n", encoding="utf-8")

        self.f3 = self.exp_dir / "exp3.yaml"
        self.f3.write_text("experiment_id: exp3\nseed: 99\nbudget: {total_timesteps: 3000}\n", encoding="utf-8")

        self.mgr = ConfigTabManager()

    def test_open_single_click_creates_preview_tab(self):
        v = self.mgr.open_file(self.f1, "experiment/demo/exp1.yaml", pinned=False)
        self.assertEqual(self.mgr.count(), 1)
        self.assertEqual(self.mgr.tabText(0), "exp1")
        self.assertEqual(self.mgr.tabBar().tabData(0), "preview")
        self.assertEqual(self.mgr.current_viewer(), v)

    def test_external_atomic_save_refreshes_open_form(self):
        from PyQt6.QtTest import QTest
        v = self.mgr.open_file(self.f1, "experiment/demo/exp1.yaml", pinned=True)
        replacement = self.f1.with_suffix(".tmp")
        replacement.write_text("experiment_id: exp1\nseed: 123\n", encoding="utf-8")
        replacement.replace(self.f1)
        QTest.qWait(1200)
        self.assertEqual(v.raw_data["seed"], 123)

    def test_external_save_does_not_overwrite_dirty_form(self):
        from PyQt6.QtTest import QTest
        v = self.mgr.open_file(self.f1, "experiment/demo/exp1.yaml", pinned=True)
        v.is_dirty = True
        v.raw_data["seed"] = 99
        self.f1.write_text("experiment_id: exp1\nseed: 123\n", encoding="utf-8")
        QTest.qWait(1200)
        self.assertEqual(v.raw_data["seed"], 99)
        self.assertTrue(v.is_dirty)

    def test_clean_preview_tab_is_reused_on_next_single_click(self):
        v1 = self.mgr.open_file(self.f1, "experiment/demo/exp1.yaml", pinned=False)
        self.assertEqual(self.mgr.count(), 1)
        self.assertEqual(self.mgr.tabText(0), "exp1")

        # Single-click opening another file reuses preview tab
        v2 = self.mgr.open_file(self.f2, "experiment/demo/exp2.yaml", pinned=False)
        self.assertEqual(self.mgr.count(), 1)
        self.assertEqual(self.mgr.tabText(0), "exp2")
        self.assertEqual(self.mgr.tabBar().tabData(0), "preview")
        self.assertEqual(v1, v2)  # Reused same ConfigViewer instance

    def test_double_click_pins_tab_and_prevents_preview_reuse(self):
        v1 = self.mgr.open_file(self.f1, "experiment/demo/exp1.yaml", pinned=True)
        self.assertEqual(self.mgr.count(), 1)
        self.assertEqual(self.mgr.tabText(0), "exp1")
        self.assertEqual(self.mgr.tabBar().tabData(0), "pinned")

        # Next single-click cannot reuse pinned tab, must open new preview tab
        v2 = self.mgr.open_file(self.f2, "experiment/demo/exp2.yaml", pinned=False)
        self.assertEqual(self.mgr.count(), 2)
        self.assertEqual(self.mgr.tabText(0), "exp1")
        self.assertEqual(self.mgr.tabText(1), "exp2")
        self.assertEqual(self.mgr.tabBar().tabData(0), "pinned")
        self.assertEqual(self.mgr.tabBar().tabData(1), "preview")
        self.assertNotEqual(v1, v2)

    def test_explicit_new_tab_duplicates_an_open_config_and_tracks_closing(self):
        rel = "experiment/demo/exp1.yaml"
        original = self.mgr.open_file(self.f1, rel, pinned=True)
        duplicate = self.mgr.open_file(self.f1, rel, pinned=True, new_tab=True)
        self.assertIsNot(original, duplicate)
        self.assertEqual(self.mgr.count(), 2)
        self.assertIs(self.mgr.current_viewer(), duplicate)
        self.assertTrue(self.mgr.close_tab_by_index(0))
        self.assertIs(self.mgr._tabs[rel], duplicate)
        self.assertIs(self.mgr.open_file(self.f1, rel), duplicate)
        self.assertTrue(self.mgr.close_tab_by_index(0))
        self.assertNotIn(rel, self.mgr._tabs)

    def test_reusing_preview_preserves_tracking_of_duplicate(self):
        rel = "experiment/demo/exp1.yaml"
        self.mgr.open_file(self.f1, rel)
        duplicate = self.mgr.open_file(self.f1, rel, new_tab=True)
        self.mgr.open_file(self.f2, "experiment/demo/exp2.yaml")
        self.assertIs(self.mgr._tabs[rel], duplicate)
        self.assertIs(self.mgr.open_file(self.f1, rel), duplicate)

    def test_new_tab_does_not_overwrite_dirty_open_config(self):
        rel = "experiment/demo/exp1.yaml"
        original = self.mgr.open_file(self.f1, rel, pinned=True)
        original.is_dirty = True
        duplicate = self.mgr.open_file(self.f1, rel, pinned=True, new_tab=True)
        self.assertTrue(original.is_dirty)
        self.assertIsNot(original, duplicate)

    def test_pin_tab_method_converts_preview_to_pinned(self):
        self.mgr.open_file(self.f1, "experiment/demo/exp1.yaml", pinned=False)
        self.assertEqual(self.mgr.tabBar().tabData(0), "preview")

        self.mgr.pin_tab("experiment/demo/exp1.yaml")
        self.assertEqual(self.mgr.tabBar().tabData(0), "pinned")

    def test_tab_double_click_pins_tab(self):
        self.mgr.open_file(self.f1, "experiment/demo/exp1.yaml", pinned=False)
        self.assertEqual(self.mgr.tabBar().tabData(0), "preview")

        self.mgr._on_tab_double_clicked(0)
        self.assertEqual(self.mgr.tabBar().tabData(0), "pinned")

    def test_dirty_tab_shows_bullet_and_auto_pins(self):
        v = self.mgr.open_file(self.f1, "experiment/demo/exp1.yaml", pinned=False)
        self.assertEqual(self.mgr.tabBar().tabData(0), "preview")
        self.assertEqual(self.mgr.tabText(0), "exp1")

        # Simulate edit marking dirty
        v.is_dirty = True
        v.dirty_changed.emit(True)

        self.assertEqual(self.mgr.tabText(0), "exp1 ●")
        self.assertEqual(self.mgr.tabBar().tabData(0), "pinned")

        # Open another file via single-click: cannot overwrite dirty tab
        self.mgr.open_file(self.f2, "experiment/demo/exp2.yaml", pinned=False)
        self.assertEqual(self.mgr.count(), 2)
        self.assertEqual(self.mgr.tabText(0), "exp1 ●")
        self.assertEqual(self.mgr.tabText(1), "exp2")

    def test_saving_clears_dirty_bullet(self):
        v = self.mgr.open_file(self.f1, "experiment/demo/exp1.yaml", pinned=True)
        v.is_dirty = True
        v.dirty_changed.emit(True)
        self.assertIn("●", self.mgr.tabText(0))

        v.is_dirty = False
        v.dirty_changed.emit(False)
        self.assertNotIn("●", self.mgr.tabText(0))

    def test_tab_switching_signals(self):
        events = []
        self.mgr.current_tab_changed.connect(lambda rel, v: events.append((rel, v)))

        self.mgr.open_file(self.f1, "experiment/demo/exp1.yaml", pinned=True)
        self.mgr.open_file(self.f2, "experiment/demo/exp2.yaml", pinned=True)

        self.mgr.setCurrentIndex(0)
        self.assertEqual(self.mgr.currentIndex(), 0)
        self.assertEqual(self.mgr.current_viewer().current_rel_path, "experiment/demo/exp1.yaml")

    def test_next_and_prev_tab_cycling(self):
        self.mgr.open_file(self.f1, "experiment/demo/exp1.yaml", pinned=True)
        self.mgr.open_file(self.f2, "experiment/demo/exp2.yaml", pinned=True)
        self.mgr.open_file(self.f3, "experiment/demo/exp3.yaml", pinned=True)

        self.mgr.setCurrentIndex(0)
        self.mgr.next_tab()
        self.assertEqual(self.mgr.currentIndex(), 1)
        self.mgr.next_tab()
        self.assertEqual(self.mgr.currentIndex(), 2)
        self.mgr.next_tab()
        self.assertEqual(self.mgr.currentIndex(), 0)

        self.mgr.prev_tab()
        self.assertEqual(self.mgr.currentIndex(), 2)

    def test_close_clean_tab(self):
        self.mgr.open_file(self.f1, "experiment/demo/exp1.yaml", pinned=True)
        self.mgr.open_file(self.f2, "experiment/demo/exp2.yaml", pinned=True)
        self.assertEqual(self.mgr.count(), 2)

        closed = self.mgr.close_current_tab()
        self.assertTrue(closed)
        self.assertEqual(self.mgr.count(), 1)
        self.assertEqual(self.mgr.tabText(0), "exp1")

    def test_close_signal_keeps_remaining_tab_permanent(self):
        self.mgr.open_file(self.f1, "experiment/demo/exp1.yaml", pinned=True)
        self.mgr.open_file(self.f2, "experiment/demo/exp2.yaml", pinned=True)
        self.mgr.tabCloseRequested.emit(1)
        self.assertEqual(self.mgr.count(), 1)
        self.assertEqual(self.mgr.tabBar().tabData(0), "pinned")
        self.mgr.open_file(self.f3, "experiment/demo/exp3.yaml")
        self.assertEqual(self.mgr.count(), 2)

    def test_close_tabs_matching(self):
        self.mgr.open_file(self.f1, "experiment/demo/exp1.yaml", pinned=True)
        self.mgr.open_file(self.f2, "experiment/demo/exp2.yaml", pinned=True)
        self.mgr.open_file(self.f3, "experiment/demo/exp3.yaml", pinned=True)
        self.assertEqual(self.mgr.count(), 3)

        self.mgr.close_tabs_matching(lambda p: "exp2" in p)
        self.assertEqual(self.mgr.count(), 2)
        self.assertEqual([self.mgr.tabText(i) for i in range(self.mgr.count())], ["exp1", "exp3"])

    def test_file_rename_updates_tab_mapping_and_label(self):
        v = self.mgr.open_file(self.f1, "experiment/demo/exp1.yaml", pinned=True)
        self.assertEqual(self.mgr.tabText(0), "exp1")

        renamed_signals = []
        self.mgr.file_renamed.connect(lambda o, n: renamed_signals.append((o, n)))

        v.file_renamed.emit("experiment/demo/exp1.yaml", "experiment/demo/exp_renamed.yaml")
        self.assertEqual(self.mgr.tabText(0), "exp_renamed")
        self.assertEqual(renamed_signals, [("experiment/demo/exp1.yaml", "experiment/demo/exp_renamed.yaml")])

    def test_single_tab_hides_tab_bar_and_multiple_tabs_shows_tab_bar(self):
        # 0 tabs: hidden
        self.assertTrue(self.mgr.tabBar().isHidden())

        # 1 tab: hidden
        self.mgr.open_file(self.f1, "experiment/demo/exp1.yaml", pinned=True)
        self.assertEqual(self.mgr.count(), 1)
        self.assertTrue(self.mgr.tabBar().isHidden())

        # 2 tabs: visible
        self.mgr.open_file(self.f2, "experiment/demo/exp2.yaml", pinned=True)
        self.assertEqual(self.mgr.count(), 2)
        self.assertFalse(self.mgr.tabBar().isHidden())

        # Close back to 1 tab: hidden
        self.mgr.close_current_tab()
        self.assertEqual(self.mgr.count(), 1)
        self.assertTrue(self.mgr.tabBar().isHidden())

    def test_tab_bar_height_alignment(self):
        self.assertEqual(self.mgr.tabBar().sizeHint().height(), 30)


if __name__ == "__main__":
    unittest.main()
