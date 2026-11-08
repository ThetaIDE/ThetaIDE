"""Artifact browser: every file a run produced, with a preview for each kind.

One ArtifactBrowser backs the Reports, Config and Files tabs; each tab only
chooses which kinds it lists. Previews reuse the Plot viewer's canvases (frontend/plots.py).
"""

from __future__ import annotations

import json
import time
from pathlib import PurePosixPath

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTextBrowser,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..plots import ImageCanvas, PlotCanvas, read_metrics_csv_text
from ..tree_tools import folder_icon
from ..widgets import SortableItem, YamlHighlighter, label
from .data import Manifest, decode_text, read_csv_rows, to_float

PREVIEW_LIMIT = 25 * 1024 * 1024  # larger files are listed with Save a copy, not previewed
TABLE_ROW_LIMIT = 5000

_ARTIFACT = Qt.ItemDataRole.UserRole
_FOLDER = Qt.ItemDataRole.UserRole + 1


def human_size(size: int | None) -> str:
    if size is None:
        return "—"
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def run_relative(artifact: dict, run_key: str) -> str:
    """'plots/group/experiment/category/file.png' -> the path inside this run's folders."""
    parts = PurePosixPath(artifact["path"]).parts
    group, experiment_id = run_key.split("/", 1)
    if len(parts) > 3 and parts[1:3] == (group, experiment_id):
        return "/".join((parts[0], *parts[3:]))
    return artifact["path"]


# ── Viewers ──────────────────────────────────────────────────────────────────


class TableViewer(QWidget):
    """Every row and column of a CSV, sortable, plus a plot of any two numeric columns."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Orientation.Vertical)
        self.table = QTableWidget(0, 0)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        splitter.addWidget(self.table)
        plot = QWidget()
        plot_layout = QVBoxLayout(plot)
        plot_layout.setContentsMargins(0, 6, 0, 0)
        axes = QHBoxLayout()
        axes.addWidget(label("Plot", "muted"))
        self.y_select, self.x_select = QComboBox(), QComboBox()
        axes.addWidget(self.y_select)
        axes.addWidget(label("against", "muted"))
        axes.addWidget(self.x_select)
        axes.addStretch()
        plot_layout.addLayout(axes)
        self.canvas = PlotCanvas()
        plot_layout.addWidget(self.canvas, 1)
        splitter.addWidget(plot)
        splitter.setSizes([400, 260])
        layout.addWidget(splitter)
        self.plot = plot
        self.columns: dict[str, list] = {}
        self.x_select.currentIndexChanged.connect(self.replot)
        self.y_select.currentIndexChanged.connect(self.replot)

    def show_data(self, name: str, data: bytes) -> str:
        text = decode_text(data)
        preview = read_csv_rows(text, TABLE_ROW_LIMIT)
        self.table.setSortingEnabled(False)
        self.table.clear()
        self.table.setColumnCount(len(preview.headers))
        self.table.setRowCount(len(preview.rows))
        self.table.setHorizontalHeaderLabels(preview.headers)
        for r, row in enumerate(preview.rows):
            for c, cell in enumerate(row[: len(preview.headers)]):
                number = to_float(cell)
                self.table.setItem(r, c, SortableItem(cell, number if number is not None else (cell or None)))
        self.table.setSortingEnabled(True)
        try:
            self.columns = read_metrics_csv_text(text)
        except (ValueError, KeyError):
            self.columns = {}
        names = list(self.columns)
        for combo, default in ((self.x_select, 0), (self.y_select, 1)):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(names)
            if names:
                combo.setCurrentIndex(min(default, len(names) - 1))
            combo.blockSignals(False)
        self.plot.setVisible(len(names) >= 2)
        self.replot()
        shown = f"first {len(preview.rows):,} of " if preview.truncated else ""
        return f"{shown}{preview.total_rows:,} rows × {len(preview.headers)} columns"

    def replot(self, *_):
        x, y = self.x_select.currentText(), self.y_select.currentText()
        if x not in self.columns or y not in self.columns:
            self.canvas.set_data([], x, y)
            return
        pairs = sorted((a, b) for a, b in zip(self.columns[x], self.columns[y]) if a is not None and b is not None)
        self.canvas.set_data([(y, pairs, "#b8bb26")], x, y)


class FigureViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = ImageCanvas()
        layout.addWidget(self.canvas)

    def show_data(self, name: str, data: bytes) -> str:
        pixmap = QPixmap()
        if not pixmap.loadFromData(data):
            raise ValueError("This image format cannot be previewed here. Use Save a copy to open it.")
        self.canvas.load(pixmap)
        return f"{pixmap.width()} × {pixmap.height()} px · scroll to zoom, drag to pan, double-click to fit"


class TextViewer(QWidget):
    """Markdown rendered; YAML highlighted; JSON pretty-printed; anything else as plain text."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(False)
        layout.addWidget(self.browser)
        self.highlighter = YamlHighlighter(None)

    def show_data(self, name: str, data: bytes) -> str:
        text = decode_text(data)
        suffix = PurePosixPath(name).suffix.lower()
        self.highlighter.setDocument(None)
        if suffix == ".md":
            self.browser.setMarkdown(text)
        elif suffix == ".json":
            try:
                text = json.dumps(json.loads(text), indent=2)
            except ValueError:
                pass
            self.browser.setPlainText(text)
        else:
            self.browser.setPlainText(text)
            if suffix in (".yaml", ".yml"):
                self.highlighter.setDocument(self.browser.document())
        return f"{text.count(chr(10)) + 1:,} lines"


class InfoViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.text = label("", "muted")
        self.text.setWordWrap(True)
        self.text.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.text)
        layout.addStretch()

    def show_info(self, artifact: dict | None, reason: str = ""):
        if artifact is None:
            self.text.setText(reason or "Select a file on the left.")
            return
        modified = time.strftime("%Y-%m-%d %H:%M", time.localtime(artifact.get("modified") or 0))
        details = [
            f"results/{artifact['path']}",
            f"Kind: {artifact.get('kind')}",
            f"Size: {human_size(artifact.get('size'))}",
            f"Modified: {modified}",
        ]
        if artifact.get("agent"):
            details.append(
                f"Agent: {artifact['agent']}" + (f" · {artifact['version']}" if artifact.get("version") else "")
            )
        self.text.setText("\n".join(details + ([""] + [reason] if reason else [])))


VIEWER_FOR_KIND = {
    "table": "table",
    "metrics": "table",
    "figure": "figure",
    "report": "text",
    "config": "text",
    "metadata": "text",
    "log": "text",
}


# ── Browser ──────────────────────────────────────────────────────────────────


class ArtifactBrowser(QWidget):
    """Files of the ticked runs (optionally only some kinds) on the left, a preview on the right."""

    def __init__(self, loader, kinds: tuple[str, ...] | None = None, empty_text: str = "", parent=None):
        super().__init__(parent)
        self.loader = loader
        self.kinds = kinds
        self.empty_text = empty_text or "The ticked runs have no files of this kind."
        self.current: tuple[str, dict] | None = None
        self._save_to: dict[tuple[str, str], str] = {}

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(8)
        splitter.setChildrenCollapsible(False)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(1)
        self.tree.setHeaderHidden(True)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tree.setMinimumWidth(160)
        self.tree.currentItemChanged.connect(self._current_changed)
        splitter.addWidget(self.tree)

        right = QWidget()
        # File paths and hidden preview widgets must not raise the splitter's
        # minimum width as files are selected. The tree owns its own width.
        right.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        right.setMinimumWidth(320)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 0, 0, 0)
        header = QHBoxLayout()
        self.title = label("", "heading")
        self.title.setWordWrap(True)
        header.addWidget(self.title, 1)
        self.save_button = QPushButton("Save a copy…")
        self.save_button.setToolTip("Download this file from the backend")
        self.save_button.clicked.connect(self._ask_save)
        self.save_button.setEnabled(False)
        header.addWidget(self.save_button)
        right_layout.addLayout(header)
        self.detail = label("", "muted")
        self.detail.setWordWrap(True)
        right_layout.addWidget(self.detail)
        self.stack = QStackedWidget()
        self.viewers = {"table": TableViewer(), "figure": FigureViewer(), "text": TextViewer()}
        self.info = InfoViewer()
        for viewer in (*self.viewers.values(), self.info):
            self.stack.addWidget(viewer)
        right_layout.addWidget(self.stack, 1)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([480, 700])
        layout.addWidget(splitter)
        loader.file_loaded.connect(self._file_loaded)
        self.set_manifests([])

    def set_manifests(self, manifests: list[Manifest]):
        """List the matching files of these runs, keeping the current file selected if it is still there."""
        keep = (self.current[0], self.current[1]["path"]) if self.current else None
        had_items = self.tree.topLevelItemCount() > 0
        expanded = {item.data(0, _FOLDER) for item in self._items() if item.isExpanded() and item.data(0, _FOLDER)}
        self.tree.blockSignals(True)
        self.tree.clear()
        restore = None
        for manifest in manifests:
            artifacts = [a for a in manifest.artifacts if self.kinds is None or a.get("kind") in self.kinds]
            if not artifacts:
                continue
            run_item = QTreeWidgetItem(self.tree, [manifest.run_key])
            run_item.setData(0, _FOLDER, (manifest.run_key, ""))
            run_item.setIcon(0, folder_icon())
            run_item.setFirstColumnSpanned(False)
            folders = {"": run_item}
            for artifact in sorted(artifacts, key=lambda a: run_relative(a, manifest.run_key)):
                parts = PurePosixPath(run_relative(artifact, manifest.run_key)).parts
                parent = run_item
                for i, part in enumerate(parts[:-1]):
                    path = "/".join(parts[:i + 1])
                    if path not in folders:
                        folder = QTreeWidgetItem(parent, [part])
                        folder.setData(0, _FOLDER, (manifest.run_key, path))
                        folder.setIcon(0, folder_icon())
                        folders[path] = folder
                    parent = folders[path]
                item = QTreeWidgetItem(
                    parent,
                    [parts[-1]],
                )
                item.setData(0, _ARTIFACT, (manifest.run_key, artifact))
                item.setToolTip(0, f"results/{artifact['path']}")
                if keep == (manifest.run_key, artifact["path"]):
                    restore = item
            run_item.setExpanded(not had_items or (manifest.run_key, "") in expanded)
            for path, folder in folders.items():
                if path and (manifest.run_key, path) in expanded:
                    folder.setExpanded(True)
        self.tree.blockSignals(False)
        if restore is not None:
            self.tree.setCurrentItem(restore)
        else:
            self.current = None
            self.title.setText("")
            self.detail.setText("")
            self.save_button.setEnabled(False)
            self.info.show_info(None, self.empty_text if self.tree.topLevelItemCount() == 0 else "")
            self.stack.setCurrentWidget(self.info)

    def artifacts(self) -> list[tuple[str, dict]]:
        return [data for item in self._items() if (data := item.data(0, _ARTIFACT))]

    def _items(self):
        def walk(item):
            yield item
            for i in range(item.childCount()):
                yield from walk(item.child(i))
        for i in range(self.tree.topLevelItemCount()):
            yield from walk(self.tree.topLevelItem(i))

    def select(self, run_key: str, path: str) -> bool:
        for item in self._items():
            data = item.data(0, _ARTIFACT)
            if data and data[0] == run_key and data[1]["path"] == path:
                parent = item.parent()
                while parent:
                    parent.setExpanded(True)
                    parent = parent.parent()
                self.tree.setCurrentItem(item)
                return True
        return False

    def _current_changed(self, item, _previous):
        data = item.data(0, _ARTIFACT) if item is not None else None
        if not data:
            self.current = None
            self.title.setText("")
            self.detail.setText("")
            self.save_button.setEnabled(False)
            self.info.show_info(None)
            self.stack.setCurrentWidget(self.info)
            return
        run_key, artifact = data
        self.current = data
        self.title.setText(run_relative(artifact, run_key))
        self.save_button.setEnabled(True)
        viewer = VIEWER_FOR_KIND.get(artifact.get("kind"))
        size = artifact.get("size") or 0
        if viewer is None or size > PREVIEW_LIMIT:
            reason = (
                "Too large to preview here." if viewer is not None else "No preview for this kind of file."
            ) + " Use Save a copy to open it."
            self.detail.setText(f"{artifact.get('kind')} · {human_size(size)}")
            self.info.show_info(artifact, reason)
            self.stack.setCurrentWidget(self.info)
            return
        self.detail.setText("Loading…")
        self.loader.load_file(run_key, artifact["path"])

    def _file_loaded(self, run_key, path):
        if (run_key, path) in self._save_to:
            self._write(run_key, path, self._save_to.pop((run_key, path)))
        if not self.current or (self.current[0], self.current[1]["path"]) != (run_key, path):
            return
        artifact = self.current[1]
        viewer = self.viewers.get(VIEWER_FOR_KIND.get(artifact.get("kind")))
        if viewer is None:
            return
        try:
            summary = viewer.show_data(path, self.loader.files[(run_key, path)])
        except ValueError as exc:
            self.info.show_info(artifact, str(exc))
            self.stack.setCurrentWidget(self.info)
            self.detail.setText(f"{artifact.get('kind')} · {human_size(artifact.get('size'))}")
            return
        self.stack.setCurrentWidget(viewer)
        self.detail.setText(f"{artifact.get('kind')} · {human_size(artifact.get('size'))} · {summary}")

    def _ask_save(self):
        if not self.current:
            return
        run_key, artifact = self.current
        target, _ = QFileDialog.getSaveFileName(self, "Save a copy", PurePosixPath(artifact["path"]).name)
        if target:
            self.save_copy(target)

    def save_copy(self, target: str):
        """Write the current file to `target`, downloading it first if needed."""
        run_key, artifact = self.current
        key = (run_key, artifact["path"])
        if key in self.loader.files:
            self._write(run_key, artifact["path"], target)
        else:
            self._save_to[key] = target
            self.loader.load_file(run_key, artifact["path"])

    def _write(self, run_key, path, target):
        try:
            with open(target, "wb") as f:
                f.write(self.loader.files[(run_key, path)])
            self.detail.setText(f"Saved a copy to {target}")
        except OSError as exc:
            self.detail.setText(f"Could not save: {exc}")
