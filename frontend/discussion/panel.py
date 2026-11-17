"""Per-experiment figures, shared parameters and autosaved discussion text."""

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (QApplication, QLabel, QPlainTextEdit, QScrollArea, QSplitter,
                             QStackedWidget, QVBoxLayout, QWidget)

from . import model
from .params import ParametersBox
from .tree import DiscussionTree
from .view import FigureBuilder, FigureView


def _empty():
    label = QLabel("Select a figure")
    label.setProperty("muted", True)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return label


class DiscussionPanel(QWidget):
    def __init__(self, paths, settings_manager=None, parent=None):
        super().__init__(parent)
        self.paths = paths
        self.doc = None
        self.parameters = None
        self._dirty = False
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(500)
        self._save_timer.timeout.connect(self.flush_pending_save)

        self.tree = DiscussionTree(paths, settings_manager=settings_manager)
        self.parameter_stack = QStackedWidget()
        self.parameter_stack.addWidget(_empty())
        self.parameter_scroll = QScrollArea()
        self.parameter_scroll.setWidgetResizable(True)
        self.parameter_stack.addWidget(self.parameter_scroll)
        self.figure_stack = QStackedWidget()
        self.figure_stack.addWidget(_empty())
        self.figure_view = FigureView()
        self.figure_stack.addWidget(self.figure_view)

        self.discussion = QPlainTextEdit()
        self.discussion.setAccessibleName("Discussion")
        self.discussion.setPlaceholderText("Write about this figure…")
        self.discussion.setEnabled(False)
        discussion_box = QWidget()
        discussion_layout = QVBoxLayout(discussion_box)
        discussion_layout.setContentsMargins(0, 0, 0, 0)
        discussion_layout.addWidget(QLabel("Discussion"))
        discussion_layout.addWidget(self.discussion)

        self.figure_splitter = QSplitter(Qt.Orientation.Vertical)
        self.figure_splitter.addWidget(self.figure_stack)
        self.figure_splitter.addWidget(discussion_box)
        self.figure_splitter.setSizes([600, 220])
        self.splitter = QSplitter()
        for widget in (self.tree, self.parameter_stack, self.figure_splitter):
            self.splitter.addWidget(widget)
        self.splitter.setSizes([240, 360, 700])
        self.splitter.setStretchFactor(2, 1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.splitter)

        self.builder = FigureBuilder(paths, self)
        self.builder.started.connect(lambda: self.figure_view.set_busy(True))
        self.builder.finished.connect(lambda: self.figure_view.set_busy(False))
        self.builder.built.connect(self._built)
        self.builder.failed.connect(self.figure_view.error)
        self.tree.figure_selected.connect(self.select_figure)
        self.tree.regenerate_requested.connect(self.regenerate)
        self.discussion.textChanged.connect(self._text_changed)
        QApplication.instance().aboutToQuit.connect(self.shutdown)

    def select_figure(self, path):
        self.flush_pending_save()
        self.builder.cancel()
        self.doc = None
        self._dirty = False
        previous = self.parameter_scroll.takeWidget()
        if previous is not None:
            previous.deleteLater()
        self.parameters = None
        self.parameter_stack.setCurrentIndex(0)
        self.figure_stack.setCurrentIndex(0)
        self.figure_view.clear()
        self.discussion.blockSignals(True)
        self.discussion.clear()
        self.discussion.setEnabled(False)
        try:
            if path is None:
                return
            doc = model.FigureDoc.load(path, paths=self.paths)
            text = model.read_discussion(path)
            self.parameters = ParametersBox(doc)
            self.parameters.live_changed.connect(self.figure_view.setParam)
            self.parameters.regenerate_needed.connect(self.regenerate_later)
            self.parameter_scroll.setWidget(self.parameters)
            self.parameter_stack.setCurrentIndex(1)
            self.doc = doc
            self.discussion.setPlainText(text)
            self.discussion.setEnabled(True)
            self.figure_stack.setCurrentIndex(1)
            self.regenerate()
        except (ValueError, OSError) as exc:
            self.figure_stack.setCurrentIndex(1)
            self.figure_view.error(str(exc))
        finally:
            self.discussion.blockSignals(False)

    def regenerate(self):
        if self.doc is not None:
            self.builder.request(self.doc, immediate=True)

    def regenerate_later(self):
        if self.doc is not None:
            self.builder.request(self.doc)

    def _built(self, spec):
        self.figure_view.render(spec)
        # Live edits made during a build must survive its older parameter snapshot.
        if self.doc is not None:
            values = self.doc.values()
            for name in self.doc.live_params():
                if name in values:
                    self.figure_view.setParam(name, values[name])

    def _text_changed(self):
        if self.doc is not None:
            self._dirty = True
            self._save_timer.start()

    def flush_pending_save(self):
        self._save_timer.stop()
        if self.doc is not None and self._dirty:
            model.write_discussion(self.doc.path, self.discussion.toPlainText())
            self._dirty = False

    def shutdown(self):
        self.flush_pending_save()
        self.builder.cancel()

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)

    def apply_theme(self):
        self.tree.refresh_icons()
        self.figure_view.apply_theme()
