"""Per-experiment figures, shared parameters and autosaved discussion text."""

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QPlainTextEdit, QScrollArea, QSplitter,
                             QStackedWidget, QVBoxLayout, QWidget)

from ..config_panel.controls import ConfigBox, _make_subtle_tool_button, _parameter_header
from ..widgets import SmoothScrollArea, label
from . import model
from .params import ParametersBox
from .picker import FigurePicker
from .tree import DiscussionTree
from .view import FigureBuilder, FigureView


def _empty():
    label = QLabel("Select a figure")
    label.setObjectName("muted")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return label


def _padded(widget, *, stretch=True):
    """``widget`` in a holder with the configuration panel's box margins."""
    holder = QWidget()
    layout = QVBoxLayout(holder)
    layout.setContentsMargins(12, 10, 12, 16)
    layout.setSpacing(14)
    layout.addWidget(widget)
    if stretch:
        layout.addStretch()
    return holder


def _scroll():
    scroll = SmoothScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.Shape.NoFrame)
    scroll.verticalScrollBar().setSingleStep(16)
    return scroll


NO_EXPERIMENT = "No experiment selected"


class DiscussionPanel(QWidget):
    """Laid out like the Experiment tab: the tree, then a titled page of ConfigBox cards.

    The page title names the selected experiment or run. Its left column holds the figure's
    parameters box (titled with the figure's name) or the figure picker; its right column holds
    the Figure and Discussion cards.
    """

    PARAMETERS_PAGE, PICKER_PAGE = 1, 2

    def __init__(self, paths, settings_manager=None, parent=None, *, backend=None, candidates=None):
        super().__init__(parent)
        self.paths = paths
        # experiment rel -> FigureCandidate list; composes the experiment by default.
        self.candidates = candidates or (lambda rel: model.figure_candidates(paths, rel))
        self.doc = None
        self.parameters = None
        self.picker = None
        self.settings_manager = settings_manager
        self._target_runs = {}
        self._collapsed = {}
        self._dirty = False
        self._runs_requested = False
        self.run_loader = None
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(500)
        self._save_timer.timeout.connect(self.flush_pending_save)

        self.tree = DiscussionTree(paths, settings_manager=settings_manager)
        if backend is not None:
            from ..results.loader import ResultsLoader
            self.run_loader = ResultsLoader(backend, self)
            self.run_loader.runs_listed.connect(self.tree.set_runs)
            self.tree.refresh_requested.connect(self.refresh_runs)

        # Left column: parameters box or figure picker.
        self.parameter_stack = QStackedWidget()
        self.parameter_stack.addWidget(_empty())
        self.parameter_scroll = _scroll()
        self.parameter_stack.addWidget(self.parameter_scroll)
        self.picker_scroll = _scroll()
        self.parameter_stack.addWidget(self.picker_scroll)

        # Right column: the Figure and Discussion cards.
        self.figure_stack = QStackedWidget()
        self.figure_stack.addWidget(_empty())
        self.figure_view = FigureView()
        self.figure_stack.addWidget(self.figure_view)
        self.figure_stack.setProperty("configBoxBody", True)  # the card's surface shows through
        self.figure_box = ConfigBox(section="figureView")
        self.figure_caption = label("", "muted")
        figure_header = _parameter_header("", heading=False)
        figure_header.replaceWidget(figure_header.caption, self.figure_caption)
        figure_header.caption.deleteLater()
        self.figure_tools = {}
        for name, icon, tip, slot in (
                ("regenerate", "regenerate", "Regenerate the figure (picks up new runs)", self.regenerate),
                ("reset", "zoom_fit", "Reset zoom, pan and legend selection", self.figure_view.reset_view),
                ("copy", "copy", "Copy the figure as an image", self.copy_image)):
            button = _make_subtle_tool_button(icon, tip, size=26, icon_size=15)
            button.setObjectName(f"figureTool_{name}")
            button.setAccessibleName(tip)
            button.clicked.connect(slot)
            figure_header.addWidget(button)
            self.figure_tools[name] = button
        self.figure_box.add_layout(figure_header)
        self.figure_box.add_widget(self.figure_stack)
        self.figure_view.image_ready.connect(self._image_ready)
        self._update_figure_tools()

        self.discussion = QPlainTextEdit()
        self.discussion.setAccessibleName("Discussion")
        self.discussion.setPlaceholderText("Write about this figure…")
        self.discussion.setEnabled(False)
        self.discussion_box = ConfigBox(section="discussion")
        self.discussion_box.set_header("Discussion")
        self.discussion_box.add_widget(self.discussion)

        self.figure_splitter = QSplitter(Qt.Orientation.Vertical)
        self.figure_splitter.setHandleWidth(14)
        self.figure_splitter.addWidget(self.figure_box)
        self.figure_splitter.addWidget(self.discussion_box)
        self.figure_splitter.setSizes([560, 260])
        self.figure_splitter.setChildrenCollapsible(False)

        self.body_splitter = QSplitter()
        self.body_splitter.addWidget(self.parameter_stack)
        self.body_splitter.addWidget(_padded(self.figure_splitter, stretch=False))
        self.body_splitter.setSizes([520, 640])  # ordinary label + editor + × without wrapping
        self.body_splitter.setStretchFactor(1, 1)

        # The page header, as the configuration viewer's.
        self.header_bar = QWidget()
        header = QHBoxLayout(self.header_bar)
        header.setContentsMargins(12, 6, 12, 6)
        header.setSpacing(10)
        self.title = label(NO_EXPERIMENT, "heading")
        header.addWidget(self.title)
        header.addStretch()
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(8)
        page_layout.addWidget(self.header_bar)
        page_layout.addWidget(self.body_splitter, 1)

        self.splitter = QSplitter()
        self.splitter.addWidget(self.tree)
        self.splitter.addWidget(page)
        self.splitter.setSizes([260, 1160])
        self.splitter.setStretchFactor(1, 1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)  # the Experiment tab's margins
        layout.addWidget(self.splitter)

        self.builder = FigureBuilder(paths, self)
        self.builder.started.connect(lambda: self.figure_view.set_busy(True))
        self.builder.finished.connect(lambda: self.figure_view.set_busy(False))
        self.builder.built.connect(self._built)
        self.builder.failed.connect(self.figure_view.error)
        self.tree.figure_selected.connect(self.select_figure)
        self.tree.add_requested.connect(self.open_picker)
        self.tree.rename_requested.connect(self.start_rename)
        self.tree.about_to_modify.connect(self._release)
        self.tree.experiment_selected.connect(self._experiment_selected)
        self.discussion.textChanged.connect(self._text_changed)
        QApplication.instance().aboutToQuit.connect(self.shutdown)

    def select_figure(self, path):
        self.flush_pending_save()
        self.builder.cancel()
        self._discard_picker()
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
            runs = self.target_runs(doc.experiment)
            self.parameters = ParametersBox(doc, metric_choices=runs.metrics, method_choices=runs.methods,
                                            collapsed=self._collapsed_groups(doc.plotter))
            self.parameters.live_changed.connect(self.figure_view.setParam)
            self.parameters.regenerate_needed.connect(self.regenerate_later)
            self.parameters.rename_requested.connect(self.rename_figure)
            self.parameters.collapsed_changed.connect(lambda groups, p=doc.plotter: self._save_collapsed(p, groups))
            self.parameter_scroll.setWidget(_padded(self.parameters))
            self.parameter_stack.setCurrentIndex(self.PARAMETERS_PAGE)
            self.doc = doc
            self.discussion.setPlainText(text)
            self.discussion.setEnabled(True)
            self.figure_stack.setCurrentIndex(1)
            package, capability = doc.capability()
            self.figure_caption.setText(f"{capability.metadata.get('label') or capability.id} · "
                                        f"{package.id} {package.version}")
            self.builder.request(doc, immediate=True)  # cached when nothing changed
        except (ValueError, OSError) as exc:
            self.figure_stack.setCurrentIndex(1)
            self.figure_view.error(str(exc))
        finally:
            self.discussion.blockSignals(False)
            self._update_figure_tools()

    def _update_figure_tools(self):
        if not self.doc:
            self.figure_caption.setText("")
        for button in self.figure_tools.values():
            button.setEnabled(self.doc is not None)

    # ── A target's runs and per-plotter preferences ──────────────────────────

    def target_runs(self, target):
        """Logged methods and metrics, reread only when the target's files change."""
        stamp = model.target_stamp(self.paths, target)
        cached = self._target_runs.get(target)
        if cached is None or cached[0] != stamp:
            cached = self._target_runs[target] = (stamp, model.target_runs(self.paths, target))
        return cached[1]

    def _collapsed_groups(self, plotter):
        if self.settings_manager is not None:
            saved = self.settings_manager.get("discussion", "collapsed_groups", default={}) or {}
            if isinstance(saved, dict):
                return set(saved.get(plotter) or [])
        return set(self._collapsed.get(plotter, ()))

    def _save_collapsed(self, plotter, groups):
        self._collapsed[plotter] = sorted(groups)
        if self.settings_manager is not None:
            saved = dict(self.settings_manager.get("discussion", "collapsed_groups", default={}) or {})
            saved[plotter] = sorted(groups)
            self.settings_manager.persist_key("discussion", "collapsed_groups", saved)

    # ── Renaming, duplicating and deleting ───────────────────────────────────

    def _release(self, path):
        """Save pending text before a figure's files change; let go of a figure being removed."""
        self.flush_pending_save()

    def start_rename(self, path):
        if self.doc is None or self.doc.path != path:
            self.tree.select_figure(path)
        if self.parameters is not None and self.doc is not None and self.doc.path == path:
            self.close_picker()
            self.parameters.start_rename()

    def rename_figure(self, name):
        if self.doc is None:
            return
        self.flush_pending_save()
        try:
            path = model.rename_figure(self.doc.path, name)
        except (ValueError, OSError) as exc:
            self.parameters.rename_failed(str(exc))
            return
        self.doc = None  # the old path is gone; nothing more is written there
        self.tree.populate(select=path)

    # ── Figure toolbar ───────────────────────────────────────────────────────

    def copy_image(self):
        if self.doc is not None:
            self.figure_view.request_image()

    def _image_ready(self, image):
        if not image.isNull():
            QApplication.clipboard().setImage(image)

    def _experiment_selected(self, experiment):
        self.title.setText(experiment.replace("/", " / ") if experiment else NO_EXPERIMENT)
        self.title.setToolTip(experiment)

    # ── Adding figures ───────────────────────────────────────────────────────

    def open_picker(self, experiment):
        """Show the experiment's compatible plot components in place of the Parameters box."""
        if not experiment:
            return
        self._discard_picker()
        self.picker = FigurePicker(experiment, self.candidates)
        self.picker.chosen.connect(self.add_figure)
        self.picker.cancelled.connect(self.close_picker)
        self.picker_scroll.setWidget(_padded(self.picker))
        self.parameter_stack.setCurrentIndex(self.PICKER_PAGE)
        if self.picker.cards:
            self.picker.cards[0].setFocus()

    def close_picker(self):
        self._discard_picker()
        self.parameter_stack.setCurrentIndex(self.PARAMETERS_PAGE if self.parameters is not None else 0)

    def _discard_picker(self):
        self.picker = None
        previous = self.picker_scroll.takeWidget()
        if previous is not None:
            previous.deleteLater()

    def add_figure(self, experiment, selector):
        """Create the figure and open it; its Parameters box replaces the picker."""
        try:
            self.tree.add_figure(experiment, selector)
        except (ValueError, OSError) as exc:
            self.close_picker()
            self.figure_stack.setCurrentIndex(1)
            self.figure_view.error(str(exc))

    def regenerate(self):
        """Rebuild the figure now, bypassing the cache (for example to pick up new runs)."""
        if self.doc is not None:
            self.builder.request(self.doc, immediate=True, force=True)

    def showEvent(self, event):
        super().showEvent(event)
        if not self._runs_requested:
            self.refresh_runs()

    def refresh_runs(self):
        if self.run_loader is not None:
            self._runs_requested = True
            self.run_loader.clear()
            self.run_loader.list_runs()

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
