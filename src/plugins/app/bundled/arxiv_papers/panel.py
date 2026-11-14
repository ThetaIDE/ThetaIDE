"""arXiv papers panel: pick a component or tags, search arXiv, read abstracts, open or cite papers.

Searches run on one background thread (searches never overlap, which also keeps within arXiv's rate
limit) and results come back to the GUI thread through a signal. A result that arrives after a newer
search started, or after the panel was closed, is dropped.
"""
from __future__ import annotations

import html
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import QObject, QRectF, QRunnable, QSize, Qt, QThreadPool, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QDesktopServices, QFont, QFontMetrics, QPainter
from PyQt6.QtWidgets import (QApplication, QComboBox, QStyle, QStyledItemDelegate, QHBoxLayout, QInputDialog, QLabel, QLineEdit,
                             QListWidget, QListWidgetItem, QPushButton, QSplitter, QTextBrowser, QTreeWidget,
                             QTreeWidgetItem, QVBoxLayout, QWidget)

from src.app.api.desktop import label, theme_color

from .catalog import KIND_TITLES, build_catalog, normalize_tags, tag_counts, tag_phrase
from .search import ACKNOWLEDGEMENT, ArxivSearcher, SearchError, build_query

if TYPE_CHECKING:
    from src.app.api.desktop import PluginContext

ROLE_KEY = Qt.ItemDataRole.UserRole
ROLE_META = Qt.ItemDataRole.UserRole + 1


class _PaperDelegate(QStyledItemDelegate):
    """A result row: the title in bold over authors, date and category in the muted colour."""

    PAD = 8

    def _fonts(self, option):
        title = QFont(option.font)
        title.setWeight(QFont.Weight.DemiBold)
        meta = QFont(option.font)
        if option.font.pixelSize() > 0:  # the theme sets sizes in pixels (zoom)
            meta.setPixelSize(max(10, round(option.font.pixelSize() * 0.9)))
        else:
            meta.setPointSizeF(max(8.0, option.font.pointSizeF() * 0.9))
        return title, meta

    def sizeHint(self, option, index):
        title, meta = self._fonts(option)
        width = max(120, option.rect.width() or 400) - 2 * self.PAD
        title_rect = QFontMetrics(title).boundingRect(0, 0, width, 1000, Qt.TextFlag.TextWordWrap,
                                                       index.data(Qt.ItemDataRole.DisplayRole) or "")
        return QSize(width, title_rect.height() + QFontMetrics(meta).height() + 2 * self.PAD + 3)

    def paint(self, painter: QPainter, option, index):
        painter.save()
        rect = option.rect
        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(rect, QColor(theme_color("border")))
        elif option.state & QStyle.StateFlag.State_MouseOver:
            painter.fillRect(rect, QColor(theme_color("surface")))
        painter.setPen(QColor(theme_color("border")))
        painter.drawLine(rect.bottomLeft(), rect.bottomRight())
        title, meta = self._fonts(option)
        inner = QRectF(rect.adjusted(self.PAD, self.PAD, -self.PAD, -self.PAD))
        text = index.data(Qt.ItemDataRole.DisplayRole) or ""
        flags = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap
        painter.setFont(title)
        painter.setPen(QColor(theme_color("text")))
        used = painter.boundingRect(inner, flags, text)
        painter.drawText(inner, flags, text)
        painter.setFont(meta)
        painter.setPen(QColor(theme_color("muted")))
        meta_metrics = QFontMetrics(meta)
        meta_rect = QRectF(inner.left(), used.bottom() + 3, inner.width(), meta_metrics.height())
        line = meta_metrics.elidedText(index.data(ROLE_META) or "", Qt.TextElideMode.ElideRight,
                                       int(meta_rect.width()))
        painter.drawText(meta_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, line)
        painter.restore()


class _Relay(QObject):
    done = pyqtSignal(int, object, bool, str)  # serial, papers, from cache, error


class _SearchTask(QRunnable):
    def __init__(self, searcher, serial, query, max_results, sort, relay):
        super().__init__()
        self.searcher, self.serial, self.relay = searcher, serial, relay
        self.args = (query, max_results, sort)

    def run(self):
        try:
            papers, cached = self.searcher.search(*self.args)
            result = (self.serial, papers, cached, "")
        except SearchError as exc:
            result = (self.serial, [], False, str(exc))
        except Exception as exc:
            result = (self.serial, [], False, f"Search failed: {exc}")
        try:
            self.relay.done.emit(*result)
        except RuntimeError:  # the panel was deleted while the search ran
            pass


class ArxivPanel(QWidget):
    def __init__(self, context: "PluginContext | None" = None, searcher: ArxivSearcher | None = None,
                 hub_client=None, config_root: Path | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("arxivPanel")
        self.context = context
        # An explicit client (tests, embedding) wins; otherwise the host's Hub service.
        self.hub_client = hub_client
        if config_root:
            self.config_root = Path(config_root)
        elif context is not None:
            self.config_root = context.workspace_paths().config
        else:
            from src.app.api.workspace import WorkspacePaths
            self.config_root = WorkspacePaths.for_process().config
        if searcher is None:
            cache = context.get_data_dir() / "plugins" / "arxiv_cache.json" if context else None
            searcher = ArxivSearcher(cache_path=cache)
        self.searcher = searcher
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.relay = _Relay(self)
        self.relay.done.connect(self._on_results)
        self.serial = 0
        self.papers = []
        self.components = []
        self._building = False
        self._build_ui()
        self.reload_components()
        if self.hub_client is not None and hasattr(self.hub_client, "indexLoaded"):
            # A bound method, not a lambda: Qt drops the connection when the panel is destroyed, so a Hub
            # index that finishes loading later never calls into a deleted panel (PyQt aborts on that).
            self.hub_client.indexLoaded.connect(self._on_hub_index)
        elif self.hub_client is None and context is not None:
            context.add_hub_index_listener(self._on_hub_index)  # removed in cleanup()

    def _hub_source(self):
        if self.hub_client is not None:
            return self.hub_client
        return self.context.hub_components() if self.context is not None else None

    def _on_hub_index(self, _components=None):
        self.reload_components()

    # ── UI ───────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(8)

        bar = QHBoxLayout()
        bar.setSpacing(8)
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Extra search words (optional)")
        self.search_input.setClearButtonEnabled(True)
        self.search_input.returnPressed.connect(self.search)
        self.search_input.textChanged.connect(self._update_query_preview)
        bar.addWidget(self.search_input, 1)
        self.match_combo = QComboBox()
        self.match_combo.addItem("Match all tags", True)
        self.match_combo.addItem("Match any tag", False)
        self.match_combo.currentIndexChanged.connect(self._update_query_preview)
        bar.addWidget(self.match_combo)
        self.sort_combo = QComboBox()
        for text, key in (("Relevance", "relevance"), ("Newest", "newest"), ("Recently updated", "updated")):
            self.sort_combo.addItem(text, key)
        bar.addWidget(self.sort_combo)
        self.count_combo = QComboBox()
        for count in (10, 25, 50):
            self.count_combo.addItem(f"{count} results", count)
        self.count_combo.setCurrentIndex(1)
        bar.addWidget(self.count_combo)
        self.search_button = QPushButton("Search arXiv")
        self.search_button.setObjectName("primary")
        self.search_button.clicked.connect(self.search)
        bar.addWidget(self.search_button)
        root.addLayout(bar)

        self.query_label = label("", "muted")
        self.query_label.setWordWrap(True)
        self.query_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self.query_label)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)

        # left: components and tags
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)
        left_layout.addWidget(label("Components", "cardTitle"))
        self.component_filter = QLineEdit()
        self.component_filter.setPlaceholderText("Filter components or tags")
        self.component_filter.setClearButtonEnabled(True)
        self.component_filter.textChanged.connect(self._filter_components)
        left_layout.addWidget(self.component_filter)
        self.component_tree = QTreeWidget()
        self.component_tree.setHeaderLabels(["Component", "Tags"])
        self.component_tree.setRootIsDecorated(True)
        self.component_tree.setColumnWidth(0, 150)
        self.component_tree.itemClicked.connect(self._on_component_clicked)
        self.component_tree.currentItemChanged.connect(lambda *_: self._update_buttons())
        left_layout.addWidget(self.component_tree, 3)
        component_buttons = QHBoxLayout()
        self.edit_tags_button = QPushButton("Edit tags")
        self.edit_tags_button.setToolTip("Change the tags searched for this component")
        self.edit_tags_button.clicked.connect(self.edit_component_tags)
        component_buttons.addWidget(self.edit_tags_button)
        component_buttons.addStretch(1)
        left_layout.addLayout(component_buttons)

        tag_header = QHBoxLayout()
        tag_header.addWidget(label("Tags", "cardTitle"))
        tag_header.addStretch(1)
        self.clear_tags_button = QPushButton("Clear")
        self.clear_tags_button.clicked.connect(self.clear_tags)
        tag_header.addWidget(self.clear_tags_button)
        left_layout.addLayout(tag_header)
        self.tag_list = QListWidget()
        self.tag_list.itemChanged.connect(self._on_tag_toggled)
        left_layout.addWidget(self.tag_list, 2)
        split.addWidget(left)

        # right: results above, the selected paper below
        right = QSplitter(Qt.Orientation.Vertical)
        right.setChildrenCollapsible(False)
        results = QWidget()
        results_layout = QVBoxLayout(results)
        results_layout.setContentsMargins(8, 0, 0, 0)
        results_layout.setSpacing(6)
        self.status_label = label("Pick a component or tags, then search arXiv.", "muted")
        self.status_label.setWordWrap(True)
        results_layout.addWidget(self.status_label)
        self.results_list = QListWidget()
        self.results_list.setWordWrap(True)
        self.results_list.setMouseTracking(True)
        self.results_list.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.results_list.setItemDelegate(_PaperDelegate(self.results_list))
        self.results_list.currentRowChanged.connect(self._show_paper)
        self.results_list.itemDoubleClicked.connect(lambda _item: self.open_abstract())
        results_layout.addWidget(self.results_list, 1)
        right.addWidget(results)

        detail = QWidget()
        detail_layout = QVBoxLayout(detail)
        detail_layout.setContentsMargins(8, 6, 0, 0)
        detail_layout.setSpacing(6)
        self.detail = QTextBrowser()
        self.detail.setOpenExternalLinks(True)
        detail_layout.addWidget(self.detail, 1)
        paper_buttons = QHBoxLayout()
        self.open_button = QPushButton("Open on arXiv")
        self.open_button.clicked.connect(self.open_abstract)
        self.pdf_button = QPushButton("Open PDF")
        self.pdf_button.clicked.connect(self.open_pdf)
        self.cite_button = QPushButton("Copy BibTeX")
        self.cite_button.clicked.connect(self.copy_citation)
        for button in (self.open_button, self.pdf_button, self.cite_button):
            paper_buttons.addWidget(button)
        paper_buttons.addStretch(1)
        detail_layout.addLayout(paper_buttons)
        acknowledgement = label(ACKNOWLEDGEMENT, "muted")
        acknowledgement.setWordWrap(True)
        detail_layout.addWidget(acknowledgement)
        right.addWidget(detail)
        right.setSizes([340, 260])
        split.addWidget(right)
        split.setSizes([300, 620])
        root.addWidget(split, 1)
        self._show_paper(-1)

    # ── components and tags ──────────────────────────────────────────────────

    def _overrides(self) -> dict:
        value = self.context.get_setting("tag_overrides", {}) if self.context else getattr(self, "_local_overrides", {})
        return value if isinstance(value, dict) else {}

    def _save_overrides(self, overrides: dict):
        if self.context:
            self.context.set_setting("tag_overrides", overrides)
        else:
            self._local_overrides = overrides

    def reload_components(self):
        checked = set(self.selected_tags())
        current = self.component_tree.currentItem()
        current_key = current.data(0, ROLE_KEY) if current else None
        self.components = build_catalog(self.config_root, self._hub_source(), self._overrides())
        self._building = True
        self.component_tree.clear()
        groups = {}
        for comp in self.components:
            group = groups.get(comp.kind)
            if group is None:
                group = QTreeWidgetItem([KIND_TITLES.get(comp.kind, comp.kind.title()), ""])
                group.setFlags(group.flags() & ~Qt.ItemFlag.ItemIsSelectable)
                self.component_tree.addTopLevelItem(group)
                group.setExpanded(True)
                groups[comp.kind] = group
            name = comp.name + ("  (Hub)" if comp.source == "hub" else "")
            item = QTreeWidgetItem([name, ", ".join(comp.tags) if comp.tags else "no tags"])
            item.setData(0, ROLE_KEY, comp.key)
            tip = comp.description or comp.key
            item.setToolTip(0, tip + ("\nTags edited in this panel" if comp.edited else ""))
            item.setToolTip(1, ", ".join(comp.tags) or "No tags yet: use Edit tags")
            if not comp.tags:
                item.setForeground(1, self.palette().placeholderText())
            group.addChild(item)
            if comp.key == current_key:
                self.component_tree.setCurrentItem(item)

        self.tag_list.clear()
        for tag, count in tag_counts(self.components).items():
            item = QListWidgetItem(f"{tag}  ({count})")
            item.setData(ROLE_KEY, tag)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if tag in checked else Qt.CheckState.Unchecked)
            item.setToolTip(f'Searches arXiv for "{tag_phrase(tag)}"')
            self.tag_list.addItem(item)
        self._building = False
        self._filter_components(self.component_filter.text())
        self._update_query_preview()
        self._update_buttons()

    def component(self, key):
        return next((c for c in self.components if c.key == key), None)

    def selected_tags(self) -> list[str]:
        return [self.tag_list.item(i).data(ROLE_KEY) for i in range(self.tag_list.count())
                if self.tag_list.item(i).checkState() == Qt.CheckState.Checked]

    def set_selected_tags(self, tags):
        wanted = set(tags)
        self._building = True
        for i in range(self.tag_list.count()):
            item = self.tag_list.item(i)
            item.setCheckState(Qt.CheckState.Checked if item.data(ROLE_KEY) in wanted else Qt.CheckState.Unchecked)
        self._building = False
        self._update_query_preview()
        first = next((self.tag_list.item(i) for i in range(self.tag_list.count())
                      if self.tag_list.item(i).checkState() == Qt.CheckState.Checked), None)
        if first is not None:
            self.tag_list.scrollToItem(first, QListWidget.ScrollHint.PositionAtTop)

    def clear_tags(self):
        self.set_selected_tags([])

    def _on_tag_toggled(self, _item):
        if not self._building:
            self._update_query_preview()

    def _on_component_clicked(self, item, _column=0):
        comp = self.component(item.data(0, ROLE_KEY))
        if comp is None:
            return
        if not comp.tags:
            self.status_label.setText(f"{comp.name} has no tags yet. Use Edit tags to add some.")
            return
        self.set_selected_tags(comp.tags)
        self.search()

    def _filter_components(self, text):
        needle = text.strip().lower()
        for i in range(self.component_tree.topLevelItemCount()):
            group = self.component_tree.topLevelItem(i)
            visible = 0
            for j in range(group.childCount()):
                child = group.child(j)
                comp = self.component(child.data(0, ROLE_KEY))
                match = not needle or (comp and (needle in comp.name.lower()
                                                 or any(needle in tag for tag in comp.tags)))
                child.setHidden(not match)
                visible += bool(match)
            group.setHidden(visible == 0)
        for i in range(self.tag_list.count()):
            item = self.tag_list.item(i)
            item.setHidden(bool(needle) and needle not in item.data(ROLE_KEY)
                           and item.checkState() != Qt.CheckState.Checked)

    def edit_component_tags(self):
        item = self.component_tree.currentItem()
        comp = self.component(item.data(0, ROLE_KEY)) if item else None
        if comp is None:
            return
        text, ok = QInputDialog.getText(self, f"Tags for {comp.name}",
                                        "Comma-separated tags (leave empty to restore the defaults):",
                                        text=", ".join(comp.tags))
        if ok:
            self.set_component_tags(comp.key, [t for t in text.split(",")])

    def set_component_tags(self, key, tags):
        """Save the user's tags for a component; empty restores its built-in or Hub tags."""
        overrides = dict(self._overrides())
        tags = normalize_tags(tags)
        if tags:
            overrides[key] = tags
        else:
            overrides.pop(key, None)
        self._save_overrides(overrides)
        self.reload_components()

    def _update_buttons(self):
        item = self.component_tree.currentItem()
        self.edit_tags_button.setEnabled(bool(item and item.data(0, ROLE_KEY)))

    # ── searching ────────────────────────────────────────────────────────────

    def current_query(self) -> str:
        return build_query(self.selected_tags(), self.search_input.text(), bool(self.match_combo.currentData()))

    def _update_query_preview(self, *_):
        query = self.current_query()
        self.query_label.setText(f"Query: {query}" if query else "Query: choose tags or type search words")
        self.search_button.setEnabled(bool(query))

    def search(self):
        query = self.current_query()
        if not query:
            self.status_label.setText("Choose a tag or type something to search for.")
            return
        self.serial += 1
        self.status_label.setText("Searching arXiv…")
        self.search_button.setEnabled(False)
        self.pool.start(_SearchTask(self.searcher, self.serial, query, int(self.count_combo.currentData()),
                                    self.sort_combo.currentData(), self.relay))

    def _on_results(self, serial, papers, cached, error):
        if serial != self.serial:
            return  # a newer search has started
        self.search_button.setEnabled(bool(self.current_query()))
        if error:
            self.status_label.setText(error)
            return
        self.papers = list(papers)
        self.results_list.clear()
        for paper in self.papers:
            authors = ", ".join(paper.authors[:3]) + (" et al." if len(paper.authors) > 3 else "")
            item = QListWidgetItem(paper.title)
            item.setData(ROLE_META, f"{authors} · {paper.published} · {paper.primary_category}")
            item.setToolTip(paper.title)
            self.results_list.addItem(item)
        count = len(self.papers)
        source = " (saved results)" if cached else ""
        self.status_label.setText(f"{count} paper{'s' if count != 1 else ''} found{source}." if count
                                  else "No papers matched. Try fewer tags or Match any tag.")
        if count:
            self.results_list.setCurrentRow(0)
        else:
            self._show_paper(-1)

    # ── the selected paper ───────────────────────────────────────────────────

    def current_paper(self):
        row = self.results_list.currentRow()
        return self.papers[row] if 0 <= row < len(self.papers) else None

    def _show_paper(self, row):
        paper = self.papers[row] if 0 <= row < len(self.papers) else None
        for button in (self.open_button, self.pdf_button, self.cite_button):
            button.setEnabled(paper is not None)
        if paper is None:
            family = html.escape(QApplication.font().family())
            self.detail.setHtml(f"<p style=\"color:{theme_color('muted')}; font-family:'{family}', sans-serif\">"
                                "Select a paper to read its abstract.</p>")
            return
        muted, accent = theme_color("muted"), theme_color("secondary")
        body = f"font-family:'{html.escape(QApplication.font().family())}', sans-serif; font-size:10pt"
        e = html.escape
        categories = ", ".join(paper.categories) or paper.primary_category
        updated = f" · updated {e(paper.updated)}" if paper.updated and paper.updated != paper.published else ""
        self.detail.setHtml(
            f"<div style=\"{body}\">"
            f"<h3 style='margin:0 0 6px 0'>{e(paper.title)}</h3>"
            f"<p style='margin:0 0 4px 0'>{e(', '.join(paper.authors))}</p>"
            f"<p style='margin:0 0 10px 0; color:{muted}'>arXiv:{e(paper.arxiv_id)} · {e(paper.published)}{updated}"
            f" · {e(categories)}</p>"
            f"<p style='margin:0 0 10px 0'>{e(paper.summary)}</p>"
            f"<p><a style='color:{accent}' href='{e(paper.abs_url)}'>Abstract page</a> &nbsp; "
            f"<a style='color:{accent}' href='{e(paper.pdf_url)}'>PDF</a></p></div>")

    def open_abstract(self):
        paper = self.current_paper()
        if paper and paper.abs_url:
            QDesktopServices.openUrl(QUrl(paper.abs_url))

    def open_pdf(self):
        paper = self.current_paper()
        if paper and paper.pdf_url:
            QDesktopServices.openUrl(QUrl(paper.pdf_url))

    def copy_citation(self):
        paper = self.current_paper()
        if paper:
            QApplication.clipboard().setText(paper.citation())
            self.status_label.setText(f"Copied BibTeX for arXiv:{paper.arxiv_id}.")

    def cleanup(self):
        """Drop results from searches still running (the panel is being removed)."""
        self.serial += 1
        try:
            self.relay.done.disconnect(self._on_results)
        except TypeError:
            pass
        if self.hub_client is not None and hasattr(self.hub_client, "indexLoaded"):
            try:
                self.hub_client.indexLoaded.disconnect(self._on_hub_index)
            except TypeError:
                pass
        elif self.context is not None:
            self.context.remove_hub_index_listener(self._on_hub_index)
