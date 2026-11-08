"""Interactive Documentation panel widget for Theta-IDE.

Left: an outline of the documentation (category headers that collapse, articles, and each
article's sections). Right: the article as one page, a header (category, title, summary) over
the text, which wraps at a readable width.
"""
from __future__ import annotations

import html
import re
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import QByteArray, QEvent, QRectF, Qt, QUrl
from PyQt6.QtGui import QColor, QFont, QIcon, QImage, QPainter, QPixmap, QTextDocument
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTextBrowser,
    QTextEdit,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from frontend.sidetabs import svg_icon
from frontend.theme import current_theme, theme_color
from frontend.widgets import label

from .content import ARTICLES, DocArticle, search_articles

if TYPE_CHECKING:
    from frontend.plugins.context import PluginContext

_BUTTON_ICON_COLORS = {(QIcon.Mode.Normal, QIcon.State.Off): "text"}
_SEARCH_ICON_COLORS = {(QIcon.Mode.Normal, QIcon.State.Off): "muted"}
_CODE_BLOCK = re.compile(r"<pre><code>(.*?)</code></pre>", re.DOTALL)
_SECTION = re.compile(r"<h3>(.*?)</h3>", re.DOTALL)
_LEADING_H2 = re.compile(r"^\s*<h2>.*?</h2>", re.DOTALL)
_TAG = re.compile(r"<[^>]+>")

TEXT_WIDTH = 860  # px the article text wraps at: long lines across a wide window are hard to read

# Articles embed diagrams/NAME.svg as <img src="diagram:NAME">; their colours are {{role}} placeholders
DIAGRAM_DIR = Path(__file__).resolve().parent / "diagrams"
DIAGRAM_SCHEME = "diagram"
_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")
_SVG_TITLE = re.compile(r"<title>(.*?)</title>", re.DOTALL)

# Outline item kinds, stored in Qt.ItemDataRole.UserRole as (kind, article_id, anchor)
CATEGORY, ARTICLE, SECTION = "category", "article", "section"


def _resized(font: QFont, factor: float) -> QFont:
    """A copy of `font` scaled by `factor`. The app font may be sized in pixels, in which case
    pointSizeF() is -1, so scale whichever unit it uses."""
    scaled = QFont(font)
    if font.pointSizeF() > 0:
        scaled.setPointSizeF(font.pointSizeF() * factor)
    elif font.pixelSize() > 0:
        scaled.setPixelSize(max(9, round(font.pixelSize() * factor)))
    return scaled


def _theme_key():
    return tuple(sorted(current_theme()["colors"].items()))


def _boxed_code_blocks(article_html: str) -> str:
    """Wrap each <pre><code> block in a one-cell table.

    Qt's rich text paints a <pre> background line by line (every line is its own block), so code
    showed as separated stripes, and it ignores border-radius. A table cell is one solid box with
    padding and a border, which is what a code block should look like.
    """
    border, fill = theme_color("border"), theme_color("surface")
    return _CODE_BLOCK.sub(
        lambda m: (f'<table class="code" width="100%" cellspacing="0" cellpadding="12" border="1" '
                   f'style="border-color: {border}; border-style: solid; background-color: {fill};">'
                   f'<tr><td><pre>{m.group(1)}</pre></td></tr></table>'),
        article_html,
    )


def article_sections(article: DocArticle) -> list[str]:
    """The plain-text titles of an article's sections (its <h3> headings), in order."""
    return [html.unescape(_TAG.sub("", title)).strip() for title in _SECTION.findall(article.html_content)]


def _anchored_sections(article_html: str) -> str:
    """Put an anchor (sec-0, sec-1, …) before each section heading, for the outline to scroll to."""
    count = iter(range(1_000_000))
    # Inside the heading: an anchor before it belongs to the previous paragraph, so jumps stopped short
    return _SECTION.sub(lambda m: f'<h3><a name="sec-{next(count)}"></a>{m.group(1)}</h3>', article_html)


def article_body(article: DocArticle) -> str:
    """The article's HTML as the page shows it: without its opening <h2> (the page header already
    titles it), with anchored sections and boxed code blocks."""
    return _boxed_code_blocks(_anchored_sections(_LEADING_H2.sub("", article.html_content, count=1)))


def diagram_svg(name: str) -> str | None:
    """diagrams/NAME.svg with its {{role}} placeholders filled in from the current theme."""
    path = DIAGRAM_DIR / f"{name}.svg"
    if not name.isidentifier() or not path.is_file():
        return None
    colors = dict(current_theme()["colors"])
    accent, base = QColor(colors["accent"]).getRgbF()[:3], QColor(colors["base"]).getRgbF()[:3]
    colors["accent_soft"] = QColor.fromRgbF(*(a * 0.18 + b * 0.82 for a, b in zip(accent, base))).name()  # pale accent
    return _PLACEHOLDER.sub(lambda m: colors.get(m.group(1), m.group(0)), path.read_text(encoding="utf-8"))


def diagram_title(name: str) -> str:
    match = _SVG_TITLE.search(diagram_svg(name) or "")
    return html.unescape(match.group(1).strip()) if match else name


def render_diagram(name: str, width: int, ratio: float = 1.0) -> QImage:
    """The diagram drawn `width` px wide in the current theme, for a screen at `ratio`."""
    svg = diagram_svg(name)
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8"))) if svg else None
    if renderer is None or not renderer.isValid():
        return QImage()
    box = renderer.viewBoxF()
    image = QImage(round(width * ratio), round(width * box.height() / box.width() * ratio),
                   QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    renderer.render(painter, QRectF(0, 0, image.width(), image.height()))
    painter.end()
    image.setDevicePixelRatio(ratio)
    return image


class DocBrowser(QTextBrowser):
    """The article view. Supplies diagram:NAME images, drawn in the current theme at the text width."""

    def loadResource(self, kind, url):
        if url.scheme() == DIAGRAM_SCHEME and kind == QTextDocument.ResourceType.ImageResource.value:
            return render_diagram(url.path(), TEXT_WIDTH, self.devicePixelRatioF())
        return super().loadResource(kind, url)


class DiagramDialog(QDialog):
    """A diagram at full size, for reading the labels the text column shrinks."""

    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(diagram_title(name))
        box = QSvgRenderer(QByteArray(diagram_svg(name).encode("utf-8"))).viewBoxF()
        natural_width, natural_height = box.width(), box.height()
        screen = (parent.screen() if parent is not None else None) or self.screen()
        room = screen.availableGeometry() if screen is not None else None
        width = round(natural_width * 1.4)
        if room is not None:
            width = min(width, round(room.width() * 0.92))
        image = render_diagram(name, width, self.devicePixelRatioF())
        self.image = QLabel()
        self.image.setPixmap(QPixmap.fromImage(image))
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scroll = QScrollArea()
        scroll.setWidget(self.image)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addWidget(scroll)
        height = round(width * natural_height / natural_width) + 40
        if room is not None:
            height = min(height, round(room.height() * 0.9))
        self.resize(width + 40, height)


class DocumentationPanel(QWidget):
    """Main documentation browser panel in Theta-IDE."""

    def __init__(self, context: PluginContext | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.context = context
        self.setObjectName("documentationPanel")
        self.current_article_id: str = "overview"
        self._styled_for = None
        # What the user unfolded or folded, kept across outline rebuilds (search, filters, theme)
        self._collapsed_categories: set[str] = set()
        self._expanded_articles: set[str] = {"overview"}
        self._anchor_tops: list[tuple[float, str]] | None = None  # (y, anchor) of each section; lazily
        self._last_heading_scroll: float | None = None
        self._active_section: str | None = None
        self._init_ui()
        self._populate_categories()
        self._apply_theme()
        self._populate_articles()
        self._fit_contents_width()
        self.select_article("overview")

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 14, 18, 14)
        root.setSpacing(12)

        # ── Top Toolbar ─────────────────────────────────────────────────────────
        toolbar = QHBoxLayout()
        toolbar.setSpacing(10)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search documentation, topics, or keywords…")
        self.search_action = self.search_input.addAction(QIcon(), QLineEdit.ActionPosition.LeadingPosition)
        self.search_input.setClearButtonEnabled(True)
        self.search_input.setMinimumWidth(260)
        self.search_input.textChanged.connect(self._on_search_changed)
        toolbar.addWidget(self.search_input, 1)

        self.cat_combo = QComboBox()
        self.cat_combo.setMinimumWidth(160)
        self.cat_combo.currentIndexChanged.connect(self._on_filter_changed)
        toolbar.addWidget(self.cat_combo)

        self.btn_hotkeys = QPushButton("Hotkeys")
        self.btn_hotkeys.setToolTip("Open Settings › Hotkeys")
        self.btn_hotkeys.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_hotkeys.clicked.connect(self._open_hotkeys)
        toolbar.addWidget(self.btn_hotkeys)

        self.btn_hub = QPushButton("Community Hub")
        self.btn_hub.setToolTip("Explore Community Hub components")
        self.btn_hub.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_hub.clicked.connect(self._open_hub)
        toolbar.addWidget(self.btn_hub)

        root.addLayout(toolbar)

        # ── Main Splitter ────────────────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(10)
        splitter.setChildrenCollapsible(False)

        # Left: the outline
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)
        contents_row = QHBoxLayout()
        contents_row.setContentsMargins(0, 0, 0, 0)
        contents_row.setSpacing(2)
        contents_row.addWidget(label("CONTENTS", "eyebrow"))
        contents_row.addStretch()
        self.btn_expand_all = QToolButton()
        self.btn_expand_all.setText("Expand all")
        self.btn_expand_all.setToolTip("Unfold every category and article")
        self.btn_expand_all.clicked.connect(lambda: self.set_all_expanded(True))
        self.btn_collapse_all = QToolButton()
        self.btn_collapse_all.setText("Collapse all")
        self.btn_collapse_all.setToolTip("Fold every category and article")
        self.btn_collapse_all.clicked.connect(lambda: self.set_all_expanded(False))
        for button in (self.btn_expand_all, self.btn_collapse_all):
            button.setObjectName("docOutlineAction")
            button.setAutoRaise(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            contents_row.addWidget(button)
        left_layout.addLayout(contents_row)

        self.toc = QTreeWidget()
        self.toc.setObjectName("docOutline")
        self.toc.setHeaderHidden(True)
        self.toc.setIndentation(14)
        self.toc.setExpandsOnDoubleClick(False)  # one click on a header toggles it
        self.toc.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.toc.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.toc.itemClicked.connect(self._on_outline_clicked)
        self.toc.itemExpanded.connect(lambda item: self._remember_expansion(item, expanded=True))
        self.toc.itemCollapsed.connect(lambda item: self._remember_expansion(item, expanded=False))
        left_layout.addWidget(self.toc, 1)
        splitter.addWidget(left_container)

        # Right: the article as one page (header over text), no box around each part
        self.page = QFrame()
        self.page.setObjectName("docPage")
        page_layout = QVBoxLayout(self.page)
        page_layout.setContentsMargins(32, 26, 16, 0)
        page_layout.setSpacing(0)

        self.header = QWidget()
        self.header.setObjectName("docHeader")
        self.header.setMaximumWidth(TEXT_WIDTH)
        header_layout = QVBoxLayout(self.header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(6)
        self.badge_cat = label("Getting Started")  # breadcrumb: Category  ›  Article
        self.badge_cat.setObjectName("docBreadcrumb")
        header_layout.addWidget(self.badge_cat)
        self.title_lbl = label("Welcome & Architecture Overview")
        self.title_lbl.setObjectName("docTitle")
        self.title_lbl.setWordWrap(True)
        header_layout.addWidget(self.title_lbl)
        self.summary_lbl = label("High-level architecture overview.")
        self.summary_lbl.setObjectName("docSummary")
        self.summary_lbl.setWordWrap(True)
        header_layout.addWidget(self.summary_lbl)
        page_layout.addWidget(self.header)

        self.divider = QFrame()
        self.divider.setObjectName("docDivider")
        self.divider.setFixedHeight(1)
        self.divider.setMaximumWidth(TEXT_WIDTH)
        page_layout.addSpacing(18)
        page_layout.addWidget(self.divider)
        page_layout.addSpacing(6)

        self.browser = DocBrowser()
        self.browser.setObjectName("docText")
        self.browser.setFrameShape(QFrame.Shape.NoFrame)
        self.browser.setOpenExternalLinks(False)
        self.browser.setOpenLinks(False)
        self.browser.anchorClicked.connect(self._on_anchor_clicked)
        self.browser.setLineWrapMode(QTextEdit.LineWrapMode.FixedPixelWidth)
        self.browser.setLineWrapColumnOrWidth(TEXT_WIDTH)
        self.browser.document().setDocumentMargin(0)
        # The outline follows the reading position: the section whose heading has scrolled to the top
        self.browser.verticalScrollBar().valueChanged.connect(self._sync_outline_to_scroll)
        self.browser.document().documentLayout().documentSizeChanged.connect(self._forget_anchor_tops)
        self.browser.viewport().installEventFilter(self)  # resizes: keep the room after the last line
        page_layout.addWidget(self.browser, 1)

        splitter.addWidget(self.page)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        self.splitter = splitter
        root.addWidget(splitter, 1)

    def _fit_contents_width(self, cap: int = 360):
        """Open the outline wide enough for the longest article title (up to `cap` px)."""
        metrics = self.toc.fontMetrics()
        longest = max((metrics.horizontalAdvance(art.title) for art in ARTICLES), default=200)
        chrome = 2 * self.toc.frameWidth() + 2 * self.toc.indentation() + 24 + 16  # indent, padding, scrollbar
        self.splitter.setSizes([min(cap, longest + chrome), 900])

    # ── Theme ───────────────────────────────────────────────────────────────────

    def changeEvent(self, event):
        # A theme switch sets the application palette, which reaches every widget as a PaletteChange
        super().changeEvent(event)
        if (event.type() == QEvent.Type.PaletteChange and self._styled_for is not None
                and self._styled_for != _theme_key()):
            self._apply_theme()
            self._populate_articles()  # outline item colors
            scroll = self.browser.verticalScrollBar().value()
            self.select_article(self.current_article_id)  # the article's HTML carries theme colors
            self.browser.verticalScrollBar().setValue(scroll)

    def _apply_theme(self):
        """Colors for the panel's own widgets, all from the current theme."""
        self._styled_for = _theme_key()
        border, accent = theme_color("border"), theme_color("accent")
        self.toc.setStyleSheet(
            f"QTreeWidget#docOutline {{ background: transparent; border: 0; outline: 0; }} "
            "QTreeWidget#docOutline::item { padding: 5px 4px; border: 0; border-radius: 4px; } "
            f"QTreeWidget#docOutline::item:hover {{ background: {theme_color('surface')}; }} "
            f"QTreeWidget#docOutline::item:selected {{ background: {theme_color('raised')}; color: {accent}; }}"
        )
        # Quiet text buttons; set on each button, since they sit beside the outline, not inside it
        action_css = (
            f"QToolButton#docOutlineAction {{ color: {theme_color('muted')}; font-size: 11px; padding: 1px 6px; "
            "border: 0; border-radius: 4px; background: transparent; } "
            f"QToolButton#docOutlineAction:hover {{ color: {theme_color('text')}; background: {theme_color('surface')}; }}"
        )
        self.btn_expand_all.setStyleSheet(action_css)
        self.btn_collapse_all.setStyleSheet(action_css)
        self.page.setStyleSheet(
            f"QFrame#docPage {{ background: {theme_color('base')}; border: 1px solid {border}; border-radius: 10px; }} "
            f"QLabel#docBreadcrumb {{ background: transparent; color: {accent}; font-size: 11px; font-weight: 600; "
            "letter-spacing: 0.5px; } "
            f"QLabel#docTitle {{ background: transparent; color: {theme_color('text')}; font-size: 24px; font-weight: 700; }} "
            f"QLabel#docSummary {{ background: transparent; color: {theme_color('muted')}; font-size: 13px; }} "
            f"QFrame#docDivider {{ background: {border}; border: 0; }} "
            "QWidget#docHeader { background: transparent; } "
            "QTextBrowser#docText { background: transparent; border: 0; padding: 0; font-size: 13px; }"
        )
        self.btn_hotkeys.setIcon(svg_icon("hotkeys", _BUTTON_ICON_COLORS, size=16))
        self.btn_hub.setIcon(svg_icon("hub", _BUTTON_ICON_COLORS, size=16))
        self.search_action.setIcon(svg_icon("search", _SEARCH_ICON_COLORS, size=16))

    def _article_css(self) -> str:
        """Styles for an article. Qt rich text supports a CSS subset (no border-radius, no rgba,
        no :hover or :nth-child), so this keeps to what it renders, in theme colors."""
        text_col, accent_col, border = theme_color("text"), theme_color("accent"), theme_color("border")
        mono = 'ui-monospace, Menlo, Consolas, "DejaVu Sans Mono", monospace'
        return f"""
            body {{ color: {text_col}; font-size: 13px; }}
            h2 {{ color: {text_col}; margin-top: 18px; margin-bottom: 10px; font-size: 18px; }}
            h3 {{ color: {text_col}; margin-top: 22px; margin-bottom: 8px; font-size: 15px; font-weight: 600; }}
            h4 {{ color: {theme_color('muted')}; margin-top: 14px; margin-bottom: 6px; font-size: 13px; }}
            p {{ margin-top: 0; margin-bottom: 10px; }}
            p, li {{ color: {text_col}; }}
            li {{ margin-bottom: 4px; }}
            code {{ color: {accent_col}; background-color: {theme_color('surface')}; font-family: {mono}; font-size: 12px; }}
            pre {{ margin: 0; color: {text_col}; font-family: {mono}; font-size: 12px; }}
            pre code {{ background-color: transparent; color: {text_col}; }}
            a {{ color: {accent_col}; text-decoration: none; font-weight: 500; }}
            table {{ margin-top: 10px; margin-bottom: 14px; border-collapse: collapse; }}
            th, td {{ padding: 8px 12px; border: 1px solid {border}; text-align: left; }}
            th {{ background-color: {theme_color('surface')}; }}
        """

    # ── Outline ─────────────────────────────────────────────────────────────────

    def _populate_categories(self):
        self.cat_combo.clear()
        self.cat_combo.addItem("All Categories", "all")
        for cat in self._categories(ARTICLES):
            self.cat_combo.addItem(cat, cat)

    @staticmethod
    def _categories(articles) -> list[str]:
        """Categories in the order their first article appears (the docs' reading order)."""
        seen: list[str] = []
        for art in articles:
            if art.category not in seen:
                seen.append(art.category)
        return seen

    def _populate_articles(self):
        """Rebuild the outline: category → article → section, for the articles matching the filters."""
        query = self.search_input.text().strip()
        cat = self.cat_combo.currentData()
        matched = search_articles(query)
        if cat and cat != "all":
            matched = [art for art in matched if art.category == cat]

        self.toc.blockSignals(True)
        self.toc.clear()
        category_font = _resized(self.toc.font(), 0.88)
        category_font.setBold(True)
        category_font.setCapitalization(QFont.Capitalization.AllUppercase)
        category_font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 106)
        section_font = _resized(self.toc.font(), 0.95)
        muted, section_color = QColor(theme_color("muted")), QColor(theme_color("comment"))

        for category in self._categories(matched):
            cat_item = QTreeWidgetItem([category])
            cat_item.setData(0, Qt.ItemDataRole.UserRole, (CATEGORY, None, None))
            cat_item.setFont(0, category_font)
            cat_item.setForeground(0, muted)
            cat_item.setFlags(Qt.ItemFlag.ItemIsEnabled)  # a header: toggles, isn't selected
            self.toc.addTopLevelItem(cat_item)
            for art in (a for a in matched if a.category == category):
                art_item = QTreeWidgetItem([art.title])
                art_item.setData(0, Qt.ItemDataRole.UserRole, (ARTICLE, art.id, None))
                art_item.setToolTip(0, f"{art.title}\n{art.summary}")
                cat_item.addChild(art_item)
                for i, section in enumerate(article_sections(art)):
                    sec_item = QTreeWidgetItem([section])
                    sec_item.setData(0, Qt.ItemDataRole.UserRole, (SECTION, art.id, f"sec-{i}"))
                    sec_item.setFont(0, section_font)
                    sec_item.setForeground(0, section_color)
                    sec_item.setToolTip(0, section)
                    art_item.addChild(sec_item)
                art_item.setExpanded(art.id in self._expanded_articles)
            # Searching shows every match; otherwise headers stay as the user left them
            cat_item.setExpanded(bool(query) or category not in self._collapsed_categories)

        if self.toc.topLevelItemCount() == 0:
            empty = QTreeWidgetItem(["No matching topics found"])
            empty.setFlags(Qt.ItemFlag.NoItemFlags)
            self.toc.addTopLevelItem(empty)
        self.toc.blockSignals(False)
        self._mark_current_article()

    def article_items(self) -> list[QTreeWidgetItem]:
        """The outline's article entries, in order."""
        items = []
        for i in range(self.toc.topLevelItemCount()):
            cat_item = self.toc.topLevelItem(i)
            items.extend(cat_item.child(j) for j in range(cat_item.childCount()))
        return items

    def _remember_expansion(self, item: QTreeWidgetItem, expanded: bool):
        kind, article_id, _ = item.data(0, Qt.ItemDataRole.UserRole) or (None, None, None)
        if kind == CATEGORY:
            (self._collapsed_categories.discard if expanded else self._collapsed_categories.add)(item.text(0))
        elif kind == ARTICLE:
            (self._expanded_articles.add if expanded else self._expanded_articles.discard)(article_id)

    def set_all_expanded(self, expanded: bool):
        """Expand all / Collapse all: every category and every article."""
        for i in range(self.toc.topLevelItemCount()):
            cat_item = self.toc.topLevelItem(i)
            cat_item.setExpanded(expanded)  # the expand/collapse signals record each change
            for j in range(cat_item.childCount()):
                cat_item.child(j).setExpanded(expanded)

    def _mark_current_article(self):
        """Select the open article in the outline. Expansion is left as the user set it."""
        for item in self.article_items():
            if item.data(0, Qt.ItemDataRole.UserRole)[1] == self.current_article_id:
                self.toc.setCurrentItem(item)
                self.toc.scrollToItem(item)
                break
        self._active_section = None
        self._sync_outline_to_scroll()

    # ── Following the reading position ──────────────────────────────────────────

    def eventFilter(self, watched, event):
        if watched is self.browser.viewport() and event.type() == QEvent.Type.Resize:
            self._pad_document_end()
        return super().eventFilter(watched, event)

    def _pad_document_end(self):
        """Leave just enough room after the last line for the last section to scroll to the top of the
        view, and no more: the page then stops scrolling exactly there. Articles whose last section
        already reaches the top get no room at all."""
        doc = self.browser.document()
        frame = doc.rootFrame()
        fmt = frame.frameFormat()
        self._section_tops()  # measures the last heading's scroll position
        target = self._last_heading_scroll
        room = 0.0
        if target is not None:
            content_end = doc.documentLayout().documentSize().height() - fmt.bottomMargin()
            room = max(0.0, target + self.browser.viewport().height() - content_end)
        if abs(fmt.bottomMargin() - room) >= 1:
            fmt.setBottomMargin(room)
            frame.setFrameFormat(fmt)

    def _forget_anchor_tops(self, *_):
        self._anchor_tops = None  # re-measured on the next scroll: the text re-flowed

    def _section_tops(self) -> list[tuple[float, str]]:
        """(y, anchor) for each section heading in the open article, top to bottom. Also notes where
        the view scrolls to when the last heading is put at its top."""
        if self._anchor_tops is None:
            self._last_heading_scroll = None
            doc = self.browser.document()
            layout = doc.documentLayout()
            tops: dict[str, float] = {}
            block = doc.begin()
            while block.isValid():
                it = block.begin()
                while not it.atEnd():
                    fragment = it.fragment()
                    if fragment.isValid():
                        for name in fragment.charFormat().anchorNames():
                            if name.startswith("sec-") and name not in tops:
                                tops[name] = layout.blockBoundingRect(block).top()
                                self._last_heading_scroll = tops[name]  # scrollToAnchor aligns the block's top
                    it += 1
                block = block.next()
            self._anchor_tops = sorted((y, name) for name, y in tops.items())
        return self._anchor_tops

    def current_section(self) -> str | None:
        """The anchor of the section being read: the last heading at or above the top of the view.
        (The room after the last line lets every heading get there, so the line can stay fixed.)"""
        line = self.browser.verticalScrollBar().value() + 24  # a heading just below the top edge counts
        current = None
        for y, name in self._section_tops():
            if y <= line:
                current = name
        return current

    def _sync_outline_to_scroll(self, *_):
        """Select the section being read (or the article, above its first section). This is the only
        indicator, so the selected entry is always where the page is."""
        anchor = self.current_section()
        if anchor == self._active_section:
            return
        self._active_section = anchor
        article = next((i for i in self.article_items()
                        if i.data(0, Qt.ItemDataRole.UserRole)[1] == self.current_article_id), None)
        if article is None:
            return
        target = next((article.child(j) for j in range(article.childCount())
                       if article.child(j).data(0, Qt.ItemDataRole.UserRole)[2] == anchor), article)
        if not target.isHidden() and (target is article or article.isExpanded()):
            self.toc.setCurrentItem(target)
            self.toc.scrollToItem(target)
        else:
            self.toc.setCurrentItem(article)  # its sections are folded away: the article stands for them

    def _on_outline_clicked(self, item: QTreeWidgetItem, _column: int = 0):
        kind, article_id, anchor = item.data(0, Qt.ItemDataRole.UserRole) or (None, None, None)
        if kind == CATEGORY:
            item.setExpanded(not item.isExpanded())
        elif kind == ARTICLE:
            if article_id == self.current_article_id:
                item.setExpanded(not item.isExpanded())  # a second click folds its sections
            else:
                self.select_article(article_id)
        elif kind == SECTION:
            if article_id != self.current_article_id:
                self.select_article(article_id)
            self.browser.scrollToAnchor(anchor)

    def _first_article_id(self):
        items = self.article_items()
        return items[0].data(0, Qt.ItemDataRole.UserRole)[1] if items else None

    def _on_search_changed(self, text: str):
        self._populate_articles()
        self._open_a_match()

    def _on_filter_changed(self, index: int):
        self._populate_articles()
        self._open_a_match()

    def _open_a_match(self):
        """Keep the open article if it still matches the search and filter; otherwise open the first match."""
        ids = [item.data(0, Qt.ItemDataRole.UserRole)[1] for item in self.article_items()]
        if ids and self.current_article_id not in ids:
            self.select_article(ids[0])

    # ── Article ─────────────────────────────────────────────────────────────────

    def select_article(self, article_id: str):
        article: DocArticle | None = next((art for art in ARTICLES if art.id == article_id), None)
        if not article:
            return
        if article_id != self.current_article_id or article_id not in self._expanded_articles:
            self._expanded_articles.add(article_id)  # opening an article unfolds its sections
            self._collapsed_categories.discard(article.category)
            self._set_expanded_in_outline(article_id)
        self.current_article_id = article_id

        self.badge_cat.setText(f"{article.category}  ›  {article.title}")
        self.title_lbl.setText(article.title)
        self.summary_lbl.setText(article.summary)

        self.browser.document().setDefaultStyleSheet(self._article_css())
        self.browser.setHtml(f"<html><body>{article_body(article)}</body></html>")
        self._anchor_tops = None  # the new article's headings; measured by the padding below
        self._pad_document_end()  # setHtml resets the root frame's format
        self._mark_current_article()

    def _set_expanded_in_outline(self, article_id: str):
        for item in self.article_items():
            if item.data(0, Qt.ItemDataRole.UserRole)[1] == article_id:
                item.parent().setExpanded(True)
                item.setExpanded(True)

    def _on_anchor_clicked(self, url: QUrl):
        url_str = url.toString()
        if url_str.startswith("ide://"):
            self._navigate_ide(url_str.replace("ide://", ""))
        elif url.scheme() == DIAGRAM_SCHEME:
            self.open_diagram(url.path())
        elif url_str.startswith("#"):
            self.browser.scrollToAnchor(url_str.lstrip("#"))  # internal anchor in the article
        elif any(art.id == url_str for art in ARTICLES):
            self.select_article(url_str)  # a link to another article

    def open_diagram(self, name: str) -> DiagramDialog | None:
        if diagram_svg(name) is None:
            return None
        dialog = DiagramDialog(name, self)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.show()
        return dialog

    # ── IDE navigation ──────────────────────────────────────────────────────────

    def _open_hotkeys(self):
        if self.context and hasattr(self.context, "open_settings"):
            self.context.open_settings("hotkeys")

    def _open_hub(self):
        if self.context and hasattr(self.context, "open_component_hub"):
            self.context.open_component_hub()

    def _navigate_ide(self, pane_id: str):
        if pane_id == "settings" and self.context and hasattr(self.context, "open_settings"):
            self.context.open_settings()  # Settings isn't a sidebar tab
        elif self.context and hasattr(self.context, "select_sidebar_tab"):
            self.context.select_sidebar_tab(pane_id)

    def cleanup(self):
        """Clean up resources on deactivation."""
        self.context = None
        self.browser.clear()
