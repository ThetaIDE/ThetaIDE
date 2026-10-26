"""Theta Hub Browser dialog for discovering and installing community components."""
from __future__ import annotations

from typing import Dict, List, Optional

from PyQt6.QtCore import QEvent, Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QDesktopServices
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..theme import current_theme, theme_color
from .client import HubClient
from .models import HubComponent

# Uninstall/remove are destructive in every theme, so they keep one red (with white text, which
# reads on it at 4.9:1) rather than borrowing a theme role that is orange or purple in some palettes.
DANGER, DANGER_HOVER, ON_DANGER = "#cc241d", "#e0352b", "#ffffff"

# The theme role that colors each kind's badge.
KIND_ROLES = {
    "plugin": "number",
    "suite": "number",
    "package": "number",
    "method": "primary",
    "model": "secondary",
    "env": "accent",
    "experiment": "focus",
}


def _luminance(color: str) -> float:
    def channel(c):
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    q = QColor(color)
    return 0.2126 * channel(q.redF()) + 0.7152 * channel(q.greenF()) + 0.0722 * channel(q.blueF())


def _contrast(a: str, b: str) -> float:
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _text_on(background: str) -> str:
    """The theme's base or text color, whichever reads better on `background`. Some palettes have
    mid-tone roles neither reaches 4.5:1 on (Catppuccin Latte's orange and green); black or white then."""
    best = max((theme_color("base"), theme_color("text")), key=lambda c: _contrast(c, background))
    if _contrast(best, background) >= 4.5:
        return best
    return max(("#000000", "#ffffff"), key=lambda c: _contrast(c, background))


def _hover(color: str) -> str:
    """`color` nudged away from the background: lighter on dark themes, darker on light ones."""
    q = QColor(color)
    dark_theme = QColor(theme_color("base")).lightnessF() < 0.5
    return (q.lighter(118) if dark_theme else q.darker(112)).name()


def _theme_key():
    """Identifies the active palette, so a restyle runs only when the theme really changed."""
    return tuple(sorted(current_theme()["colors"].items()))


def _filled_button_css(background: str, hover: str, foreground: str) -> str:
    return (
        "QPushButton {"
        f"  background-color: {background};"
        f"  color: {foreground};"
        f"  border: 1px solid {background};"
        "  border-radius: 4px;"
        "  font-weight: 700;"
        "  padding: 4px 14px;"
        "}"
        "QPushButton:hover {"
        f"  background-color: {hover};"
        f"  border-color: {hover};"
        "}"
        + _disabled_button_css()
    )


def _disabled_button_css():
    """Disabled look for the action buttons, in the current theme's muted colors."""
    return (
        "QPushButton:disabled {"
        f"  background-color: {theme_color('surface')};"
        f"  color: {theme_color('disabled')};"
        f"  border-color: {theme_color('raised')};"
        "}"
    )


class HubComponentCard(QFrame):
    """Card widget representing an individual component in the Hub."""

    def __init__(self, component: HubComponent, client: HubClient, parent=None):
        super().__init__(parent)
        self.component = component
        self.client = client
        self.setObjectName("card")
        self._styled_for = None  # the theme the colors below were last set for
        self._build_ui()

    def changeEvent(self, event):
        # A theme switch sets the application palette, which reaches every widget as a PaletteChange
        super().changeEvent(event)
        if (event.type() == QEvent.Type.PaletteChange and self._styled_for is not None  # built and styled
                and self._styled_for != _theme_key()):
            self._apply_theme()

    def _apply_theme(self):
        """Set every color on the card from the current theme. Runs when built and on theme switches."""
        self._styled_for = _theme_key()
        badge_bg = theme_color(KIND_ROLES.get(self.component.kind, "disabled"))
        self.kind_badge.setStyleSheet(
            f"background-color: {badge_bg}; color: {_text_on(badge_bg)}; font-size: 10px; "
            f"font-weight: 700; padding: 2px 6px; border-radius: 4px;"
        )
        self.version_label.setStyleSheet(f"color: {theme_color('comment')}; font-size: 12px;")
        self.author_label.setStyleSheet(f"color: {theme_color('muted')}; font-size: 11px;")
        self.desc_label.setStyleSheet(f"color: {theme_color('text')}; font-size: 12px; line-height: 1.4;")
        for tag_label in self.tag_labels:
            tag_label.setStyleSheet(f"color: {theme_color('secondary')}; font-size: 11px;")
        if self.btn_repo is not None:
            self.btn_repo.setStyleSheet(
                f"color: {theme_color('secondary')}; font-size: 11px; border: none; background: transparent;")
        self._style_action_button()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        # ── Top Row: Badge, Title, Version, Author & Action Button ───────────
        top_row = QHBoxLayout()
        top_row.setSpacing(8)

        # Kind badge (colored by _apply_theme, like everything below)
        self.kind_badge = QLabel(self.component.kind.upper())
        top_row.addWidget(self.kind_badge)

        # Title
        title_label = QLabel(self.component.name)
        title_label.setStyleSheet("font-size: 14px; font-weight: 700;")
        top_row.addWidget(title_label)

        # Version
        self.version_label = QLabel(f"v{self.component.version}")
        top_row.addWidget(self.version_label)

        # Author
        self.author_label = QLabel(f"by {self.component.author.name}")
        top_row.addWidget(self.author_label)

        top_row.addStretch()

        # Action Button
        self.btn_action = QPushButton()
        self.btn_action.setFixedHeight(28)
        self.btn_action.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_action.clicked.connect(self._on_action_clicked)
        top_row.addWidget(self.btn_action)

        # An installed component with an available update still needs an uninstall
        # path. Keep Update as the primary action and expose Uninstall alongside it.
        self.btn_uninstall_update = QPushButton("Uninstall")
        self.btn_uninstall_update.setObjectName("actionUninstall")
        self.btn_uninstall_update.setFixedHeight(28)
        self.btn_uninstall_update.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_uninstall_update.setToolTip("Uninstall this component without updating it")
        self.btn_uninstall_update.clicked.connect(self._on_uninstall_clicked)
        top_row.addWidget(self.btn_uninstall_update)

        layout.addLayout(top_row)

        # ── Description ──────────────────────────────────────────────────────
        self.desc_label = QLabel(self.component.description)
        self.desc_label.setWordWrap(True)
        layout.addWidget(self.desc_label)

        # ── Bottom Row: Tags & Links ─────────────────────────────────────────
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(6)

        self.tag_labels = [QLabel(f"#{tag}") for tag in self.component.tags[:5]]
        for tag_label in self.tag_labels:
            bottom_row.addWidget(tag_label)

        bottom_row.addStretch()

        self.btn_repo = None
        if self.component.repository:
            self.btn_repo = QToolButton()
            self.btn_repo.setText("GitHub")
            self.btn_repo.setCursor(Qt.CursorShape.PointingHandCursor)
            self.btn_repo.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(self.component.repository)))
            bottom_row.addWidget(self.btn_repo)

        layout.addLayout(bottom_row)

        # ── Download Progress Bar (hidden by default) ────────────────────────
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        self.update_action_state()  # text and role for the install state
        self._apply_theme()

    def update_action_state(self):
        """Set the action button's text for the local installation state, and color it."""
        if self.component.is_installed and self.component.has_update:
            self.btn_action.setText(f"Update to v{self.component.version}")
            self.btn_action.setObjectName("actionUpdate")
        elif self.component.is_installed:
            self.btn_action.setText("Uninstall")
            self.btn_action.setObjectName("actionUninstall")
        else:
            self.btn_action.setText("Install")
            self.btn_action.setObjectName("actionInstall")
        self.btn_action.setEnabled(True)
        self.btn_uninstall_update.setVisible(bool(self.component.is_installed and self.component.has_update))
        self.btn_uninstall_update.setEnabled(True)
        self._style_action_button()

    def _style_action_button(self):
        """Color the action button for its role. Leaves its text and enabled state alone, so a theme
        switch during an install doesn't reset "Installing…" or re-enable the button."""
        role = self.btn_action.objectName()
        if role == "actionUninstall":
            css = _filled_button_css(DANGER, DANGER_HOVER, ON_DANGER)
        elif role == "actionUpdate":
            update = theme_color("secondary")
            css = _filled_button_css(update, _hover(update), _text_on(update))
        else:
            install = theme_color("primary")
            css = _filled_button_css(install, theme_color("primary_hover"), _text_on(install))
        self.btn_action.setStyleSheet(css)
        self.btn_action.style().unpolish(self.btn_action)
        self.btn_action.style().polish(self.btn_action)
        danger_css = _filled_button_css(DANGER, DANGER_HOVER, ON_DANGER)
        self.btn_uninstall_update.setStyleSheet(danger_css)
        self.btn_uninstall_update.style().unpolish(self.btn_uninstall_update)
        self.btn_uninstall_update.style().polish(self.btn_uninstall_update)

    def _on_action_clicked(self):
        self.btn_action.setEnabled(False)
        self.btn_uninstall_update.setEnabled(False)
        if self.component.is_installed and not self.component.has_update:
            self._prompt_and_uninstall()
        else:
            self.btn_action.setText("Installing…")
            self.progress_bar.setValue(0)
            self.progress_bar.show()
            self.client.install_async(self.component)

    def _on_uninstall_clicked(self):
        self.btn_action.setEnabled(False)
        self.btn_uninstall_update.setEnabled(False)
        self._prompt_and_uninstall()

    def _prompt_and_uninstall(self):
        dlg = UninstallComponentDialog(self.component.name, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            self.update_action_state()
            return
        self.btn_action.setText("Uninstalling…")
        self.client.uninstall_async(self.component)

    def set_progress(self, percent: int):
        self.progress_bar.setValue(percent)


class UninstallComponentDialog(QDialog):
    """Confirm uninstall; component configs are hidden and restored on reinstall."""

    def __init__(self, component_name: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Uninstall Component")
        self.setFixedWidth(430)
        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel(f"Uninstall {component_name}?")
        title.setStyleSheet(f"font-size: 15px; font-weight: bold; color: {theme_color('text')};")
        layout.addWidget(title)

        desc = QLabel(
            "Its configuration files will be hidden while the component is uninstalled. "
            "Your edits and copied profiles will be restored if you install it again."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {theme_color('muted')}; font-size: 12px; line-height: 1.4;")
        layout.addWidget(desc)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self._on_cancel)
        btn_row.addWidget(btn_cancel)

        btn_row.addStretch()

        btn_uninstall = QPushButton("Uninstall")
        btn_uninstall.setStyleSheet(f"background-color: {DANGER}; color: {ON_DANGER}; font-weight: bold;")
        btn_uninstall.clicked.connect(self.accept)
        btn_row.addWidget(btn_uninstall)

        layout.addLayout(btn_row)

    def _on_cancel(self):
        self.reject()


class HubView(QWidget):
    """Full-featured marketplace view for exploring and installing components."""
    close_requested = pyqtSignal()

    def __init__(
        self,
        client: HubClient,
        initial_kind: str | None = None,
        mode: str = "components",  # "components" (all package kinds) | "plugins" (only plugins) | "all"
        close_button_text: str = "Back to Parameters",
        show_footer: bool = True,
        title: str | None = None,
        include_plugins: bool = True,
        all_label: str = "All Components",
        parent=None,
    ):
        super().__init__(parent)
        self.client = client
        self.mode = mode
        self.active_kind: str | None = "plugin" if mode == "plugins" else initial_kind
        self.close_button_text = close_button_text
        self.show_footer = show_footer
        self.title = title
        self.include_plugins = include_plugins
        self.all_label = all_label
        self.cards: Dict[str, HubComponentCard] = {}
        self.search_input = None
        self._styled_for = None

        self._build_ui()
        self._apply_theme()
        self._wire_client()

        # Load components
        self.client.fetch_index_async()

    def changeEvent(self, event):
        super().changeEvent(event)  # cards restyle themselves; this covers the header and footer
        if (event.type() == QEvent.Type.PaletteChange and self._styled_for is not None  # built and styled
                and self._styled_for != _theme_key()):
            self._apply_theme()

    def _apply_theme(self):
        self._styled_for = _theme_key()
        self.sort_lbl.setStyleSheet(f"color: {theme_color('muted')}; font-size: 11px;")
        if self.title_label is not None:
            self.title_label.setStyleSheet(
                f"color: {theme_color('text')}; font-size: 24px; font-weight: bold;"
            )
        if self.status_label is not None:
            self.status_label.setStyleSheet(f"color: {theme_color('comment')}; font-size: 11px;")

    def _matches_mode(self, comp: HubComponent) -> bool:
        """Hide package internals while filtering standalone packages by view."""
        # Package-internal components are available to dependency resolution but
        # are installed through their containing package.
        if comp.is_grouped_component or comp.hidden:
            return False

        if not self.include_plugins and comp.kind == "plugin":
            return False

        if self.mode == "plugins":
            return comp.kind == "plugin"
        return True

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 12, 16, 14)
        main_layout.setSpacing(12)

        self.title_label = None
        if self.title:
            self.title_label = QLabel(self.title)
            main_layout.addWidget(self.title_label)

        # ── Top Bar: Organizational Tabs & Sort / Refresh Controls ───────────
        top_bar = QHBoxLayout()
        top_bar.setSpacing(6)

        self.pills = {}
        if self.mode == "components":
            kind_options = [
                ("all", self.all_label),
                ("experiment", "Experiments"),
                ("suite", "Suites"),
                ("method", "Algorithms"),
                ("model", "Models"),
                ("env", "Environments"),
                ("data", "Data Loaders"),
                ("eval", "Evaluators"),
                ("plot", "Plotters"),
            ]
            if self.include_plugins:
                kind_options.insert(1, ("plugin", "Plugins"))
        elif self.mode == "plugins":
            kind_options = [
                ("plugin", "All Plugins"),
            ]
        else:
            kind_options = [
                ("all", "All"),
                ("experiment", "Experiments"),
                ("plugin", "Plugins"),
                ("suite", "Suites"),
                ("method", "Algorithms"),
                ("model", "Models"),
                ("env", "Environments"),
                ("data", "Data Loaders"),
                ("eval", "Evaluators"),
                ("plot", "Plotters"),
            ]

        for kind_id, label_text in kind_options:
            btn = QPushButton(label_text)
            btn.setCheckable(True)
            default_kind = "plugin" if self.mode == "plugins" else "all"
            if (self.active_kind or default_kind) == kind_id:
                btn.setChecked(True)
            btn.clicked.connect(lambda chk, k=kind_id: self._select_kind(k))
            self.pills[kind_id] = btn
            top_bar.addWidget(btn)

        top_bar.addStretch()

        self.sort_lbl = QLabel("Sort:")
        top_bar.addWidget(self.sort_lbl)

        self.sort_combo = QComboBox()
        self.sort_combo.addItem("Default", "default")
        self.sort_combo.addItem("Uninstalled first", "uninstalled_first")
        self.sort_combo.addItem("Installed first", "installed_first")
        self.sort_combo.addItem("Name (A–Z)", "name_asc")
        self.sort_combo.setToolTip("Sort components list")
        self.sort_combo.currentIndexChanged.connect(self._apply_filters)
        top_bar.addWidget(self.sort_combo)

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.setToolTip("Reload registry from GitHub")
        self.btn_refresh.clicked.connect(lambda: self.client.fetch_index_async(force=True))
        top_bar.addWidget(self.btn_refresh)

        main_layout.addLayout(top_bar)

        # ── Components Scroll Area ───────────────────────────────────────────
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)

        self.card_container = QWidget()
        self.card_layout = QVBoxLayout(self.card_container)
        self.card_layout.setContentsMargins(0, 4, 6, 4)
        self.card_layout.setSpacing(10)
        self.card_layout.addStretch()

        self.scroll_area.setWidget(self.card_container)
        main_layout.addWidget(self.scroll_area, 1)

        # The embedded Component Hub uses the panel controls to switch views,
        # so it omits the dialog footer entirely.
        self.status_label = None
        self.btn_close = None
        if self.show_footer:
            footer = QHBoxLayout()
            self.status_label = QLabel("Loading community registry…")
            footer.addWidget(self.status_label, 1)

            self.btn_close = QPushButton(self.close_button_text)
            self.btn_close.clicked.connect(self.close_requested.emit)
            footer.addWidget(self.btn_close)

            main_layout.addLayout(footer)

    def _set_status(self, text: str):
        if self.status_label is not None:
            self.status_label.setText(text)

    def _wire_client(self):
        self.client.indexLoaded.connect(self._on_index_loaded)
        self.client.fetchFailed.connect(self._on_fetch_failed)
        self.client.installProgress.connect(self._on_install_progress)
        self.client.installFinished.connect(self._on_install_finished)
        self.client.uninstallFinished.connect(self._on_uninstall_finished)

    def _select_kind(self, kind_id: str):
        if kind_id not in self.pills:
            return
        if self.mode == "plugins":
            self.active_kind = "plugin"
        else:
            self.active_kind = None if kind_id == "all" else kind_id

        for kid, btn in self.pills.items():
            btn.blockSignals(True)
            btn.setChecked(kid == kind_id)
            btn.blockSignals(False)
        self._apply_filters()

    def _on_index_loaded(self, components: List[HubComponent]):
        filtered = [c for c in components if self._matches_mode(c)]
        item_name = "plugins" if self.mode == "plugins" else "components"
        self._set_status(f"Registry ready · {len(filtered)} {item_name} available")
        self._populate_cards(components)

    def _on_fetch_failed(self, error: str):
        self._set_status(f"Using local cache (Network notice: {error})")
        self._populate_cards(self.client.components)

    def _populate_cards(self, components: List[HubComponent]):
        # Remove existing cards
        for card in self.cards.values():
            self.card_layout.removeWidget(card)
            card.deleteLater()
        self.cards.clear()

        # Build cards for components matching current mode
        for comp in components:
            if not self._matches_mode(comp):
                continue
            card = HubComponentCard(comp, self.client, self.card_container)
            self.cards[comp.id] = card
            # Insert before the trailing stretch spacer
            self.card_layout.insertWidget(self.card_layout.count() - 1, card)

        self._apply_filters()

    def _apply_filters(self):
        query = self.search_input.text() if self.search_input is not None else ""
        target_kind = "plugin" if self.mode == "plugins" else self.active_kind
        matched = list(self.client.search(query=query, kind=target_kind))
        # Enforce mode filter
        matched = [c for c in matched if self._matches_mode(c)]

        sort_mode = self.sort_combo.currentData() if hasattr(self, "sort_combo") else "default"
        if sort_mode == "uninstalled_first":
            matched.sort(key=lambda c: (c.is_installed, c.name.lower()))
        elif sort_mode == "installed_first":
            matched.sort(key=lambda c: (not c.is_installed, c.name.lower()))
        elif sort_mode == "name_asc":
            matched.sort(key=lambda c: c.name.lower())

        visible_count = 0
        matched_ids = {c.id for c in matched}

        # Re-pack layout in sorted order
        for comp in matched:
            card = self.cards.get(comp.id)
            if card:
                self.card_layout.removeWidget(card)
                self.card_layout.insertWidget(visible_count, card)
                card.setVisible(True)
                visible_count += 1

        for comp_id, card in self.cards.items():
            if comp_id not in matched_ids:
                card.setVisible(False)

        item_name = "plugins" if self.mode == "plugins" else "components"
        total_mode_count = len([c for c in self.client.components if self._matches_mode(c)])
        self._set_status(f"Showing {visible_count} of {total_mode_count} {item_name}")

    def _on_install_progress(self, comp_id: str, percent: int):
        if comp_id in self.cards:
            self.cards[comp_id].set_progress(percent)

    def _on_install_finished(self, comp_id: str, success: bool, message: str):
        if comp_id in self.cards:
            card = self.cards[comp_id]
            card.progress_bar.hide()
            card.update_action_state()
            if success:
                self._set_status(f"Installed {card.component.name} successfully.")
            else:
                self._set_status(f"Installation failed: {message}")
        for other_card in self.cards.values():
            other_card.update_action_state()
        self._apply_filters()

    def _on_uninstall_finished(self, comp_id: str, success: bool, message: str):
        if comp_id in self.cards:
            card = self.cards[comp_id]
            card.update_action_state()
            if success:
                self._set_status(f"Uninstalled {card.component.name}.")
            else:
                self._set_status(f"Uninstall failed: {message}")
        for other_card in self.cards.values():
            other_card.update_action_state()
        self._apply_filters()


class HubDialog(QDialog):
    """Full-featured marketplace dialog for exploring and installing components or plugins."""

    def __init__(
        self,
        client: HubClient,
        initial_kind: str | None = None,
        mode: str = "plugins",  # "plugins" for Community Plugins | "components" for Component Hub
        parent=None,
    ):
        super().__init__(parent)
        self.mode = mode
        title = "Theta Community Plugins" if mode == "plugins" else "Theta Community Hub"
        self.setWindowTitle(title)
        self.resize(860, 680)
        self.setMinimumSize(700, 520)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.hub_view = HubView(
            client=client,
            initial_kind=initial_kind,
            mode=mode,
            close_button_text="Close",
            parent=self,
        )
        self.hub_view.close_requested.connect(self.accept)
        layout.addWidget(self.hub_view)

    @property
    def client(self):
        return self.hub_view.client

    @property
    def cards(self):
        return self.hub_view.cards

    @property
    def pills(self):
        return self.hub_view.pills

    @property
    def sort_combo(self):
        return self.hub_view.sort_combo

    @property
    def card_layout(self):
        return self.hub_view.card_layout

    @property
    def search_input(self):
        return self.hub_view.search_input

    @property
    def _on_index_loaded(self):
        return self.hub_view._on_index_loaded

    def _populate_cards(self, components: List[HubComponent]):
        return self.hub_view._populate_cards(components)

    def _apply_filters(self):
        return self.hub_view._apply_filters()

    def __getattr__(self, name):
        if name == "hub_view" or "hub_view" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.hub_view, name)
