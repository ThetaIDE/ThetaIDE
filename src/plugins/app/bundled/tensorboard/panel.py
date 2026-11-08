"""Native server management card for TensorBoard without Chromium webview overhead."""
from __future__ import annotations
from typing import Callable, Optional

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from frontend.theme import theme_color
from frontend.widgets import label

from .server import TensorBoardServer


class TensorBoardPanel(QWidget):
    """Native server management card for TensorBoard without Chromium webview overhead."""

    def __init__(self, server: TensorBoardServer, log: Optional[Callable[[str], None]] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.server = server
        self.log = log or (lambda msg: None)
        self.url: Optional[str] = None
        self.server.starting.connect(lambda: self.set_ui_state(starting=True))
        self.server.ready.connect(self._on_ready)
        self.server.stopped.connect(self._on_stopped)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(16)

        header = QVBoxLayout()
        header.setSpacing(4)
        title = label("TensorBoard", "heading")
        subtitle = label(
            "Inspect binary training event logs, computational graphs, and histograms in full fidelity.",
            "muted"
        )
        header.addWidget(title)
        header.addWidget(subtitle)
        layout.addLayout(header)

        # Main Server Control Card
        card = QFrame()
        card.setObjectName("card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(20, 20, 20, 20)
        card_layout.setSpacing(14)

        status_row = QHBoxLayout()
        self.status_badge = label("● Stopped", "muted")
        self.server_url_label = label("Server offline", "muted")
        status_row.addWidget(self.status_badge)
        status_row.addSpacing(12)
        status_row.addWidget(self.server_url_label)
        status_row.addStretch()
        card_layout.addLayout(status_row)

        self.info_text = label(
            "ThetaIDE runs TensorBoard over results/tensorboard/.\n"
            "Runs configured with 'Log to TensorBoard' write event files here.",
            "muted"
        )
        self.info_text.setWordWrap(True)
        card_layout.addWidget(self.info_text)

        button_row = QHBoxLayout()
        self.start_btn = QPushButton("Start server")
        self.start_btn.clicked.connect(self.server.start)
        self.stop_btn = QPushButton("Stop server")
        self.stop_btn.clicked.connect(self.stop)
        self.browser_btn = QPushButton("Open in browser")
        self.browser_btn.clicked.connect(self.open_in_browser)
        self.browser_btn.setEnabled(False)

        button_row.addWidget(self.start_btn)
        button_row.addWidget(self.stop_btn)
        button_row.addWidget(self.browser_btn)
        button_row.addStretch()
        card_layout.addLayout(button_row)

        layout.addWidget(card)
        layout.addStretch(1)

        self.set_ui_state(running=False)

    def activate(self) -> None:
        """Called when the tab is shown."""
        if self.server.url:
            self.set_ui_state(running=True)
        elif self.server.is_running():
            self.set_ui_state(starting=True)
        elif not self.server.available():
            self.set_ui_state(running=False, error="tensorboard is not installed in this environment")

    def _on_ready(self, url: str) -> None:
        self.url = url
        self.log(f"TensorBoard running at {url}")
        self.set_ui_state(running=True)

    def _on_stopped(self, error: str) -> None:
        self.url = None
        if error:
            self.log(f"TensorBoard stopped: {error}")
        self.set_ui_state(running=False, error=error or None)

    def set_ui_state(self, running: bool = False, starting: bool = False, error: Optional[str] = None) -> None:
        if starting:
            self.status_badge.setText("● Starting…")
            self.status_badge.setStyleSheet(f"color: {theme_color('comment')}; font-weight: bold;")
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.browser_btn.setEnabled(False)
        elif running:
            self.status_badge.setText("● Running")
            self.status_badge.setStyleSheet(f"color: {theme_color('primary')}; font-weight: bold;")
            self.server_url_label.setText(self.url or "")
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.browser_btn.setEnabled(True)
        else:
            self.status_badge.setText("● Stopped")
            self.status_badge.setStyleSheet(f"color: {theme_color('muted')};")
            self.server_url_label.setText("Server offline" if not error else f"Error: {error}")
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.browser_btn.setEnabled(False)

    def stop(self) -> None:
        self.server.stop()
        self.url = None
        self.set_ui_state(running=False)

    def open_in_browser(self) -> None:
        if self.url:
            QDesktopServices.openUrl(QUrl(self.url))
