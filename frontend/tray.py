"""System tray icon: closing the window while training hides ThetaIDE to the tray so runs keep going.

Training runs are children of the app (QProcess), so they only survive the window closing if the app
itself keeps running. The tray keeps it running, shows what is training, notifies when runs finish, and
offers Show / Stop all training / Quit.
"""
from __future__ import annotations

from PyQt6.QtCore import QObject, QRect, QRectF, Qt
from PyQt6.QtGui import QAction, QColor, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from .app_icon import app_icon

FINAL_STATUSES = ("completed", "failed", "cancelled", "error", "interrupted")
# Badge in the icon's bottom-right corner: green while a run is training, blue while the queue waits to start
# the next job. No badge when idle.
BADGE_COLORS = {"training": "#2ea043", "queued": "#3b82f6"}
ICON_SIZES = (16, 20, 24, 32, 40, 48, 64)


def badged_icon(base: QIcon, color: str) -> QIcon:
    """`base` with a status dot in the bottom-right corner, at every tray size.

    A transparent gap is cut around the dot rather than drawing a rim, so the dot stays round and separate
    from the logo's rounded corner at 16 px and reads on light and dark taskbars alike. The dot is sized and
    placed in whole pixels so small sizes stay crisp.
    """
    icon = QIcon()
    for size in ICON_SIZES:
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        base.paint(painter, QRect(0, 0, size, size))
        diameter = 2 * round(size * 0.22)  # even, so the centre falls on a pixel boundary
        gap = max(1, round(size / 16))
        dot = QRectF(size - diameter, size - diameter, diameter, diameter)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        painter.setBrush(QColor(0, 0, 0))
        painter.drawEllipse(dot.adjusted(-gap, -gap, gap, gap))
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        painter.setBrush(QColor(color))
        painter.drawEllipse(dot)
        painter.end()
        icon.addPixmap(pixmap)
    return icon


class TrayController(QObject):
    def __init__(self, window, parent=None):
        super().__init__(parent or window)
        self.window = window
        base = app_icon()
        self._icons = {"idle": base, **{state: badged_icon(base, color) for state, color in BADGE_COLORS.items()}}
        self.indicator = "idle"
        self.icon = QSystemTrayIcon(base, self)
        self.icon.activated.connect(self._on_activated)

        self.menu = QMenu()
        self.show_action = QAction("Show ThetaIDE", self.menu)
        self.show_action.triggered.connect(self.show_window)
        self.menu.addAction(self.show_action)
        self.menu.addSeparator()
        self.status_action = QAction("No training running", self.menu)
        self.status_action.setEnabled(False)
        self.menu.addAction(self.status_action)
        self.stop_action = QAction("Stop all training", self.menu)
        self.stop_action.triggered.connect(self.stop_all)
        self.menu.addAction(self.stop_action)
        self.menu.addSeparator()
        quit_action = QAction("Quit ThetaIDE", self.menu)
        quit_action.triggered.connect(self.window.quit_app)
        self.menu.addAction(quit_action)
        self.icon.setContextMenu(self.menu)

        window.jobs.job_changed.connect(self._on_job_changed)
        window.jobs.queue_changed.connect(lambda _state: self.refresh())
        self.refresh()
        if self.available():
            self.icon.show()

    @staticmethod
    def available() -> bool:
        return QSystemTrayIcon.isSystemTrayAvailable()

    def keeps_running(self) -> bool:
        """Closing the window should hide it here: something is training, or the queue will start more."""
        jobs = self.window.jobs
        return self.available() and (jobs.is_busy() or (jobs.queue_running and bool(jobs.queue_order)))

    # ── Window ───────────────────────────────────────────────────────────────

    def hide_to_tray(self):
        self.window.hide()
        running = self._running_names()
        what = ", ".join(running) if running else "the job queue"
        self.icon.showMessage("ThetaIDE is still training",
                              f"{what} keeps running in the background. Click the tray icon to reopen ThetaIDE.",
                              QSystemTrayIcon.MessageIcon.Information, 6000)

    def show_window(self):
        self.window.showNormal()
        self.window.raise_()
        self.window.activateWindow()

    def notify(self, title: str, body: str, warning: bool = False):
        """Show one run-completion notification through the system tray."""
        if not self.available():
            return
        icon = (QSystemTrayIcon.MessageIcon.Warning if warning
                else QSystemTrayIcon.MessageIcon.Information)
        self.icon.showMessage(title, body, icon, 8000)

    def _on_activated(self, reason):
        if reason in (QSystemTrayIcon.ActivationReason.Trigger, QSystemTrayIcon.ActivationReason.DoubleClick):
            self.show_window()

    def stop_all(self):
        jobs = self.window.jobs
        jobs.pause_queue()
        for job in jobs.active_jobs():
            jobs.cancel(job["job_id"])

    # ── Status ───────────────────────────────────────────────────────────────

    def _running_names(self) -> list[str]:
        return [job["experiment_id"] for job in self.window.jobs.active_jobs()]

    def refresh(self):
        jobs = self.window.jobs
        running = self._running_names()
        queued = len(jobs.queue_order)
        if running:
            status = f"Training: {', '.join(running)}"
        elif queued:
            status = f"{queued} queued" + ("" if jobs.queue_running else " (queue paused)")
        else:
            status = "No training running"
        if running and queued:
            status += f" · {queued} queued"
        self.status_action.setText(status)
        self.stop_action.setEnabled(bool(running) or (jobs.queue_running and bool(queued)))
        self.icon.setToolTip(f"ThetaIDE · {status}")
        indicator = "training" if running else ("queued" if jobs.queue_running and queued else "idle")
        if indicator != self.indicator:
            self.indicator = indicator
            self.icon.setIcon(self._icons[indicator])

    def _on_job_changed(self, job):
        self.refresh()
        status = job.get("status")
        if status not in FINAL_STATUSES or self.window.isVisible():
            return
        run = self.window.run_for_job(job.get("job_id"))
        notifier = getattr(self.window, "run_notifier", None)
        if run is not None and notifier is not None and notifier.has_notified(run):
            return
        name = job.get("experiment_id", "A run")
        if status == "completed":
            title, icon = f"{name} finished", QSystemTrayIcon.MessageIcon.Information
            body = "Training completed. Open ThetaIDE to see the results."
        elif status == "cancelled":
            title, icon, body = f"{name} stopped", QSystemTrayIcon.MessageIcon.Information, "The run was stopped."
        else:
            code = job.get("returncode")
            title, icon = f"{name} failed", QSystemTrayIcon.MessageIcon.Warning
            body = f"Training ended with exit code {code}." if code is not None else "Training could not run."
        self.icon.showMessage(title, body, icon, 8000)

    def remove(self):
        """Take the icon out of the tray (the app is quitting)."""
        self.icon.hide()
        QApplication.processEvents()
