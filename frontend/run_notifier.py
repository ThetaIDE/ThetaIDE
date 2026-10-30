"""When a run ends: a toast (a system notification when ThetaIDE is not in front) and a short chime.

Every ending notifies: completed, failed and stopped, each with its own mark, colour and sound. Settings →
Training pipeline → Run notifications turns the toast and the sound on or off ([notifications] toast, sound).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from PyQt6.QtCore import QObject, Qt, QUrl
from PyQt6.QtWidgets import QApplication

from .monitor.runs_sidebar import run_label
from .toast import Toast, ToastStack

SOUNDS_DIR = Path(__file__).parent / "sounds"
SOUNDS = {"completed": "run_completed.wav", "failed": "run_failed.wav", "stopped": "run_stopped.wav"}
SOUND_VOLUME = 0.6

# Run status -> how the ending is shown: (kind, mark, verb in the title, theme colour, word in the detail line)
OUTCOMES = {
    "completed": ("completed", "✓", "finished", "primary", "Completed"),
    "failed": ("failed", "✗", "failed", "number", "Failed"),
    "error": ("failed", "✗", "failed", "number", "Could not run"),
    "interrupted": ("failed", "✗", "failed", "number", "Interrupted"),
    "stopped": ("stopped", "■", "stopped", "muted", "Stopped"),
    "cancelled": ("stopped", "■", "stopped", "muted", "Stopped"),
}


def duration_text(seconds: float | None) -> str:
    """'45s', '2m 14s', '1h 05m'."""
    if seconds is None or seconds < 0:
        return ""
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def describe(status: str, name: str, seconds: float | None = None, exit_code: int | None = None) -> dict:
    """The toast's content for a run that ended with `status`."""
    kind, mark, verb, color, word = OUTCOMES.get(status, OUTCOMES["failed"])
    parts = [word]
    if kind == "failed" and exit_code not in (None, 0):
        parts.append(f"exit code {exit_code}")
    if duration_text(seconds):
        parts.append(duration_text(seconds))
    return {"kind": kind, "mark": mark, "title": f"{name} {verb}", "detail": " · ".join(parts), "color": color}


class RunNotifier(QObject):
    """Window.finish calls run_finished once for each run that ends (demo runs too)."""

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.toasts = ToastStack(window)
        self._effects: dict[str, object] = {}
        self._notified: set[str] = set()

    def enabled(self, what: str) -> bool:
        settings = getattr(self.window, "settings_manager", None)
        return bool(settings.get("notifications", what, default=True)) if settings is not None else True

    def run_finished(self, run: dict, status: str):
        if run["id"] in self._notified:
            return
        self._notified.add(run["id"])
        info = describe(status, run_label(run), self._seconds(run), self._exit_code(run))
        if self.enabled("toast"):
            self.show(info, run)
        if self.enabled("sound"):
            self.play(info["kind"])

    # ── Toast or system notification ─────────────────────────────────────────

    def in_front(self) -> bool:
        """ThetaIDE is the app in use and its window is on screen."""
        return (QApplication.applicationState() == Qt.ApplicationState.ApplicationActive
                and self.window.isVisible() and not self.window.isMinimized())

    def show(self, info: dict, run: dict):
        tray = getattr(self.window, "tray", None)
        if not self.in_front() and tray is not None and tray.available():
            tray.notify(info["title"], info["detail"], warning=info["kind"] == "failed")
            return None
        toast = Toast(info["mark"], info["title"], info["detail"], "View results", info["color"])
        toast.action.connect(lambda r=run: self.window.view_run(r))
        return self.toasts.show(toast)

    # ── Sound ────────────────────────────────────────────────────────────────

    def play(self, kind: str):
        effect = self._effect(kind)
        if effect is not None:
            effect.play()

    def _effect(self, kind: str):
        """The chime for `kind`, loaded on first use; None without Qt Multimedia or the sound file."""
        if kind not in self._effects:
            effect = None
            path = SOUNDS_DIR / SOUNDS.get(kind, SOUNDS["completed"])
            try:
                from PyQt6.QtMultimedia import QSoundEffect
            except (ImportError, OSError):
                QSoundEffect = None
            if QSoundEffect is not None and path.is_file():
                effect = QSoundEffect(self)
                effect.setSource(QUrl.fromLocalFile(str(path)))
                effect.setVolume(SOUND_VOLUME)
            self._effects[kind] = effect
        return self._effects[kind]

    # ── Run details ──────────────────────────────────────────────────────────

    def _job(self, run: dict) -> dict:
        job_id = (run.get("backend") or {}).get("job_id")
        jobs = getattr(getattr(self.window, "jobs", None), "jobs", {})
        return (jobs.get(job_id) or {}) if job_id else {}

    def _seconds(self, run: dict) -> float | None:
        """How long the run took: the job's own start and end, else since the run was created."""
        job = self._job(run)
        if job.get("started") and job.get("finished"):
            return job["finished"] - job["started"]
        try:
            created = datetime.fromisoformat(run["created"])
        except (KeyError, TypeError, ValueError):
            return None
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - created).total_seconds()

    def _exit_code(self, run: dict) -> int | None:
        return self._job(run).get("returncode")
