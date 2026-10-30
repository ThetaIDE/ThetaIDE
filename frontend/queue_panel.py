"""Live cards for processes currently training in the workspace."""
import time

from PyQt6.QtCore import QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPainterPath
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from .widgets import label
from .theme import theme_color


def clock(seconds):
    if seconds is None:
        return "0:00"
    try:
        seconds = int(max(0, seconds))
        return f"{seconds // 60}:{seconds % 60:02d}"
    except (TypeError, ValueError):
        return "0:00"


class ProcessCard(QFrame):
    """Clickable summary of one active process."""

    activated = pyqtSignal(str, str)

    def __init__(self, job, method_name=None, parent=None, *, show_progress=False):
        super().__init__(parent)
        self.setObjectName("card")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName("Open training monitor for process")
        self.job_id = ""
        self.method_name = str(method_name or "")
        self.show_progress = show_progress and bool(self.method_name)
        self.progress_fraction = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 13, 16, 13)
        layout.setSpacing(7)

        top = QHBoxLayout()
        self.experiment_id = label("", "cardTitle")
        self.status = QLabel()
        self.status.setObjectName("badge")
        self.elapsed = label("", "muted")
        top.addWidget(self.experiment_id, 1)
        top.addWidget(self.status)
        top.addSpacing(10)
        top.addWidget(self.elapsed)
        layout.addLayout(top)

        self.description = label("", "muted")
        self.description.setWordWrap(True)
        layout.addWidget(self.description)

        # Let clicks on any text activate the enclosing card.
        for widget in (self.experiment_id, self.status, self.elapsed, self.description):
            widget.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.update_job(job)

    def update_job(self, job):
        self.job_id = str(job.get("job_id") or "")
        experiment_id = str(job.get("experiment_id") or self.job_id or "Unknown experiment")
        self.experiment_id.setText(f"{experiment_id} · {self.method_name}" if self.method_name else experiment_id)
        self.experiment_id.setToolTip(self.experiment_id.text())
        self.description.setText(str(job.get("description") or "No description"))
        self.status.setText(str(job.get("status") or "Starting").capitalize())
        self.update_elapsed(job)

    def update_elapsed(self, job):
        if job.get("status") == "queued":
            self.elapsed.setText("Waiting")
            return
        started = job.get("started")
        if not started and job.get("status") == "pending":
            self.elapsed.setText("Starting")
            return
        started = started or job.get("created")
        if started:
            self.elapsed.setText(f"{clock(time.time() - started)} elapsed")
        else:
            self.elapsed.setText("Starting")

    def update_progress(self, lives):
        if not self.show_progress:
            return
        from .monitor.pipeline_strip import pipeline_segments
        live = lives.get(self.method_name)
        segment = pipeline_segments([self.method_name], {self.method_name: live} if live else {})[0]
        self.progress_fraction = segment.fraction
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.progress_fraction is None or self.progress_fraction <= 0:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        bounds = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        clip = QPainterPath()
        clip.addRoundedRect(bounds, 5, 5)
        painter.setClipPath(clip)
        fill = QColor(theme_color("secondary"))
        fill.setAlpha(55)
        painter.fillRect(QRectF(bounds.left(), bounds.top(), bounds.width() * min(1.0, self.progress_fraction),
                                bounds.height()), fill)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.activated.emit(self.job_id, self.method_name)
            event.accept()
            return
        super().mousePressEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.activated.emit(self.job_id, self.method_name)
            event.accept()
            return
        super().keyPressEvent(event)


class QueuePanel(QWidget):
    """Shows only active training processes; selecting a card opens its Training Monitor."""

    def __init__(self, on_open, parent=None, *, split_methods=False, lives_for_job=None):
        super().__init__(parent)
        self.on_open = on_open
        self.split_methods = split_methods
        self.lives_for_job = lives_for_job
        self.card_jobs = []
        self.cards = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.content = QWidget()
        self.cards_layout = QVBoxLayout(self.content)
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.setSpacing(10)
        self.empty = label("No running processes", "muted")
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cards_layout.addWidget(self.empty)
        self.cards_layout.addStretch(1)
        self.scroll.setWidget(self.content)
        layout.addWidget(self.scroll, 1)

        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._update_times)
        self.timer.start()

    def set_data(self, data):
        for card in self.cards.values():
            self.cards_layout.removeWidget(card)
            card.deleteLater()
        self.cards = {}
        self.card_jobs = []

        for job in data.get("active", []):
            job_id = str(job.get("job_id") or "")
            agents = job.get("agents") or []
            methods = agents if agents and (self.split_methods or (job.get("run_in_parallel") and len(agents) > 1)) else [None]
            for method_name in methods:
                key = f"{job_id}:{method_name}" if method_name else job_id
                self.card_jobs.append((key, job))
                card = ProcessCard(job, method_name, self.content, show_progress=self.lives_for_job is not None)
                card.activated.connect(self.on_open)
                self.cards[key] = card
                self.cards_layout.insertWidget(self.cards_layout.count() - 1, card)

        self.empty.setVisible(not self.card_jobs)
        self._update_times()

    def content_height(self, width):
        """Height needed for the cards at this width, before the scroll area clips them."""
        margins = self.layout().contentsMargins()
        available = max(1, width - margins.left() - margins.right() - 4)
        widgets = list(self.cards.values()) or [self.empty]
        heights = [max(widget.sizeHint().height(), widget.heightForWidth(available)) for widget in widgets]
        return margins.top() + margins.bottom() + sum(heights) + self.cards_layout.spacing() * (len(widgets) - 1) + 4

    def _update_times(self):
        lives_by_job = {}
        for key, job in self.card_jobs:
            card = self.cards.get(key)
            if card is not None:
                card.update_elapsed(job)
                if self.lives_for_job is not None and card.method_name:
                    job_id = job.get("job_id")
                    if job_id not in lives_by_job:
                        lives_by_job[job_id] = self.lives_for_job(job_id)
                    card.update_progress(lives_by_job[job_id])
