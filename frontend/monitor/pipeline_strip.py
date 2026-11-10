"""Pipeline strip: the selected run's methods in training order, one segment each, with their events.

run_pipeline.py trains an experiment's methods one after another, so a run is a row of segments: done,
training, or still waiting. A segment is as wide as that method took (or is expected to take), and its
checkpoints, its latest new best and its warnings sit under it as marks. Clicking a segment shows only that method's
curves; clicking it again shows them all.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from ..live_run import LiveRun, format_duration
from ..theme import theme_color

BAR_HEIGHT = 26
MARK_ROW = 16
GAP = 3
MARK_SPACING = 14.0  # px: marks of one kind closer than this are drawn as one
# How each kind of mark is drawn: (glyph, color role or hex)
MARKS = {"checkpoint": ("◆", "comment"), "best": ("★", "accent"), "warning": ("⚠", "#fe8019"), "error": ("✖", "#fb4934")}


@dataclass
class Segment:
    agent: str
    state: str  # "done", "failed", "running" or "pending"
    weight: float  # share of the strip's width, before normalizing
    fraction: float | None = None  # how far a running method is through its budget
    duration: float | None = None  # seconds taken (done) or elapsed so far (running)
    marks: list[tuple[float, str]] = field(default_factory=list)  # (0..1 along the segment, kind)


def _started(live: LiveRun) -> float | None:
    if live.start:
        return (live.start.get("run") or {}).get("started") or live.start.get("t")
    times = [e.get("t") for e in live.events if e.get("t")]
    return min(times) if times else None


def _marks(live: LiveRun, start: float | None, length: float | None) -> list[tuple[float, str]]:
    if not start or not length or length <= 0:
        return []
    found = []
    for event in live.events:
        level, kind = event.get("level"), event.get("kind")
        mark = "best" if kind == "new_best" else level if level in ("warning", "error") else None
        if mark and event.get("t"):
            found.append((event["t"], mark))
    # Early in training a new best comes with nearly every evaluation: mark only where the latest was set.
    bests = [entry for entry in found if entry[1] == "best"]
    found = [entry for entry in found if entry[1] != "best"] + bests[-1:]
    seen = set()
    for artifact in live.artifacts:  # a checkpoint saved again is one mark, at its first save
        if artifact.get("kind") == "checkpoint" and artifact.get("t") and artifact.get("path") not in seen:
            seen.add(artifact.get("path"))
            found.append((artifact["t"], "checkpoint"))
    return sorted((min(1.0, max(0.0, (t - start) / length)), kind) for t, kind in found)


def pipeline_segments(agents: list[str], lives: dict[str, LiveRun], now: float | None = None) -> list[Segment]:
    """A segment per method: the run's declared agents in order, then any other agent that reported."""
    now = time.time() if now is None else now
    order = list(dict.fromkeys([*agents, *lives]))
    segments = []
    for agent in order:
        live = lives.get(agent)
        start = _started(live) if live is not None else None
        if live is None or (not live.start and not live.rows and live.finish is None):
            segments.append(Segment(agent, "pending", 0.0))
            continue
        if live.finish is not None:
            end = live.finish.get("t") or live.last_update or now
            duration = max(0.0, end - start) if start else None
            state = "failed" if live.finish.get("status") == "failed" else "done"
            segments.append(Segment(agent, state, duration or 1.0, 1.0, duration, _marks(live, start, duration)))
            continue
        elapsed = max(0.0, now - start) if start else None
        progress = live.progress()
        fraction = min(1.0, progress / live.budget) if live.budget and progress is not None else None
        # Expected length: elapsed time scaled up by how far through its budget the method is.
        expected = elapsed / fraction if elapsed and fraction and fraction > 0.02 else (elapsed or 1.0) * 2
        state = {"testing": "testing", "starting": "starting", "diagnostics": "diagnostics"}.get(getattr(live, "stage", None), "running")
        segments.append(Segment(agent, state, expected, fraction, elapsed, _marks(live, start, expected)))
    known = [s.weight for s in segments if s.state != "pending" and s.weight > 0]
    for s in segments:
        if s.state == "pending":  # nothing to go on: as long as the methods seen so far, on average
            s.weight = sum(known) / len(known) if known else 1.0
    return segments


class PipelineStrip(QWidget):
    """Paints pipeline_segments(); `segment_clicked(agent)` on a click, "" when the focus is cleared."""

    segment_clicked = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(BAR_HEIGHT + MARK_ROW)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.segments: list[Segment] = []
        self.focused: str | None = None
        self._rects: list[tuple[QRectF, Segment]] = []

    def show_run(self, agents: list[str], lives: dict[str, LiveRun], now: float | None = None):
        self.segments = pipeline_segments(agents, lives, now)
        if self.focused not in {s.agent for s in self.segments}:
            self.focused = None
        self.setVisible(any(s.state != "pending" for s in self.segments))
        self.update()

    def _layout(self) -> list[tuple[QRectF, Segment]]:
        total = sum(s.weight for s in self.segments) or 1.0
        room = self.width() - GAP * max(0, len(self.segments) - 1)
        x, rects = 0.0, []
        for s in self.segments:
            w = max(24.0, room * s.weight / total)
            rects.append((QRectF(x, 0, w, BAR_HEIGHT), s))
            x += w + GAP
        return rects

    @staticmethod
    def segment_text(s: Segment) -> str:
        if s.state in ("testing", "starting", "diagnostics"):
            return f"{s.agent} · {s.state.capitalize()}"
        if s.state == "running":
            share = f" · {s.fraction:.0%}" if s.fraction is not None else f" · {format_duration(s.duration)}"
            return f"▶ {s.agent}{share}"
        if s.state in ("done", "failed"):
            took = f" · {format_duration(s.duration)}" if s.duration else ""
            return f"{'✓' if s.state == 'done' else '✗'} {s.agent}{took}"
        return s.agent

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._rects = self._layout()
        tones = {"done": "primary", "failed": "#fb4934", "running": "secondary"}
        for rect, s in self._rects:
            tone = QColor(theme_color(tones.get(s.state, "comment")))
            fill = QColor(tone)
            fill.setAlpha(45 if s.state != "pending" else 0)
            border = QColor(tone)
            border.setAlpha(255 if s.agent == self.focused else 140 if s.state != "pending" else 90)
            pen = QPen(border, 2 if s.agent == self.focused else 1, Qt.PenStyle.SolidLine if s.state != "pending"
                       else Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(fill)
            painter.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), 4, 4)
            if s.state == "running" and s.fraction:
                done = QColor(tone)
                done.setAlpha(55)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(done)
                painter.drawRoundedRect(QRectF(rect.left() + 1, rect.top() + 1, (rect.width() - 2) * s.fraction,
                                               rect.height() - 2), 3, 3)
            painter.setFont(QFont("Segoe UI", 8))
            painter.setPen(QColor(theme_color("text" if s.state != "pending" else "muted")))
            text_rect = rect.adjusted(8, 0, -6, 0)
            text = painter.fontMetrics().elidedText(self.segment_text(s), Qt.TextElideMode.ElideRight,
                                                    int(text_rect.width()))
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)
            painter.setFont(QFont("Segoe UI", 8))
            last_x: dict[str, float] = {}  # early on a new best comes every evaluation: merge marks that touch
            for position, kind in s.marks:
                x = rect.left() + rect.width() * position
                if x - last_x.get(kind, -MARK_SPACING) < MARK_SPACING:
                    continue
                last_x[kind] = x
                glyph, color = MARKS[kind]
                painter.setPen(QColor(theme_color(color)))
                painter.drawText(QRectF(x - 6, BAR_HEIGHT + 1, 12, MARK_ROW - 1), Qt.AlignmentFlag.AlignCenter, glyph)
        painter.end()

    def _segment_at(self, pos) -> Segment | None:
        for rect, s in self._rects or self._layout():
            if rect.left() <= pos.x() <= rect.right():
                return s
        return None

    def mouseMoveEvent(self, event):
        s = self._segment_at(event.position())
        if s is None:
            self.setToolTip("")
            return
        marks = {kind: sum(1 for _, k in s.marks if k == kind) for kind in MARKS}
        counted = ", ".join(f"{n} {kind}{'s' if n > 1 else ''}" for kind, n in marks.items() if n)
        hint = "Click to show all methods again" if s.agent == self.focused else "Click to show only this method"
        self.setToolTip(f"{self.segment_text(s)}  ({s.state})" + (f"\n{counted}" if counted else "") + f"\n{hint}")

    def mousePressEvent(self, event):
        s = self._segment_at(event.position())
        if event.button() == Qt.MouseButton.LeftButton and s is not None and s.state != "pending":
            self.focused = None if s.agent == self.focused else s.agent
            self.update()
            self.segment_clicked.emit(self.focused or "")
        super().mousePressEvent(event)
