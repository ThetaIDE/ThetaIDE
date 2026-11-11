"""Live cards for processes currently training in the workspace."""
import time

from PyQt6.QtCore import QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPainterPath
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget, QToolButton

from .live_run import format_duration
from .widgets import label
from .theme import theme_color
from .tree_tools import style_tool_button


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
    cancel_requested = pyqtSignal(str, str)

    def __init__(self, job, method_name=None, parent=None, *, show_progress=False, can_cancel=False):
        super().__init__(parent)
        self.setObjectName("card")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName("Open training monitor for process")
        self._hovered = False
        self._can_cancel = can_cancel
        self._sweep_backlog = False
        self.job_id = ""
        self.sweep_key = ""
        self.method_name = str(method_name or "")
        self.show_progress = show_progress and bool(self.method_name)
        self.progress_fraction = None
        self._speed = None
        self._speed_live = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 13, 16, 13)
        layout.setSpacing(7)

        top = QHBoxLayout()
        self.experiment_id = label("", "cardTitle")
        self.experiment_id.setWordWrap(True)
        self.status = QLabel()
        self.status.setObjectName("badge")
        self.elapsed = label("", "muted")
        self.expand_button = QToolButton()
        self.expand_button.setAutoRaise(True)
        self.expand_button.setCheckable(True)
        self.expand_button.setAccessibleName("Expand or collapse queued work")
        self.expand_button.hide()
        top.addWidget(self.expand_button)
        top.addWidget(self.experiment_id, 1)
        top.addWidget(self.status)
        self.cancel_button = style_tool_button(QToolButton(), "trash", "Cancel this item")
        policy = self.cancel_button.sizePolicy()
        policy.setRetainSizeWhenHidden(True)
        self.cancel_button.setSizePolicy(policy)
        self.cancel_button.clicked.connect(self._cancel)
        self.cancel_button.hide()
        top.addWidget(self.cancel_button)
        if not self.show_progress:
            top.addSpacing(10)
            top.addWidget(self.elapsed)
        layout.addLayout(top)

        self.description = label("", "muted")
        self.description.setWordWrap(True)
        layout.addWidget(self.description)
        self.progress_text = label('', 'muted')
        layout.addWidget(self.progress_text)
        self.speed = label("", "muted")
        self.eta = label("", "muted")
        if self.show_progress:
            for field in (self.elapsed, self.speed, self.eta):
                field.setWordWrap(True)
                field.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)
            metrics = QHBoxLayout()
            metrics.addStretch()
            metrics.addWidget(self.elapsed)
            metrics.addWidget(self.speed)
            metrics.addWidget(self.eta)
            layout.addLayout(metrics)

        # Let clicks on any text activate the enclosing card.
        for widget in (self.experiment_id, self.status, self.elapsed, self.speed, self.eta, self.description, self.progress_text):
            widget.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.update_job(job)

    def update_job(self, job):
        self.job_id = str(job.get("job_id") or "")
        self.open_method = job.get("queue_open_method", self.method_name)
        from src.app.pipeline.config import normalize_agent_name
        self.cancel_target = job.get("cancel_target") or (f"method:{normalize_agent_name(self.method_name)}" if self.method_name else "")
        self.cancel_button.setToolTip(f"Cancel {self.method_name or job.get('experiment_id') or self.job_id}")
        self._sweep_backlog = bool(job.get('sweep_backlog'))
        self._execution_started = bool(job.get("execution_started"))
        self._update_cancel_visibility()
        self.sweep_key = str(job.get("sweep_key") or "") if job.get("sweep_parent") or job.get('sweep_backlog') else ""
        from .method_labels import monitor_method_label
        method_label = monitor_method_label(self.method_name) if self.method_name else ""
        experiment_id = str(job.get("experiment_id") or self.job_id or "Unknown experiment")
        suffix = ' · Pending trials' if job.get('sweep_backlog') else " · Hyperparameter tuning" if job.get("sweep_parent") else ""
        self.experiment_id.setText(f"{experiment_id} · {method_label}" if self.method_name else experiment_id + suffix)
        self.experiment_id.setToolTip(self.experiment_id.text())
        self.description.setText(str(job.get("description") or "No description"))
        if job.get('waiting_reason'):
            self.description.setText(self.description.text() + '\n' + str(job['waiting_reason']))
        if job.get('queue_title'):
            self.experiment_id.setText(str(job['queue_title']))
        self.status.setText(str(job.get("status") or "Starting").capitalize())
        self.status.setToolTip(str(job.get("waiting_reason") or ""))
        self._training_budget = job.get("effective_timesteps") or job.get("total_timesteps")
        progress = job.get('queue_progress') or {
            'completed': job.get('trials_finished', 0), 'total': job.get('trials_total'), 'unit': 'trials'}
        if self.method_name and progress['total'] is None:
            self._set_progress(0, self._training_budget, 'steps')
        else:
            self._set_progress(progress['completed'], progress['total'], progress['unit'])
        self.update()
        self.update_elapsed(job)

    def _set_progress(self, completed, total, unit):
        if isinstance(total, (int, float)) and total > 0 and completed is not None:
            self.progress_fraction = max(0.0, min(1.0, completed / total))
            suffix = ' finished' if unit != 'steps' else ''
            self.progress_text.setText(f'{completed:,.0f}/{total:,.0f} {unit}{suffix} · {self.progress_fraction:.0%}')
            self.progress_text.show()
        else:
            self.progress_fraction = None
            self.progress_text.clear()
            self.progress_text.hide()

    def update_elapsed(self, job):
        if job.get("status") == "queued" or job.get('sweep_backlog'):
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
        self.status.setText({"running": "Training", "testing": "Testing", "starting": "Starting",
                             "diagnostics": "Diagnostics", "done": "Completed", "failed": "Failed"}.get(
                                 segment.state, "Starting" if self._execution_started else "Waiting"))
        self._set_progress(live.progress() if live else 0, live.budget if live else self._training_budget, 'steps')
        self.progress_fraction = segment.fraction
        if segment.state == "pending":
            self.elapsed.setText("Starting training" if self._execution_started else "Waiting")
            self.speed.setText("Speed: —")
            self.eta.setText("Time left: —")
            self._speed = self._speed_live = None
        else:
            self.elapsed.setText(f"Elapsed: {format_duration(segment.duration)}")
            rate = live.rate() if segment.state == "running" else None
            if rate and self._speed and live is self._speed_live:
                rate = 0.7 * self._speed + 0.3 * rate
            self._speed, self._speed_live = rate, live
            self.speed.setText(f"Speed: {rate:,.0f} steps/s" if rate else "Speed: —")
            progress = live.progress()
            eta = max(0.0, (live.budget - progress) / rate) if rate and live.budget and progress is not None else None
            if eta is not None and eta >= 90:
                eta = round(eta / 10) * 10
            self.eta.setText(f"Time left: {format_duration(eta)}")
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

    def _cancel(self):
        self.cancel_button.setEnabled(False)
        self.cancel_requested.emit(self.job_id, self.cancel_target)

    def _update_cancel_visibility(self):
        self.cancel_button.setVisible(self._hovered and self._can_cancel and not self._sweep_backlog)

    def enterEvent(self, event):
        self._hovered = True
        self._update_cancel_visibility()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        self._update_cancel_visibility()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.activated.emit(self.job_id, f"sweep:{self.sweep_key}" if self.sweep_key else self.open_method)
            event.accept()
            return
        super().mousePressEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.activated.emit(self.job_id, f"sweep:{self.sweep_key}" if self.sweep_key else self.open_method)
            event.accept()
            return
        super().keyPressEvent(event)


class QueuePanel(QWidget):
    """Shows only active training processes; selecting a card opens its Training Monitor."""

    structure_changed = pyqtSignal()

    def __init__(self, on_open, parent=None, *, split_methods=False, lives_for_job=None, on_cancel=None, phases_for_job=None):
        super().__init__(parent)
        self.on_open = on_open
        self.on_cancel = on_cancel
        self.split_methods = split_methods
        self.lives_for_job = lives_for_job
        self.phases_for_job = phases_for_job
        self.card_jobs = []
        self.cards = {}
        self._expanded = {}
        self._user_expanded = set()
        self._compact = False
        self._data = None
        self._nodes = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)

        self.resource_status = label('', 'muted')
        self.resource_status.setWordWrap(True)
        layout.addWidget(self.resource_status)
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
        waiting = data.get('waiting_reason', '')
        count = len(data.get('queued', []))
        self.resource_status.setText(waiting or (f'{count} jobs waiting · queue paused' if count and not data.get('running')
                                               else f'{count} jobs waiting for a slot' if count else ''))
        self.resource_status.setVisible(bool(self.resource_status.text()))
        if self._data == data:
            self._update_times(force=True)
            return
        # Own a snapshot: callers may mutate their dictionaries between polls.
        import copy
        self._data = copy.deepcopy(data)
        self._nodes = self._build_nodes(data)
        self._render_nodes()

    @staticmethod
    def _card_key(job, method):
        job_id = str(job.get('job_id') or '')
        return (f"{job_id}:{job.get('sweep_key')}:pending" if job.get('sweep_backlog') else
                f"{job_id}:{method}" if method else
                f"{job_id}:{job.get('sweep_key')}" if job.get('sweep_parent') else job_id)

    def _build_nodes(self, data):
        nodes = {}
        for job in data.get('active', []):
            job_id = str(job.get('job_id') or '')
            agents = job.get('agents') or []
            methods = agents if agents and (job.get('sweep_trial') or job.get('queue_method') or job.get('queue_root') or self.split_methods or
                                            job.get('run_in_parallel') and len(agents) > 1) else [None]
            for method in methods:
                key = self._card_key(job, method)
                parent = None
                queue_method = job.get('queue_method')
                if queue_method or job.get('queue_root'):
                    experiment_key = f'{job_id}:experiment'
                    if experiment_key not in nodes:
                        nodes[experiment_key] = [dict(job, agents=[], sweep_parent=False,
                            queue_title=str(job.get('experiment_id') or job_id),
                            cancel_target='', status=job.get('queue_job_status', job.get('status')),
                            trials_total=None, queue_progress=None, description=job.get('experiment_description', 'Experiment')), None, None, []]
                    parent = experiment_key
                if queue_method:
                    method_key = f'{job_id}:method:{queue_method}'
                    if method_key not in nodes:
                        group = dict(job, agents=[], sweep_parent=False, trials_total=None,
                                     queue_title=queue_method, queue_open_method=queue_method, cancel_target=f'method:{queue_method}', description='Method')
                        nodes[method_key] = [group, None, experiment_key, []]
                        nodes[experiment_key][3].append(method_key)
                    parent = method_key
                    if job.get('sweep_parent') or job.get('queue_method_parent'):
                        nodes[method_key][0] = dict(job, queue_title=queue_method + (' · Hyperparameter tuning' if job.get('sweep_parent') else ''),
                                                  queue_open_method=queue_method)
                        continue
                    trial = job.get('queue_trial')
                    if trial is not None and (job.get('queue_seed') is not None or job.get('queue_trial_parent')):
                        trial_key = f'{method_key}:{job.get("queue_study")}:trial:{trial}'
                        if trial_key not in nodes:
                            group = dict(job, agents=[], queue_title=f'Trial {trial}',
                                         queue_open_method=f'{queue_method} · {job.get("queue_study")} · trial {trial}',
                                         cancel_target=f'trial:{job.get("trial_log_version", "")}', description='Training variance')
                            nodes[trial_key] = [group, None, method_key, []]
                            nodes[method_key][3].append(trial_key)
                        if job.get('queue_trial_parent'):
                            nodes[trial_key][0] = dict(job,
                                queue_title=f'Trial {trial}', description='Training variance',
                                queue_open_method=f'{queue_method} · {job.get("queue_study")} · trial {trial}',
                                cancel_target=f'trial:{job.get("trial_log_version", "")}')
                            continue
                        parent = trial_key
                if job.get('queue_seed') is not None:
                    job = dict(job, queue_title=f"Training seed {job['queue_seed']}")
                nodes[key] = [job, method, parent, []]
                if parent:
                    nodes[parent][3].append(key)
        return nodes

    def _toggle_node(self, key, expanded):
        self._expanded[key] = expanded
        self._user_expanded.add(key)
        self._render_nodes()
        self.updateGeometry()
        self.structure_changed.emit()

    def _render_nodes(self):
        self._phase_ancestors = {}
        for key, (job, method, parent, _children) in self._nodes.items():
            if method:
                ancestors = []
                while parent:
                    ancestors.append(parent)
                    parent = self._nodes[parent][2]
                self._phase_ancestors[(job.get('job_id'), method)] = ancestors
        previous_cards = self.cards
        self.cards = {}
        self.card_jobs = []
        # Large queues start compact. Hidden branches have no widgets and do
        # not load telemetry; expanding a branch materializes only that branch.
        compact = len(self._nodes) > 30
        if compact and not self._compact:
            self._expanded = {key: value for key, value in self._expanded.items() if key in self._user_expanded}
        self._compact = compact
        self._user_expanded.intersection_update(self._nodes)
        visible = []
        def visit(key, depth):
            job, method, parent, children = self._nodes[key]
            visible.append((key, job, method, depth, children))
            expanded = self._expanded.setdefault(key, depth == 0 or not compact)
            if expanded:
                for child in sorted(children, key=lambda k: (
                        self._nodes[k][0].get('sweep_backlog', False),
                        self._nodes[k][0].get('queue_trial') if self._nodes[k][0].get('queue_trial') is not None else -1)):
                    visit(child, depth + 1)
        for key, node in self._nodes.items():
            if node[2] is None:
                visit(key, 0)
        for key, job, method, depth, children in visible:
            self.card_jobs.append((key, job))
            card = previous_cards.pop(key, None)
            if card is None:
                card = ProcessCard(job, method, self.content,
                                   show_progress=self.lives_for_job is not None,
                                   can_cancel=self.on_cancel is not None and not job.get('remote'))
                card.activated.connect(self.on_open)
                card.expand_button.toggled.connect(lambda checked, k=key: self._toggle_node(k, checked))
                if self.on_cancel is not None:
                    card.cancel_requested.connect(self.on_cancel)
            else:
                card.update_job(job)
            card.expand_button.blockSignals(True)
            card.expand_button.setVisible(bool(children))
            card.expand_button.setChecked(self._expanded[key])
            card.expand_button.setArrowType(Qt.ArrowType.DownArrow if self._expanded[key] else Qt.ArrowType.RightArrow)
            card.expand_button.setToolTip('Collapse' if self._expanded[key] else 'Expand')
            card.expand_button.blockSignals(False)
            card.layout().setContentsMargins(16 + depth * 18, 13, 16, 13)
            self.cards[key] = card
            index = len(self.cards) - 1
            if self.cards_layout.indexOf(card) != index:
                self.cards_layout.insertWidget(index, card)
        for card in previous_cards.values():
            self.cards_layout.removeWidget(card)
            card.hide()
            card.deleteLater()
        self._expanded = {key: value for key, value in self._expanded.items() if key in self._nodes}
        self.empty.setVisible(not self.card_jobs)
        self._update_times(force=True)

    def content_height(self, width):
        """Height needed for the cards at this width, before the scroll area clips them."""
        margins = self.layout().contentsMargins()
        available = max(1, width - margins.left() - margins.right() - 4)
        widgets = list(self.cards.values()) or [self.empty]
        heights = [max(widget.sizeHint().height(), widget.heightForWidth(available)) for widget in widgets]
        return margins.top() + margins.bottom() + sum(heights) + self.cards_layout.spacing() * (len(widgets) - 1) + 4

    def _update_times(self, *, force=False):
        if not force and not self.isVisible():
            return
        clip_to_viewport = self.isVisible()
        lives_by_job = {}
        for key, job in self.card_jobs:
            card = self.cards.get(key)
            if card is not None:
                if clip_to_viewport and not self.scroll.viewport().rect().intersects(card.geometry().translated(-self.scroll.horizontalScrollBar().value(), -self.scroll.verticalScrollBar().value())):
                    continue
                card.update_elapsed(job)
                if self.lives_for_job is not None and card.method_name:
                    job_id = job.get("job_id")
                    if job_id not in lives_by_job:
                        lives_by_job[job_id] = self.lives_for_job(job_id)
                    card.update_progress(lives_by_job[job_id])
        # Aggregate cached active phases once; hidden branches never read files.
        phase_nodes, seen_jobs = {}, set()
        if self.phases_for_job:
            for card in self.cards.values():
                if card.job_id in seen_jobs:
                    continue
                seen_jobs.add(card.job_id)
                for name, stage in self.phases_for_job(card.job_id).items():
                    if stage:
                        for ancestor in self._phase_ancestors.get((card.job_id, name), ()):
                            phase_nodes.setdefault(ancestor, set()).add(stage)
        # Reflect active child phases on method/experiment cards too.
        for key in reversed(list(self.cards)):
            card = self.cards[key]
            children = self._nodes.get(key, (None, None, None, []))[3]
            phases = {self.cards[child].status.text() for child in children if child in self.cards}
            if children and self.phases_for_job:
                job = self._nodes[key][0]
                stages = phase_nodes.get(key, ())
                phases = {dict(training='Training', testing='Testing', diagnostics='Diagnostics', starting='Starting').get(stage, stage.capitalize())
                          for stage in stages}
                card.status.setText(str(job.get('status') or 'Starting').capitalize())
            active = phases - {'Waiting', 'Starting', 'Completed', 'Failed'}
            if active:
                card.status.setText('Training' if 'Training' in active else
                                    'Testing' if 'Testing' in active else next(iter(sorted(active))))
