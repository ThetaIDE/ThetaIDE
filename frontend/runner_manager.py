"""Headless queue engine owned by frontend.job_runner.

QProcess children belong to the independent runner, never to a desktop window.
Logs and status persist while desktops disconnect and reconnect.
"""
from __future__ import annotations

import codecs
from datetime import datetime
import json
import shutil
import sys
import time
import uuid
import psutil
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QFileSystemWatcher, QObject, QProcess, QProcessEnvironment, pyqtSignal

from src.app.services import JOBS_DIR, LOGS_DIR, PROJECT_ROOT
from src.app.services.config_service import LaunchError, prepare_launch
from src.app.services.job_store import JobStore
from src.app.services.metrics import AgentFeed
from src.app.services.processes import kill_process_tree, process_stats
from src.app.services.resource_budget import ResourceBudget, ResourcePool, host_capacity, tree_memory, managed_environment
from src.app.telemetry.records import metrics_row

ACTIVE_STATUSES = ("pending", "running")
FINAL_STATUSES = ("completed", "failed", "cancelled", "error", "interrupted")
MAX_LOG_LINES = 20000
HARDWARE_SAMPLE_SECONDS = 1.0


def _queue_run_progress(runs):
    """Count all planned runs, including terminal runs omitted from active cards."""
    return {'completed': sum(run.get('status') in ('completed', 'failed', 'cancelled') for run in runs),
            'total': len(runs), 'unit': 'training runs'}


class JobManager(QObject):
    """Launches, queues and stops pipeline runs. One job trains at a time from the queue."""

    job_changed = pyqtSignal(dict)  # the job's public record after any status change
    job_output = pyqtSignal(str, list)  # job_id, new log lines
    job_metrics = pyqtSignal(str, str, list, bool)  # job_id, agent, new metric rows, reset (start over)
    job_records = pyqtSignal(str, str, list, bool)  # job_id, agent, new telemetry records of every type, reset
    job_hardware = pyqtSignal(str, dict)  # job_id, {"cpu_percent", "memory_mb", ...}
    queue_changed = pyqtSignal(dict)  # {"running", "active", "queued", "finished"}

    def __init__(self, store_path: Path | None = None, python: str | None = None,
                 project_root: Path | None = None, logs_dir: Path | None = None, parent=None,
                 settings_path: Path | None = None, enabled_plugins_provider=None):
        super().__init__(parent)
        self.project_root = Path(project_root or PROJECT_ROOT)
        if not python:
            venv_bin = self.project_root / "venv" / ("Scripts" if sys.platform == "win32" else "bin")
            venv_python = venv_bin / ("python.exe" if sys.platform == "win32" else "python")
            if venv_python.is_file():
                python = str(venv_python)
        self.python = python or sys.executable
        self.logs_dir = Path(logs_dir or LOGS_DIR)  # where run_pipeline.py writes results/logs/<group>/<id>
        self.settings_path = Path(settings_path) if settings_path else None
        self.enabled_plugins_provider = enabled_plugins_provider
        self.jobs_dir = Path(store_path).parent if store_path else JOBS_DIR
        self.store = JobStore(store_path or JOBS_DIR / "jobs.db")
        self.jobs: dict[str, dict[str, Any]] = {}
        self.queue_order: list[str] = []
        self.queue_running = False  # queued jobs start only after start_queue(); it pauses itself when drained
        self.max_parallel = None  # zero/None means the shared worker cap is automatic
        self.resource_budget = ResourceBudget()
        self.resource_pool = ResourcePool(self.jobs_dir / 'resource_usage.db', self.settings_path)
        self.waiting_reason = ''
        self._last_resource_tick = 0.0
        self._load_resource_settings()
        self._processes: dict[str, QProcess] = {}
        self._decoders: dict[str, Any] = {}
        self._partial: dict[str, str] = {}
        self._logs: dict[str, list[str]] = {}
        self._log_files: dict[str, Any] = {}
        self._stream_files: dict[str, dict[str, Any]] = {}
        self._feeds: dict[tuple[str, str], AgentFeed] = {}
        self._focused_replicates: dict[str, str] = {}
        self._execution_signatures: dict[str, tuple] = {}
        self._last_hardware: dict[str, float] = {}
        self._streamed_hardware: set[str] = set()  # jobs whose training process reports its own resource use
        self.watcher = QFileSystemWatcher(self)
        self.watcher.directoryChanged.connect(self._on_path_changed)
        self.watcher.fileChanged.connect(self._on_path_changed)
        self._recover()

    # ── State ────────────────────────────────────────────────────────────────

    def _recover(self):
        """Load past jobs. A job still marked active belonged to a previous session: QProcess jobs end
        with the app, so it can't be reattached."""
        self.store.recover_active_jobs()
        for job in self.store.list_jobs():
            if job.get("status") in ACTIVE_STATUSES:
                job.update(status="interrupted", finished=job.get("finished") or time.time(),
                           error="Started by a previous ThetaIDE session that has closed.")
                self.store.save_job(job)
            self.jobs[job["job_id"]] = job
        self.queue_order = [job_id for job_id in self.store.get_queue() if self.jobs.get(job_id, {}).get("status") == "queued"]
        self.store.set_queue(self.queue_order)

    def _save(self, job: dict[str, Any]):
        self.store.save_job(job)
        self.job_changed.emit(self.public(job))

    @staticmethod
    def public(job: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in job.items() if not k.startswith("_")}

    def job(self, job_id: str) -> dict[str, Any] | None:
        job = self.jobs.get(job_id)
        return self.public(job) if job else None

    def log_lines(self, job_id: str) -> list[str]:
        return list(self._logs.get(job_id, []))

    def active_jobs(self) -> list[dict[str, Any]]:
        return [self.public(j) for j in self.jobs.values() if j.get("status") in ACTIVE_STATUSES]

    def is_busy(self) -> bool:
        return any(j.get("status") in ACTIVE_STATUSES for j in self.jobs.values())

    def running_count(self) -> int:
        return sum(1 for j in self.jobs.values() if j.get("status") in ACTIVE_STATUSES)

    def has_capacity(self) -> bool:
        """Legacy query; actual admission checks each job's resource demand."""
        return True

    def _load_resource_settings(self):
        import tomllib
        values = {}
        paths = [Path.home() / '.config/thetaide/settings.toml', self.settings_path] if self.settings_path else []
        for path in paths:
            try:
                with path.open('rb') as stream:
                    section = tomllib.load(stream).get('training', {})
                    if isinstance(section, dict):
                        values.update(section)
            except (OSError, ValueError, TypeError):
                pass
        self.resource_budget = ResourceBudget.from_settings(values)
        self.max_parallel = self.resource_budget.max_parallel or None

    def _resource_reason(self, job):
        if not job.get('request', {}).get('local', True):
            return ''
        try:
            request = job.get('request', {})
            # Pipeline parents only coordinate work. Actual methods and trials
            # acquire their own learned budget immediately before training.
            spec = {'key': '__coordinator__', 'memory': .1 * 1024**3, 'cpu': 0}
            _token, reason = self.resource_pool.acquire(spec, reserve=False,
                accelerator=False, capacity=host_capacity())
            return reason
        except (OSError, RuntimeError, psutil.Error):
            return 'Waiting: system resource measurements are unavailable.'

    def resource_tick(self):
        """Retry pressure-blocked queues even when no job finishes or writes output."""
        if time.monotonic() - self._last_resource_tick < 1:
            return
        self._last_resource_tick = time.monotonic()
        self._load_resource_settings()
        try:
            self.resource_pool.sample()
        except (OSError, RuntimeError, psutil.Error):
            self.waiting_reason = 'Waiting: system resource measurements are unavailable.'
            return
        before = self.waiting_reason
        self._start_next_queued()
        if before != self.waiting_reason:
            self._emit_queue()

    def set_max_parallel(self, count: int):
        """Update the workspace-wide leaf-worker cap, including detached workers."""
        from frontend.settings import _parse_toml_file, _write_toml_file
        if self.settings_path is None:
            self.settings_path = self.project_root / '.thetaide/settings.toml'
        values = _parse_toml_file(self.settings_path)
        values.setdefault('training', {})['max_parallel'] = max(0, int(count))
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        _write_toml_file(self.settings_path, values)
        self._start_next_queued()
        self._emit_queue()

    def queue_state(self) -> dict[str, Any]:
        """The active job(s), queued jobs in run order, and the ten most recent finished jobs."""
        finished = sorted((j for j in self.jobs.values() if j.get("status") in FINAL_STATUSES),
                          key=lambda j: j.get("finished") or 0, reverse=True)[:10]
        active = [item for job in self.active_jobs() for item in self._execution_queue_items(job)]
        return {
            "running": self.queue_running,
            "hierarchy_version": 2,
            "waiting_reason": self.waiting_reason,
            "active": active,
            "queued": [dict(self.public(self.jobs[job_id]), position=n)
                       for n, job_id in enumerate(self.queue_order) if job_id in self.jobs],
            "finished": [self.public(j) for j in finished],
        }

    def _execution_manifests(self, job):
        """Only manifests owned by this launch, even when an experiment name is reused."""
        if job.get("status") == "queued":
            return
        root = self.project_root / "results" / "executions" / str(job.get("group")) / str(job.get("experiment_id"))
        seen = set()
        for path in sorted(root.glob("*/manifest.json")):
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
                if job.get("pid") and manifest.get("owner_pid") != job["pid"]:
                    continue
                if job.get("started"):
                    created = datetime.fromisoformat(manifest["created_at"]).timestamp()
                    if created < job["started"] or (job.get("finished") and created > job["finished"]):
                        continue
            except (OSError, ValueError, KeyError, TypeError):
                continue
            parent = manifest.get("parent") or {}
            seen.add((parent.get("study"), parent.get("trial_number"), parent.get("log_version")))
            yield manifest
        # Running workers from an earlier build can still be followed without
        # rewriting their files or restarting their study.
        for _method, summary in self._sweep_summaries(job):
            for trial in summary.get("trials", []):
                manifest = trial.get("execution")
                if manifest:
                    if job.get("pid") and manifest.get("owner_pid") and manifest["owner_pid"] != job["pid"]:
                        continue
                    if job.get("started") and manifest.get("created_at"):
                        if datetime.fromisoformat(manifest["created_at"]).timestamp() < job["started"]:
                            continue
                    parent = manifest.get("parent") or {}
                    key = (parent.get("study"), parent.get("trial_number"), parent.get("log_version"))
                    if key not in seen:
                        seen.add(key)
                        yield manifest

    def _replicate_state(self, replicate):
        """Legacy coordinators called resource waits running; require worker startup evidence."""
        state = replicate.get("status", "queued")
        if state == "running" and not replicate.get("execution_started") and replicate.get("output_log"):
            output = Path(replicate["output_log"])
            if not output.is_absolute():
                output = self.project_root / output
            if not output.is_file():
                return "pending"
        return state

    def _execution_queue_items(self, job):
        """Expand one pipeline job into its visible replicate cards when a manifest exists."""
        sweeps = self._sweep_summaries(job)
        if sweeps:
            items = []
            for method, summary in sweeps:
                if summary.get("stop_reason") != "running":
                    continue
                trials = sorted(summary.get("trials", []), key=lambda trial: trial.get("id", 0))
                initial = summary.get("initial_trials") or 0
                requested = summary.get("requested_trials")
                recent = trials[initial:]
                finished = sum(t.get("state") in ("complete", "fail", "pruned") for t in recent)
                remaining = max(0, requested - len(recent)) if isinstance(requested, int) else None
                pending_count = summary.get('pending_trials', (remaining or 0) + sum(t.get('state') == 'pending' for t in recent))
                parent = dict(job, queue_method=method, queue_progress={'completed': finished, 'total': requested, 'unit': 'trials'}, trials_finished=finished, trials_total=requested, sweep_parent=True, sweep_key=f"{method}:{summary.get('study', '')}", agents=[], cancel_target=f"method:{method}",
                              description=f"{method} · {finished}/{requested} trials finished" if requested is not None else method)
                parent['description'] += f' · {pending_count} pending'
                items.append(parent)
                if remaining:
                    items.append(dict(job, queue_method=method, sweep_backlog=True, sweep_key=parent['sweep_key'], status='pending', agents=[],
                                      cancel_target=f'method:{method}',
                                      description=f'{remaining} future trials pending · parameters sampled when capacity opens'))
                for trial in recent:
                    if trial.get("state") not in ('running', 'pending'):
                        continue
                    execution = trial.get("execution")
                    if execution:
                        from src.app.pipeline.execution_data import replicate_label
                        runs = execution.get("runs", [])
                        trial_progress = _queue_run_progress(runs)
                        items.append(dict(job, queue_method=method, queue_trial=trial['id'],
                                          queue_study=summary.get('study', ''), queue_trial_parent=True,
                                          queue_progress=trial_progress, trial_log_version=trial.get('log_version'),
                                          agents=[], status=trial.get('state', 'pending')))
                        for replicate in runs:
                            state = self._replicate_state(replicate)
                            if state in ("completed", "failed", "cancelled"):
                                continue
                            label = replicate_label(execution, replicate)
                            target = (f"trial:{trial.get('log_version')}" if replicate.get('legacy_execution')
                                      else f"replicate:{replicate['run_id']}")
                            items.append(dict(job, queue_method=method, queue_trial=trial["id"], queue_study=summary.get("study", ""), sweep_trial=True, status=state, agents=[label],
                                              waiting_reason=replicate.get("waiting_reason", ""),
                                              cancel_target=target,
                                              execution_started=replicate.get("execution_started"),
                                              queue_seed=replicate.get("training_seed"),
                                              trial_log_version=trial.get("log_version"),
                                              description=f"{method} / Trial {trial['id']} / Seed {replicate['training_seed']}"))
                        continue
                    label = f"{method} · {summary.get('study', '')} · trial {trial['id']}"
                    version = trial.get("log_version")
                    trial_dir = self._run_dir(job) / (summary.get("log_method") or method) / str(version or "")
                    state = 'pending' if trial.get('state') == 'pending' else (
                        'running' if trial.get('execution_started') or version and any((trial_dir / name).exists()
                        for name in ('events.jsonl', 'metrics.csv')) else 'pending')
                    items.append(dict(job, queue_method=method, queue_trial=trial["id"], queue_study=summary.get("study", ""), sweep_trial=True, status=state, agents=[label],
                                      cancel_target=f"trial:{version}",
                                      description=f"{method} / Trial {trial['id']}"))
            if items:
                return items
        items = []
        manifests = list(self._execution_manifests(job))
        runs_by_method = {}
        for manifest in manifests:
            for run in manifest.get('runs', []):
                runs_by_method.setdefault(str(run.get('method', 'run')), []).append(run)
        for method, runs in runs_by_method.items():
            progress = _queue_run_progress(runs)
            items.append(dict(job, queue_method=method, queue_method_parent=True,
                              queue_progress=progress, agents=[], queue_job_status=job.get('status'),
                              status='completed' if progress['completed'] == progress['total'] else job.get('status')))
        for manifest in manifests:
            for replicate in manifest.get("runs", []):
                if replicate.get("status") in ("completed", "failed", "cancelled"):
                    continue
                method = str(replicate.get("method", "run"))
                from src.app.pipeline.execution_data import replicate_label
                label = replicate_label(manifest, replicate)
                item = dict(job)
                state = self._replicate_state(replicate)
                item.update(status=state, queue_method=method, queue_seed=replicate.get("training_seed"),
                            waiting_reason=replicate.get("waiting_reason", ""),
                            execution_started=replicate.get("execution_started"),
                            experiment_id=manifest.get("result_experiment_id", job.get("experiment_id")),
                            agents=[label], cancel_target=f"replicate:{replicate.get('run_id')}", description=f"Replicate {replicate.get('version', '')}".strip())
                items.append(item)
        return items if manifests else [dict(job, queue_root=True)]

    def _sweep_summaries(self, job):
        from src.app.pipeline.execution_data import study_execution_states
        root = self.project_root / "results" / "checkpoints" / str(job.get("group")) / str(job.get("experiment_id"))
        summaries = []
        for path in sorted(root.glob("*/tuning_summary.json")):
            try:
                if job.get("started") and job.get("status") in ACTIVE_STATUSES and path.stat().st_mtime < job["started"]:
                    continue
                summary = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if summary.get("component") == "optuna":
                method = summary.get("log_method") or path.parent.name
                summaries.append((path.parent.name, study_execution_states(summary, self._run_dir(job), method)))
        return summaries

    def replicate(self, job_id: str, replicate_label: str) -> dict[str, Any] | None:
        """Find a replicate and its stamped result ID from an active execution manifest."""
        job = self.jobs.get(job_id)
        if not job:
            return None
        for manifest in self._execution_manifests(job):
            for item in manifest.get("runs", []):
                from src.app.pipeline.execution_data import replicate_label as source_label
                label = source_label(manifest, item)
                if label == str(replicate_label):
                    return {**item, "group": manifest.get("group", job.get("group")),
                            "result_experiment_id": manifest.get("result_experiment_id", job.get("experiment_id"))}
        return None

    def _emit_queue(self):
        self.store.set_queue(self.queue_order)
        self.queue_changed.emit(self.queue_state())

    # ── Launching ────────────────────────────────────────────────────────────

    def launch(self, experiment: str, overrides: list[str] | None = None, queue: bool = False,
               overwrite: bool = False, _plan: dict | None = None) -> dict[str, Any]:
        """Validate exactly as the pipeline will, then start the run (or queue it). Raises LaunchError."""
        self._load_resource_settings()
        taken = [self.public(j) for j in self.jobs.values() if j.get("status") in ("queued", *ACTIVE_STATUSES)]
        plan = _plan or prepare_launch(experiment, overrides, overwrite=overwrite, active_jobs=taken)
        if _plan and any((j.get("group"), j.get("experiment_id")) == (plan["group"], plan["experiment_id"]) for j in taken):
            raise LaunchError("An active or queued job already uses these results.", conflict=True)
        job_id = str(uuid.uuid4())
        job = {
            "job_id": job_id,
            "status": "queued" if queue else "pending",
            "request": {"experiment": experiment, "overrides": list(overrides or []), "overwrite": overwrite,
                        "local": plan.get('local', True), "accelerator": plan.get('accelerator', True),
                        "resource_estimate": plan.get('resource_estimate')},
            "experiment": plan["experiment"],
            "group": plan["group"],
            "experiment_id": plan["experiment_id"],
            "agents": plan["agents"],
            "total_timesteps": plan["total_timesteps"],
            "effective_timesteps": plan["effective_timesteps"],
            "run_in_parallel": plan["run_in_parallel"],
            "created": time.time(),
        }
        must_wait = bool(self._resource_reason(job))
        self.jobs[job_id] = job
        if queue or must_wait:
            job['status'] = 'queued'
            if not queue:
                self.queue_running = True
            self.queue_order.append(job_id)
            self._save(job)
            self._emit_queue()
            return dict(self.public(job), position=self.queue_order.index(job_id), queue_running=self.queue_running)
        self._save(job)
        self._start(job)
        return self.public(job)

    def _start(self, job: dict[str, Any]):
        job_id = job["job_id"]
        request = job.get("request", {})
        process = QProcess(self)
        process.setProgram(self.python)
        from src.app.workspace import PROJECT_ROOT as APPLICATION_ROOT
        process.setArguments(["-u", str(APPLICATION_ROOT / "run_pipeline.py"), request.get("experiment", job["experiment"]),
                              *request.get("overrides", [])])
        process.setWorkingDirectory(str(self.project_root))
        env = QProcessEnvironment.systemEnvironment()
        for name in ("THETA_MCP_URL", "THETA_MCP_TOKEN"):  # runs get no access to an MCP server's tools
            env.remove(name)
        from src.app.workspace import WorkspacePaths
        for key, value in WorkspacePaths(self.project_root).process_environment().items():
            env.insert(key, value)
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("THETA_CANCEL_DIR", str((self.jobs_dir / "cancellations" / job_id).resolve()))
        env.insert("PYTHONIOENCODING", "utf-8")
        if self.settings_path is not None:
            env.insert("THETAIDE_SETTINGS_FILE", str(self.settings_path))
        if self.enabled_plugins_provider is not None:
            env.insert("THETA_ENABLED_PLUGINS", json.dumps(self.enabled_plugins_provider()))
        if request.get('local', True):
            cores, _total, _available = host_capacity()
            request['reserved_threads'] = self.resource_budget.threads(cores)
            job['resource_queue_ticket'] = self.resource_pool.enqueue()[0]
            env.insert('THETA_RESOURCE_QUEUE_PARENT', job['resource_queue_ticket'])
            env.insert('THETA_RESOURCE_POOL', str(self.resource_pool.path.resolve()))
            env.insert('THETA_RESOURCE_SPEC', json.dumps(request.get('resource_estimate') or
                       {'key': job['experiment'], 'memory': .75 * 1024**3, 'cpu': 1}))
            env.insert('THETA_RESOURCE_ACCELERATOR', '1' if request.get('accelerator', True) else '0')
            for key, value in managed_environment(request['reserved_threads']).items():
                env.insert(key, value)
        process.setProcessEnvironment(env)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        process.readyReadStandardOutput.connect(lambda j=job_id: self._on_output(j, "stdout"))
        process.readyReadStandardError.connect(lambda j=job_id: self._on_output(j, "stderr"))
        process.finished.connect(lambda code, _status, j=job_id: self._on_finished(j, code))
        process.errorOccurred.connect(lambda error, j=job_id: self._on_error(j, error))

        self._processes[job_id] = process
        self._decoders[job_id] = {stream: codecs.getincrementaldecoder("utf-8")(errors="replace")
                                  for stream in ("stdout", "stderr")}
        self._partial[job_id] = {"stdout": "", "stderr": ""}
        self._logs[job_id] = []
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._log_files[job_id] = open(self.jobs_dir / f"{job_id}.log", "w", encoding="utf-8")
        self._stream_files[job_id] = {stream: open(self.jobs_dir / f"{job_id}.{stream}.log", "wb")
                                      for stream in ("stdout", "stderr")}
        job.update(status="running", started=time.time())
        process.start()
        if not process.waitForStarted(10000):
            return  # errorOccurred(FailedToStart) finishes the job
        job["pid"] = int(process.processId())
        self._save(job)
        self._watch(job)
        self._execution_signatures[job_id] = self._execution_status_signature(job)
        self._emit_queue()

    # ── Process events ───────────────────────────────────────────────────────

    def _on_output(self, job_id: str, stream: str = "stdout", final: bool = False):
        process = self._processes.get(job_id)
        if process is None:
            return
        data = bytes(process.readAllStandardOutput() if stream == "stdout" else process.readAllStandardError())
        stream_file = self._stream_files[job_id][stream]
        stream_file.write(data)
        stream_file.flush()
        text = self._partial[job_id][stream] + self._decoders[job_id][stream].decode(data, final=final)
        *lines, self._partial[job_id][stream] = text.replace("\r\n", "\n").split("\n")
        if final and self._partial[job_id][stream]:
            lines.append(self._partial[job_id][stream])
            self._partial[job_id][stream] = ""
        if lines:
            self._record(job_id, lines)
        self._scan_metrics(job_id)
        self._sample_hardware(job_id)

    def _record(self, job_id: str, lines: list[str]):
        log = self._logs.setdefault(job_id, [])
        log.extend(lines)
        if len(log) > MAX_LOG_LINES:
            dropped = len(log) - MAX_LOG_LINES
            del log[:dropped]
            job = self.jobs[job_id]
            job["log_dropped"] = job.get("log_dropped", 0) + dropped
        log_file = self._log_files.get(job_id)
        if log_file:
            log_file.write("\n".join(lines) + "\n")
            log_file.flush()
        self.job_output.emit(job_id, lines)

    def _sample_hardware(self, job_id: str):
        """Fallback for runs without a telemetry stream: sample the pipeline process from outside."""
        if job_id in self._streamed_hardware:
            return
        now = time.monotonic()
        if now - self._last_hardware.get(job_id, 0) < HARDWARE_SAMPLE_SECONDS:
            return
        self._last_hardware[job_id] = now
        stats = process_stats(self.jobs[job_id].get("pid"))
        if stats:
            self.job_hardware.emit(job_id, stats)

    def _on_error(self, job_id: str, error):
        if error == QProcess.ProcessError.FailedToStart:
            job = self.jobs[job_id]
            job["error"] = f"Could not start {self.python}: {self._processes[job_id].errorString()}"
            self._stream_files[job_id]["stderr"].write((job["error"] + "\n").encode("utf-8"))
            self._record(job_id, [job["error"]])
            self._finish(job_id, None)

    def _on_finished(self, job_id: str, exit_code: int):
        for stream in ("stdout", "stderr"):
            self._on_output(job_id, stream, final=True)
        self._finish(job_id, exit_code)

    def _finish(self, job_id: str, exit_code: int | None):
        job = self.jobs[job_id]
        if job.get("status") in FINAL_STATUSES and job.get("finished"):
            return
        if job.get('resource_queue_ticket'):
            self.resource_pool.dequeue(job.pop('resource_queue_ticket'))
        self._scan_metrics(job_id)  # pick up the last rows written before exit
        job["returncode"] = exit_code
        if job.get("status") != "cancelled":
            if exit_code is None:
                job["status"] = "error"
            else:
                job["status"] = "completed" if exit_code == 0 else "failed"
        job["finished"] = time.time()
        log_file = self._log_files.pop(job_id, None)
        if log_file:
            log_file.close()
        for stream_file in self._stream_files.pop(job_id, {}).values():
            stream_file.close()
        self._decoders.pop(job_id, None)
        self._partial.pop(job_id, None)
        if job["status"] in ("failed", "error"):
            run_dir = self._run_dir(job)
            run_dir.mkdir(parents=True, exist_ok=True)
            for stream in ("stdout", "stderr"):
                shutil.copyfile(self.jobs_dir / f"{job_id}.{stream}.log", run_dir / f"{job_id}.{stream}.log")
        self._unwatch(job)
        process = self._processes.pop(job_id, None)
        if process is not None:
            process.deleteLater()
        self._save(job)
        self._start_next_queued()
        self._emit_queue()

    # ── Metrics: watch the run's folders and metrics.csv, read only what was appended ──

    def _run_dir(self, job: dict[str, Any]) -> Path:
        return self.logs_dir / job["group"] / job["experiment_id"]

    def _run_sources(self, job: dict[str, Any]):
        """Telemetry sources for a normal run or each replicate in its execution manifest."""
        sources = []
        for manifest in self._execution_manifests(job):
            for replicate in manifest.get("runs", []):
                run_dir = Path(replicate.get("logs", ""))
                if not run_dir.is_absolute():
                    run_dir = self.project_root / run_dir
                method = str(replicate.get("method", "run"))
                agent_dir_name = str(replicate.get("agent") or method.replace("/", "_"))
                from src.app.pipeline.execution_data import replicate_label
                label = replicate_label(manifest, replicate)
                version = replicate.get("version")
                if version is None:  # adapt manifests created before runs shared one experiment folder
                    sources.append((label, run_dir, agent_dir_name, "", self._replicate_state(replicate)))
                else:
                    sources.append((label, run_dir, agent_dir_name, str(version), self._replicate_state(replicate)))
        if sources:
            return sources
        run_dir = self._run_dir(job)
        sweeps = self._sweep_summaries(job)
        if sweeps:
            for method, summary in sweeps:
                for trial in summary.get("trials", []):
                    version = trial.get("log_version")
                    if version:
                        sources.append((f"{method} · {summary.get('study', '')} · trial {trial['id']}", run_dir,
                                        summary.get("log_method") or method, version,
                                        trial.get("state", "running")))
            if sources:
                return sources
        return [(str(agent), run_dir, str(agent), "", job.get("status", "running")) for agent in job.get("agents", [])]

    def _watch(self, job: dict[str, Any]):
        paths = [self.logs_dir, self._run_dir(job).parent]
        execution_root = self.project_root / "results" / "executions" / str(job.get("group")) / str(job.get("experiment_id"))
        checkpoint_root = self.project_root / "results" / "checkpoints" / str(job.get("group")) / str(job.get("experiment_id"))
        paths.extend([self.project_root / "results", self.project_root / "results" / "checkpoints",
                      checkpoint_root.parent, checkpoint_root])
        paths.extend(checkpoint_root.glob("*/tuning_summary.json"))
        paths.extend(checkpoint_root.glob("*"))
        if execution_root.exists():
            paths.append(execution_root)
            paths.extend(execution_root.glob("*/manifest.json"))
        for _label, run_dir, agent, explicit_version, _status in self._run_sources(job):
            paths.extend([run_dir.parent, run_dir])
            agent_dir = run_dir / agent
            paths.append(agent_dir)
            version_dirs = [agent_dir / explicit_version] if explicit_version else list(agent_dir.glob("version_*"))
            for version_dir in version_dirs:
                paths.extend([version_dir, version_dir / "events.jsonl", version_dir / "metrics.csv"])
        existing = {str(p) for p in paths if Path(p).exists()}
        watched = set(self.watcher.directories()) | set(self.watcher.files())
        new = sorted(existing - watched)
        if new:
            self.watcher.addPaths(new)

    def _unwatch(self, job: dict[str, Any]):
        roots = [str(run_dir) for _, run_dir, _, _, _ in self._run_sources(job)]
        roots.append(str(self._run_dir(job)))
        roots.append(str(self.project_root / "results" / "checkpoints" / str(job.get("group")) / str(job.get("experiment_id"))))
        stale = [p for p in self.watcher.directories() + self.watcher.files()
                 if any(p.startswith(root) for root in roots)]
        if stale:
            self.watcher.removePaths(stale)
        for key in [k for k in self._feeds if k[0] == job["job_id"]]:
            del self._feeds[key]
        self._streamed_hardware.discard(job["job_id"])
        self._execution_signatures.pop(job["job_id"], None)

    def _on_path_changed(self, _path: str):
        queue_changed = False
        for job_id, job in self.jobs.items():
            if job.get("status") == "running":
                before = self._execution_signatures.get(job_id, ())
                self._watch(job)  # folders created since the last change (agent, version_N, metrics.csv)
                self._scan_metrics(job_id)
                after = self._execution_status_signature(job)
                self._execution_signatures[job_id] = after
                if before != after:
                    queue_changed = True
        if queue_changed:
            self._emit_queue()

    def _execution_status_signature(self, job):
        signature = []
        for manifest in self._execution_manifests(job):
            for run in manifest.get("runs", []):
                output = Path(run.get("output_log") or "__missing_worker_log__")
                if not output.is_absolute():
                    output = self.project_root / output
                signature.append((run.get("run_id"), run.get("status"), run.get("waiting_reason"), output.is_file()))
        for method, summary in self._sweep_summaries(job):
            signature.append((method, summary.get("stop_reason"),
                              tuple((t.get("id"), t.get("state"),
                                     any((self._run_dir(job) / (summary.get("log_method") or method)
                                          / str(t.get("log_version") or "") / name).exists()
                                         for name in ("events.jsonl", "metrics.csv")))
                                    for t in summary.get("trials", []))))
        return tuple(signature)

    def _scan_metrics(self, job_id: str):
        """Read what each agent's run appended since the last scan and signal it."""
        job = self.jobs.get(job_id)
        if not job:
            return
        sources = self._run_sources(job)
        for source in sources:
            if source[4] in ("queued", "pending"):
                continue
            label, run_dir, agent_dir, version, _status = source
            feed = self._feeds.get((job_id, label))
            if feed is None:
                feed_path = run_dir / agent_dir
                self._feeds[(job_id, label)] = feed = AgentFeed(feed_path, version=version or None)
            limits = getattr(self, "_telemetry_read_limits", None)
            records, reset = feed.read(**limits) if limits else feed.read()
            if not records and not reset:
                continue
            # A telemetry run owns its resource policy, even when it deliberately
            # sends no system records. Do not restart external probing then.
            if any(record.get("type") == "start" for record in records):
                self._streamed_hardware.add(job_id)
            self.job_records.emit(job_id, label, records, reset)
            rows = [metrics_row(r) for r in records if r.get("type") == "metrics"]
            rows = [row for row in rows if row]
            if rows or reset:
                self.job_metrics.emit(job_id, label, rows, reset)
            samples = [r["data"] for r in records if r.get("type") == "system" and r.get("data")]
            if samples:
                self._streamed_hardware.add(job_id)
                latest = dict(samples[-1])
                if "rss_mb" in latest:
                    latest["memory_mb"] = latest["rss_mb"]  # the name job_hardware has always used
                self.job_hardware.emit(job_id, latest)

    def focus_replicate(self, job_id: str, replicate_label: str):
        """Follow one replicate's live feed while sibling runs continue independently."""
        self._focused_replicates[job_id] = str(replicate_label)
        self._scan_metrics(job_id)

    # ── Stopping and the queue ───────────────────────────────────────────────

    def cancel(self, job_id: str, target: str = "") -> dict[str, Any]:
        job = self.jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        status = job.get("status")
        if target and (status in ACTIVE_STATUSES or (status == "queued" and len(job.get("agents") or []) > 1)):
            from src.app.services.cancellation import request_path
            directory = self.jobs_dir / "cancellations" / job_id
            directory.mkdir(parents=True, exist_ok=True)
            request_path(directory, target).touch()
            return self.public(job)
        if status == "queued":  # never started: take it out of the queue
            if job_id in self.queue_order:
                self.queue_order.remove(job_id)
            job.update(status="cancelled", finished=time.time())
            self._save(job)
            self._emit_queue()
            return self.public(job)
        if status not in ACTIVE_STATUSES:
            return self.public(job)
        job["status"] = "cancelled"
        self._save(job)
        process = self._processes.get(job_id)
        if process is not None and process.state() != QProcess.ProcessState.NotRunning:
            process_id = int(process.processId())
            kill_process_tree(process_id)
            if process.state() != QProcess.ProcessState.NotRunning:
                process.kill()
            from src.app.pipeline.variability import mark_cancelled
            mark_cancelled(job.get("group", ""), job.get("experiment_id", ""), process_id)
        return self.public(job)

    def start_queue(self) -> dict[str, Any]:
        """Start queued work as resources become available."""
        self.resource_pool.set_paused(False)
        self.queue_running = bool(self.queue_order or self.running_count())
        self._start_next_queued()
        self._emit_queue()
        return {"running": self.queue_running, "queued": len(self.queue_order)}

    def pause_queue(self) -> dict[str, Any]:
        """Stop starting queued jobs. A job that is already training keeps running."""
        self.queue_running = False
        self.resource_pool.set_paused(True)
        self._emit_queue()
        return {"running": False, "queued": len(self.queue_order)}

    def move(self, job_id: str, position: int) -> int:
        if job_id not in self.queue_order:
            raise LaunchError("Only queued jobs can be moved.", conflict=True)
        self.queue_order.remove(job_id)
        self.queue_order.insert(max(0, min(position, len(self.queue_order))), job_id)
        self._emit_queue()
        return self.queue_order.index(job_id)

    def _start_next_queued(self):
        """Admit by resources, staggering coordinator startup for measurement."""
        self._load_resource_settings()
        self.waiting_reason = ''
        while self.queue_running and self.queue_order:
            self.waiting_reason = self._resource_reason(self.jobs[self.queue_order[0]])
            if self.waiting_reason:
                break
            job = self.jobs[self.queue_order.pop(0)]
            job["status"] = "pending"
            self._start(job)
            # Let the next resource tick observe this coordinator's startup.
            break
        if self.queue_running and not self.queue_order and not self.running_count():
            self.queue_running = False  # drained and the last job finished: jobs added later wait for Start

    def shutdown(self):
        """Stop every running job (the app is closing; QProcess children end with it)."""
        self.queue_running = False
        for job_id in [j for j, job in self.jobs.items() if job.get("status") in ACTIVE_STATUSES]:
            self.cancel(job_id)
        for process in list(self._processes.values()):
            process.waitForFinished(5000)
