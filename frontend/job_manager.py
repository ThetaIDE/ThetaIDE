"""Runs training jobs inside the desktop app: run_pipeline.py as a QProcess, reported through signals.

Replaces the FastAPI job endpoints. Output arrives through readyReadStandardOutput; telemetry through a
QFileSystemWatcher on the run's events.jsonl (or metrics.csv, for engines that write only that), read by
an AgentFeed per agent; status and queue changes are signals, so nothing polls.
Jobs persist in results/jobs/jobs.db (JobStore), as before.
"""
from __future__ import annotations

import codecs
import json
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QFileSystemWatcher, QObject, QProcess, QProcessEnvironment, pyqtSignal

from src.app.services import JOBS_DIR, LOGS_DIR, PROJECT_ROOT
from src.app.services.config_service import LaunchError, prepare_launch
from src.app.services.job_store import JobStore
from src.app.services.metrics import AgentFeed
from src.app.services.processes import kill_process_tree, process_stats
from src.app.telemetry.records import metrics_row

ACTIVE_STATUSES = ("pending", "running")
FINAL_STATUSES = ("completed", "failed", "cancelled", "error", "interrupted")
MAX_LOG_LINES = 20000
HARDWARE_SAMPLE_SECONDS = 1.0


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
        self.max_parallel = 1  # how many jobs train at once (the Queue tab's "Run up to N at once")
        self._processes: dict[str, QProcess] = {}
        self._decoders: dict[str, Any] = {}
        self._partial: dict[str, str] = {}
        self._logs: dict[str, list[str]] = {}
        self._log_files: dict[str, Any] = {}
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
        """Whether another job may start now (fewer than max_parallel are training)."""
        return self.running_count() < self.max_parallel

    def set_max_parallel(self, count: int):
        """Train up to `count` jobs at once; a running queue starts more right away if there is room."""
        self.max_parallel = max(1, int(count))
        self._start_next_queued()
        self._emit_queue()

    def queue_state(self) -> dict[str, Any]:
        """The active job(s), queued jobs in run order, and the ten most recent finished jobs."""
        finished = sorted((j for j in self.jobs.values() if j.get("status") in FINAL_STATUSES),
                          key=lambda j: j.get("finished") or 0, reverse=True)[:10]
        active = [item for job in self.active_jobs() for item in self._execution_queue_items(job)]
        return {
            "running": self.queue_running,
            "active": active,
            "queued": [dict(self.public(self.jobs[job_id]), position=n)
                       for n, job_id in enumerate(self.queue_order) if job_id in self.jobs],
            "finished": [self.public(j) for j in finished],
        }

    def _execution_queue_items(self, job):
        """Expand one pipeline job into its visible replicate cards when a manifest exists."""
        root = self.project_root / "results" / "executions" / str(job.get("group")) / str(job.get("experiment_id"))
        items = []
        for path in root.glob("*/manifest.json"):
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            for replicate in manifest.get("runs", []):
                if replicate.get("status") in ("completed", "failed", "cancelled"):
                    continue
                method = str(replicate.get("method", "run"))
                label = f"{method} · seed {replicate.get('training_seed')}"
                item = dict(job)
                item.update(status=replicate.get("status", "queued"),
                            experiment_id=manifest.get("result_experiment_id", job.get("experiment_id")),
                            agents=[label], description=f"Replicate {replicate.get('version', '')}".strip())
                items.append(item)
        return items or [job]

    def replicate(self, job_id: str, replicate_label: str) -> dict[str, Any] | None:
        """Find a replicate and its stamped result ID from an active execution manifest."""
        job = self.jobs.get(job_id)
        if not job:
            return None
        root = self.project_root / "results" / "executions" / str(job.get("group")) / str(job.get("experiment_id"))
        for path in root.glob("*/manifest.json"):
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            for item in manifest.get("runs", []):
                label = f"{item.get('method', 'run')} · seed {item.get('training_seed')}"
                if label == str(replicate_label):
                    return {**item, "group": manifest.get("group", job.get("group")),
                            "result_experiment_id": manifest.get("result_experiment_id", job.get("experiment_id"))}
        return None

    def _emit_queue(self):
        self.store.set_queue(self.queue_order)
        self.queue_changed.emit(self.queue_state())

    # ── Launching ────────────────────────────────────────────────────────────

    def launch(self, experiment: str, overrides: list[str] | None = None, queue: bool = False,
               overwrite: bool = False) -> dict[str, Any]:
        """Validate exactly as the pipeline will, then start the run (or queue it). Raises LaunchError."""
        taken = [self.public(j) for j in self.jobs.values() if j.get("status") in ("queued", *ACTIVE_STATUSES)]
        plan = prepare_launch(experiment, overrides, overwrite=overwrite, active_jobs=taken)
        job_id = str(uuid.uuid4())
        job = {
            "job_id": job_id,
            "status": "queued" if queue else "pending",
            "request": {"experiment": experiment, "overrides": list(overrides or []), "overwrite": overwrite},
            "experiment": plan["experiment"],
            "group": plan["group"],
            "experiment_id": plan["experiment_id"],
            "agents": plan["agents"],
            "total_timesteps": plan["total_timesteps"],
            "effective_timesteps": plan["effective_timesteps"],
            "run_in_parallel": plan["run_in_parallel"],
            "created": time.time(),
        }
        self.jobs[job_id] = job
        if queue:
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
        process.setArguments(["-u", "run_pipeline.py", request.get("experiment", job["experiment"]),
                              *request.get("overrides", [])])
        process.setWorkingDirectory(str(self.project_root))
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        if self.settings_path is not None:
            env.insert("THETAIDE_SETTINGS_FILE", str(self.settings_path))
        if self.enabled_plugins_provider is not None:
            env.insert("THETA_ENABLED_PLUGINS", json.dumps(self.enabled_plugins_provider()))
        process.setProcessEnvironment(env)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(lambda j=job_id: self._on_output(j))
        process.finished.connect(lambda code, _status, j=job_id: self._on_finished(j, code))
        process.errorOccurred.connect(lambda error, j=job_id: self._on_error(j, error))

        self._processes[job_id] = process
        self._decoders[job_id] = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._partial[job_id] = ""
        self._logs[job_id] = []
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._log_files[job_id] = open(self.jobs_dir / f"{job_id}.log", "w", encoding="utf-8")
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

    def _on_output(self, job_id: str):
        process = self._processes.get(job_id)
        if process is None:
            return
        text = self._partial.get(job_id, "") + self._decoders[job_id].decode(bytes(process.readAllStandardOutput()))
        *lines, self._partial[job_id] = text.replace("\r\n", "\n").split("\n")
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
            self._finish(job_id, None)

    def _on_finished(self, job_id: str, exit_code: int):
        self._on_output(job_id)  # output can still be buffered when the process reports it exited
        leftover = self._partial.pop(job_id, "")
        if leftover:
            self._record(job_id, [leftover])
        self._finish(job_id, exit_code)

    def _finish(self, job_id: str, exit_code: int | None):
        job = self.jobs[job_id]
        if job.get("status") in FINAL_STATUSES and job.get("finished"):
            return
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
        execution_root = self.project_root / "results" / "executions" / job["group"] / job["experiment_id"]
        for manifest_path in execution_root.glob("*/manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            for replicate in manifest.get("runs", []):
                run_dir = Path(replicate.get("logs", ""))
                if not run_dir.is_absolute():
                    run_dir = self.project_root / run_dir
                method = str(replicate.get("method", "run"))
                agent_dir_name = str(replicate.get("agent") or method.replace("/", "_"))
                label = f"{method} · seed {replicate.get('training_seed')}"
                version = replicate.get("version")
                if version is None:  # adapt manifests created before runs shared one experiment folder
                    sources.append((label, run_dir, agent_dir_name, "", replicate.get("status", "queued")))
                else:
                    sources.append((label, run_dir, agent_dir_name, str(version), replicate.get("status", "queued")))
        if sources:
            return sources
        run_dir = self._run_dir(job)
        return [(str(agent), run_dir, str(agent), "", job.get("status", "running")) for agent in job.get("agents", [])]

    def _watch(self, job: dict[str, Any]):
        paths = [self.logs_dir, self._run_dir(job).parent]
        execution_root = self.project_root / "results" / "executions" / str(job.get("group")) / str(job.get("experiment_id"))
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
        root = self.project_root / "results" / "executions" / str(job.get("group")) / str(job.get("experiment_id"))
        signature = []
        for path in root.glob("*/manifest.json"):
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
                signature.extend((run.get("run_id"), run.get("status")) for run in manifest.get("runs", []))
            except (OSError, ValueError):
                continue
        return tuple(signature)

    def _scan_metrics(self, job_id: str):
        """Read what each agent's run appended since the last scan and signal it."""
        job = self.jobs.get(job_id)
        if not job:
            return
        sources = self._run_sources(job)
        for source in sources:
            if source[4] == "queued":
                continue
            label, run_dir, agent_dir, version, _status = source
            feed = self._feeds.get((job_id, label))
            if feed is None:
                feed_path = run_dir / agent_dir
                self._feeds[(job_id, label)] = feed = AgentFeed(feed_path, version=version or None)
            records, reset = feed.read()
            if not records and not reset:
                continue
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

    def cancel(self, job_id: str) -> dict[str, Any]:
        job = self.jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        status = job.get("status")
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
        """Start running queued jobs in order, up to max_parallel at once. Starting an empty queue leaves it paused."""
        self.queue_running = bool(self.queue_order)
        self._start_next_queued()
        self._emit_queue()
        return {"running": self.queue_running, "queued": len(self.queue_order)}

    def pause_queue(self) -> dict[str, Any]:
        """Stop starting queued jobs. A job that is already training keeps running."""
        self.queue_running = False
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
        """Start queued jobs, in order, while the queue runs and fewer than max_parallel are training."""
        while self.queue_running and self.queue_order and self.has_capacity():
            job = self.jobs[self.queue_order.pop(0)]
            job["status"] = "pending"
            self._start(job)
        if self.queue_running and not self.queue_order and not self.running_count():
            self.queue_running = False  # drained and the last job finished: jobs added later wait for Start

    def shutdown(self):
        """Stop every running job (the app is closing; QProcess children end with it)."""
        self.queue_running = False
        for job_id in [j for j, job in self.jobs.items() if job.get("status") in ACTIVE_STATUSES]:
            self.cancel(job_id)
        for process in list(self._processes.values()):
            process.waitForFinished(5000)
