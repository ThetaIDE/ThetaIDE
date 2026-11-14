"""TensorBoard background server runner via QProcess."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import socket
import sys
from typing import Optional

from PyQt6.QtCore import QObject, QProcess, pyqtSignal

from src.app.api.execution import kill_process_tree

READY_LINE = re.compile(r"TensorBoard \S+ at (http://\S+)")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class TensorBoardServer(QObject):
    """Local TensorBoard server managed as a child QProcess.

    Signals:
        starting: Emitted when process start is initiated.
        ready(url): Emitted when TensorBoard logs that it is serving HTTP.
        stopped(error): Emitted when the server terminates.
    """

    starting = pyqtSignal()
    ready = pyqtSignal(str)  # url
    stopped = pyqtSignal(str)  # error text, or "" when stopped intentionally

    def __init__(self, python: Optional[str] = None, logdir: Optional[Path] = None, parent: Optional[QObject] = None,
                 working_dir: Optional[Path] = None):
        super().__init__(parent)
        self.python = python or sys.executable
        if logdir is None or working_dir is None:
            from src.app.api.workspace import WorkspacePaths
            paths = WorkspacePaths.for_process()
            logdir = logdir or paths.results / "tensorboard"
            working_dir = working_dir or paths.root
        self.logdir = Path(logdir)
        self.working_dir = Path(working_dir)
        self.process: Optional[QProcess] = None
        self.url: Optional[str] = None
        self._output = ""
        self._stopping = False
        self._tracker_callback = None

    def set_tracker_callback(self, cb) -> None:
        """Register a callback (e.g. context.track_process) to track this process."""
        self._tracker_callback = cb

    @staticmethod
    def available() -> bool:
        """Return True if tensorboard package is importable."""
        return importlib.util.find_spec("tensorboard") is not None

    def is_running(self) -> bool:
        """Return True if the server process is currently running."""
        return self.process is not None and self.process.state() != QProcess.ProcessState.NotRunning

    def start(self) -> None:
        """Start the TensorBoard server process if not already running."""
        if self.is_running():
            return
        if not self.available():
            self.stopped.emit("tensorboard is not installed in this environment")
            return
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.url, self._output, self._stopping = None, "", False
        self.process = QProcess(self)
        self.process.setProgram(self.python)
        self.process.setArguments([
            "-m", "tensorboard.main",
            "--logdir", str(self.logdir),
            "--host", "127.0.0.1",
            "--port", str(_free_port()),
            "--reload_interval", "5",
        ])
        self.process.setWorkingDirectory(str(self.working_dir))
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._on_output)
        self.process.finished.connect(self._on_finished)
        self.process.errorOccurred.connect(self._on_error)

        if self._tracker_callback is not None:
            self._tracker_callback(self.process)

        self.starting.emit()
        self.process.start()

    def _on_output(self, *_) -> None:
        if self.process is None:
            return
        self._output += bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._output = self._output[-4000:]
        if self.url is None and (match := READY_LINE.search(self._output)):
            self.url = match.group(1)
            self.ready.emit(self.url)

    def _on_error(self, error) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self.stopped.emit(f"Could not start TensorBoard: {self.process.errorString()}")

    def _on_finished(self, *_) -> None:
        error = "" if self._stopping else (self._output.strip().splitlines() or ["TensorBoard exited"])[-1]
        self.url = None
        if self.process:
            self.process.deleteLater()
            self.process = None
        self.stopped.emit(error)

    def stop(self) -> None:
        """Terminate the running server and all its child processes."""
        if not self.is_running():
            return
        self._stopping = True
        try:
            pid = int(self.process.processId())
            if pid > 0:
                kill_process_tree(pid)
        except Exception:
            pass
        if self.process is not None:
            if self.process.state() != QProcess.ProcessState.NotRunning:
                self.process.kill()
                self.process.waitForFinished(2000)
            if self.url is not None:
                self._on_finished()
