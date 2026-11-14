"""A persistent, lazily started validation interpreter owned by one desktop."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from uuid import uuid4


class ValidationProcess:
    def __init__(self, *, workspace=None):
        self._directory = tempfile.TemporaryDirectory(prefix="theta-validation-", ignore_cleanup_errors=True)
        self.directory = Path(self._directory.name)
        self._closed = threading.Event()
        self._process = None
        from src.app.workspace import WorkspacePaths
        self.paths = workspace or WorkspacePaths.for_process()
        self._environment = {**os.environ, **self.paths.process_environment()}

    def call(self, operation, *args):
        if self._closed.is_set():
            raise RuntimeError("Validation service is closed")
        process = self._process
        if process is None or process.poll() is not None:
            source = self.paths.application
            env = dict(self._environment)
            env["PYTHONPATH"] = str(source) + os.pathsep + env.get("PYTHONPATH", "")
            # Keep provider output away from the protocol. Diagnostics survive
            # until this desktop closes, including native crashes in providers.
            with (self.directory / "worker.log").open("ab") as output:
                process = subprocess.Popen(
                    [sys.executable, "-m", "frontend.validation_worker"], cwd=self.paths.root, env=env,
                    stdin=subprocess.PIPE, stdout=output, stderr=output,
                )
            self._process = process
            if self._closed.is_set():
                process.kill()
                process.wait()
                raise RuntimeError("Validation service is closed")
        reply = self.directory / f"{uuid4().hex}.json"
        request = {"operation": operation, "args": args, "reply": str(reply)}
        try:
            process.stdin.write((json.dumps(request) + "\n").encode())
            process.stdin.flush()
            deadline = time.monotonic() + 120
            while not reply.is_file():
                if self._closed.is_set():
                    raise RuntimeError("Validation service is closed")
                if process.poll() is not None:
                    detail = (self.directory / "worker.log").read_text(errors="replace")[-3000:]
                    raise RuntimeError("Validation process exited: " + detail)
                if time.monotonic() >= deadline:
                    process.kill()
                    process.wait()
                    raise RuntimeError("Validation process timed out")
                time.sleep(.02)  # This runs on LocalBackend's worker, never the GUI thread.
            response = json.loads(reply.read_text())
            if response.get("error"):
                raise RuntimeError(response["error"])
            return response["data"]
        finally:
            reply.unlink(missing_ok=True)

    def shutdown(self):
        self._closed.set()
        process = self._process
        if process is not None and process.poll() is None:
            process.kill()  # It owns validation only, never training or the queue.
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
        self._directory.cleanup()
