"""Drain both process pipes while retaining combined output and real stderr."""
from __future__ import annotations

import threading


class ProcessOutput:
    """Keep stderr identifiable through nested workers without blocking either pipe."""

    def __init__(self, process, combined, errors, *, stdout=None, stderr=None):
        self.combined, self.errors = combined, errors
        self.stdout, self.stderr = stdout, stderr
        self.lock = threading.RLock()
        self.failures = []
        self.threads = []
        for stream, is_error in ((process.stdout, False), (process.stderr, True)):
            thread = threading.Thread(target=self._forward, args=(stream, is_error), daemon=True)
            thread.start()
            self.threads.append(thread)

    def _write(self, text, is_error):
        with self.lock:
            console = self.stderr if is_error else self.stdout
            targets = (self.combined, self.errors if is_error else None, console)
            failure = None
            for target in targets:
                if target is None:
                    continue
                try:
                    target.write(text)
                    target.flush()
                except Exception as exc:
                    if failure is None:
                        failure = exc
            if failure is not None:
                raise failure

    def _forward(self, stream, is_error):
        try:
            with stream:
                for line in stream:
                    try:
                        self._write(line, is_error)
                    except Exception as exc:
                        if not self.failures:
                            self.failures.append(exc)
                        # Drain even when storage fails; never block the child on a full pipe.
        except Exception as exc:
            if not self.failures:
                self.failures.append(exc)

    def write_error(self, text):
        try:
            self._write(text, True)
        except Exception as exc:
            if not self.failures:
                self.failures.append(exc)
            # All destinations, including scheduler stderr, were attempted above.
            # Logging failure must not prevent termination and final publication.

    def finish(self):
        for thread in self.threads:
            thread.join()
        if self.failures:
            raise OSError('Could not retain process output') from self.failures[0]
