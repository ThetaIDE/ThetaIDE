"""Record uncaught desktop exceptions instead of aborting the process.

PyQt calls ``qFatal`` (abort) when a slot raises and ``sys.excepthook`` is the
default. A desktop launched from the app bundle has no stderr, so the abort
also loses the traceback. Training runs in the independent runner, so a failed
UI callback must neither close the window nor go unrecorded.
"""
from __future__ import annotations

import faulthandler
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

MAX_BYTES = 1_000_000
_reporter = None
_fault_stream = None


def log_path(state_dir) -> Path:
    return Path(state_dir) / "logs" / "desktop-errors.log"


def set_reporter(callback):
    """Show later errors to the user, e.g. in the console; ``None`` stops reporting."""
    global _reporter
    _reporter = callback


def _append(path: Path, text: str):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and path.stat().st_size > MAX_BYTES:
            path.replace(path.with_suffix(".log.1"))
        with path.open("a", encoding="utf-8") as stream:
            stream.write(text)
    except OSError:
        pass


def install(state_dir) -> Path:
    """Log uncaught exceptions to ``<state>/logs/desktop-errors.log`` and keep running."""
    global _fault_stream
    path = log_path(state_dir)
    stderr = sys.__stderr__
    if stderr is None or not stderr.isatty():
        # Native faults (segfaults, Qt aborts) have nowhere to go without a terminal.
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            _fault_stream = open(path.with_name("desktop-faults.log"), "a", encoding="utf-8")
            faulthandler.enable(file=_fault_stream, all_threads=True)
        except OSError:
            _fault_stream = None

    def hook(exc_type, exc, tb):
        if issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
            sys.__excepthook__(exc_type, exc, tb)
            return
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        stamp = datetime.now().isoformat(timespec="seconds")
        _append(path, f"--- {stamp} pid {os.getpid()}\n{text}\n")
        if stderr is not None:
            try:
                stderr.write(text)
                stderr.flush()
            except (OSError, ValueError):
                pass
        reporter = _reporter
        if reporter is not None:
            try:
                reporter(f"{exc_type.__name__}: {exc}", path)
            except Exception:  # reporting must never raise from the hook
                pass

    sys.excepthook = hook
    return path
