"""Opt-in startup milestones. Set THETAIDE_STARTUP_TRACE to a JSONL file."""
import json
import os
import time

_STARTED = float(os.environ.get("THETAIDE_STARTUP_EPOCH", time.perf_counter()))


def mark(event, **details):
    target = os.environ.get("THETAIDE_STARTUP_TRACE")
    if not target:
        return
    record = {"event": event, "elapsed_ms": round((time.perf_counter() - _STARTED) * 1000, 3),
              "pid": os.getpid(), "startup_id": f"{os.getpid()}-{_STARTED:.9f}", **details}
    try:
        with open(target, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
    except OSError:
        pass  # Diagnostics must never prevent launching the editor.
