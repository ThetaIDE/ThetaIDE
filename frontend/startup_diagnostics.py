"""Opt-in main-thread profiles and event-loop latency for the real launcher."""
import atexit
import cProfile
import threading
import traceback
import sys
import json
import os
from pathlib import Path
import time

_profile = None
_stacks = None
_directory = None
_stop = threading.Event()
_sampler = None


def begin(directory, profile=True):
    global _profile, _stacks, _directory, _sampler
    _directory = Path(directory) / str(os.getpid())
    _directory.mkdir(parents=True, exist_ok=True)
    os.environ['THETAIDE_STARTUP_TRACE'] = str(_directory / 'events.jsonl')
    _stacks = (_directory / 'stacks.log').open('w')
    _stop.clear()
    main_id = threading.main_thread().ident
    def sample():
        while not _stop.wait(.5):
            frame = sys._current_frames().get(main_id)
            if frame is not None:
                _stacks.write(f"\nSAMPLE {time.perf_counter()}\n" + ''.join(traceback.format_stack(frame)))
                _stacks.flush()
    if profile:
        _sampler = threading.Thread(target=sample, name="startup-stack-sampler", daemon=True)
        _sampler.start()
        _profile = cProfile.Profile()
        _profile.enable()
    atexit.register(finish)


def finish():
    global _profile, _stacks, _sampler
    if _profile is not None:
        _profile.disable()
        _profile.dump_stats(str(_directory / 'main-thread.pstats'))
        _profile = None
    if _stacks is not None:
        _stop.set()
        if _sampler is not None:
            _sampler.join(timeout=1)
            _sampler = None
        _stacks.close()
        _stacks = None


def attach(app, window, exit_after=0):
    from PyQt6.QtCore import QTimer
    from .startup import mark
    previous = [time.perf_counter()]
    maximum = [0.0]
    def tick():
        now = time.perf_counter()
        gap = (now - previous[0]) * 1000
        previous[0] = now
        maximum[0] = max(maximum[0], gap)
        modal = app.activeModalWidget()
        mark('event_loop_heartbeat', gap_ms=round(gap, 3), enabled=window.isEnabled(),
             modal=type(modal).__name__ if modal is not None else None)
    timer = QTimer(app)
    timer.setInterval(50)
    timer.timeout.connect(tick)
    timer.start()
    app._startup_diagnostic_timer = timer
    def completed():
        (_directory / 'latency.json').write_text(json.dumps({'max_gap_ms': maximum[0]}))
        finish()
    app.aboutToQuit.connect(completed)
    if exit_after > 0:
        QTimer.singleShot(round(exit_after * 1000), window.quit_app)
