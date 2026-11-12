"""Synthetic monitor refresh timings; run with venv/bin/python scripts/benchmark_monitor_refresh.py.

Measures synchronous refresh/data work without painting a desktop window. No user
runs are read or changed. --baseline loads these modules from Git HEAD in memory.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def head_module(name, path):
    source = subprocess.check_output(['git', 'show', f'HEAD:{path}'], cwd=ROOT, text=True)
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(source, path, 'exec'), module.__dict__)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--rows', type=int, default=50000)
    parser.add_argument('--metrics', type=int, default=12)
    args = parser.parse_args()
    from PyQt6.QtWidgets import QApplication
    from frontend import live_run
    from frontend.monitor import live_view
    app = QApplication.instance() or QApplication([])
    if args.baseline:
        live_run = head_module('frontend._benchmark_live_run', 'frontend/live_run.py')
        live_view = head_module('frontend.monitor._benchmark_live_view', 'frontend/monitor/live_view.py')

    def timed(label, operation):
        start = time.perf_counter()
        operation()
        print(f'{label}: {(time.perf_counter() - start) * 1000:.2f} ms')

    run = live_run.LiveRun()
    run.rows = [{'step': i, **{f'loss/{j}': float((i + j) % 100) for j in range(args.metrics)}}
                for i in range(args.rows)]
    view = live_view.ChartsView()
    timed('initial chart history', lambda: view.show_lives({'a': run}))
    for dense in (False, True):
        for iteration in range(3):
            row = {'step': len(run.rows), **{f'loss/{j}': 1. for j in range(args.metrics if dense else 1)}}
            run.rows.append(row)
            timed(f'{"dense" if dense else "sparse"} append {iteration + 1}', lambda: view.show_lives({'a': run}))
    timed('unchanged charts', lambda: view.show_lives({'a': run}))
    view.close()
    with tempfile.TemporaryDirectory() as directory:
        version = Path(directory) / 'g/e/a/version_0'
        version.mkdir(parents=True)
        stream = version / 'events.jsonl'

        def record(step):
            return json.dumps({'v': 1, 'type': 'metrics', 'step': step, 'data': {'loss': float(step)}}) + '\n'

        stream.write_text(''.join(record(i) for i in range(args.rows)))
        cache = live_run.LiveRunCache()
        load = lambda: cache.load(directory, 'g', 'e', ['a'])
        timed('initial stream history', load)
        for iteration in range(3):
            with stream.open('a') as output:
                output.write(record(args.rows + iteration))
            timed(f'stream append {iteration + 1}', load)
    app.processEvents()


if __name__ == '__main__':
    main()
