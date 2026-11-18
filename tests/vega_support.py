"""Run figure specs in the vendored Vega and Vega-Lite builds with Node, when Node is installed.

``run_specs([(spec, signals), …], config=…)`` returns one ``{ok, error, warnings, svg}`` per spec. Tests
that use it skip when ``node`` is not on PATH.
"""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

VENDOR = Path(__file__).resolve().parent.parent / "frontend" / "discussion" / "vendor"
CHECK = Path(__file__).with_name("vega_check.js")
NODE = shutil.which("node")
requires_node = pytest.mark.skipif(NODE is None, reason="node is not installed")


def run_specs(items, *, config=None):
    with tempfile.TemporaryDirectory() as modules:
        for name, build in (("vega", "vega.min.js"), ("vega-lite", "vega-lite.min.js")):
            package = Path(modules) / name
            package.mkdir()
            (package / "index.js").write_text(f"module.exports = require({json.dumps(str(VENDOR / build))});\n")
        payload = json.dumps([{"spec": spec, "signals": signals or {}, "config": config} for spec, signals in items])
        result = subprocess.run([NODE, str(CHECK)], input=payload, capture_output=True, text=True, timeout=120,
                                env={"NODE_PATH": modules, "PATH": ""}, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return json.loads(result.stdout)
