"""Static check that package code reaches the host only through the Theta SDK (`src.app.api`).

Used by `tests/test_sdk_boundary.py` for the packages in this repository and by package
builders for packages maintained elsewhere.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

HOST = re.compile(r"^(src\.app|frontend)(\.|$)")
SDK = re.compile(r"^src\.app\.api(\.|$)")
# A string naming a private host module, e.g. a `python -m src.app.x` argument.
MODULE_STRING = re.compile(r"^(src\.app|frontend)(\.[A-Za-z_][A-Za-z0-9_]*)+$")
WINDOW_ACCESS = re.compile(r"\bcontext\.window\b|\bcontext\._window\b|\.context\.window\b")


def private_references(path: Path) -> list[tuple[int, str]]:
    """(line, module) for every private host module `path` imports or names."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    found = []
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        elif isinstance(node, ast.Call):
            func = node.func
            called = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
            if called in ("__import__", "import_module") and node.args and isinstance(node.args[0], ast.Constant):
                names = [str(node.args[0].value)]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and MODULE_STRING.match(node.value):
            names = [node.value]
        found += [(node.lineno, name) for name in names if HOST.match(name) and not SDK.match(name)]
    return sorted(set(found))


def window_access(path: Path) -> list[int]:
    """Lines of `path` that reach into the host window through a plugin context."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [number for number, line in enumerate(lines, 1) if WINDOW_ACCESS.search(line)]


def violations(root: Path) -> list[str]:
    """Every private import and window access under `root`, as `relative/path.py:line: detail`."""
    root = Path(root)
    found = []
    for path in sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts):
        relative = path.relative_to(root).as_posix()
        found += [f"{relative}:{line}: imports private host module {name}" for line, name in private_references(path)]
        found += [f"{relative}:{line}: reaches into the host window" for line in window_access(path)]
    return found
