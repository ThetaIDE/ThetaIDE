"""Static checks that package code reaches the host only through the Theta SDK (`src.app.api`),
reaches other packages only through declared exports, and imports only declared libraries.

Used by `tests/test_sdk_boundary.py` for the packages in this repository, by the installer and
the exporter (`package_errors`), and by package builders for packages maintained elsewhere
(`violations`, `undeclared_imports`).
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

HOST = re.compile(r"^(src\.app|frontend)(\.|$)")
SDK = re.compile(r"^src\.app\.api(\.|$)")
# Another package's modules, by path. Packages use each other through `provider_export`.
PACKAGE = re.compile(r"^(src\.usr|src\.plugins|optional_components)(\.|$)")
# A string naming a private host module, e.g. a `python -m src.app.x` argument.
MODULE_STRING = re.compile(r"^(src\.app|frontend)(\.[A-Za-z_][A-Za-z0-9_]*)+$")
WINDOW_ACCESS = re.compile(r"\bcontext\.window\b|\bcontext\._window\b|\.context\.window\b")
HOST_ROOTS = {"src", "frontend", "hydra_plugins", "run_pipeline", "__future__"}


def _imports(tree):
    """(node, module) for every absolute import and literal dynamic import in `tree`."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from ((node, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node, node.module
        elif isinstance(node, ast.Call):
            func = node.func
            called = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
            if called in ("__import__", "import_module") and node.args and isinstance(node.args[0], ast.Constant):
                yield node, str(node.args[0].value)


def private_references(path: Path) -> list[tuple[int, str]]:
    """(line, module) for every private host module `path` imports or names."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    found = [(node.lineno, name) for node, name in _imports(tree)]
    found += [(node.lineno, node.value) for node in ast.walk(tree)
              if isinstance(node, ast.Constant) and isinstance(node.value, str) and MODULE_STRING.match(node.value)]
    return sorted({(line, name) for line, name in found if HOST.match(name) and not SDK.match(name)})


def package_references(path: Path) -> list[tuple[int, str]]:
    """(line, module) for every import of another package's module by its location."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    return sorted({(node.lineno, name) for node, name in _imports(tree) if PACKAGE.match(name)})


def window_access(path: Path) -> list[int]:
    """Lines of `path` that reach into the host window through a plugin context."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [number for number, line in enumerate(lines, 1) if WINDOW_ACCESS.search(line)]


def _python_files(root: Path):
    return sorted(p for p in Path(root).rglob("*.py") if "__pycache__" not in p.parts and "node_modules" not in p.parts)


def _file_violations(path: Path, relative: str) -> list[str]:
    return ([f"{relative}:{line}: imports private host module {name}" for line, name in private_references(path)]
            + [f"{relative}:{line}: imports package module {name} by path; use a declared export"
               for line, name in package_references(path)]
            + [f"{relative}:{line}: reaches into the host window" for line in window_access(path)])


def violations(root: Path) -> list[str]:
    """Every private import, package import by path and window access under `root`, as `relative/path.py:line: detail`."""
    root = Path(root)
    found = []
    for path in _python_files(root):
        found += _file_violations(path, path.relative_to(root).as_posix())
    return found


def implementation_files(manifest: dict, root: Path) -> list[Path]:
    """The package's declared implementation code, without its tests."""
    root = Path(root)
    files = []
    for value in (manifest.get("payload") or {}).get("implementations") or []:
        path = root / value
        files += _python_files(path) if path.is_dir() else [path] if path.suffix == ".py" and path.is_file() else []
    return [path for path in files if not path.name.startswith("test_")]


def package_errors(manifest: dict, root: Path) -> list[str]:
    """Boundary violations in a package's implementation files (the installer and exporter refuse them)."""
    root = Path(root)
    return [line for path in implementation_files(manifest, root)
            for line in _file_violations(path, path.relative_to(root).as_posix())]


def _name(requirement: str) -> str:
    return re.sub(r"[-_.]+", "-", re.split(r"[<>=!~;\[ @]", requirement.strip(), maxsplit=1)[0]).lower()


def _optional(tree) -> set[int]:
    """Nodes guarded by ``try/except ImportError`` or under a condition: optional imports."""
    guarded = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try) and any(
                handler.type is None or any(getattr(t, "id", "") in ("ImportError", "ModuleNotFoundError", "Exception")
                                            for t in (handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]))
                for handler in node.handlers):
            body = node.body
        elif isinstance(node, ast.If):
            body = node.body + node.orelse
        else:
            continue
        guarded.update(id(child) for statement in body for child in ast.walk(statement))
    return guarded


def undeclared_imports(manifest: dict, root: Path, application_requirements) -> list[str]:
    """Unconditional third-party imports that neither the application nor ``dependencies.pip`` provides.

    A module maps to its distribution through the running environment's metadata, or to its own
    name when that distribution is not installed. The package's own modules, the standard library
    and the host are not third-party.
    """
    from importlib import metadata
    root = Path(root)
    owners = metadata.packages_distributions()
    provided = {_name(value) for value in application_requirements}
    provided |= {_name(value) for value in ((manifest.get("dependencies") or {}).get("pip") or [])}
    local = {path.stem for path in root.rglob("*.py")} | {path.name for path in root.rglob("*") if path.is_dir()}
    found = []
    for path in implementation_files(manifest, root):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        optional = _optional(tree)
        for node, module in _imports(tree):
            top = module.split(".")[0]
            if id(node) in optional or top in sys.stdlib_module_names or top in HOST_ROOTS or top in local:
                continue
            if not {_name(owner) for owner in owners.get(top, [top])} & provided:
                found.append(f"{path.relative_to(root).as_posix()}:{node.lineno}: imports {top}, "
                             "which neither the application nor dependencies.pip provides")
    return sorted(set(found))
