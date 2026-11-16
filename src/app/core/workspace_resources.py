"""Workspace resource directories that packages declare they read.

A package declares ``"workspace_resources": {"<name>": {"depth": N, "path": "<dir>"}}``.
The user's entries live in ``src/usr/<name>``; ``path`` optionally names the
package's own entries. An entry is a directory ``depth`` levels below a root,
identified by its relative POSIX path (for example ``cartpole/default``). A
parameter with ``{"x-theta-resource": "<name>"}`` is a picker of these entries.
Core attaches no meaning to a name or its entries.
"""
from __future__ import annotations

from pathlib import Path

from src.app.workspace import WorkspacePaths


def declarations(paths=None, packages=None) -> dict[str, list]:
    """Resource name -> [(package, declaration)] for every (or the given) package."""
    from src.app.core.provider_catalog import get_catalog
    paths = paths or WorkspacePaths.for_process()
    result: dict[str, list] = {}
    for package in packages if packages is not None else get_catalog(paths).packages:
        for name, declaration in (package.manifest.get("workspace_resources") or {}).items():
            result.setdefault(name, []).append((package, declaration))
    return result


def roots(name, paths=None) -> list[Path]:
    """The workspace directory followed by each declaring package's own directory."""
    paths = paths or WorkspacePaths.for_process()
    declared = declarations(paths).get(name)
    if not declared:
        raise ValueError(f"No installed package declares the workspace resource {name!r}")
    found = [paths.resource(name)]
    found.extend(package.resource(declaration["path"]) for package, declaration in declared if "path" in declaration)
    return list(dict.fromkeys(found))


def depth(name, paths=None) -> int:
    depths = {declaration["depth"] for _, declaration in declarations(paths).get(name, [])}
    if len(depths) != 1:
        raise ValueError(f"Packages declare the workspace resource {name!r} with different depths: {sorted(depths)}"
                         if depths else f"No installed package declares the workspace resource {name!r}")
    return depths.pop()


def _entries_at(root: Path, levels: int):
    if not root.is_dir():
        return []
    current = [root]
    for _ in range(levels):
        current = [child for parent in current for child in sorted(parent.iterdir())
                   if child.is_dir() and not child.name.startswith((".", "_"))]
    found = []
    for directory in current:
        if not directory.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Resource entry escapes its directory: {directory}")
        found.append((directory.relative_to(root).as_posix(), directory.resolve()))
    return found


def entries(name, paths=None) -> dict[str, Path]:
    """Entry ID -> directory across every root. Differing duplicates are an error."""
    from src.app.core.provider_catalog import content_hash
    paths = paths or WorkspacePaths.for_process()
    levels = depth(name, paths)
    result: dict[str, Path] = {}
    for root in roots(name, paths):
        for entry, directory in _entries_at(root, levels):
            if entry in result and result[entry] != directory:
                if content_hash(result[entry]) != content_hash(directory):
                    raise ValueError(f"Duplicate {name} entry {entry!r}: {result[entry]}, {directory}")
                continue
            result[entry] = directory
    return dict(sorted(result.items(), key=lambda item: item[0].casefold()))


def entry(name, entry_id, paths=None) -> Path:
    """The directory of one entry, by its full relative ID."""
    if not isinstance(entry_id, str) or not entry_id or "\\" in entry_id or Path(entry_id).is_absolute() \
            or any(part in ("", ".", "..") for part in entry_id.split("/")):
        raise ValueError(f"Invalid {name} entry: {entry_id!r}")
    found = entries(name, paths).get(entry_id)
    if found is None:
        raise ValueError(f"No {name} entry {entry_id!r}")
    return found


def workspace_directories(paths=None, packages=None) -> list[Path]:
    """The ``src/usr/<name>`` directories the given (or all) packages declare."""
    paths = paths or WorkspacePaths.for_process()
    return [paths.resource(name) for name in sorted(declarations(paths, packages))]
