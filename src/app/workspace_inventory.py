"""Read-only, reproducible boundary inventory, including ignored research files.

Usage: python -m src.app.workspace_inventory --workspace PATH --output FILE
No provider implementation is imported and no workspace data is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import stat
import subprocess
from pathlib import Path

from src.app.workspace import WorkspacePaths

EXCLUDED = {"__pycache__", ".git", "node_modules", ".pytest_cache"}


def _files(root, excluded):
    if root.is_file() or root.is_symlink():
        yield root
    elif root.is_dir():
        if (root / "pyvenv.cfg").is_file():
            excluded.add(str(root))
            return
        for child in sorted(root.iterdir()):
            if child.name not in EXCLUDED:
                yield from _files(child, excluded)


def inventory(paths: WorkspacePaths) -> dict:
    """Record input/source/config bytes, symlinks, ownership state and references."""
    roots = [paths.root / "in", paths.research, *paths.managed_roots,
             paths.application / "src/app", paths.application / "frontend", *paths.bundled_roots]
    state = [paths.data_dir / name for name in ("components", "provider_overrides.json", "workspace.json",
                                               "config_overrides.json", "config_backups", "retained_packages")]
    excluded = set()
    files = sorted(set(file for root in [*roots, *state] for file in _files(root, excluded)))
    tracked = set()
    try:
        result = subprocess.run(["git", "ls-files", "-z"], cwd=paths.root, check=True, capture_output=True)
        tracked = {paths.root / p.decode() for p in result.stdout.split(b"\0") if p}
    except (OSError, subprocess.CalledProcessError):
        pass  # An external workspace does not need a Git repository.
    records, references, diagnostics = [], [], []
    for file in files:
        application_roots = (paths.application / "src/app", paths.application / "frontend", *paths.bundled_roots)
        origin = "application" if any(file.is_relative_to(root) for root in application_roots) else "workspace" if file.is_relative_to(paths.root) else "state"
        owner_root = {"workspace": paths.root, "application": paths.application, "state": paths.data_dir}[origin]
        relative = file.relative_to(owner_root).as_posix()
        destination = relative
        if relative.startswith("in/config/"):
            destination = "src/usr/config/" + relative[len("in/config/"):]
        elif relative.startswith("in/datasets/"):
            destination = "src/usr/datasets/" + relative[len("in/datasets/"):]
        try:
            link = file.is_symlink()
            payload = str(file.readlink()).encode() if link else file.read_bytes()
            records.append({"path": relative, "origin": origin, "tracked": file in tracked,
                            "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
                            "mode": stat.S_IMODE(file.lstat().st_mode), "symlink": link,
                            "intended_destination": destination,
                            "selector": relative[len("in/config/"):] if relative.startswith("in/config/") else
                                        relative[len("src/usr/config/"):] if relative.startswith("src/usr/config/") else None})
            if not link and file.suffix in (".yaml", ".yml"):
                import yaml
                def visit(value, key=""):
                    if isinstance(value, dict):
                        for name, item in value.items():
                            field = f"{key}.{name}" if key else str(name)
                            if name in ("dataset_path", "data_path", "input_path") and isinstance(item, str):
                                target = Path(item).expanduser()
                                references.append({"file": relative, "field": field, "value": item,
                                                   "external": target.is_absolute() and not target.is_relative_to(paths.root)})
                            visit(item, field)
                    elif isinstance(value, list):
                        for index, item in enumerate(value): visit(item, f"{key}[{index}]")
                try:
                    visit(yaml.safe_load(payload))
                except (ValueError, yaml.YAMLError) as exc:
                    diagnostics.append({"file": relative, "error": str(exc)})
        except OSError as exc:
            diagnostics.append({"file": relative, "error": str(exc)})
    from src.app.core.provider_catalog import get_catalog
    catalog = get_catalog(paths)
    return {"inventory_version": 1, "workspace": str(paths.root), "application": str(paths.application),
            "excluded_virtual_environments": sorted(excluded),
            "files": records, "dataset_references": references,
            "packages": [p.record() for p in catalog.packages],
            "diagnostics": [*diagnostics, *catalog.diagnostics]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--application", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = WorkspacePaths(args.workspace, **({"application": args.application} if args.application else {}))
    args.output.write_text(json.dumps(inventory(paths), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
