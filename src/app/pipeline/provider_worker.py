"""Generic script launcher with workspace resolution, leases and provider verification."""
from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path

from src.app.components.transactions import provider_lease
from src.app.core.provider_catalog import get_catalog, content_hash, file_hash, provider_hash
from src.app.core.provider_loader import resolve_resource
from src.app.workspace import WorkspacePaths


def provider_snapshot(paths=None):
    paths = paths or WorkspacePaths.for_process()
    return {"workspace": str(paths.root), "providers": [
        {"id": p.id, "version": p.version, "origin": p.origin, "hash": provider_hash(p) if p.origin != "python_distribution" else None,
         "distribution":p.manifest.get("distribution"),
         "parent": p.manifest.get("parent"), "configs": {rel: file_hash(paths.root / rel) for rel in p.manifest.get("config_paths", []) if (paths.root / rel).is_file()}}
        for p in get_catalog(paths).packages],
        "overrides": get_catalog(paths).overrides,
        "user_rules": {file.relative_to(paths.rules).as_posix(): file_hash(file) for file in paths.rules.rglob('*')
                       if file.is_file() and '__pycache__' not in file.parts}}


def verify_snapshot(snapshot, paths=None):
    paths = paths or WorkspacePaths.for_process()
    current = {p.id: p for p in get_catalog(paths).packages}
    for record in snapshot.get("providers", []):
        package = current.get(record["id"])
        if package is None or package.version != record["version"] or (record.get("hash") is not None and provider_hash(package) != record["hash"]):
            raise RuntimeError(f"Provider {record['id']} differs from launch snapshot; synchronize provider versions/resources before running")
        for relative, digest in record.get('configs', {}).items():
            config = paths.config_file(relative)
            if not config.is_file() or file_hash(config) != digest:
                raise RuntimeError(f'Provider config {relative} differs from launch snapshot')
    if get_catalog(paths).overrides != snapshot.get('overrides', get_catalog(paths).overrides):
        raise RuntimeError('Provider override selection differs from launch snapshot')
    rules = {file.relative_to(paths.rules).as_posix(): file_hash(file) for file in paths.rules.rglob('*')
             if file.is_file() and '__pycache__' not in file.parts}
    if rules != snapshot.get('user_rules', rules):
        raise RuntimeError('Authored rules differ from launch snapshot')


def run(script, args=(), run_name="__main__"):
    paths = WorkspacePaths.for_process()
    os.environ.update(paths.process_environment())
    script = resolve_resource(script, paths).resolve()
    with provider_lease(paths):
        snapshot = os.environ.get("THETA_PROVIDER_SNAPSHOT")
        if snapshot: verify_snapshot(json.loads(Path(snapshot).read_text()), paths)
        sys.argv = [str(script), *args]
        os.chdir(paths.root)
        owners = [p for p in get_catalog(paths).packages if p.origin != 'python_distribution' and script.resolve().is_relative_to(p.root)]
        if not owners:
            return runpy.run_path(str(script), run_name=run_name)
        import importlib, types
        from src.app.core.provider_loader import module_prefix
        owner = max(owners, key=lambda p: len(p.root.parts))
        relative = script.relative_to(owner.root).with_suffix('')
        parent = '.'.join((module_prefix(owner, paths), *relative.parts[:-1]))
        importlib.import_module(parent)
        module = types.ModuleType(run_name)
        module.__dict__.update(__file__=str(script), __package__=parent)
        previous = sys.modules.get(run_name)
        sys.modules[run_name] = module
        try:
            exec(compile(script.read_bytes(), str(script), 'exec'), module.__dict__)
            return module.__dict__
        finally:
            if previous is None: sys.modules.pop(run_name, None)
            else: sys.modules[run_name] = previous


def main():
    if len(sys.argv) < 2: raise SystemExit("Usage: python -m src.app.pipeline.provider_worker SCRIPT [ARGS...]")
    run(sys.argv[1], sys.argv[2:])


if __name__ == "__main__": main()
