"""Private copies of provider packages and declared resources for one launch."""
from __future__ import annotations

import json
import shutil
from functools import lru_cache
from pathlib import Path

INPUTS_SNAPSHOT_ENV = "THETA_INPUTS_SNAPSHOT"
_IGNORED = shutil.ignore_patterns("__pycache__", ".git", "node_modules", ".pytest_cache", ".mypy_cache", ".ruff_cache")


@lru_cache(maxsize=64)
def read_inputs(path):
    return json.loads(Path(path).read_text())


def _copy(source, target):
    if source.is_dir():
        # Dereference symlinks: a link back to an editable file is not a snapshot.
        shutil.copytree(source, target, ignore=_IGNORED)
        # Hub directory modes must not prevent cleanup of our private copy.
        for directory in (target, *(p for p in target.rglob("*") if p.is_dir())):
            directory.chmod(directory.stat().st_mode | 0o700)
    else:
        target.mkdir(parents=True)


def capture_inputs(paths, launch):
    from src.app.core.provider_catalog import get_catalog, provider_hash, content_hash
    from src.app.core.workspace_resources import declarations

    catalog = get_catalog(paths)
    extensions = paths.data_dir / "extensions.json"
    result = {"workspace": str(paths.root), "packages": [], "overrides": catalog.overrides, "resources": {},
              "extensions": json.loads(extensions.read_text()) if extensions.is_file() else {"schema": "theta.extension-state/v1", "disabled": []}}
    for index, package in enumerate(catalog.packages):
        if package.origin == "python_distribution":
            continue  # Dependencies belong to the launch's Python environment.
        digest = provider_hash(package)
        target = launch / "packages" / str(index)
        _copy(package.root, target)
        if provider_hash(package) != digest:
            raise RuntimeError(f"Provider {package.id} changed while capturing launch inputs; retry the launch")
        result["packages"].append({"manifest": str(target / ".theta_component.json"),
                                   "origin": package.origin, "source": str(package.root)})
    for name in declarations(paths):
        source, target = paths.resource(name), launch / "resources" / name
        digest = content_hash(source)
        _copy(source, target)
        if content_hash(source) != digest:
            raise RuntimeError(f"Workspace resource {name} changed while capturing launch inputs; retry the launch")
        result["resources"][name] = str(target)
    _copy(paths.framework_config, launch / "framework_config")
    result["framework_config"] = str(launch / "framework_config")
    target = launch / "inputs.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    return target
