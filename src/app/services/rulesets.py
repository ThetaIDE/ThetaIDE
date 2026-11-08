"""Environment-scoped logic resources shared by editors, models and packages.

Authored resources live in src/usr/rules/<environment>/<name>; managed resources
remain inside their owning packages. A ruleset can
provide its own valuation.py without replacing the simulator's predicates or
requiring a different environment ID.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.app.workspace import WorkspacePaths


def reject_legacy_rules_paths(value):
    """Reject legacy asset references; never rewrite a research resource path."""
    if isinstance(value, (str, Path)):
        path = str(value).replace("\\", "/").rstrip("/")
        if path == "in/rules" or path.startswith("in/rules/") or "/in/rules/" in path or path.endswith("/in/rules"):
            raise ValueError(f"Unsupported legacy rules path: {value}. Use src/usr/rules and update the package or reference explicitly.")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            reject_legacy_rules_paths(key)
            reject_legacy_rules_paths(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            reject_legacy_rules_paths(item)


def require_canonical_rules_layout(workspace):
    old = Path(workspace) / "in/rules"
    if old.exists() or old.is_symlink():
        raise ValueError(f"Unsupported legacy rules location: {old}. Rules must exist only in src/usr/rules; move the assets and update references explicitly.")


def symbolic_environment(environment, workspace) -> str | None:
    """Resolve a config selector/profile and its explicit symbolic_name override."""
    root = Path(workspace)
    if isinstance(environment, Mapping):
        explicit = environment.get("symbolic_name")
        if explicit:
            return _environment_id(str(explicit))
        environment = environment.get("name")
    if not environment:
        return None
    name = str(environment)
    _environment_id(name)  # validate before constructing a filesystem path
    for suffix in (".yaml", ".yml"):
        profile = root / "in/config/env" / (name + suffix)
        if profile.is_file():
            data = yaml.safe_load(profile.read_text()) or {}
            if data.get("symbolic_name"):
                return _environment_id(str(data["symbolic_name"]))
    return Path(name).name


def _environment_id(name):
    if not name or "\\" in name or Path(name).is_absolute() or any(p in (".", "..") for p in name.split("/")):
        raise ValueError(f"Invalid symbolic environment: {name!r}")
    return name


@dataclass(frozen=True)
class Ruleset:
    environment: str
    name: str
    directory: Path
    workspace: Path

    @property
    def metadata(self):
        path = self.directory / "ruleset.yaml"
        data = yaml.safe_load(path.read_text()) if path.is_file() else {}
        if data is not None and not isinstance(data, dict):
            raise ValueError(f"Ruleset metadata must be a mapping: {path}")
        if data and data.get("format_version", 1) != 1:
            raise ValueError(f"Unsupported ruleset format: {path}")
        return data or {}

    @property
    def valuation_path(self):
        source = self.metadata.get("valuation")
        if source:
            path = (self.directory / str(source)).resolve()
            if not path.is_relative_to(self.directory.resolve()) or not path.is_file():
                raise ValueError(f"Ruleset valuation must be a file inside {self.directory}")
            return path
        from src.app.core.provider_loader import environment_resource
        return environment_resource(self.environment, "valuation.py", WorkspacePaths(self.workspace))


def available_rulesets(workspace, environment=None) -> list[Ruleset]:
    paths = WorkspacePaths(workspace)
    require_canonical_rules_layout(paths.root)
    env = symbolic_environment(environment, paths.root)
    bases = [paths.rules]
    from src.app.core.provider_catalog import get_catalog
    for package in get_catalog(paths).packages:
        if not package.managed and package.origin != "bundled": continue
        resources = package.manifest.get("rules_root")
        if resources:
            bases.append(package.resource(resources))
    found = []
    for base in dict.fromkeys(bases):
        found.extend(_rulesets_at(base, env, paths))
    owners = {}
    for item in found:
        key = (item.environment, item.name)
        if key in owners and owners[key].directory != item.directory:
            from src.app.core.provider_catalog import content_hash
            if content_hash(owners[key].directory) != content_hash(item.directory):
                raise ValueError(f"Duplicate ruleset {key}: {owners[key].directory}, {item.directory}")
            continue
        owners[key] = item
    return sorted(owners.values(), key=lambda r: (r.environment.casefold(), r.name.casefold()))


def _rulesets_at(base, env, paths):
    roots = [base / env] if env else (sorted(base.iterdir()) if base.is_dir() else [])
    found = []
    for directory in roots:
        if not directory.is_dir():
            continue
        for candidate in sorted(directory.iterdir()):
            if not candidate.is_dir() or not (candidate / "clauses.txt").is_file():
                continue
            resolved = candidate.resolve()
            if not resolved.is_relative_to(base.resolve()):
                raise ValueError(f"Ruleset directory escapes src/usr/rules: {candidate}")
            found.append(Ruleset(env or directory.name, candidate.name, resolved, paths.root))
    return sorted(found, key=lambda r: (r.environment.casefold(), r.name.casefold()))


def resolve_ruleset(workspace, environment, name) -> Ruleset:
    env = symbolic_environment(environment, workspace)
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("Select one ruleset")
    parts = name.split("/")
    if len(parts) == 2:
        qualified_env, name = parts
        qualified_env = symbolic_environment(qualified_env, workspace)
        if env and env != qualified_env:
            raise ValueError(f"Ruleset belongs to {qualified_env}, not {env}")
        env = qualified_env
    elif len(parts) != 1:
        raise ValueError("Ruleset selectors must be a name or environment/name")
    if name in (".", ".."):
        raise ValueError("Invalid ruleset name")
    if not env:
        raise ValueError("Select an environment or use an environment/ruleset selector")
    matches = [r for r in available_rulesets(workspace, env) if r.name == name]
    if not matches:
        raise ValueError(f"No ruleset {name!r} for environment {env!r}")
    return matches[0]


def rules_folder_targets(*, value, context):
    """Open the selected resource folder; never search all matching basenames."""
    root = Path(context["workspace_root"])
    require_canonical_rules_layout(root)
    env = context.get("config", {}).get("env")
    if not symbolic_environment(env, root) and (not isinstance(value, str) or "/" not in value):
        paths = WorkspacePaths(root)
        directory = paths.rules
    else:
        directory = resolve_ruleset(root, env, value).directory
    return [{"path": str(directory), "label": "Open rules folder"}]
