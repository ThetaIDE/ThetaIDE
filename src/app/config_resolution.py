"""Source-aware configuration selection without importing provider code.

Native payload mappings are the authority for package profile ownership. User
overrides are whole profiles; their selected provider still owns the schema.
"""
from __future__ import annotations

import base64
import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

from src.app.core.provider_catalog import DuplicateProvider, get_catalog, owner_selected
from src.app.workspace import WorkspacePaths

GROUP_KINDS = {"agent": "algorithm", "model": "model", "env": "env", "data": "datamodule",
               "eval": "evaluator", "plot": "plotter", "site": "site", "tuner": "tuner"}
_COMPOSITION_SOURCE = ContextVar("theta_composition_source", default=None)


@contextmanager
def configuration_context(paths, config_root=None):
    resolver = ConfigResolver(paths, config_root=config_root)
    resolver.validate()
    token = _COMPOSITION_SOURCE.set(resolver.search_uri())
    try:
        yield resolver
    finally:
        _COMPOSITION_SOURCE.reset(token)


def composition_source():
    return _COMPOSITION_SOURCE.get()


def selector_path(relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("Config selector must be relative")
    if path.suffix not in (".yaml", ".yml"):
        path = Path(str(path) + ".yaml")
    return path


@dataclass(frozen=True)
class ResolvedConfig:
    selector: str
    file: Path
    origin: str
    package: object | None = None
    overridden: bool = False

    @property
    def schema(self):
        if self.package is None:
            return {}
        schema = self.package.manifest.get("parameter_schema") or {}
        return schema.get("x-theta-config-schemas", {}).get("in/config/" + self.selector, schema)


class ConfigResolver:
    def __init__(self, paths: WorkspacePaths, *, config_root=None):
        self.paths = paths
        self.config_root = Path(config_root) if config_root else paths.config
        self.user_roots = tuple(dict.fromkeys((
            *((self.config_root,) if self.config_root != paths.config else ()),
            paths.user_config, paths.config,
        )))
        self.catalog = get_catalog(paths)
        self.profiles = {}
        self._selectors = None
        for package in self.catalog.packages:
            for destination, source in package.manifest.get("payload", {}).get("configs", {}).items():
                if destination.startswith("in/config/"):
                    relative = destination[len("in/config/"):]
                    self.profiles.setdefault(relative, []).append((package, package.resource(source)))

    def owner(self, relative):
        relative = selector_path(relative).as_posix()
        candidates = self.profiles.get(relative, [])
        group, _, name = relative.rpartition("/")
        kind = GROUP_KINDS.get(group.split("/", 1)[0])
        # Capability selection also selects profile/schema ownership. This is
        # deliberately the same catalog and override mechanism as execution.
        # Declared forwarding-only selectors preserve historical Hydra aliases.
        # Their defaults resolve the actual profile/provider; they own no capability.
        forwarding = bool(candidates) and all(relative in p.manifest.get("compatibility_selectors", []) for p, _ in candidates)
        selected = self.catalog.resolve(kind, relative.split("/", 1)[1].rsplit(".", 1)[0]) if kind and not forwarding else None
        # Named site variants inherit their site's explicit provider ownership.
        if selected is None and kind == "site" and "/" in relative.split("/", 1)[1]:
            selected = self.catalog.resolve(kind, relative.split("/")[1])
        override = self.catalog.overrides.get("profile:" + relative)
        if selected:
            claimed = bool(candidates)
            candidates = [(p, file) for p, file in candidates if p is selected[0]]
            if override and not owner_selected(selected[0], override):
                raise DuplicateProvider(f"Profile {relative} ownership conflicts with selected {kind} provider {selected[0].id}")
            if not candidates:
                if claimed:
                    raise DuplicateProvider(f"Profile {relative} is not supplied by selected {kind} provider {selected[0].id}")
                return selected[0], None
        elif override:
            candidates = [(p, file) for p, file in candidates if owner_selected(p, override)]
            if not candidates:
                raise DuplicateProvider(f"Selected profile owner for {relative} is not installed or does not provide it")
        if len(candidates) > 1:
            raise DuplicateProvider(f"Ambiguous config profile {relative}: " + ", ".join(f"{p.id}@{p.root}" for p, _ in candidates))
        return candidates[0] if candidates else (None, None)

    def resolve(self, relative, *, required=False):
        relative = selector_path(relative).as_posix()
        package, source = self.owner(relative)
        for root in self.user_roots:
            user = root / relative
            if user.is_file():
                return ResolvedConfig(relative, user, "user", package, source is not None)
        if source is not None and source.is_file():
            return ResolvedConfig(relative, source, package.origin, package)
        fallback = self.paths.framework_config / relative
        if fallback.is_file():
            return ResolvedConfig(relative, fallback, "framework", package)
        if required:
            raise FileNotFoundError(f"Configuration profile {relative!r} is unavailable; install/select its provider or create a user override")
        return None

    def invalidate_selectors(self):
        """Re-scan profile files on the next listing, after config files were added or removed."""
        self._selectors = None

    def selectors(self, group=""):
        prefix = group.rstrip("/") + "/" if group else ""
        if self._selectors is None:
            values = set(self.profiles)
            for root in (*self.user_roots, self.paths.framework_config):
                if root.is_dir():
                    values.update(file.relative_to(root).as_posix() for file in root.rglob("*")
                                  if file.is_file() and file.suffix in (".yaml", ".yml"))
            self._selectors = sorted(values)
        return [name for name in self._selectors if name.startswith(prefix)]

    def validate(self):
        for relative in self.profiles:
            self.owner(relative)

    def search_uri(self):
        context = {"root": str(self.paths.root), "application": str(self.paths.application), "data_dir": str(self.paths.data_dir)}
        if self.config_root != self.paths.config:
            context["config_root"] = str(self.config_root)
        return "theta://" + base64.urlsafe_b64encode(json.dumps(context).encode()).decode().rstrip("=")


def resolver_from_uri(value):
    context = json.loads(base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)))
    config_root = context.pop("config_root", None)
    return ConfigResolver(WorkspacePaths(**context), config_root=config_root)
