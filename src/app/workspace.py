"""Authoritative storage policy. Distribution code and workspace content are separate."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STORAGE_VERSION = 2
# Authoring destinations only. Never use these as installer destinations.
COMPONENT_DIRECTORIES = {
    "model": "src/usr/models", "algorithm": "src/usr/algorithms",
    "method": "src/usr/algorithms", "agent": "src/usr/algorithms",
    "datamodule": "src/usr/data", "data": "src/usr/data",
    "evaluator": "src/usr/eval", "eval": "src/usr/eval",
    "env": "src/usr/environments", "environment": "src/usr/environments",
    "plotter": "src/usr/plotters", "plot": "src/usr/plotters",
    "objective": "src/usr/objectives", "rules": "src/usr/rules",
    "site": "src/usr/sites",
    "monitor": "src/usr/monitors", "tuner": "src/usr/tuners", "plugin": "src/usr/app_plugins",
    "experiment": "src/usr/experiments", "suite": "src/usr/packages", "package": "src/usr/packages",
}


@dataclass(frozen=True)
class WorkspacePaths:
    root: Path = PROJECT_ROOT
    data_dir: Path | None = None
    application: Path = PROJECT_ROOT

    def __post_init__(self):
        object.__setattr__(self, "root", Path(self.root).expanduser().resolve())
        object.__setattr__(self, "application", Path(self.application).expanduser().resolve())
        selected = os.environ.get("THETA_WORKSPACE")
        inherited_state = os.environ.get("THETA_DATA_DIR") if selected and Path(selected).expanduser().resolve() == self.root else None
        state = self.data_dir if self.data_dir is not None else inherited_state
        object.__setattr__(self, "data_dir", Path(state).expanduser().resolve() if state is not None else self.root / ".thetaide")

    @property
    def config(self): return self.root / "in/config"
    @property
    def user_config(self): return self.research / "config"
    @property
    def user_inputs(self): return self.research / "datasets"
    @property
    def framework_defaults(self): return self.framework_config

    def provider_config(self, package): return package.root / "configs"

    def config_resolver(self):
        from src.app.config_resolution import ConfigResolver
        from src.app.core.provider_discovery import current_snapshot
        snapshot = current_snapshot()
        if snapshot is None:
            return ConfigResolver(self)
        # One read operation shares one resolver; building it maps every package profile.
        key = ("config_resolver", self)
        if key not in snapshot.values:
            snapshot.values[key] = ConfigResolver(self)
        return snapshot.values[key]

    def config_searchpath(self):
        resolver = self.config_resolver()
        resolver.validate()
        return [resolver.search_uri()]
    @property
    def identity(self):
        import hashlib, json
        identity = self.root / ".thetaide/workspace.json"
        return json.loads(identity.read_text())["id"] if identity.is_file() else hashlib.sha256(str(self.root).encode()).hexdigest()

    @property
    def primary_config(self):
        return self.config if (self.config / 'config.yaml').is_file() else self.framework_config

    @property
    def framework_config(self): return self.application / "src/app/config_defaults"

    def config_roots(self):
        from src.app.core.provider_catalog import get_catalog
        roots = [self.user_config, self.config]
        for package in get_catalog(self).packages:
            root = package.root / "configs"
            if root.is_dir(): roots.append(root)
        roots.append(self.framework_config)
        return tuple(dict.fromkeys(roots))

    def resolve_config(self, relative):
        relative = Path(relative)
        if relative.is_absolute() or ".." in relative.parts: raise ValueError("Config selector must be relative")
        resolved = self.config_resolver().resolve(relative)
        if resolved is not None: return resolved.file
        return self.config / relative

    @property
    def datasets(self): return self.root / "in/datasets"
    @property
    def research(self): return self.root / "src/usr"
    @property
    def rules(self): return self.research / "rules"
    @property
    def results(self): return self.root / "results"
    @property
    def collection(self): return self.data_dir / "components"
    @property
    def bundled_components(self): return self.application / "src/plugins/components/bundled"
    @property
    def installed_components(self): return self.root / "src/plugins/components/installed"
    @property
    def bundled_app(self): return self.application / "src/plugins/app/bundled"
    @property
    def installed_app(self): return self.root / "src/plugins/app/installed"
    @property
    def user_app(self): return self.research / "app_plugins"
    @property
    def managed_roots(self): return (self.installed_components, self.installed_app)
    @property
    def bundled_roots(self): return (self.bundled_components, self.bundled_app)
    @property
    def component_roots(self):
        return tuple(dict.fromkeys((*self.bundled_roots, *self.managed_roots, *(self.user(kind) for kind in COMPONENT_DIRECTORIES))))
    @property
    def app_roots(self):
        return (self.bundled_app, self.installed_app, self.user_app)

    def user(self, kind): return self.root / COMPONENT_DIRECTORIES[kind]

    def installation(self, package_id, kind):
        import re
        if not re.fullmatch(r"[A-Za-z0-9_-]+", package_id):
            raise ValueError(f"Invalid package ID: {package_id!r}")
        root = self.installed_app if kind == "plugin" else self.installed_components
        target = root / package_id
        self.require_managed(target)
        return target

    def require_managed(self, path):
        for managed in self.managed_roots:
            if managed.resolve() != managed.absolute():
                raise ValueError(f"Managed roots cannot use symbolic links: {managed}")
        path = Path(path).resolve()
        if not any(path.is_relative_to(root.resolve()) and path != root.resolve() for root in self.managed_roots):
            raise ValueError(f"Lifecycle destination is outside managed installed roots: {path}. Use a new package ID or an editable user copy.")
        # A symlinked managed root must never turn into a source-tree write.
        protected = (self.research, self.application / "src/app", self.application / "frontend", *self.bundled_roots)
        if any(path.is_relative_to(p.resolve()) for p in protected):
            raise ValueError(f"Protected source destination: {path}")
        return path

    def manifests(self):
        import json
        from src.app.core.provider_discovery import current_snapshot
        snapshot = current_snapshot()
        if snapshot is not None and self in snapshot.manifests:
            return list(snapshot.manifests[self])
        candidates = sorted({p.resolve() for root in self.component_roots if root.is_dir()
                       for p in root.rglob(".theta_component.json")
                       if not any(part in ("__pycache__", ".git", "node_modules") for part in p.parts)}, key=lambda p: (len(p.parts), str(p)))
        result, package_roots = [], []
        for path in candidates:
            if any(path.is_relative_to(root) for root in package_roots): continue
            result.append(path)
            try:
                if json.loads(path.read_text()).get("storage_version") == STORAGE_VERSION:
                    package_roots.append(path.parent)
            except (ValueError, OSError): pass
        result = sorted(result)
        if snapshot is not None:
            snapshot.manifests[self] = tuple(result)
        return result

    def config_file(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.root / path if path.parts[:2] == ("in", "config") or path.parts[:3] == ("src", "usr", "config") else self.config / path
        if not path.suffix: path = path.with_suffix(".yaml")
        path = path.resolve()
        if not any(path.is_relative_to(root.resolve()) for root in (self.config, self.user_config)) or path.suffix not in (".yaml", ".yml"):
            raise ValueError("Configuration paths must be YAML files inside in/config/ (legacy) or src/usr/config/.")
        return path

    def process_environment(self):
        return {"THETA_WORKSPACE": str(self.root), "THETA_DATA_DIR": str(self.data_dir)}

    @classmethod
    def from_config_root(cls, config_root):
        root = Path(config_root).resolve()
        if root.parts[-3:] == ("src", "usr", "config"):
            return cls(root.parents[2])
        if root.parts[-3:] == ("src", "app", "config_defaults"):
            return cls(root.parents[2], application=root.parents[2])
        return cls(root.parent.parent)

    @classmethod
    def for_process(cls):
        if os.environ.get("THETA_WORKSPACE"):
            return cls(Path(os.environ["THETA_WORKSPACE"]), os.environ.get("THETA_DATA_DIR"))
        # A composed Hydra job carries the coordinator's explicit launch workspace.
        # Inspect already-loaded Hydra state without importing it during headless listings.
        import sys
        hydra = sys.modules.get("hydra.core.hydra_config")
        if hydra is not None:
            try:
                if hydra.HydraConfig.initialized():
                    return cls(Path(hydra.HydraConfig.get().runtime.cwd))
            except (AttributeError, ValueError, TypeError): pass
        current = Path.cwd()
        return cls(current) if (current / "in/config").is_dir() or (current / "src/usr").is_dir() else cls()
