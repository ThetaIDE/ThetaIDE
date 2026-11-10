"""Headless package inventory and capability resolution; discovery executes no code.

Package IDs are case-sensitive ASCII storage identifiers. Capability selectors are
case-insensitive, may contain slash-separated namespaces, and are qualified by kind.
Python module names and config profile selectors are separate manifest fields.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from functools import lru_cache

@lru_cache(maxsize=1)
def _entrypoint_metadata():
    return entry_points()
from pathlib import Path
from typing import Any

from src.app.workspace import WorkspacePaths

KINDS = {"method": "algorithm", "agent": "algorithm", "data": "datamodule", "eval": "evaluator",
         "plot": "plotter", "environment": "env"}
GROUPS = {"model": "theta.models", "algorithm": "theta.agents", "env": "theta.environments",
          "datamodule": "theta.datamodules", "evaluator": "theta.evaluators", "plotter": "theta.plotters",
          "monitor": "theta.monitors", "tuner": "theta.tuners", "run_logger": "thetaide.run_loggers", "component": "theta.components",
          "plugin": "theta.plugins"}


def kind_name(kind): return KINDS.get(kind, kind)


def file_hash(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def content_hash(root):
    digest = hashlib.sha256()
    for path in sorted(Path(root).rglob("*")):
        if path.is_file() and not any(p in ("__pycache__", ".git", "node_modules") for p in path.parts):
            digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


@dataclass
class Capability:
    id: str
    kind: str
    entry_point: str
    aliases: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class Package:
    id: str
    root: Path
    origin: str
    version: str
    manifest: dict
    capabilities: list[Capability] = field(default_factory=list)
    entrypoint: Any = None

    @property
    def writable(self): return self.origin == "user"
    @property
    def managed(self): return self.origin in ("hub", "local_archive")

    def resource(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError(f"Resource escapes package {self.id}: {relative}")
        return path

    def record(self, hashes=False):
        result = {"id": self.id, "version": self.version, "origin": self.origin, "root": str(self.root),
                  "writable": self.writable, "aliases": self.manifest.get("aliases", []),
                  "capabilities": [vars(c) for c in self.capabilities],
                  "configs": self.manifest.get("config_paths", []), "dependencies": self.manifest.get("dependencies", {}),
                  "source": self.manifest.get("installed_from"), "legacy": False}
        if hashes: result["hashes"] = {p.relative_to(self.root).as_posix(): file_hash(p) for p in self.root.rglob("*")
                                      if p.is_file() and not any(v in p.parts for v in ("__pycache__","node_modules",".git"))}
        return result


def owner_selected(package, selection):
    package_id = selection.get('package') if isinstance(selection,dict) else selection
    origin = selection.get('origin') if isinstance(selection,dict) else None
    return package.id == package_id and (origin is None or package.origin == origin)


class DuplicateProvider(ValueError): pass


class ProviderCatalog:
    def __init__(self, paths=None, app_roots=()):
        self.paths = paths if isinstance(paths, WorkspacePaths) else WorkspacePaths(paths) if paths else WorkspacePaths.for_process()
        inventory = self.paths.application / "src/plugins/inventory.json"
        self.distribution_inventory = json.loads(inventory.read_text()).get("packages", []) if inventory.is_file() else []
        self.extra_app_roots = tuple(Path(p) for p in app_roots)
        self.packages: list[Package] = []
        self.diagnostics: list[str] = []
        overrides = self.paths.data_dir / "provider_overrides.json"
        self.overrides = json.loads(overrides.read_text()) if overrides.is_file() else {}
        self._discover()

    def _origin(self, root, meta):
        if any(root.is_relative_to(p.resolve()) for p in self.paths.bundled_roots):
            return "bundled" if any(root == (self.paths.application / item["root"]).resolve() and meta.get("id") == item["id"]
                                    for item in self.distribution_inventory) else "unknown"
        if any(root.is_relative_to(p.resolve()) for p in self.paths.managed_roots):
            return meta.get("origin") if meta.get("origin") in ("hub", "local_archive") else "local_archive"
        if root.is_relative_to(self.paths.research) or any(root.is_relative_to(p.resolve()) for p in self.extra_app_roots): return "user"
        return "unknown"

    def _discover(self):
        manifests = list(self.paths.manifests())
        manifests.extend(p for root in self.extra_app_roots if root.is_dir() for p in root.rglob(".theta_component.json"))
        seen = set()
        for path in sorted(set(manifests)):
            root = path.parent.resolve()
            if root in seen: continue
            seen.add(root)
            try:
                meta = json.loads(path.read_text())
                from src.app.components.archive import validate_native_package
                validate_native_package(meta, root)
                meta.setdefault("config_paths", list(meta["payload"]["configs"]))
                package = Package(meta["id"], root, self._origin(root, meta), meta["version"], meta)
                self._capabilities(package)
                self.packages.append(package)
            except (OSError, ValueError, TypeError, SyntaxError) as exc:
                self.diagnostics.append(f"{path}: {exc}")
        for kind, group in GROUPS.items():
            for ep in _entrypoint_metadata().select(group=group):
                dist = getattr(ep, "dist", None)
                meta = {"distribution": getattr(dist, "name", None), "entry_point": ep.value}
                self.packages.append(Package(f"python:{meta['distribution']}:{ep.name}", Path("."), "python_distribution",
                    getattr(dist, "version", "0.0.0"), meta, [Capability(ep.name, kind, ep.value)], ep))
        for key in {p.id for p in self.packages}:
            owners = [p for p in self.packages if p.id == key]
            if len(owners) > 1:
                self.diagnostics.append(f"Duplicate package {key}: " + ", ".join(str(p.root) for p in owners))
        for key in {(c.kind, selector.lower()) for p in self.packages for c in p.capabilities for selector in (c.id, *c.aliases)}:
            owners = self.candidates(*key)
            if len(owners) > 1 and f"{key[0]}:{key[1]}" not in self.overrides:
                self.diagnostics.append(f"Duplicate capability {key[0]}:{key[1]}: " + ", ".join(f"{p.id}@{p.root}" for p, _ in owners))

    def _capabilities(self, package):
        meta = package.manifest
        for item in meta.get("capabilities", []):
            package.capabilities.append(Capability(item["id"], kind_name(item["kind"]), item["entry_point"], item.get("aliases", []), item))
        for cap in package.capabilities:
            if cap.kind == "plugin" and (package.root / "plugin.json").is_file():
                cap.metadata = {**cap.metadata, **json.loads((package.root / "plugin.json").read_text())}

    def candidates(self, kind, selector):
        key = selector.strip().lower()
        return [(p, c) for p in self.packages for c in p.capabilities if c.kind == kind_name(kind)
                and key in [c.id.lower(), *(a.lower() for a in c.aliases)]]

    def resolve(self, kind, selector):
        owners = self.candidates(kind, selector)
        override = self.overrides.get(f"{kind_name(kind)}:{selector.lower()}")
        if override:
            owners = [(p, c) for p, c in owners if owner_selected(p, override)]
        if len(owners) > 1: raise DuplicateProvider(f"Ambiguous {kind}:{selector}: " + ", ".join(f"{p.id}@{p.root}" for p, c in owners))
        return owners[0] if owners else None

    def package(self, package_id):
        owners = [p for p in self.packages if package_id in (p.id, *p.manifest.get("aliases", []))]
        if len(owners) > 1: raise DuplicateProvider(f"Duplicate package/alias {package_id}: " + ", ".join(str(p.root) for p in owners))
        return owners[0] if owners else None

    def load(self, kind, selector):
        resolved = self.resolve(kind, selector)
        if not resolved: return None
        from src.app.core.provider_loader import load_capability
        return load_capability(*resolved, paths=self.paths)

    def inventory(self, hashes=True):
        active_configs = []
        for package in self.packages:
            for relative in package.manifest.get("config_paths", []):
                path = self.paths.root / relative
                active_configs.append({"owner": package.id, "path": str(path), "exists": path.is_file(),
                    "hash": file_hash(path) if hashes and path.is_file() else None,
                    "installed_hash": package.manifest.get("config_hashes", {}).get(relative)})
        return {"active_configs": active_configs,"storage_version": 2, "workspace": str(self.paths.root), "application": str(self.paths.application),
                "state": str(self.paths.data_dir), "packages": [p.record(hashes) for p in self.packages],
                "diagnostics": self.diagnostics,
                "unsupported_rules": str(self.paths.root / "in/rules") if (self.paths.root / "in/rules").exists() else None}

_CATALOGS = {}


def get_catalog(paths=None, app_roots=()):
    """Cache metadata by workspace plus filesystem generation, never imported provider objects."""
    paths = paths if isinstance(paths, WorkspacePaths) else WorkspacePaths(paths) if paths else WorkspacePaths.for_process()
    extra = tuple(Path(p) for p in app_roots)
    files = [*paths.manifests(), paths.application / "src/plugins/inventory.json", paths.data_dir / "provider_overrides.json"]
    files.extend(p for root in extra if root.is_dir() for p in root.rglob(".theta_component.json"))
    stamp = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in sorted(set(files)) if p.is_file())
    key = (paths, extra)
    cached = _CATALOGS.get(key)
    if cached is None or cached[0] != stamp:
        cached = _CATALOGS[key] = (stamp, ProviderCatalog(paths, extra))
    return cached[1]


def provider_hash(package):
    """Hash executable payload and stable metadata, excluding machine-local install records."""
    volatile = {'target_dir','origin','installed_from','sha256','source_hashes','asset_hashes','config_hashes',
                'config_backups','installed_files','user_config_paths','retained_edits','authoring_hash','releases'}
    def stable(value):
        if isinstance(value,dict): return {k:stable(v) for k,v in value.items() if k not in volatile}
        if isinstance(value,list): return [stable(v) for v in value]
        return value
    digest=hashlib.sha256(json.dumps(stable(package.manifest),sort_keys=True).encode())
    for file in sorted(package.root.rglob('*')):
        if not file.is_file() or file.name=='.theta_component.json' or any(p in ('__pycache__','.git','node_modules') for p in file.parts):continue
        digest.update(file.relative_to(package.root).as_posix().encode()+b'\0'+file.read_bytes()+b'\0')
    return digest.hexdigest()
