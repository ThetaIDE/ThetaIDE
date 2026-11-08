"""Package-relative loading and stable native checkpoint namespaces."""
from __future__ import annotations

import hashlib
import importlib
import importlib.abc
import importlib.util
import logging
import sys
import types
from pathlib import Path

from src.app.workspace import WorkspacePaths

_LOADED = {}  # (workspace, package root, version, file) -> module
_OWNED = {}   # module name -> (workspace, package root)
_ACTIVE_WORKSPACE = None


def _namespace(name, directory):
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = [str(directory)]
        module.__package__ = name
        sys.modules[name] = module
    return sys.modules[name]


def module_prefix(package, paths):
    if package.origin == "bundled":
        return ".".join(package.root.relative_to(paths.application).parts)
    identity = hashlib.sha256((paths.identity + "\0" + package.id).encode()).hexdigest()[:24]
    return "src.plugins._workspace.p_" + identity


def load_module(package, relative="__init__.py", *, paths=None, fresh=False):
    paths = paths or WorkspacePaths.for_process()
    if package.entrypoint is not None: return package.entrypoint.load()
    source = package.resource(relative)
    if source.is_dir(): source = source / "__init__.py"
    if not source.is_file(): raise ImportError(f"Provider {package.id}: missing entry point {source}")
    prefix = module_prefix(package, paths)
    if package.origin == "bundled":
        suffix = source.relative_to(package.root).with_suffix("").parts
        if suffix[-1] == "__init__": suffix = suffix[:-1]
        name = ".".join((prefix, *suffix))
        return importlib.import_module(name)
    importlib.import_module("src.plugins")
    _namespace("src.plugins._workspace", package.root.parent)
    root_init = package.root / "__init__.py"
    if source != root_init and root_init.is_file() and prefix not in sys.modules:
        load_module(package, "__init__.py", paths=paths)
    else:
        _namespace(prefix, package.root)
    suffix = source.relative_to(package.root).with_suffix("").parts
    if suffix[-1] == "__init__": suffix = suffix[:-1]
    name = ".".join((prefix, *suffix))
    key = (str(paths.root), package.id, str(package.root), package.version, str(source))
    if fresh:
        _LOADED.pop(key, None)
        sys.modules.pop(name, None)
    if key in _LOADED: return _LOADED[key]
    if name in sys.modules and getattr(sys.modules[name], "__file__", None) == str(source):
        module = sys.modules[name]
    else:
        # Construct package-relative namespaces for nested source directories.
        parent = prefix
        directory = package.root
        for part in suffix[:-1]:
            parent += "." + part
            directory /= part
            init = directory / "__init__.py"
            if init.is_file() and parent not in sys.modules:
                load_module(package, init.relative_to(package.root).as_posix(), paths=paths)
            else:
                _namespace(parent, directory)
        spec = importlib.util.spec_from_file_location(name, source,
            submodule_search_locations=[str(source.parent)] if source.name == "__init__.py" else None)
        if spec is None or spec.loader is None: raise ImportError(str(source))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            # Compile directly so an equal-size update within one timestamp tick cannot reuse pyc.
            exec(compile(source.read_bytes(), str(source), "exec"), module.__dict__)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    _LOADED[key] = module
    _OWNED[name] = (str(paths.root), str(package.root))
    return module


def load_capability(package, capability, *, paths=None):
    if package.entrypoint is not None: return package.entrypoint.load()
    entry, _, attr = capability.entry_point.partition(":")
    if capability.kind == "objective":
        from src.app.objectives.api import load_objective
        return load_objective(capability.id, workspace=(paths or WorkspacePaths.for_process()).root)
    if not entry.endswith(".py"): raise ValueError(f"Executable entry point must name a Python file: {entry}")
    module = load_module(package, entry, paths=paths)
    value = module
    for part in attr.split(".") if attr else ():
        value = getattr(value, part)
    return value


def invalidate(paths=None, package_root=None, package_id=None):
    """Remove only objects owned by this workspace/package, including decorator entries."""
    paths = paths or WorkspacePaths.for_process()
    prefixes = {module_prefix(p, paths) for p in __import__("src.app.core.provider_catalog", fromlist=["get_catalog"]).get_catalog(paths).packages
                if p.origin not in ("python_distribution", "bundled") and (package_root is None or p.root == Path(package_root).resolve()) and (package_id is None or p.id == package_id)}
    canonical_prefix = "src.plugins._workspace.p_" + hashlib.sha256((paths.identity + "\0" + str(package_id)).encode()).hexdigest()[:24] if package_id else None
    if canonical_prefix: prefixes.add(canonical_prefix)
    prefixes.update(name for name, (ws, root) in _OWNED.items()
                    if ws == str(paths.root) and (package_root is None or root == str(Path(package_root).resolve()))
                    and (canonical_prefix is None or name == canonical_prefix or name.startswith(canonical_prefix + '.')))
    for name in list(sys.modules):
        module = sys.modules.get(name)
        identity = getattr(module, "__name__", name)
        if any(name == prefix or name.startswith(prefix + ".") or identity == prefix or identity.startswith(prefix + ".") for prefix in prefixes): sys.modules.pop(name, None)
    for modname, attr in (("src.app.core.model_registry", "MODEL_REGISTRY"), ("src.app.core.agent_registry", "AGENT_REGISTRY"),
                          ("src.app.core.datamodule_registry", "DATAMODULE_REGISTRY"), ("src.app.core.eval_registry", "EVALUATOR_REGISTRY"),
                          ("src.app.core.plotter_registry", "PLOTTER_REGISTRY"), ("src.app.core.paradigm_loader", "_COMPONENT_REGISTRY"),
                          ("src.app.core.plugin_discovery", "ENVIRONMENT_REGISTRY")):
        module = sys.modules.get(modname)
        registry = getattr(module, attr, {})
        for key, obj in list(registry.items()):
            if any(getattr(obj, "__module__", "") == prefix or getattr(obj, "__module__", "").startswith(prefix + ".") for prefix in prefixes):
                registry.pop(key, None)
        if module and hasattr(module, "_DISCOVERY_COMPLETE"): module._DISCOVERY_COMPLETE = False
        if module and hasattr(module, "_MODEL_CANDIDATES"):
            for key, values in list(module._MODEL_CANDIDATES.items()):
                remaining = [obj for obj in values if not any(getattr(obj, "__module__", "").startswith(prefix) for prefix in prefixes)]
                if remaining: module._MODEL_CANDIDATES[key] = remaining
                else: module._MODEL_CANDIDATES.pop(key, None)
    for key in list(_LOADED):
        if key[0] == str(paths.root) and (package_root is None or key[2] == str(Path(package_root).resolve())) and (package_id is None or key[1] == package_id): _LOADED.pop(key)
    from src.app.core.provider_catalog import _CATALOGS, _entrypoint_metadata
    _CATALOGS.clear()
    _entrypoint_metadata.cache_clear()
    discovery = sys.modules.get("src.app.core.plugin_discovery")
    if discovery:
        discovery._DISCOVERY_CACHE.clear()
        discovery._DISCOVERY_PERFORMED = False
    importlib.invalidate_caches()


def resolve_provider(kind, selector, paths=None):
    global _ACTIVE_WORKSPACE
    from src.app.core.provider_catalog import get_catalog
    paths = paths or WorkspacePaths.for_process()
    if _ACTIVE_WORKSPACE and _ACTIVE_WORKSPACE != paths:
        invalidate(_ACTIVE_WORKSPACE)
    _ACTIVE_WORKSPACE = paths
    install_namespace_finder()
    catalog = get_catalog(paths)
    resolved = catalog.resolve(kind, selector)
    if resolved is None and kind == "algorithm":
        keys = [key for p in catalog.packages for c in p.capabilities if c.kind == kind for key in (c.id, *c.aliases)
                if selector == key or selector.startswith(key + "_")]
        if keys:
            resolved = catalog.resolve(kind, max(keys, key=len))
    return load_capability(*resolved, paths=paths) if resolved else None


def discover_kind(kind, paths=None):
    """Metadata discovery is lazy; selected providers load through resolve_provider."""
    from src.app.core.provider_catalog import get_catalog
    return [(p, c) for p in get_catalog(paths).packages for c in p.capabilities if c.kind == kind]


def resolve_resource(value, paths=None):
    """Resolve an explicit package URI or workspace input path."""
    from src.app.core.provider_catalog import get_catalog
    paths = paths or WorkspacePaths.for_process()
    value = str(value).replace("\\", "/")
    if value == 'src/app/train.py':
        return paths.application / value
    if value.startswith("package://"):
        package_id, _, relative = value[10:].partition("/")
        package = get_catalog(paths).package(package_id)
        if package is None: raise FileNotFoundError(f"Package {package_id!r} is not installed")
        candidate = package.resource(relative)
    else:
        candidate = Path(value)
        if not candidate.is_absolute(): candidate = paths.root / candidate
    return candidate


def provider_object(package_id, entry_point, paths=None):
    """Load an explicit public helper from a declared dependency package."""
    from src.app.core.provider_catalog import get_catalog, Capability
    paths = paths or WorkspacePaths.for_process()
    package = get_catalog(paths).package(package_id)
    if package is None: raise ImportError(f"Required provider package {package_id!r} is not installed")
    return load_capability(package, Capability(entry_point, "resource", entry_point), paths=paths)


class _NamespaceLoader(importlib.abc.Loader):
    def __init__(self, package, relative, paths): self.package, self.relative, self.paths = package, relative, paths
    def create_module(self, spec): return load_module(self.package, self.relative, paths=self.paths)
    def exec_module(self, module): pass


class _NamespaceFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "src.plugins._workspace":
            return importlib.util.spec_from_loader(fullname, loader=None, is_package=True)
        if fullname.startswith("src.plugins._workspace."):
            from src.app.core.provider_catalog import get_catalog
            paths = WorkspacePaths.for_process()
            for package in get_catalog(paths).packages:
                if package.origin in ("bundled", "python_distribution"): continue
                prefix = module_prefix(package, paths)
                if fullname == prefix or fullname.startswith(prefix + "."):
                    suffix = fullname[len(prefix):].strip(".").replace(".", "/")
                    source = package.root / suffix if suffix else package.root
                    if source.is_dir():
                        if not (source / "__init__.py").is_file():
                            return importlib.util.spec_from_loader(fullname, loader=None, is_package=True)
                        source /= "__init__.py"
                    else: source = source.with_suffix(".py")
                    if source.is_file():
                        return importlib.util.spec_from_loader(fullname, _NamespaceLoader(package, source.relative_to(package.root).as_posix(), paths), is_package=source.name == "__init__.py")
            return None
        return None


def install_namespace_finder():
    if not any(isinstance(f, _NamespaceFinder) for f in sys.meta_path): sys.meta_path.insert(0, _NamespaceFinder())


def environment_resource(environment, relative, paths=None):
    from src.app.core.provider_catalog import get_catalog
    paths = paths or WorkspacePaths.for_process()
    selected = get_catalog(paths).resolve("env", str(environment))
    if selected is None: raise FileNotFoundError(f"No declared environment provider: {environment}")
    package, cap = selected
    resource_root = cap.metadata.get("resource_root")
    if resource_root is None: raise FileNotFoundError(f"Environment {environment} has no declared resource_root")
    candidate = package.resource(str(Path(resource_root) / relative))
    if not candidate.is_file(): raise FileNotFoundError(f"Environment resource is missing: {candidate}")
    return candidate
