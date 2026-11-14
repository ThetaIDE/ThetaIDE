"""Narrow distribution-owned import aliases. Remove after saved targets/checkpoints migrate."""
import importlib
import importlib.abc
import importlib.util
import sys

ALIASES = {
    "src.app.tuners.optuna": "src.plugins.components.bundled.optuna",
    "src.usr.models.sequential_spec": "src.plugins.components.bundled.sequential_spec",
    "plot.standard_figures": "src.plugins.components.bundled.standard_figures",
    **{f"frontend.plugins.core.{name}": f"src.plugins.app.bundled.{name}" for name in
       ("text_editor", "model_builder", "tensorboard", "arxiv_papers", "documentation")},
}


class _Loader(importlib.abc.Loader):
    def __init__(self, target): self.target = target
    def create_module(self, spec): return importlib.import_module(self.target)
    def exec_module(self, module): pass


class _OptionalLoader(importlib.abc.Loader):
    def __init__(self, package, relative, paths):
        self.package, self.relative, self.paths = package, relative, paths
    def create_module(self, spec):
        from .provider_loader import load_module
        return load_module(self.package, self.relative, paths=self.paths)
    def exec_module(self, module): pass


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        # Fixed API-1 saved-target adapter, never a source-tree/default-provider fallback.
        old = "src.plugins.components.bundled.optuna"
        if fullname == old or fullname.startswith(old + "."):
            from src.app.workspace import WorkspacePaths
            from .provider_catalog import get_catalog
            from .contributions import ensure_enabled
            paths = WorkspacePaths.for_process()
            selected = get_catalog(paths).resolve("tuner", "optuna")
            if selected is None:
                raise ModuleNotFoundError("Saved Optuna target requires the optional Optuna component; install it in the selected workspace")
            package, capability = selected
            ensure_enabled(package, capability, paths=paths)
            suffix = fullname[len(old):].lstrip('.').replace('.', '/')
            relative = suffix + '.py' if suffix else '__init__.py'
            return importlib.util.spec_from_loader(fullname, _OptionalLoader(package, relative, paths), is_package=not suffix)
        for old, new in sorted(ALIASES.items(), key=lambda pair: -len(pair[0])):
            if fullname == old or fullname.startswith(old + "."):
                canonical = new + fullname[len(old):]
                spec = importlib.util.find_spec(canonical)
                if spec is not None:
                    return importlib.util.spec_from_loader(fullname, _Loader(canonical), is_package=spec.submodule_search_locations is not None)
        if fullname.startswith('src.usr.') and len(fullname.split('.')) >= 4 and not fullname.endswith(('agent_registry','base_agent')):
            raise ModuleNotFoundError(f"Undeclared legacy import {fullname}; use the native provider API")
        return None


def install():
    if not any(isinstance(f, _Finder) for f in sys.meta_path): sys.meta_path.insert(0, _Finder())


def alias_module(old, new):
    install()
    sys.modules[old] = importlib.import_module(new)
