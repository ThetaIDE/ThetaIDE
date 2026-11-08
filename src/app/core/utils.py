"""Core utility functions for dynamic module loading and inspection."""

import importlib.util
import sys
from pathlib import Path


def load_module(path: str):
    """Dynamically load a Python module from a file path without polluting sys.modules."""
    from src.app.core.provider_loader import resolve_resource, load_module as load_provider_module
    from src.app.core.provider_catalog import get_catalog
    p = resolve_resource(path)
    for package in get_catalog().packages:
        if package.origin != "python_distribution" and p.resolve().is_relative_to(package.root.resolve()):
            return load_provider_module(package, p.resolve().relative_to(package.root.resolve()).as_posix())
    import hashlib

    module_name = "dynamic_" + hashlib.sha256(str(p.resolve()).encode()).hexdigest()
    spec = importlib.util.spec_from_file_location(module_name, str(p))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from path: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
