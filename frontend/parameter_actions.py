"""Generic transport for component-defined file-opening actions."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path


def opening_targets(action, *, value, context, workspace):
    """Execute only the declared lightweight handler; never infer resource paths."""
    if action.get("type") != "open-in-editor":
        raise ValueError("Unsupported parameter action")
    root = Path(workspace).resolve()
    if action.get("domain") == "ruleset":
        from src.app.services.rulesets import rules_folder_targets
        targets = rules_folder_targets(value=value, context=context)
    elif "handler" in action:
        handler = action["handler"]
        from src.app.core.provider_loader import provider_object, resolve_resource
        from src.app.workspace import WorkspacePaths
        paths = WorkspacePaths(root)
        if handler['path'].startswith('package://'):
            package_id, _, relative = handler['path'][10:].partition('/')
            callback = provider_object(package_id, relative + ':' + handler['function'], paths)
        else:
            source = resolve_resource(handler['path'], paths).resolve()
            if not source.is_relative_to(root) or not source.is_file():
                raise ValueError("Action handler must be a declared package resource or a file in this workspace")
            name = "theta_parameter_action_" + hashlib.sha256(str(source).encode()).hexdigest()
            spec = importlib.util.spec_from_file_location(name, source)
            if spec is None or spec.loader is None: raise ValueError("Cannot load component action")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            callback = getattr(module, handler['function'])
        targets = callback(value=value, context=context)
    else:
        targets = action["path"]
    if isinstance(targets, (str, Path, dict)):
        targets = [targets]
    result = []
    for target in targets or []:
        target = target if isinstance(target, dict) else {"path": target}
        path = Path(target["path"])
        path = path if path.is_absolute() else root / path
        path = path.resolve()
        if not path.is_file() and not path.is_dir():
            raise ValueError(f"Resource does not exist: {path}")
        if path not in [item[1] for item in result]:
            result.append((str(target.get("label") or path.name), path))
    if not result:
        raise ValueError("The component supplied no files for this parameter value")
    return result
