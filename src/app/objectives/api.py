"""Small, engine-independent objective API. Component metadata never imports user code."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


@dataclass
class ObjectiveResult:
    value: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    seed: int
    status: str
    metrics: dict[str, float] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    error: str | None = None

    def curve(self, metric: str, axis: str = "transitions") -> list[tuple[float, float]]:
        """Sorted finite observations; repeated steps use the latest observation."""
        points = {}
        for event in self.events:
            data = event.get("data", {})
            if event.get("metric") == metric and event.get("episode_outcomes"):
                values = event["episode_outcomes"]
                value = sum(values) / len(values)
            else:
                value = data.get(metric)
            x = (event.get("training_step", data.get("transitions")) if axis == "transitions"
                 else event.get("step", data.get(axis)))
            if x is not None and value is not None and math.isfinite(float(x)) and math.isfinite(float(value)):
                points[float(x)] = float(value)
        return sorted(points.items())


@dataclass
class TrialResults:
    runs: list[RunResult]
    config: dict[str, Any]


@dataclass
class ObjectiveComponent:
    name: str
    source: Path
    metadata: dict[str, Any]

    def parameters(self, overrides=None):
        import copy
        def merge(defaults, values):
            result = copy.deepcopy(defaults)
            for key, value in values.items():
                result[key] = merge(result[key], value) if isinstance(result.get(key), dict) and isinstance(value, dict) else value
            return result
        params = merge(self.metadata.get("defaults", {}), overrides or {})
        from jsonschema import ValidationError
        try:
            Draft202012Validator(self.metadata.get("schema", {})).validate(params)
        except ValidationError as exc:
            raise ValueError(f"Invalid objective parameter {'.'.join(map(str, exc.path))}: {exc.message}") from exc
        return params

    def evaluate(self, results: TrialResults, params) -> ObjectiveResult:
        # Execute freshly saved source on every scoring invocation, without stale pyc files.
        from src.app.core.provider_catalog import get_catalog
        from src.app.core.provider_loader import load_module
        from src.app.workspace import WorkspacePaths
        paths = WorkspacePaths.for_process()
        owners = [p for p in get_catalog(paths).packages if p.origin != "python_distribution" and self.source.is_relative_to(p.root)]
        if owners:
            owner = max(owners, key=lambda p: len(p.root.parts))
            module = load_module(owner, self.source.relative_to(owner.root).as_posix(), paths=paths, fresh=True)
        else:
            # Explicitly provided standalone objectives still support fresh source execution.
            import types
            module = types.ModuleType("theta_objective_" + hashlib.sha256(str(self.source).encode()).hexdigest())
            module.__file__ = str(self.source)
            exec(compile(self.source.read_text(encoding="utf-8"), str(self.source), "exec"), module.__dict__)
        result = getattr(module, self.metadata.get("function", "evaluate"))(results, params)
        if not isinstance(result, ObjectiveResult):
            result = ObjectiveResult(float(result))
        result.value = float(result.value)
        if not math.isfinite(result.value):
            raise ValueError("Objective must return a finite scalar")
        json.dumps(result.diagnostics, allow_nan=False)
        return result


def objective_root(workspace=None):
    from src.app.workspace import WorkspacePaths
    # Desktop launchers and workers need not start in the workspace directory.
    root = Path(workspace) if workspace is not None else WorkspacePaths.for_process().root
    return root / "src" / "usr" / "objectives"


def available_objectives(workspace=None):
    from src.app.core.provider_catalog import get_catalog
    from src.app.workspace import WorkspacePaths
    paths = WorkspacePaths(workspace) if workspace is not None else WorkspacePaths.for_process()
    catalog = get_catalog(paths)
    components = {}
    for name in sorted({c.id for p in catalog.packages for c in p.capabilities if c.kind == "objective"}):
        package, cap = catalog.resolve("objective", name)
        components[name] = package.resource(cap.entry_point)
    return components


def load_objective(name: str, workspace=None) -> ObjectiveComponent:
    paths = available_objectives(workspace)
    if name not in paths:
        raise ValueError(f"Unknown objective {name!r}; create it in src/usr/objectives")
    path = paths[name]
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("api_version") != 1:
        raise ValueError(f"Objective {name!r} must declare api_version: 1")
    source = (path.parent / metadata["source"]).resolve()
    if not source.is_relative_to(path.parent.resolve()): raise ValueError("Objective source must remain inside its package")
    if not source.is_file():
        raise ValueError(f"Objective source does not exist: {source}")
    Draft202012Validator.check_schema(metadata.get("schema", {}))
    return ObjectiveComponent(name, source, metadata)


def create_objective(name: str, template="learning_curve", workspace=None):
    """Create an editable workspace copy without changing installed components."""
    import re
    if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_-]*", name):
        raise ValueError("Use letters, numbers, underscores or hyphens for an objective name")
    component = load_objective(template, workspace)
    root = objective_root(workspace) / name
    if root.exists() or name in available_objectives(workspace):
        raise ValueError(f"Objective {name!r} already exists")
    root.mkdir(parents=True, exist_ok=True)
    source, manifest = root / f"{name}.py", root / f"{name}.json"
    if source.exists() or manifest.exists() or name in available_objectives(workspace):
        raise ValueError(f"Objective {name!r} already exists")
    metadata = {**component.metadata, "source": source.name}
    source.write_text(component.source.read_text(encoding="utf-8"), encoding="utf-8")
    manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    package = {'id':name,'name':name,'kind':'objective','version':'0.1.0','storage_version':2,'api_version':1,
               'capabilities':[{'id':name,'kind':'objective','entry_point':manifest.name}],
               'payload':{'implementations':[source.name],'resources':[manifest.name],'configs':{}}}
    (root / '.theta_component.json').write_text(json.dumps(package,indent=2)+'\n')
    return load_objective(name, workspace)


def validate_objective(cfg, *, workspace=None, paths=None):
    """Validate parameters and declared data needs before starting training."""
    from omegaconf import OmegaConf
    spec = cfg.get("objective")
    if not hasattr(spec, "get") or not spec.get("component"):
        raise ValueError("objective requires a component and optional params")
    if cfg.get("recover") or cfg.get("workflow"):
        raise ValueError("Objective sampling does not yet support recovery or workflow launches")
    from src.app.pipeline.tuning_features import validate_tuning_compatibility
    from src.app.workspace import WorkspacePaths
    paths = paths or (WorkspacePaths(workspace) if workspace is not None else WorkspacePaths.for_process())
    validate_tuning_compatibility(cfg, paths=paths)
    component = load_objective(str(spec["component"]), paths.root)
    try:
        compile(component.source.read_text(encoding="utf-8"), str(component.source), "exec")
    except SyntaxError as exc:
        raise ValueError(f"Objective source has a syntax error at line {exc.lineno}: {exc.msg}") from exc
    params = spec.get("params") or {}
    if OmegaConf.is_config(params):
        params = OmegaConf.to_container(params, resolve=True)
    params = component.parameters(params)
    requirements = component.metadata.get("requires", {})
    from src.app.telemetry.selection import selection_from_config
    selection = selection_from_config(cfg)
    metrics = list(requirements.get("metrics", []))
    if requirements.get("metric_parameter"):
        metrics.append(params[requirements["metric_parameter"]])
    for metric in metrics:
        if not selection.keeps(metric):
            raise ValueError(f"Objective requires recorded metric {metric!r}")
    if requirements.get("episode_outcomes") and (cfg.get("paradigm") != "online_rl" or int(cfg.get("eval_episodes") or 0) < 1):
        raise ValueError("Objective requires online evaluation with eval_episodes > 0")
    if requirements.get("episode_outcomes"):
        from src.app.pipeline.engine import resolve_engine, runner_supports_episode_outcomes
        from src.app.pipeline.config import parse_methods_dict
        for name, method in parse_methods_dict(cfg, paths=paths).items():
            resolve_engine(method, cfg, paths=paths)
            if not runner_supports_episode_outcomes({**dict(cfg.items()), **method}, paths=paths):
                raise ValueError(f"Method {name!r} does not supply the episode outcomes required by this objective")
    return component, params
