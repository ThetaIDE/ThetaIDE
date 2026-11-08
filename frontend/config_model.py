"""Reads the Hydra config tree and answers experiment-compatibility questions.

Pure Python: no Qt and no torch imports, so it loads in headless CI and can be
unit tested without a display or the ML stack.

The panel asks this module three things: which options are valid for a given
paradigm, which form fields that paradigm permits, and what command line a
selection turns into.
"""

from __future__ import annotations

import copy
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

# Paradigms opt into fields with an `allows_<flag>` key whose name does not
# always match the Hydra key the field maps to.
_ALLOW_FLAGS = {
    "eval_episodes": "allows_eval_episodes",
}

_GROUPS = ("paradigms", "env", "agent", "model", "site")


# path -> (file bytes, parsed value). The tree is rebuilt often (every save invalidates it) but
# usually only one file changed, and parsing is what costs: reading the bytes is ~100x cheaper.
# Keyed on the exact contents, not mtime, so two quick same-size writes can't return stale data.
_PARSED: dict[Path, tuple[bytes, Any]] = {}


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        data = path.read_bytes()
        cached = _PARSED.get(path)
        if cached is not None and cached[0] == data:
            parsed = cached[1]
        else:
            parsed = yaml.safe_load(data) or {}
            _PARSED[path] = (data, parsed)
    except (OSError, yaml.YAMLError):
        return {}
    return copy.deepcopy(parsed)  # callers may mutate what they get; the cache must not change


def config_fingerprint(config_root: Path) -> int:
    """Fingerprint configs, component schemas and workspace visibility preferences."""
    root = Path(config_root)
    workspace = root.parent.parent
    candidates = set(root.rglob("*.yaml")) | set(root.rglob("*.yml"))
    from src.app.workspace import WorkspacePaths
    candidates.update(WorkspacePaths(workspace).manifests())
    candidates.add(workspace / ".thetaide/components/collection.json")
    entries = []
    for path in sorted(candidates):
        try:
            entries.append((str(path), path.read_bytes()))
        except OSError:
            continue
    return hash(tuple(entries))


def _yaml_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.rglob("*.yaml")
        if not p.name.startswith("_") and not any(part.startswith(".") for part in p.parts)
    )


@dataclass(frozen=True)
class Environment:
    name: str
    compatible_paradigms: tuple[str, ...] = ()
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @property
    def environment_source(self) -> str | None:
        """Return whether this config describes static data or a live simulator."""
        declared = self.raw.get("environment_source")
        if declared in ("static_dataset", "live_environment"):
            return declared
        if any(key in self.raw for key in ("dataset_name", "dataset_path", "data_path")):
            return "static_dataset"
        if self.raw.get("env_id"):
            return "live_environment"
        return None

    @classmethod
    def from_file(cls, path: Path, root: Path | None = None) -> Environment:
        raw = _read_yaml(path)
        rel_name = path.stem
        if root is not None:
            try:
                rel_name = path.relative_to(root).with_suffix("").as_posix()
            except ValueError:
                rel_name = path.stem

        env_name = raw.get("name")
        if not env_name and isinstance(raw.get("env"), dict):
            env_name = raw["env"].get("name")

        if env_name and "/" in env_name:
            chosen_name = env_name
        elif "/" in rel_name:
            chosen_name = rel_name
        else:
            chosen_name = env_name or rel_name

        paradigms = raw.get("compatible_paradigms") or raw.get("paradigms")
        if not paradigms and isinstance(raw.get("env"), dict):
            paradigms = raw["env"].get("paradigms") or raw["env"].get("compatible_paradigms")

        if isinstance(paradigms, list):
            cp = tuple(str(p) for p in paradigms)
        elif isinstance(paradigms, str):
            cp = (paradigms,)
        else:
            cp = ()

        return cls(
            name=chosen_name,
            compatible_paradigms=cp,
            raw=raw,
        )


@dataclass(frozen=True)
class Agent:
    name: str
    algorithm: str
    compatible_paradigms: tuple[str, ...] = ()
    hyperparameters: Mapping[str, Any] = field(default_factory=dict, repr=False)
    parameter_schema: Mapping[str, Any] = field(default_factory=dict, repr=False)
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_file(cls, path: Path, root: Path | None = None) -> Agent:
        raw = _read_yaml(path)
        name = path.stem
        if root is not None:
            try:
                name = path.relative_to(root).with_suffix("").as_posix()
            except ValueError:
                name = path.stem

        paradigms = raw.get("compatible_paradigms") or raw.get("paradigms")
        if isinstance(paradigms, list):
            cp = tuple(str(p) for p in paradigms)
        elif isinstance(paradigms, str):
            cp = (paradigms,)
        else:
            # Compatibility is component metadata, not an inference from an
            # implementation package name.  An omitted declaration means the
            # component is unconstrained for legacy third-party packages.
            cp = ()

        return cls(
            name=name,
            algorithm=raw.get("algorithm") or path.stem,
            compatible_paradigms=cp,
            hyperparameters={k: v for k, v in raw.items() if k not in ("algorithm", "paradigms", "compatible_paradigms", "parameter_schema")},
            raw=raw,
        )


@dataclass(frozen=True)
class Model:
    name: str
    architecture: str
    compatible_paradigms: tuple[str, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict, repr=False)
    parameter_schema: Mapping[str, Any] = field(default_factory=dict, repr=False)
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_file(cls, path: Path, root: Path | None = None) -> Model:
        raw = _read_yaml(path)
        name = path.stem
        if root is not None:
            try:
                name = path.relative_to(root).with_suffix("").as_posix()
            except ValueError:
                name = path.stem

        paradigms = raw.get("compatible_paradigms") or raw.get("paradigms")
        if isinstance(paradigms, list):
            cp = tuple(str(p) for p in paradigms)
        elif isinstance(paradigms, str):
            cp = (paradigms,)
        else:
            cp = ()

        return cls(
            name=name,
            architecture=raw.get("architecture") or path.stem,
            compatible_paradigms=cp,
            parameters={k: v for k, v in raw.items() if k not in ("architecture", "paradigms", "compatible_paradigms", "parameter_schema")},
            parameter_schema=dict(raw.get("parameter_schema") or {}),
            raw=raw,
        )

@dataclass(frozen=True)
class Experiment:
    name: str
    group: str
    path: Path
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_file(cls, path: Path, root: Path) -> Experiment:
        relative = path.relative_to(root)
        if relative.parts[0] == "experiment" and len(relative.parts) > 2:
            group = relative.parts[1]
        elif len(relative.parts) > 1:
            group = relative.parts[0]
        else:
            group = relative.parent.name or "ungrouped"

        exp_name = relative.with_suffix("").as_posix()
        if exp_name.startswith("experiment/"):
            exp_name = exp_name[len("experiment/"):]

        return cls(
            name=exp_name,
            group=group,
            path=path,
            raw=_read_yaml(path),
        )


@dataclass(frozen=True)
class Paradigm:
    name: str
    description: str
    allowed_agents: tuple[str, ...]
    forbidden_agents: tuple[str, ...]
    requires: Mapping[str, Any] = field(default_factory=dict, repr=False)
    forbids: Mapping[str, Any] = field(default_factory=dict, repr=False)
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_file(cls, path: Path) -> Paradigm:
        raw = _read_yaml(path)
        constraints = raw.get("constraints") or {}
        return cls(
            name=path.stem,
            description=raw.get("description", ""),
            allowed_agents=tuple(raw.get("allowed_agents") or ()),
            forbidden_agents=tuple(raw.get("forbidden_agents") or ()),
            requires=constraints.get("requires") or {},
            forbids=constraints.get("forbids") or {},
            raw=raw,
        )

    def permits_agent(self, agent: Agent | str) -> bool:
        algo = (agent.algorithm if isinstance(agent, Agent) else str(agent)).lower()
        name = (agent.name if isinstance(agent, Agent) else str(agent)).lower()
        clean = (algo, name, name.split("/")[-1], algo.split("/")[-1])
        if any(
            f.lower() in clean
            or name.startswith(f.lower() + "/")
            or name.endswith("/" + f.lower())
            for f in self.forbidden_agents
        ):
            return False

        if isinstance(agent, Agent) and agent.compatible_paradigms:
            return self.name in agent.compatible_paradigms

        # Fallback for agents without explicit compatible_paradigms
        if "allowed_agents" in self.raw and not self.allowed_agents:
            # Explicitly declared empty allow list (e.g. supervised)
            return False

        if self.allowed_agents:
            for allowed in self.allowed_agents:
                al = allowed.lower()
                if al in (algo, name) or name.startswith(al + "/") or algo.startswith(al + "_") or name.endswith("/" + al):
                    return True
            return False

        return True

    def permits_model(self, model: Model | str) -> bool:
        if isinstance(model, Model) and model.compatible_paradigms:
            return self.name in model.compatible_paradigms
        return True

    def permits_environment(self, environment: Environment) -> bool:
        # A paradigm and environment must agree on whether data comes from a
        # live simulator or a static dataset, even when older component
        # metadata omits or misstates compatible_paradigms.
        paradigm_source = self.environment_source
        environment_source = environment.environment_source
        if (
            paradigm_source in {"live_environment", "static_dataset"}
            and environment_source in {"live_environment", "static_dataset"}
            and paradigm_source != environment_source
        ):
            return False
        if not environment.compatible_paradigms:
            return True
        return self.name in environment.compatible_paradigms

    @property
    def environment_source(self) -> str:
        """The kind of research environment an experiment supplies.

        A static dataset is the offline form of an environment or task.  It is
        deliberately distinct from ``data_module``: the latter only adapts
        the environment's observations into batches a model can consume.

        Older component packages did not declare this capability, so retain a
        conservative compatibility inference while all paradigm manifests are
        migrated.
        """
        declared = self.raw.get("environment_source")
        if isinstance(declared, str) and declared:
            return declared
        data_module = self.raw.get("data_module") or self.raw.get("default_data_module")
        return "live_environment" if data_module == "SimulatorDataModule" else "static_dataset"

    @property
    def uses_static_dataset(self) -> bool:
        """Whether this paradigm's environment is realized by static data."""
        return self.environment_source == "static_dataset"

    @property
    def budget_unit(self) -> str:
        """The primary training budget displayed when authoring this paradigm."""
        training = self.raw.get("training")
        declared = training.get("budget") if isinstance(training, Mapping) else None
        if declared in {"timesteps", "epochs"}:
            return declared
        # Migration behavior for third-party paradigms authored before this
        # capability was made explicit.
        return "epochs" if self.uses_static_dataset else "timesteps"

    @property
    def algorithm_requirement(self) -> str:
        """Return ``required``, ``optional``, or ``forbidden`` for methods.

        This is a form contract, not a statement about installed components.
        In particular, a required algorithm field must remain visible when no
        compatible algorithm is installed, so the UI can explain the missing
        dependency instead of quietly authoring an invalid model-only method.
        """
        method = self.raw.get("method")
        declared = method.get("algorithm") if isinstance(method, Mapping) else None
        if declared in ("required", "optional", "forbidden"):
            return declared
        if "allowed_agents" in self.raw and not self.allowed_agents:
            return "forbidden"
        if self.requires.get("agent") == "required" or self.allowed_agents:
            return "required"
        return "optional"

    @property
    def allows_algorithm(self) -> bool:
        return self.algorithm_requirement != "forbidden"

    @property
    def requires_algorithm(self) -> bool:
        return self.algorithm_requirement == "required"

    def field_enabled(self, name: str) -> bool:
        if name in self.forbids:
            return False
        flag = _ALLOW_FLAGS.get(name)
        if flag is not None:
            # These are opt-in: validation.py reads paradigm_def.get(flag, False),
            # so a paradigm that does not declare the flag forbids the field.
            return bool(self.raw.get(flag, False))
        return True

    def disabled_reason(self, name: str) -> str | None:
        if self.field_enabled(name):
            return None
        if name in self.forbids:
            return f"{self.name} forbids {name} ({self.forbids[name]})"
        return f"{self.name} does not enable {name}"


class ConfigTree:
    """The `in/config/` directory, parsed into queryable objects."""

    def __init__(self, config_root: Path):
        self.config_root = Path(config_root)
        self._experiment_groups: dict[str, ExperimentGroup] | None = None
        from src.app.workspace import WorkspacePaths
        framework = WorkspacePaths.from_config_root(self.config_root).framework_config
        self.paradigms = {p.stem: Paradigm.from_file(p) for p in _yaml_files(framework / "paradigms")}
        self.paradigms.update({p.stem: Paradigm.from_file(p) for p in _yaml_files(self.config_root / "paradigms")})
        agent_dir = self.config_root / "agent"
        self.agents = {}
        for a in _yaml_files(agent_dir):
            rel_name = a.relative_to(agent_dir).with_suffix("").as_posix()
            agent_obj = Agent.from_file(a, agent_dir)
            self.agents[rel_name] = agent_obj
            if a.stem not in self.agents:
                self.agents[a.stem] = agent_obj

        model_dir = self.config_root / "model"
        self.models = {}
        for m in _yaml_files(model_dir):
            rel_name = m.relative_to(model_dir).with_suffix("").as_posix()
            model_obj = Model.from_file(m, model_dir)
            self.models[rel_name] = model_obj
            if m.stem not in self.models:
                self.models[m.stem] = model_obj
        # ``models`` keeps short-name aliases for backwards-compatible lookups
        # (e.g. ``profile`` for ``package/profile``). UI pickers should show each config
        # file once, under its canonical path.
        from src.app.services.local_components import LocalComponents
        hidden_paths = LocalComponents(self.config_root.parent.parent).hidden_config_paths()
        hidden_model_names = {path.relative_to(model_dir.resolve()).with_suffix("").as_posix() for path in hidden_paths if path.is_relative_to(model_dir.resolve())}
        self._hidden_names = {
            kind: {path.relative_to((self.config_root / kind).resolve()).with_suffix("").as_posix()
                   for path in hidden_paths if path.is_relative_to((self.config_root / kind).resolve())}
            for kind in ("model", "agent", "env", "data", "eval", "plot")
        }
        self.model_names = sorted({model.name for model in self.models.values()} - hidden_model_names)
        # ``agents`` also includes short-name aliases for older configs. Keep a
        # canonical list for UI pickers so each installed profile appears once.
        self.agent_names = sorted({agent.name for agent in self.agents.values()} - self._hidden_names["agent"])

        env_dir = self.config_root / "env"
        self.environments = {}
        for e in _yaml_files(env_dir):
            rel_name = e.relative_to(env_dir).with_suffix("").as_posix()
            env_obj = Environment.from_file(e, env_dir)
            self.environments[rel_name] = env_obj
            if e.stem not in self.environments:
                self.environments[e.stem] = env_obj

        self.sites = tuple(s.stem for s in _yaml_files(self.config_root / "site"))
        self.component_parameter_schemas: dict[str, dict[str, Any]] = {}

        self._discover_plugin_manifests()

        experiment_root = self.config_root / "experiment"
        self.experiments: dict[str, Experiment] = {}
        if experiment_root.is_dir():
            for path in sorted(experiment_root.glob("**/*.yaml")):
                if path.name.startswith("_") or path.parent.name == "methods" or "/methods/" in path.as_posix():
                    continue
                experiment = Experiment.from_file(path, experiment_root)
                self.experiments[experiment.name] = experiment

    def _discover_plugin_manifests(self) -> None:
        """Scan workspace for installed plugin manifests (.theta_component.json / plugin.json)
        and register them into self.agents, self.models, and self.environments if not already present.
        """
        import json

        # Infer workspace root
        if self.config_root.name == "config" and self.config_root.parent.name == "in":
            ws_root = self.config_root.parent.parent
        else:
            ws_root = self.config_root.parent

        from src.app.core.provider_catalog import get_catalog
        from src.app.workspace import WorkspacePaths
        for package in get_catalog(WorkspacePaths(ws_root)).packages:
                try:
                    meta = package.manifest
                    default_kind = meta.get("kind", "component")
                    meta_file = package.root / ".theta_component.json"
                    kind = str(meta.get("kind", default_kind)).lower()
                    comp_id = str(meta.get("id") or meta_file.parent.name)
                    parameter_schema = meta.get("parameter_schema")
                    if isinstance(parameter_schema, dict):
                        profile_schemas = parameter_schema.get("x-theta-config-schemas", {})
                        declared_paths = list(meta.get("config_paths") or [])
                        if meta.get("config_path") and meta["config_path"] not in declared_paths:
                            declared_paths.insert(0, meta["config_path"])
                        for config_path in declared_paths:
                            if not isinstance(config_path, str) or not config_path:
                                continue
                            normalized = Path(config_path).as_posix()
                            profile_schema = profile_schemas.get(normalized, parameter_schema) if isinstance(profile_schemas, dict) else parameter_schema
                            self.component_parameter_schemas[normalized] = profile_schema
                            if normalized.startswith("in/config/"):
                                self.component_parameter_schemas[normalized[len("in/config/"):]] = profile_schema
                    paradigms = meta.get("paradigms") or meta.get("compatible_paradigms") or ()
                    if isinstance(paradigms, list):
                        paradigms_tuple = tuple(str(p) for p in paradigms)
                    elif isinstance(paradigms, str):
                        paradigms_tuple = (paradigms,)
                    else:
                        paradigms_tuple = ()

                    if kind in ("agent", "algorithm"):
                        existing = self.agents.get(comp_id)
                        if existing is not None:
                            self.agents[comp_id] = Agent(
                                name=existing.name,
                                algorithm=existing.algorithm,
                                compatible_paradigms=existing.compatible_paradigms or paradigms_tuple,
                                hyperparameters=existing.hyperparameters,
                                parameter_schema=existing.parameter_schema or dict(parameter_schema or {}),
                                raw=existing.raw,
                            )
                        else:
                            self.agents[comp_id] = Agent(
                                name=comp_id,
                                algorithm=str(meta.get("algorithm") or comp_id),
                                compatible_paradigms=paradigms_tuple,
                                hyperparameters=dict(meta.get("hyperparameters", {})),
                                parameter_schema=dict(parameter_schema or {}),
                                raw=meta,
                            )
                    elif kind == "model":
                        existing = self.models.get(comp_id)
                        if existing is not None:
                            self.models[comp_id] = Model(
                                name=existing.name,
                                architecture=existing.architecture,
                                compatible_paradigms=existing.compatible_paradigms or paradigms_tuple,
                                parameters=existing.parameters,
                                parameter_schema=existing.parameter_schema or dict(parameter_schema or {}),
                                raw=existing.raw,
                            )
                        else:
                            self.models[comp_id] = Model(
                                name=comp_id,
                                architecture=str(meta.get("architecture") or comp_id),
                                compatible_paradigms=paradigms_tuple,
                                parameters=dict(meta.get("parameters", {})),
                                parameter_schema=dict(parameter_schema or {}),
                                raw=meta,
                            )
                    # Environment manifests describe installable packages, not
                    # selectable environments. The actual choices come from
                    # each package's YAML configs under in/config/env/.
                except Exception:
                    continue

        # Associate manifest schemas with the matching profile-backed components.
        # The component ID for an algorithm/model can be its config path (e.g.
        # ``package/profile``), while a package manifest may describe several profiles.
        for config_path, schema in self.component_parameter_schemas.items():
            rel_path = config_path[len("in/config/"):] if config_path.startswith("in/config/") else config_path
            parts = Path(rel_path).parts
            if len(parts) < 2 or not parts[-1].endswith(".yaml"):
                continue
            component_name = Path(*parts[1:]).with_suffix("").as_posix()
            aliases = {component_name, Path(parts[-1]).stem}
            for name in aliases:
                if parts[0] == "agent" and name in self.agents:
                    existing = self.agents[name]
                    self.agents[name] = Agent(
                        name=existing.name,
                        algorithm=existing.algorithm,
                        compatible_paradigms=existing.compatible_paradigms,
                        hyperparameters=existing.hyperparameters,
                        parameter_schema=schema,
                        raw=existing.raw,
                    )
                elif parts[0] == "model" and name in self.models:
                    existing = self.models[name]
                    self.models[name] = Model(
                        name=existing.name,
                        architecture=existing.architecture,
                        compatible_paradigms=existing.compatible_paradigms,
                        parameters=existing.parameters,
                        parameter_schema=schema,
                        raw=existing.raw,
                    )

    def create_folder(self, group: str, subpath: str) -> Path:
        """Create a new organizational folder under an experiment group."""
        folder = self.config_root / "experiment" / group / subpath
        folder.mkdir(parents=True, exist_ok=True)
        self.refresh_groups()
        return folder

    def rename_folder(self, old_path: Path | str, new_name: str) -> Path:
        """Rename an organizational folder under an experiment group."""
        old_p = Path(old_path)
        new_p = old_p.with_name(new_name)
        if new_p.exists() and not old_p.samefile(new_p):
            raise FileExistsError(f"A folder named '{new_name}' already exists.")
        old_p.rename(new_p)
        self.refresh_groups()
        return new_p

    def delete_folder(self, folder_path: Path | str) -> bool:
        """Delete an organizational folder under an experiment group."""
        import shutil
        folder_p = Path(folder_path)
        if folder_p.is_dir():
            shutil.rmtree(folder_p)
            self.refresh_groups()
            return True
        return False

    @classmethod
    def discover(cls, start: Path | None = None) -> ConfigTree:
        """Walk upward from `start` looking for an `in/config` directory."""
        current = Path(start or Path(__file__).resolve().parent)
        for candidate in (current, *current.parents):
            config_root = candidate / "in" / "config"
            if config_root.is_dir():
                return cls(config_root)
        raise FileNotFoundError(f"No in/config directory found above {current}")

    def visible_environments(self):
        return {env.name: env for env in self.environments.values()
                if env.name not in self._hidden_names["env"]}.values()

    def agents_for(self, paradigm: str) -> list[Agent]:
        if paradigm not in self.paradigms:
            return []
        rules = self.paradigms[paradigm]
        seen: set[str] = set()
        result: list[Agent] = []
        for a in self.agents.values():
            if a.name in self._hidden_names["agent"]:
                continue
            # Manifest schema association can replace canonical and alias
            # entries independently, so object identity is not stable here.
            if a.name not in seen:
                seen.add(a.name)
                if rules.permits_agent(a):
                    result.append(a)
        return result

    def environments_for(self, paradigm: str) -> list[Environment]:
        if paradigm not in self.paradigms:
            return []
        rules = self.paradigms[paradigm]
        seen: set[int] = set()
        result: list[Environment] = []
        for e in self.environments.values():
            if e.name in self._hidden_names["env"]:
                continue
            if id(e) not in seen:
                seen.add(id(e))
                if rules.permits_environment(e):
                    result.append(e)
        return result

    def models_for(self, paradigm: str) -> list[Model]:
        if paradigm not in self.paradigms:
            return [m for m in self.models.values() if m.name not in self._hidden_names["model"]]
        rules = self.paradigms[paradigm]
        seen: set[int] = set()
        result: list[Model] = []
        for m in self.models.values():
            if m.name in self._hidden_names["model"]:
                continue
            if id(m) not in seen:
                seen.add(id(m))
                if rules.permits_model(m):
                    result.append(m)
        return result

    def field_enabled(self, paradigm: str, name: str) -> bool:
        if paradigm not in self.paradigms:
            return True
        return self.paradigms[paradigm].field_enabled(name)

    def experiments_in(self, group: str) -> list[Experiment]:
        return [e for e in self.experiments.values() if e.group == group]

    @property
    def groups(self) -> list[str]:
        return sorted({e.group for e in self.experiments.values()} | set(self.experiment_groups))

    @property
    def experiment_groups(self) -> dict[str, ExperimentGroup]:
        # Built once per tree: group() alone reads it up to three times. Folder edits through this
        # tree reset it; anything else that changes in/config invalidates the whole tree.
        if self._experiment_groups is None:
            root = self.config_root / "experiment"
            self._experiment_groups = (
                {d.name: ExperimentGroup.from_dir(d) for d in sorted(root.iterdir()) if d.is_dir()}
                if root.is_dir() else {}
            )
        return self._experiment_groups

    def refresh_groups(self) -> None:
        """Re-read the experiment groups on next use (after writing a group's folder or _base.yaml)."""
        self._experiment_groups = None

    def group(self, name: str) -> ExperimentGroup:
        if not name:
            return ExperimentGroup(name="")
        if name in self.experiment_groups:
            return self.experiment_groups[name]
        for k, g in self.experiment_groups.items():
            if k.lower() == name.lower():
                return g
        return ExperimentGroup(name=name)

    def agent_hyperparameters(self, agent_name: str | None) -> dict[str, Any]:
        if not agent_name or agent_name not in self.agents:
            return {}
        return dict(self.agents[agent_name].hyperparameters)

    def model_parameters(self, model_name: str | None) -> dict[str, Any]:
        if not model_name or model_name not in self.models:
            return {}
        return dict(self.models[model_name].parameters)

    def ruleset_names(self, environment=None) -> list[str]:
        """Use the runtime catalog; unscoped profiles get qualified selectors."""
        from src.app.services.rulesets import available_rulesets, symbolic_environment

        workspace = self.config_root.parent.parent
        scoped = symbolic_environment(environment, workspace)
        return [r.name if scoped else f"{r.environment}/{r.name}"
                for r in available_rulesets(workspace, environment)]

    def model_parameter_schema(self, model_name: str | None, path: tuple[str, ...] = ()) -> dict[str, Any]:
        """Return the JSON Schema node describing a model parameter or nested object."""
        if not model_name or model_name not in self.models:
            return {}
        schema: Any = self.models[model_name].parameter_schema
        for key in path:
            schema = schema.get("properties", {}).get(key, {}) if isinstance(schema, dict) else {}
        return dict(schema) if isinstance(schema, dict) else {}

    def parameter_schema_for_config(self, config_path: str | Path) -> dict[str, Any]:
        """Find a component-manifest schema for a config profile path."""
        normalized = Path(config_path).as_posix()
        candidates = [normalized]
        if normalized.startswith("in/config/"):
            candidates.append(normalized[len("in/config/"):])
        else:
            candidates.append(f"in/config/{normalized}")
        for candidate in candidates:
            schema = self.component_parameter_schemas.get(candidate)
            if schema:
                return dict(schema)
        return {}

    def agent_parameter_schema(self, agent_name: str | None, path: tuple[str, ...] = ()) -> dict[str, Any]:
        """Return the JSON Schema node describing an agent parameter."""
        if not agent_name or agent_name not in self.agents:
            return {}
        schema: Any = self.agents[agent_name].parameter_schema
        for key in path:
            schema = schema.get("properties", {}).get(key, {}) if isinstance(schema, dict) else {}
        return dict(schema) if isinstance(schema, dict) else {}

    def model_parameter_keys(self, model_name: str | None, path: tuple[str, ...] = ()) -> list[str]:
        """Return declared/default parameter keys at a model nesting level."""
        if not model_name or model_name not in self.models:
            return []
        defaults: Any = self.models[model_name].parameters
        for key in path:
            defaults = defaults.get(key, {}) if isinstance(defaults, dict) else {}
        keys = list(defaults.keys()) if isinstance(defaults, dict) else []
        schema = self.model_parameter_schema(model_name, path)
        for key in schema.get("properties", {}):
            if key not in keys:
                keys.append(key)
        return keys

@dataclass(frozen=True)
class ExperimentGroup:
    """A directory under in/config/experiment/, described by its _base.yaml.

    The base is what binds an environment and a paradigm, so every experiment in
    the group inherits them; individual experiments only choose methods and a
    training budget.
    """

    name: str
    paradigm: str | None = None
    env: str | None = None
    has_base: bool = False
    base_methods: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dir(cls, directory: Path) -> ExperimentGroup:
        base = directory / "_base.yaml"
        if not base.is_file():
            return cls(name=directory.name)
        raw = _read_yaml(base)
        env = raw.get("env")
        for entry in raw.get("defaults") or []:
            if isinstance(entry, dict):
                env = entry.get("override /env", env)
        methods = raw.get("methods")
        return cls(
            name=directory.name,
            paradigm=raw.get("paradigm"),
            env=env,
            has_base=True,
            base_methods=methods if isinstance(methods, dict) else {},
        )


def default_method_model(
    group: ExperimentGroup | None,
    known: Any = (),
    fallback: str | None = None,
    paradigm: Paradigm | None = None,
) -> str | None:
    """Return a declared model default, or no selection when none is declared.

    A group whose base already declares methods has settled on an architecture,
    so follow that rather than impose a default on a group that has already chosen.

    Failing that, use the paradigm's own declared default, then an explicitly
    requested fallback if it is among the installed models. Never guess a model.
    """
    for spec in (group.base_methods if group else {}).values():
        if isinstance(spec, dict) and isinstance(spec.get("model"), str):
            return spec["model"]
    names = set(known or ())
    declared = paradigm.raw.get("default_model") if paradigm is not None else None
    if isinstance(declared, str):
        # Keep the paradigm contract visible even if the corresponding Hub
        # component has not been installed yet. The UI can then name the
        # missing dependency instead of silently substituting an incompatible
        # architecture.
        return declared
    if fallback is not None and fallback in names:
        return fallback
    return None


def group_base_yaml(env: str | None, paradigm: str, rules: Paradigm | None = None, seed: int = 1) -> str:
    """The _base.yaml for a new group, which is what binds env and paradigm."""
    lines = [
        "# @package _global_",
    ]
    if env and str(env).strip().lower() not in ("", "none"):
        lines += [
            "defaults:",
            f"  - override /env: {env}",
            "",
        ]
    else:
        lines += [
            "defaults: []",
            "",
        ]
    lines += [
        f"paradigm: {paradigm}",
        "",
    ]
    if rules is not None and not rules.field_enabled("eval_episodes"):
        lines.append("eval_episodes: 0")
    lines += [f"seed: {seed}", "save_dataset: false", "recover: false"]
    return "\n".join(lines) + "\n"


def experiment_yaml(
    group: str,
    experiment_id: str,
    paradigm: Paradigm | None = None,
    agent: str | None = None,
    model: str | None = None,
    seed: int = 42,
    total_timesteps: int = 10000,
) -> str:
    """A valid experiment for `group`, omitting whatever its paradigm forbids."""
    lines = [
        "# @package _global_",
        "defaults:",
        f"  - {group}/_base",
        "",
        f"experiment_id: {experiment_id}",
        f"seed: {seed}",
        f"total_timesteps: {total_timesteps}",
    ]
    if paradigm is None or paradigm.field_enabled("eval_episodes"):
        lines.append("eval_episodes: 100")

    # A paradigm with no agents still needs a method, so a model on its own is
    # enough to write one: supervised groups name methods for the architecture.
    if agent or model:
        lines += ["", "methods:", f"  {unique_method_name((), agent, model)}:"]
        if agent:
            lines.append(f"    agent: {agent}")
        if model:
            lines.append(f"    model: {model}")
    return "\n".join(lines) + "\n"


@dataclass
class Selection:
    """A chosen experiment plus the overrides the panel will apply to it."""

    experiment: str
    paradigm: str | None = None
    environment: str | None = None
    agent: str | None = None
    model: str | None = None
    site: str = "local"
    experiment_id: str | None = None
    seed: int | None = None
    total_timesteps: int | None = None
    eval_episodes: int | None = None
    extra: list[str] = field(default_factory=list)

    _GROUP_FIELDS = (
        ("paradigm", "paradigm"),
        ("environment", "env"),
        ("agent", "agent"),
        ("model", "model"),
        ("site", "site"),
    )
    _SCALAR_FIELDS = (
        ("experiment_id", "experiment_id"),
        ("seed", "seed"),
        ("total_timesteps", "total_timesteps"),
        ("eval_episodes", "eval_episodes"),
    )

    def to_overrides(self, tree: ConfigTree | None = None) -> list[str]:
        """Render the selection as Hydra `key=value` strings.

        Passing `tree` drops overrides the chosen paradigm forbids, so the panel
        cannot submit a combination the pipeline would reject at startup.
        """
        overrides: list[str] = []
        for attribute, key in self._GROUP_FIELDS:
            value = getattr(self, attribute)
            if value is not None:
                overrides.append(f"{key}={value}")
        for attribute, key in self._SCALAR_FIELDS:
            value = getattr(self, attribute)
            if value is None:
                continue
            if tree is not None and self.paradigm and not tree.field_enabled(self.paradigm, key):
                continue
            overrides.append(f"{key}={value}")
        overrides.extend(self.extra)
        return overrides

    def command(
        self,
        tree: ConfigTree | None = None,
        python: str = "python",
        script: str = "run_pipeline.py",
        dry_run: bool = False,
    ) -> list[str]:
        argv = [python, script, self.experiment, *self.to_overrides(tree)]
        if dry_run:
            argv.append("dry_run=true")
        return argv

    def command_line(self, **kwargs: Any) -> str:
        return shlex.join(self.command(**kwargs))


def unique_method_name(existing: Any, agent: str | None, model: str | None) -> str:
    """A method name free in `existing`, following the <agent>_<model> convention.

    Method names combine whichever algorithm and model identifiers are present.
    """
    base = f"{agent}_{model}" if agent and model else (agent or model or "method")
    taken = set(existing or ())
    if base not in taken:
        return base
    n = 2
    while f"{base}_{n}" in taken:
        n += 1
    return f"{base}_{n}"


def add_method(
    data: dict[str, Any],
    agent: str | None = None,
    model: str | None = None,
    paradigm: Paradigm | None = None,
) -> str:
    """Add a method to an experiment's `methods` block. Returns its name.

    `agent` is omitted for paradigms that declare none: supervised learning has
    no policy to choose actions with, so its methods name an architecture only.
    """
    if model is None:
        model = paradigm.raw.get("default_model") if paradigm else None
    methods = data.setdefault("methods", {})
    name = unique_method_name(methods, agent, model)
    spec = {"agent": agent} if agent else {}
    if model:
        spec["model"] = model
    methods[name] = spec
    return name


def remove_method(data: dict[str, Any], name: str) -> bool:
    """Drop a method. Returns False if it was not there.

    `params` holds settings shared by every method rather than a method of its
    own, so it is never removable this way.
    """
    methods = data.get("methods")
    if not isinstance(methods, dict) or name == "params" or name not in methods:
        return False
    del methods[name]
    if not [k for k in methods if k != "params"]:
        # A methods block with only `params` left satisfies no paradigm, and an
        # absent one lets the group base supply its own.
        data.pop("methods", None)
    return True


def deletion_blocked_reason(path: Path) -> str | None:
    """Why deleting this config would break something, or None if it is safe.

    A group's _base.yaml binds the environment and paradigm every experiment in
    the group inherits. Removing one while experiments remain leaves them
    pointing at nothing, and Hydra cannot even load the result:
    "Could not load 'experiment/<group>/_base'".
    """
    path = Path(path)
    if not path.exists():
        return f"{path.name} no longer exists"
    if path.name != "_base.yaml":
        return None
    siblings = [p for p in path.parent.glob("*.yaml") if p.name != "_base.yaml"]
    if siblings:
        names = ", ".join(sorted(p.stem for p in siblings)[:4])
        return (
            f"{path.parent.name}/_base.yaml defines the environment and paradigm for "
            f"{len(siblings)} experiment(s) that inherit it ({names}). "
            f"Delete those first."
        )
    return None


def rename_experiment_text(text: str, new_id: str) -> str:
    """The file's contents with experiment_id updated to match a new filename.

    Rewriting the one line rather than round-tripping through yaml keeps the
    comments and key order the rest of the team wrote.
    """
    pattern = re.compile(r"^(\s*experiment_id\s*:\s*).*$", re.MULTILINE)
    if pattern.search(text):
        return pattern.sub(lambda m: f"{m.group(1)}{new_id}", text, count=1)
    return text
