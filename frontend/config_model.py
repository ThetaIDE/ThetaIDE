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

COMPONENT_ID_KEYS = ("name", "algorithm", "type", "algo", "architecture", "base")


def is_group_defaults_path(rel_path: str | Path) -> bool:
    """Only a base directly inside an experiment group defines group defaults."""
    path = Path(rel_path)
    return (
        len(path.parts) == 3
        and path.parts[0] in ("experiment", "experiments")
        and path.stem == "_base"
        and path.suffix in (".yaml", ".yml")
    )


def component_selection(value: Any) -> tuple[str | None, dict[str, Any]]:
    """Read scalar, named, and component-keyed method selections consistently."""
    if isinstance(value, str):
        return value, {}
    if not isinstance(value, Mapping):
        return None, {}
    for key in COMPONENT_ID_KEYS:
        if isinstance(value.get(key), str) and value[key]:
            return value[key], {k: v for k, v in value.items() if k not in COMPONENT_ID_KEYS}
    if len(value) == 1:
        key, params = next(iter(value.items()))
        if params is None or isinstance(params, Mapping):
            return key, dict(params or {})
    return None, dict(value)


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


def config_fingerprint(config_root: Path, *, workspace=None) -> int:
    """Fingerprint configs, component schemas and workspace visibility preferences."""
    root = Path(config_root)
    candidates = set(root.rglob("*.yaml")) | set(root.rglob("*.yml"))
    from src.app.workspace import WorkspacePaths
    paths = workspace or WorkspacePaths.from_config_root(root)
    candidates.update(paths.manifests())
    candidates.add(paths.data_dir / "provider_overrides.json")
    candidates.add(paths.data_dir / "extensions.json")
    for source in paths.config_roots():
        candidates.update(source.rglob("*.yaml"))
        candidates.update(source.rglob("*.yml"))
    candidates.add(paths.data_dir / "components/collection.json")
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


from src.app.core.requirements import SELECTION_KINDS as _SELECTION_KINDS


@dataclass(frozen=True)
class Environment:
    name: str
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

        return cls(name=chosen_name, raw=raw)


@dataclass(frozen=True)
class Agent:
    name: str
    algorithm: str
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

        return cls(
            name=name,
            algorithm=raw.get("algorithm") or path.stem,
            hyperparameters={k: v for k, v in raw.items() if k not in ("algorithm", "parameter_schema")},
            raw=raw,
        )


@dataclass(frozen=True)
class Model:
    name: str
    architecture: str
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

        return cls(
            name=name,
            architecture=raw.get("architecture") or path.stem,
            parameters={k: v for k, v in raw.items() if k not in ("architecture", "parameter_schema")},
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
    """A paradigm describes only itself; components declare their own compatibility."""
    name: str
    description: str
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_file(cls, path: Path) -> Paradigm:
        raw = _read_yaml(path)
        return cls(name=path.stem, description=raw.get("description", ""), raw=raw)

    def permits_environment(self, environment: Environment) -> bool:
        """A paradigm and environment must agree on whether data comes from a live simulator or a static dataset."""
        environment_source = environment.environment_source
        return environment_source is None or environment_source == self.environment_source

    @property
    def environment_source(self) -> str:
        """The kind of research environment an experiment supplies (validated as required).

        A static dataset is the offline form of an environment or task. It is
        deliberately distinct from ``data_module``: the latter only adapts
        the environment's observations into batches a model can consume.
        """
        return self.raw.get("environment_source")

    @property
    def uses_static_dataset(self) -> bool:
        """Whether this paradigm's environment is realized by static data."""
        return self.environment_source == "static_dataset"

    @property
    def budget_unit(self) -> str:
        """The primary training budget displayed when authoring this paradigm."""
        return (self.raw.get("training") or {}).get("budget")

    @property
    def algorithm_requirement(self) -> str:
        """Return ``required``, ``optional``, or ``forbidden`` for methods.

        This is a form contract, not a statement about installed components.
        In particular, a required algorithm field must remain visible when no
        compatible algorithm is installed, so the UI can explain the missing
        dependency instead of quietly authoring an invalid model-only method.
        """
        return (self.raw.get("method") or {}).get("algorithm")

    @property
    def allows_algorithm(self) -> bool:
        return self.algorithm_requirement != "forbidden"

    @property
    def requires_algorithm(self) -> bool:
        return self.algorithm_requirement == "required"

    def field_enabled(self, name: str) -> bool:
        flag = _ALLOW_FLAGS.get(name)
        if flag is not None:
            # These are opt-in: validation.py reads paradigm_def.get(flag, False),
            # so a paradigm that does not declare the flag forbids the field.
            return bool(self.raw.get(flag, False))
        return True

    def disabled_reason(self, name: str) -> str | None:
        if self.field_enabled(name):
            return None
        return f"{self.name} does not enable {name}"


class ConfigTree:
    """The workspace config root (``src/usr/config``) with package profiles, parsed into queryable objects."""

    def __init__(self, config_root: Path, *, workspace=None):
        self.config_root = Path(config_root)
        self._experiment_groups: dict[str, ExperimentGroup] | None = None
        from src.app.workspace import WorkspacePaths
        self.workspace = workspace or WorkspacePaths.from_config_root(self.config_root)
        self.resolver = self.workspace.config_resolver()
        self.resolver.validate()
        def profile_files(group):
            for selector in self.resolver.selectors(group):
                resolved = self.resolver.resolve(selector, required=True)
                relative = Path(selector).relative_to(group)
                if relative.name.startswith("_") or any(part.startswith(".") for part in relative.parts):
                    continue
                yield relative.with_suffix("").as_posix(), resolved.file
        self.paradigms = {name: Paradigm.from_file(p) for name, p in profile_files("paradigms")}
        agent_dir = self.config_root / "agent"
        self.agents = {}
        from dataclasses import replace
        for rel_name, a in profile_files("agent"):
            agent_obj = replace(Agent.from_file(a), name=rel_name)
            self.agents[rel_name] = agent_obj

        self.models = {}
        for rel_name, m in profile_files("model"):
            model_obj = replace(Model.from_file(m), name=rel_name)
            self.models[rel_name] = model_obj
        from src.app.services.local_components import LocalComponents
        hidden_selectors = LocalComponents(self.workspace).hidden_selectors()
        self._hidden_names = {
            kind: {Path(selector).relative_to(kind).with_suffix("").as_posix()
                   for selector in hidden_selectors if selector.startswith(kind + "/")}
            for kind in ("model", "agent", "env", "data", "eval", "plot")
        }
        self.model_names = sorted({model.name for model in self.models.values()} - self._hidden_names["model"])
        self.agent_names = sorted({agent.name for agent in self.agents.values()} - self._hidden_names["agent"])

        env_dir = self.config_root / "env"
        self.environments = {}
        for rel_name, e in profile_files("env"):
            env_obj = Environment.from_file(e, env_dir)
            self.environments[rel_name] = env_obj

        from src.app.sites import profiles
        self.sites = tuple(profiles(self.workspace))
        self.component_parameter_schemas: dict[str, dict[str, Any]] = {}

        self._discover_plugin_manifests()
        # Manifest scan order must never select a different owner's schema.
        for selector in self.resolver.selectors():
            resolved = self.resolver.resolve(selector, required=True)
            if resolved.schema:
                self.component_parameter_schemas[selector] = resolved.schema
                group, _, name = selector.partition("/")
                name = str(Path(name).with_suffix(""))
                collection = self.agents if group == "agent" else self.models if group == "model" else {}
                if name in collection:
                    collection[name] = replace(collection[name], parameter_schema=resolved.schema)

        experiment_root = self.config_root / "experiment"
        self.experiments: dict[str, Experiment] = {}
        for name, path in profile_files("experiment"):
                if path.name.startswith("_") or path.parent.name == "methods" or "/methods/" in path.as_posix():
                    continue
                experiment = Experiment(name=name, group=name.split("/", 1)[0] if "/" in name else "ungrouped", path=path, raw=_read_yaml(path))
                self.experiments[experiment.name] = experiment

    def _discover_plugin_manifests(self) -> None:
        """Scan workspace for installed plugin manifests (.theta_component.json / plugin.json)
        and register them into self.agents, self.models, and self.environments if not already present.
        """
        import json

        from src.app.core.provider_catalog import get_catalog
        from src.app.workspace import WorkspacePaths
        for package in get_catalog(self.workspace).packages:
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
                    if kind in ("agent", "algorithm"):
                        existing = self.agents.get(comp_id)
                        if existing is not None:
                            self.agents[comp_id] = Agent(
                                name=existing.name,
                                algorithm=existing.algorithm,
                                hyperparameters=existing.hyperparameters,
                                parameter_schema=existing.parameter_schema or dict(parameter_schema or {}),
                                raw=existing.raw,
                            )
                        else:
                            self.agents[comp_id] = Agent(
                                name=comp_id,
                                algorithm=str(meta.get("algorithm") or comp_id),
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
                                parameters=existing.parameters,
                                parameter_schema=existing.parameter_schema or dict(parameter_schema or {}),
                                raw=existing.raw,
                            )
                        else:
                            self.models[comp_id] = Model(
                                name=comp_id,
                                architecture=str(meta.get("architecture") or comp_id),
                                parameters=dict(meta.get("parameters", {})),
                                parameter_schema=dict(parameter_schema or {}),
                                raw=meta,
                            )
                    # Environment manifests describe installable packages, not
                    # selectable environments. The actual choices come from
                    # each package's env/ config profiles.
                except Exception:
                    continue

        # Associate manifest schemas with the matching profile-backed components.
        # The component ID for an algorithm/model can be its config path (e.g.
        # ``package/profile``), while a package manifest may describe several profiles.
        for config_path, schema in self.component_parameter_schemas.items():
            parts = Path(config_path).parts
            if len(parts) < 2 or not parts[-1].endswith(".yaml"):
                continue
            name = Path(*parts[1:]).with_suffix("").as_posix()
            if parts[0] == "agent" and name in self.agents:
                existing = self.agents[name]
                self.agents[name] = Agent(
                    name=existing.name,
                    algorithm=existing.algorithm,
                    hyperparameters=existing.hyperparameters,
                    parameter_schema=schema,
                    raw=existing.raw,
                )
            elif parts[0] == "model" and name in self.models:
                existing = self.models[name]
                self.models[name] = Model(
                    name=existing.name,
                    architecture=existing.architecture,
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
    def discover(cls) -> ConfigTree:
        """The config tree of the process workspace (``WorkspacePaths.for_process()``)."""
        from src.app.workspace import WorkspacePaths
        paths = WorkspacePaths.for_process()
        return cls(paths.config, workspace=paths)

    def visible_environments(self):
        return {env.name: env for env in self.environments.values()
                if env.name not in self._hidden_names["env"]}.values()

    def candidate_findings(self, key: str, candidate: str, draft: Mapping[str, Any] | None = None,
                           method: Mapping[str, Any] | None = None) -> list:
        """Declared compatibility findings if ``candidate`` became the ``key`` selection of ``method``.

        Rules a candidate makes about its own settings are left out: those describe
        how to configure it once selected, which its own form shows inline.
        """
        from src.app.core.requirements import candidate_findings
        findings = candidate_findings(dict(draft or {}), dict(method or {}), key, candidate, paths=self.workspace)
        own = key + "."
        return [f for f in findings
                if not (f.rule and f.capability.startswith(_SELECTION_KINDS[key] + ":")
                        and str(f.rule.get("config", "")).startswith(own))]

    def _compatible(self, key, items, hidden, draft, method):
        seen: set[str] = set()
        result = []
        for item in items:
            if item.name in hidden or item.name in seen:
                continue
            seen.add(item.name)
            if not any(f.level == "block" for f in self.candidate_findings(key, item.name, draft, method)):
                result.append(item)
        return result

    def agents_for(self, paradigm: str, draft: Mapping[str, Any] | None = None,
                   method: Mapping[str, Any] | None = None) -> list[Agent]:
        """Algorithms whose declared requirements do not block them in this draft."""
        if paradigm not in self.paradigms:
            return []
        draft = {**(draft or {}), "paradigm": paradigm}
        return self._compatible("agent", self.agents.values(), self._hidden_names["agent"], draft, method)

    def environments_for(self, paradigm: str, draft: Mapping[str, Any] | None = None) -> list[Environment]:
        if paradigm not in self.paradigms:
            return []
        rules = self.paradigms[paradigm]
        draft = {**(draft or {}), "paradigm": paradigm}
        candidates = [e for e in self.environments.values() if rules.permits_environment(e)]
        return self._compatible("env", candidates, self._hidden_names["env"], draft, None)

    def models_for(self, paradigm: str, draft: Mapping[str, Any] | None = None,
                   method: Mapping[str, Any] | None = None) -> list[Model]:
        draft = {**(draft or {}), "paradigm": paradigm} if paradigm else dict(draft or {})
        return self._compatible("model", self.models.values(), self._hidden_names["model"], draft, method)

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
        # tree reset it; anything else that changes the config root invalidates the whole tree.
        if self._experiment_groups is None:
            names = {Path(key).parts[1] for key in self.resolver.selectors("experiment") if len(Path(key).parts) > 2}
            # A workspace group folder is a group before it holds any profile.
            own = self.workspace.config / "experiment"
            names |= {d.name for d in own.iterdir() if d.is_dir() and not d.name.startswith((".", "_"))} if own.is_dir() else set()
            self._experiment_groups = {
                name: ExperimentGroup.from_profile(name, self.workspace.resolve_config(f"experiment/{name}/_base.yaml"))
                for name in sorted(names)
            }
        return self._experiment_groups

    def refresh_groups(self) -> None:
        """Re-read the experiment groups on next use (after writing a group's folder or _base.yaml)."""
        self._experiment_groups = None
        self.resolver.invalidate_selectors()

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

    def resource_entries(self, name: str) -> list[str]:
        """Entry IDs of a workspace resource packages declare (``x-theta-resource``)."""
        from src.app.core.workspace_resources import entries
        return list(entries(name, self.workspace))

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
        schema = self.component_parameter_schemas.get(Path(config_path).as_posix().removeprefix("src/usr/config/"))
        return dict(schema) if schema else {}

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
    """An experiment config group (``experiment/<group>/``), described by its _base.yaml.

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
        return cls.from_profile(directory.name, directory / "_base.yaml")

    @classmethod
    def from_profile(cls, name: str, base: Path) -> ExperimentGroup:
        if not base.is_file():
            return cls(name=name)
        raw = _read_yaml(base)
        env = raw.get("env")
        for entry in raw.get("defaults") or []:
            if isinstance(entry, dict):
                env = entry.get("override /env", env)
        methods = raw.get("methods")
        return cls(
            name=name,
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
    site: str = "local/default"
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
