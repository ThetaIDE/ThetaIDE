"""Domain bindings for Hydra selections, methods, providers and normalization.

Functions take the document host explicitly. UI composition lives in sections;
these adapters retain the existing configuration services and structural actions.
"""

import copy
import math
import re
from pathlib import Path
from typing import Any, Mapping

import yaml
from PyQt6.QtCore import QTimer

from src.app.services import PROJECT_ROOT
from src.app.telemetry.selection import SECTIONS as METRIC_SECTIONS

from ..config_model import COMPONENT_ID_KEYS, ConfigTree, component_selection, is_group_defaults_path
from ..config_model import remove_method as remove_method_from
from .bindings import component_identity
from .catalogs import (
    CHECKPOINT_KEYS,
    DEFAULT_CHECKPOINTS,
    _monitor_settings,
    _read_group_base,
    config_tree,
    get_agent_candidate_hyperparameters,
)


def _get_resolved_defaults(self) -> dict:
    tree = self.tree
    group = self._group_name()
    resolved = {}
    if tree:
        root_cfg = getattr(tree, "config_root", None)
        if root_cfg and tree.workspace.resolve_config("config.yaml").is_file():
            try:
                resolved.update(yaml.safe_load(tree.workspace.resolve_config("config.yaml").read_text(encoding="utf-8")) or {})
            except Exception:
                pass
        if group:
            base_data = _read_group_base(tree, group)
            if base_data:
                resolved.update(base_data)
    return resolved


def _group_name(self):
    """The experiment group this file sits in, from experiment/<group>/<file>."""
    parts = Path(str(self.current_rel_path or "")).parts
    return parts[1] if len(parts) > 2 and parts[0] == "experiment" else None


def env_selection(self):
    """The chosen environment, or None when it is inherited from the base."""
    combo = getattr(self, "combo_env", None)
    if combo is None or combo.currentData() == self.INHERIT:
        return None
    return combo.currentText()


def _inherited_paradigm(self):
    """The paradigm this experiment inherits from its group's _base.yaml."""
    tree, group = self.tree, self._group_name()
    if tree is None or group is None:
        return None
    return tree.group(group).paradigm


def _inherited_env(self):
    """The environment this experiment inherits from its group's _base.yaml."""
    tree, group = self.tree, self._group_name()
    if tree is None or group is None or group not in tree.groups:
        return None
    return tree.group(group).env


def _is_base_config(self) -> bool:
    """True if the currently open configuration is a group defaults (_base.yaml) file."""
    return bool(self.current_path and is_group_defaults_path(self.current_rel_path or ""))


def _get_explicit_env(self) -> str | None:
    """The environment explicitly overridden in defaults or raw_data."""
    if not self.raw_data or not isinstance(self.raw_data, dict):
        return None
    for entry in self.raw_data.get("defaults") or []:
        if isinstance(entry, dict):
            if "override /env" in entry:
                val = entry.get("override /env")
                return str(val) if val is not None else None
            elif "/env" in entry:
                val = entry.get("/env")
                return str(val) if val is not None else None
    val = self.raw_data.get("env")
    return str(val) if isinstance(val, str) else None


def _is_environment_choice(name: Any) -> bool:
    """Reject directory-like values; selectors should name environment configs."""
    return isinstance(name, str) and bool(name.strip()) and not name.strip().endswith(("/", "\\"))


def _environment_for_name(tree: ConfigTree | None, name: str | None):
    if not tree or not name:
        return None
    if name in tree.environments:
        return tree.environments[name]
    return next((env for env in tree.environments.values() if env.name == name), None)


def _environment_is_compatible(self, tree: ConfigTree | None, name: str | None, paradigm: str | None) -> bool:
    if not self._is_environment_choice(name):
        return False
    environment = self._environment_for_name(tree, name)
    rules = tree.paradigms.get(paradigm) if tree and paradigm else None
    if not environment:
        return False
    if not paradigm:
        return True
    return rules.permits_environment(environment) if rules else False


def _set_experiment_env(self, env_name: str | None):
    """Update or insert override /env in defaults, removing legacy scalar env."""
    if not self.raw_data or not isinstance(self.raw_data, dict):
        return
    self.raw_data.pop("env", None)
    defaults = self.raw_data.get("defaults")

    if env_name is None or env_name == self.INHERIT or env_name in ("", "— Select Environment —"):
        if isinstance(defaults, list):
            self.raw_data["defaults"] = [
                d for d in defaults if not (isinstance(d, dict) and ("override /env" in d or "/env" in d))
            ]
        self._mark_dirty()
        return

    if not isinstance(defaults, list):
        defaults = []
        self.raw_data["defaults"] = defaults

    found = False
    for entry in defaults:
        if isinstance(entry, dict) and ("override /env" in entry or "/env" in entry):
            entry.clear()
            entry["override /env"] = env_name
            found = True
            break
    if not found:
        defaults.append({"override /env": env_name})
    self._mark_dirty()


def _get_base_env(self) -> str | None:
    """The environment configured in a group's _base.yaml defaults."""
    return self._get_explicit_env()


def _set_base_env(self, env_name: str | None):
    """Update or insert override /env in the group's _base.yaml defaults."""
    self._set_experiment_env(env_name)


def _clean_redundant_params(self):
    """Strip redundant parameters inherited from defaults or matching filename stem."""
    if not self.raw_data or not isinstance(self.raw_data, dict):
        return
    is_base = self._is_base_config()
    if is_base:
        self.raw_data.pop("group", None)
        self.raw_data.pop("experiment_id", None)
        # A group base may declare one default method, which new experiments inherit, so its
        # concrete methods are kept. (They used to be stripped here, on load and on save, which
        # deleted the default method from disk the first time the base was opened and saved.)
        # The one-method limit is enforced where methods are added.
        if "methods" in self.raw_data and isinstance(self.raw_data["methods"], dict):
            if not self.raw_data["methods"]:
                self.raw_data.pop("methods", None)
        return

    is_experiment = (
        (self.current_rel_path and self.current_rel_path.startswith("experiment/"))
        or "methods" in self.raw_data
        or any("_base" in str(d) for d in self.raw_data.get("defaults", []))
    )
    if is_experiment:
        self.raw_data.pop("group", None)
        stem = self.current_path.stem if self.current_path else ""
        if self.raw_data.get("experiment_id") == stem:
            self.raw_data.pop("experiment_id", None)
        inh_paradigm = self._inherited_paradigm()
        if inh_paradigm is not None and self.raw_data.get("paradigm") == inh_paradigm:
            self.raw_data.pop("paradigm", None)
        inh_env = self._inherited_env()
        explicit_env = self._get_explicit_env()
        if explicit_env is not None and (
            (inh_env is not None and explicit_env == inh_env) or explicit_env == self.INHERIT
        ):
            self._set_experiment_env(None)
        elif isinstance(self.raw_data.get("env"), str):
            self._set_experiment_env(self.raw_data.get("env"))
        if "description" in self.raw_data and not self.raw_data["description"]:
            self.raw_data.pop("description", None)
        if "resources" in self.raw_data and not self.raw_data["resources"]:
            self.raw_data.pop("resources", None)
        if "trainer" in self.raw_data and not self.raw_data["trainer"]:
            self.raw_data.pop("trainer", None)
        if "methods" in self.raw_data and isinstance(self.raw_data["methods"], dict):
            if "params" in self.raw_data["methods"] and not self.raw_data["methods"]["params"]:
                self.raw_data["methods"].pop("params", None)
        # Strip redundant default flags matching root/base defaults
        if self.raw_data.get("tensorboard") is False:
            self.raw_data.pop("tensorboard", None)
        if self.raw_data.get("save_dataset") is False:
            self.raw_data.pop("save_dataset", None)
        if self.raw_data.get("recover") is False:
            self.raw_data.pop("recover", None)


def _has_component_override(kind: str, data: dict) -> bool:
    return (data.get("plots") is not None or "plot_overrides" in data) if kind == "plots" else kind in data


def _effective_objective_spec(data, resolved):
    inherited = resolved.get("objective")
    explicit = data.get("objective")
    if "objective" in data and explicit is None:
        return None
    if not isinstance(inherited, dict):
        return explicit
    if not isinstance(explicit, dict):
        return inherited
    if explicit.get("component") and explicit.get("component") != inherited.get("component"):
        return explicit

    def merge(base, override):
        result = copy.deepcopy(base)
        for key, value in override.items():
            result[key] = (
                merge(result[key], value) if isinstance(result.get(key), dict) and isinstance(value, dict) else value
            )
        return result

    return merge(inherited, explicit)


def add_component_override(self, kind: str, resolved: dict | None = None) -> None:
    resolved = resolved or self._get_resolved_defaults()
    if kind in ("data_module", "evaluator"):
        inherited = self._component_name(self._inherited_component(kind, resolved))
        choices = self._component_choices(kind)
        name = inherited or (sorted(choices)[0] if choices else "")
        if name:
            self.raw_data[kind] = {"name": name, "params": self._component_default_params(kind, name)}
    elif kind == "plots":
        self.raw_data["plots"] = []
        self.raw_data.pop("plot_overrides", None)
    else:
        return
    self._mark_dirty()
    self._render_boxes()


def _component_name(value: Any) -> str:
    if isinstance(value, Mapping):
        return str(value.get("name") or value.get("class") or value.get("type") or "")
    return str(value or "")


def _component_default_params(self, kind: str, name: str) -> dict:
    profile = self._component_choices(kind).get(name, {})
    reserved = {
        "data_module": {"name", "class", "type", "data_module", "role", "compatible_paradigms", "paradigms"},
        "symbolic_state_loader": {"name", "class", "type", "data_module", "role", "compatible_paradigms", "paradigms"},
        "evaluator": {"name", "class", "type", "evaluator", "compatible_paradigms", "paradigms"},
    }.get(kind, {"name", "class", "type", "plotter", "compatible_paradigms", "paradigms"})
    return {key: copy.deepcopy(value) for key, value in profile.items() if key not in reserved}


def _inherited_component(self, kind: str, resolved: dict) -> Any:
    value = resolved.get(kind)
    if value is not None:
        return value
    tree = self.tree
    env_name = self._get_explicit_env() or self._inherited_env()
    env = tree.environments.get(env_name) if tree and env_name else None
    if kind != "data_module" and env and env.raw.get(kind) is not None:
        return env.raw[kind]
    paradigm = self.raw_data.get("paradigm") or self._inherited_paradigm()
    rules = tree.paradigms.get(paradigm) if tree and paradigm else None
    if rules:
        key = "default_data_module" if kind == "data_module" else "eval_protocol"
        return rules.raw.get(key)
    return None


def _set_component_choice(self, kind: str, name: str) -> None:
    if not name:
        self.raw_data.pop(kind, None)
    else:
        old = self.raw_data.get(kind)
        if isinstance(old, dict) and self._component_name(old) == name:
            return
        self.raw_data[kind] = {"name": name, "params": self._component_default_params(kind, name)}
    self._mark_dirty()
    QTimer.singleShot(0, self._render_boxes)


def _add_component_parameter(self, kind: str, key: str, default: Any) -> None:
    if not key:
        return
    value = self.raw_data.get(kind)
    if not isinstance(value, dict):
        value = {"name": self._component_name(value)}
        self.raw_data[kind] = value
    value.setdefault("params", {})[key] = copy.deepcopy(default)
    self._mark_dirty()
    QTimer.singleShot(0, self._render_boxes)


def _inherited_plot_names(self, resolved: dict) -> list[str]:
    plots = resolved.get("plots")
    if isinstance(plots, list):
        return [str(item) for item in plots]
    if isinstance(plots, dict):
        choices = self._component_choices("plots")
        return [
            name
            for name, value in plots.items()
            if value is not None and (name in choices or isinstance(value, dict) and name != "style")
        ]
    return []


def _set_plot_override(self, name: str, enabled: bool, resolved: dict | None = None) -> None:
    plots = self.raw_data.get("plots")
    inherited = (resolved or self._get_resolved_defaults()).get("plots")
    if isinstance(plots, list):
        plots = [str(item) for item in plots]
    elif plots is None and isinstance(inherited, list):
        plots = [str(item) for item in inherited]
    elif not isinstance(plots, dict):
        plots = {item: {} for item in self._inherited_plot_names(resolved or self._get_resolved_defaults())}
    if isinstance(plots, list):
        if enabled and name not in plots:
            plots.append(name)
        elif not enabled:
            plots = [item for item in plots if item != name]
    elif enabled:
        plots[name] = plots.get(name) or {}
    else:
        if isinstance(inherited, dict) and name in inherited:
            plots[name] = None  # tombstone: Hydra deep-merges group dictionaries
        else:
            plots.pop(name, None)
    if enabled:
        params = self.raw_data.setdefault("plot_overrides", {})
        params.setdefault(name, self._component_default_params("plots", name))
    self.raw_data["plots"] = plots
    if not enabled and isinstance(self.raw_data.get("plot_overrides"), dict):
        self.raw_data["plot_overrides"].pop(name, None)
        if not self.raw_data["plot_overrides"]:
            self.raw_data.pop("plot_overrides", None)
    self._mark_dirty()
    QTimer.singleShot(0, self._render_boxes)


def _add_plot_parameter(self, name: str, key: str, default: Any) -> None:
    plot = self.raw_data.setdefault("plot_overrides", {}).setdefault(name, {})
    plot[key] = copy.deepcopy(default)
    self._mark_dirty()
    QTimer.singleShot(0, self._render_boxes)


def _loader_paths(self):
    from src.app.workspace import WorkspacePaths

    tree = self.tree
    return tree.workspace if tree else WorkspacePaths.for_process()



def _tuner_contributions(self, slot="configuration.tuning"):
    from src.app.core.contributions import ContributionRegistry
    return [item for item in ContributionRegistry(self._loader_paths()).contributions(
        slot, config=self._parameter_context_config()) if item.capability.kind == "tuner"]


def _tuner_ready(self):
    from src.app.pipeline.tuners import tuner_name
    from src.app.core.contributions import ContributionRegistry
    from src.app.core.provider_catalog import get_catalog
    config = self._parameter_context_config()
    if not isinstance(config.get("tuning"), dict):
        return False
    paths = self._loader_paths()
    from src.app.pipeline.exceptions import ConfigurationError
    try:
        component = tuner_name(config)
    except ConfigurationError:
        return False
    selected = get_catalog(paths).resolve('tuner', component)
    return bool(selected and any(item.package.root == selected[0].root and item.capability.id == selected[1].id
        for item in ContributionRegistry(paths).contributions("configuration.parameter_sweeps", config=config)))


def _set_loader_binding(self, path, name):
    if not name:
        self._remove_generic_path(path)
        return
    slot = path[-1]
    value = {"name": name, "params": self._component_default_params(slot, name)}
    self._on_generic_path_edited(path, value)
    QTimer.singleShot(0, self._render_boxes)


def _remove_figures_box(self) -> None:
    # A null selection masks a group-level selection when Hydra composes it.
    if not self._is_base_config() and self._get_resolved_defaults().get("plots") is not None:
        self.raw_data["plots"] = None
    else:
        self.raw_data.pop("plots", None)
    self.raw_data.pop("plot_overrides", None)
    self._mark_dirty()
    self._render_boxes()


def _add_plot_parameter_defaults(self, name: str) -> None:
    """Materialize all first-level defaults for a selected inherited figure."""
    params = self._component_default_params("plots", name)
    self.raw_data.setdefault("plot_overrides", {})[name] = params
    self._mark_dirty()
    QTimer.singleShot(0, self._render_boxes)


def _site_selection(data):
    for entry in data.get("defaults") or []:
        if isinstance(entry, dict):
            for key in ("override /site", "/site", "site"):
                if key in entry:
                    return entry[key]
    value = data.get("site")
    return value if isinstance(value, str) else None


def _set_experiment_site(self, value):
    # Store a Hydra group selection, preserving unrelated defaults and overrides.
    defaults = [
        entry
        for entry in self.raw_data.get("defaults", [])
        if not (isinstance(entry, dict) and any(key in entry for key in ("override /site", "/site", "site")))
    ]
    if isinstance(self.raw_data.get("site"), str):
        self.raw_data.pop("site", None)
    if value is not None:
        defaults.append({"override /site": value})
    if defaults:
        self.raw_data["defaults"] = defaults
    else:
        self.raw_data.pop("defaults", None)
    self._mark_dirty()
    self._render_boxes()


def _add_site_resource(self, key, value):
    self.raw_data.setdefault("resources", {})[key] = value
    self._mark_dirty()
    self._render_boxes()


def _has_experiment_lower_params(self, data: dict, resolved: dict) -> bool:
    """Return whether the lower parameter area has anything visible."""
    if self._is_base_config():
        return True  # group defaults always expose runtime defaults below the divider
    trainer_data = data.get("trainer") if isinstance(data.get("trainer"), dict) else {}
    if any(key in trainer_data for key in ("accelerator", "max_epochs")):
        return True
    tree = self.tree
    paradigm = data.get("paradigm") or self._inherited_paradigm()
    rules = tree.paradigms.get(paradigm) if tree and paradigm else None
    if rules and rules.uses_static_dataset and ("dataset_path" in data or self.view_mode == "resolved"):
        return True
    lower_keys = {
        "total_timesteps",
        "epochs",
        "seed",
        "eval_episodes",
        "eval_freq_steps",
        "eval_freq",
        "eval_every_n_epochs",
        "num_checkpoints",
        "checkpoint_every_steps",
        "run_in_parallel",
        "save_dataset",
        "recover",
        "no_plot",
    }
    if self._plugin_enabled("tensorboard"):
        lower_keys.add("tensorboard")
    if any(key in data for key in lower_keys):
        return True
    if self.view_mode == "resolved" and any(key in resolved for key in lower_keys):
        return True
    standard_keys = lower_keys | {
        "defaults",
        "paradigm",
        "experiment_id",
        "description",
        "methods",
        "env",
        "site",
        "trainer",
        "resources",
        "dataset",
        "dataset_path",
        "hydra",
        "plots",
        "tensorboard",
        "tuning",
        "sweeper",
        "sweep",
    }
    return any(key not in standard_keys and isinstance(value, (int, float, str, bool)) for key, value in data.items())


def _add_experiment_context_param(self, path: str) -> None:
    """Add a context override, initialized from the group's resolved value."""
    resolved = self._get_resolved_defaults()
    parts = path.split(".")
    default_value = resolved
    for part in parts:
        default_value = default_value.get(part) if isinstance(default_value, dict) else None
    if path == "dataset_path":
        self._set_experiment_dataset_override()
        return
    if default_value is None:
        default_value = {
            "trainer.accelerator": "cpu",
            "trainer.max_epochs": 25,
        }.get(path)
    if path.startswith("env."):
        self.raw_data.setdefault("env", {})[parts[-1]] = default_value
    elif path.startswith("trainer."):
        self.raw_data.setdefault("trainer", {})[parts[-1]] = default_value
    self._mark_dirty()
    QTimer.singleShot(0, self._render_boxes)


def _set_experiment_dataset_override(self) -> None:
    """Add a static dataset override without rebuilding under the open menu."""
    tree = self.tree
    paradigm = self.raw_data.get("paradigm") or self._inherited_paradigm()
    rules = tree.paradigms.get(paradigm) if tree else None
    if not rules or not rules.uses_static_dataset:
        return
    env_name = (
        self.env_selection() or self._get_explicit_env() or self._inherited_env() or self._group_name() or "dataset"
    )
    self.raw_data["dataset_path"] = self._get_resolved_defaults().get("dataset_path") or f"in/datasets/{env_name}"
    self._mark_dirty()
    QTimer.singleShot(0, self._render_boxes)


def _experiment_budget_candidates(self, data: dict, resolved: dict, rules) -> list[tuple]:
    """Return budget and runtime options available from the Experiment header."""
    candidates = []
    if rules and rules.budget_unit == "timesteps":
        possible = [
            ("total_timesteps", "Total Timesteps", int(resolved.get("total_timesteps") or 10000)),
            ("seed", "Random Seed", int(resolved.get("seed") if resolved.get("seed") is not None else 42)),
        ]
        if rules.field_enabled("eval_episodes"):
            possible.append(("eval_episodes", "Eval Episodes", int(resolved.get("eval_episodes") or 100)))
        possible.extend(
            [
                ("eval_freq_steps", "Eval Frequency (Steps)", int(resolved.get("eval_freq_steps") or 10000)),
                ("num_checkpoints", "Checkpoints", DEFAULT_CHECKPOINTS),
            ]
        )
    else:
        possible = [
            ("epochs", "Epochs", int(resolved.get("epochs") or 10)),
            ("seed", "Random Seed", int(resolved.get("seed") if resolved.get("seed") is not None else 42)),
            ("eval_every_n_epochs", "Eval Frequency (Epochs)", int(resolved.get("eval_every_n_epochs") or 1)),
        ]
    for key, title, default in possible:
        if key not in data and not (key == "eval_freq_steps" and "eval_freq" in data):
            if key == "num_checkpoints" and any(k in data for k in CHECKPOINT_KEYS):
                continue
            candidates.append((key, title, default))
    optional_integrations = []
    if self._plugin_enabled("tensorboard"):
        optional_integrations.append(("tensorboard", "Log to TensorBoard"))
    for key, title in (
        *optional_integrations,
        ("save_dataset", "Save Dataset"),
        ("recover", "Recover Checkpoint"),
        ("no_plot", "No Plots"),
    ):
        if key not in data:
            candidates.append((key, title, True))
    candidates.append(("custom", "+ Custom Parameter...", None))
    return candidates


def _add_selected_experiment_param(self, item: tuple) -> None:
    key, _title, default = item
    if key == "custom":
        self._prompt_add_custom_experiment_param()
    elif key == "dataset_path":
        self._set_experiment_dataset_override()
    elif "." in key:
        self._add_experiment_context_param(key)
    else:
        self._add_experiment_param_from_menu(key, default)


def _remove_experiment_context_param(self, path: str) -> None:
    parts = path.split(".")
    if len(parts) == 1:
        self.raw_data.pop(path, None)
        self._mark_dirty()
        QTimer.singleShot(0, self._render_boxes)
        return
    section, key = parts
    values = self.raw_data.get(section)
    if isinstance(values, dict):
        values.pop(key, None)
        if not values:
            self.raw_data.pop(section, None)
    self._mark_dirty()
    QTimer.singleShot(0, self._render_boxes)


def _parameter_context_config(self):
    config = copy.deepcopy(self._get_resolved_defaults())

    def merge(target, source):
        for key, item in source.items():
            if isinstance(item, dict) and isinstance(target.get(key), dict):
                merge(target[key], item)
            else:
                target[key] = copy.deepcopy(item)

    merge(config, self.raw_data)
    # An environment selected through Hydra defaults can coexist with a
    # mapping of overrides in env:. Keep both, including symbolic_name.
    for resolver in ("env_selection", "_get_explicit_env", "_inherited_env"):
        method = getattr(self, resolver, None)
        if callable(method):
            environment = method()
            if environment:
                overrides = config.get("env")
                config["env"] = {"name": environment, **(overrides if isinstance(overrides, dict) else {})}
                break
    return config


def _component_parameter_schema(self, path: tuple[str, ...]) -> dict[str, Any]:
    """Look up the active component's declared schema node for a YAML path."""
    from src.app.core.data_loaders import LOADER_SLOTS, loader_metadata, loader_selection

    prefix = 3 if path[:1] == ("methods",) and len(path) > 2 else 1
    if len(path) > prefix and path[prefix - 1] in LOADER_SLOTS and path[prefix] == "params":
        from src.app.pipeline.config import deep_merge, parse_methods_dict

        context = deep_merge(self._get_resolved_defaults(), self.raw_data)
        method = parse_methods_dict(context, paths=self._loader_paths()).get(path[1]) if prefix == 3 else None
        selected, _ = loader_selection(context, path[prefix - 1], method, paths=self._loader_paths())
        schema = loader_metadata(selected, paths=self._loader_paths()).get("parameter_schema", {})
        for key in path[prefix + 1 :]:
            schema = schema.get("properties", {}).get(key, {})
        return schema
    if path[:1] == ("monitors",):
        from src.app.core.provider_catalog import get_catalog
        from src.app.monitoring.config import schema_for
        from src.app.pipeline.config import deep_merge
        from src.app.workspace import WorkspacePaths

        tree = self.tree
        paths = tree.workspace if tree else WorkspacePaths.for_process()
        effective = deep_merge(self._get_resolved_defaults(), self.raw_data)
        return schema_for(path, effective, get_catalog(paths))
    if path[:2] == ("objective", "params"):
        from src.app.objectives.api import load_objective

        spec = self._effective_objective_spec(self.raw_data, self._get_resolved_defaults()) or {}
        tree = self.tree
        workspace = tree.config_root.parent.parent if tree else PROJECT_ROOT
        try:
            schema = load_objective(spec["component"], workspace).metadata.get("schema", {})
        except (ValueError, KeyError, OSError):
            schema = {}
        for key in path[2:]:
            schema = schema.get("properties", {}).get(key, {})
        return schema
    tree = self.tree
    schema = tree.parameter_schema_for_config(str(self.current_rel_path or "")) if tree else {}
    for key in path:
        schema = schema.get("properties", {}).get(key, {}) if isinstance(schema, dict) else {}
    return schema if isinstance(schema, dict) else {}


def _parameter_required(self, path):
    if not path:
        return False
    if path == ("objective", "component"):
        return True
    return path[-1] in self._component_parameter_schema(path[:-1]).get("required", [])


def _validate_method_name(self, name: str, existing_names: set[str]) -> str | None:
    """Return an error message if invalid, or None if valid."""
    name = name.strip()
    if not name:
        return "Method name cannot be empty."
    if name == "params":
        return "The method name 'params' is reserved for universal parameters."
    if name in existing_names:
        return f"A method named '{name}' already exists in this experiment."
    if not re.match(r"^[a-zA-Z0-9_\-]+$", name):
        return "Method name contains invalid characters. Use only letters, digits, hyphens, and underscores."
    return None


def duplicate_method(self, name: str) -> str | None:
    """Duplicate an existing method card with all its configuration."""
    if self._is_base_config():
        return None
    methods = self.raw_data.get("methods")
    if not isinstance(methods, dict) or name not in methods:
        return None
    source_spec = methods[name]

    existing_keys = set(methods.keys())
    base = f"{name}_copy"
    if base not in existing_keys:
        dup_name = base
    else:
        n = 2
        while f"{base}_{n}" in existing_keys:
            n += 1
        dup_name = f"{base}_{n}"

    new_methods = {}
    for k, v in methods.items():
        new_methods[k] = v
        if k == name:
            new_methods[dup_name] = copy.deepcopy(source_spec)

    self.raw_data["methods"] = new_methods
    self._mark_dirty()
    self._render_boxes()
    return dup_name


def _get_group_default_method(self) -> tuple[str | None, dict | None]:
    """Return (method_name, method_spec) for the default method declared in group's _base.yaml."""
    base_data = None
    if self.current_path and self._group_name():
        parts = Path(self.current_rel_path).parts
        group_dir = self.current_path.parents[len(parts) - 3]
        local_base = group_dir / "_base.yaml"
        if local_base.is_file():
            try:
                base_data = yaml.safe_load(local_base.read_text(encoding="utf-8")) or {}
            except Exception:
                pass
    if not base_data:
        tree = self.tree
        group = self._group_name()
        if tree and group:
            base_data = _read_group_base(tree, group)
    if base_data:
        base_methods = base_data.get("methods", {})
        if isinstance(base_methods, dict):
            for k, v in base_methods.items():
                if k != "params" and isinstance(v, dict):
                    return k, copy.deepcopy(v)
    return None, None


def add_method(self, agent=None, model=None, name=None):
    """Add a method to the open experiment or base defaults. Returns its name, or None."""
    if self._is_base_config():
        methods = self.raw_data.setdefault("methods", {})
        if any(k != "params" and isinstance(v, dict) for k, v in methods.items()):
            return None
        meth_name = name or "default_method"
        spec = {}
        if agent:
            spec["agent"] = agent
        if model:
            spec["model"] = model
        methods[meth_name] = spec
        self._mark_dirty()
        self._render_boxes()
        return meth_name

    def_name, def_spec = self._get_group_default_method()
    paradigm = self.raw_data.get("paradigm") or self._inherited_paradigm()
    tree = self.tree
    if not tree or paradigm not in tree.paradigms:
        return None
    p_obj = tree.paradigms.get(paradigm) if tree and paradigm in tree.paradigms else None
    default_model = (def_spec.get("model") if def_spec else None) or (p_obj.raw.get("default_model") if p_obj else None)

    methods = self.raw_data.setdefault("methods", {})
    if name:
        meth_name = name
    elif agent or model:
        from ..config_model import unique_method_name

        meth_name = unique_method_name(methods, agent, model or default_model)
    elif def_name and def_name not in methods:
        meth_name = def_name
    else:
        base = "method"
        if base not in methods:
            meth_name = base
        else:
            n = 2
            while f"{base}_{n}" in methods:
                n += 1
            meth_name = f"{base}_{n}"

    spec = copy.deepcopy(def_spec) if def_spec else {}
    if agent:
        spec["agent"] = agent
    if model:
        spec["model"] = model
    methods[meth_name] = spec
    self._mark_dirty()
    self._render_boxes()
    return meth_name


def _on_experiment_env_changed(self, value):
    if self._block_updates:
        return
    selected = self.env_selection()
    self._set_experiment_env(selected)
    self._mark_dirty()
    self._render_boxes()


def remove_method(self, name):
    """Remove a method from the open experiment. Returns True if it went."""
    if not remove_method_from(self.raw_data, name):
        return False
    self._mark_dirty()
    self._render_boxes()
    return True


def rename_method(self, old_name: str, new_name: str) -> bool:
    """Rename a method in the open experiment while preserving dict ordering."""
    if not old_name or not new_name or old_name == new_name:
        return False
    methods = self.raw_data.get("methods")
    if not isinstance(methods, dict) or old_name not in methods:
        return False
    new_methods = {}
    for k, v in methods.items():
        if k == old_name:
            new_methods[new_name] = v
        else:
            new_methods[k] = v
    self.raw_data["methods"] = new_methods
    self._mark_dirty()
    self._render_boxes()
    return True


def reorder_method(self, method_name: str, target_index: int) -> bool:
    """Reorder method cards by moving method_name to target_index."""
    methods = self.raw_data.get("methods")
    if not isinstance(methods, dict) or method_name not in methods:
        return False
    has_params = "params" in methods
    params_val = methods.get("params")

    method_keys = [k for k in methods.keys() if k != "params"]
    if method_name not in method_keys:
        return False
    method_keys.remove(method_name)
    target_index = max(0, min(target_index, len(method_keys)))
    method_keys.insert(target_index, method_name)

    new_methods = {}
    if has_params:
        new_methods["params"] = params_val
    for k in method_keys:
        new_methods[k] = methods[k]

    self.raw_data["methods"] = new_methods
    self._mark_dirty()
    self._render_boxes()
    return True


def _on_paradigm_changed(self, value):
    if self._block_updates:
        return
    self.raw_data["paradigm"] = value
    # An env or agent valid under the old paradigm may be forbidden under the
    # new one, so drop stale choices and rebuild against the new constraints.
    tree = self.tree
    if tree and value in tree.paradigms:
        rules = tree.paradigms[value]
        env_name = self._get_explicit_env()
        environment = self._environment_for_name(tree, env_name)
        if environment and not rules.permits_environment(environment):
            self._set_experiment_env(None)
        self._pin_fields_the_paradigm_forbids(rules)
    self._mark_dirty()
    self._render_boxes()


def _on_nested_edited(self, parent_key, child_key, value):
    if self._block_updates:
        return
    return self.document.set((parent_key, child_key), value)


def _on_method_enabled_toggled(self, method_name: str, enabled: bool):
    if self._block_updates:
        return
    methods = self.raw_data.setdefault("methods", {})
    method = methods.setdefault(method_name, {})
    method["enabled"] = bool(enabled)
    self._mark_dirty()
    QTimer.singleShot(0, self._render_boxes)


def _on_method_param_edited(self, method_name, param_key, value):
    if self._block_updates:
        return
    from .bindings import method_parameter_path

    path = ("methods", method_name, param_key)
    m_spec = self.raw_data.get("methods", {}).get(method_name, {})
    if param_key not in ("agent", "model", "enabled", "base", "from_study"):
        for kind in ("agent", "model"):
            target = self._method_component_parameter_root(method_name, kind)
            if target is not m_spec and param_key in target:
                path = method_parameter_path(self.raw_data, method_name, kind, (param_key,))
                break
        else:
            agent_name, _ = component_selection(m_spec.get("agent"))
            tree = self.tree
            keys = set(tree.agent_hyperparameters(agent_name)) if tree and agent_name else set()
            keys.update(get_agent_candidate_hyperparameters(agent_name, tree))
            if param_key in keys:
                path = method_parameter_path(self.raw_data, method_name, "agent", (param_key,))
    return self.document.set(path, value)


def _on_method_nested_param_edited(self, method_name: str, parent_key: str, child_key: str, value: Any):
    if self._block_updates:
        return
    from .bindings import method_parameter_path

    model = self._method_component_parameter_root(method_name, "model")
    path = (
        method_parameter_path(self.raw_data, method_name, "model", (parent_key, child_key))
        if isinstance(model.get(parent_key), dict)
        else ("methods", method_name, parent_key, child_key)
    )
    return self.document.set(path, value)


def _remove_method_nested_param(self, method_name: str, parent_key: str, child_key: str):
    """Remove a nested method parameter, including parameters inside a model config."""
    methods = self.raw_data.get("methods", {})
    m_spec = methods.get(method_name, {}) if isinstance(methods, dict) else {}
    if isinstance(m_spec, dict):
        model_config = self._method_component_parameter_root(method_name, "model")
        if isinstance(model_config.get(parent_key), dict):
            model_config[parent_key].pop(child_key, None)
        nested = m_spec.get(parent_key)
        if isinstance(nested, dict):
            nested.pop(child_key, None)
    self._mark_dirty()
    self._render_boxes()


def _method_component_parameter_root(self, method_name: str, kind: str, create: bool = False) -> dict:
    methods = self.raw_data.setdefault("methods", {}) if create else self.raw_data.get("methods", {})
    m_spec = methods.setdefault(method_name, {}) if create else methods.get(method_name, {})
    model_spec = m_spec.get(kind)
    if isinstance(model_spec, dict):
        if any(isinstance(model_spec.get(key), str) for key in COMPONENT_ID_KEYS):
            return model_spec
        model_name, _ = component_selection(model_spec)
        value = model_spec.get(model_name)
        if isinstance(value, dict):
            return value
        if create and model_name:
            model_spec[model_name] = {}
            return model_spec[model_name]
    return m_spec


def _method_model_parameter_root(self, method_name: str, model_name: str, create: bool = False) -> dict:
    return self._method_component_parameter_root(method_name, "model", create=create)


def _on_method_model_parameter_edited(
    self,
    method_name: str,
    model_name: str,
    path: tuple[str, ...],
    value: Any,
    schema_model_name: str | None = None,
    schema_path: tuple[str, ...] | None = None,
):
    if self._block_updates or not path:
        return
    tree = self.tree
    schema_owner = schema_model_name or model_name
    schema = tree.model_parameter_schema(schema_owner, schema_path if schema_path is not None else path) if tree else {}
    from .document import validation_error

    if validation_error(value, schema):
        return
    from .bindings import method_parameter_path

    storage = method_parameter_path(self.raw_data, method_name, "model", path)
    document = self.document

    def identity(storage_path, previous):
        selected = component_identity(storage_path, previous)
        if selected is not None:
            return selected
        if schema_owner != model_name:
            return {"name": previous if isinstance(previous, str) else schema_owner}
        return None

    document.selector_identity = identity
    return document.set(storage, value, schema=schema)


def _remove_method_model_parameter(
    self, method_name: str, model_name: str, path: tuple[str, ...], schema_model_name=None, schema_path=None
):
    if not path:
        return
    tree = self.tree
    lookup_path = schema_path if schema_path is not None else path
    schema = tree.model_parameter_schema(schema_model_name or model_name, lookup_path[:-1]) if tree else {}
    from .bindings import method_parameter_path

    storage = method_parameter_path(self.raw_data, method_name, "model", path)
    result = self.document.remove(storage, required=bool(lookup_path and lookup_path[-1] in schema.get("required", [])))
    if not result.changed:
        return
    spec = self.raw_data.get("methods", {}).get(method_name, {})
    dimensions = spec.get("tune", {})
    if isinstance(dimensions, dict):
        prefix = "model." + ".".join(path)
        for key in list(dimensions):
            if key == prefix or key.startswith(prefix + "."):
                dimensions.pop(key)
        if not dimensions:
            spec.pop("tune", None)
    self._render_boxes()


def _add_method_model_parameter(
    self,
    method_name: str,
    model_name: str,
    path: tuple[str, ...],
    schema_model_name: str | None = None,
    schema_path: tuple[str, ...] | None = None,
):
    if not path:
        return
    tree = self.tree
    schema_owner = schema_model_name or model_name
    default = tree.model_parameters(schema_owner) if tree else {}
    lookup_path = schema_path if schema_path is not None else path
    found = True
    for key in lookup_path:
        if isinstance(default, dict) and key in default:
            default = default[key]
        else:
            found = False
            break
    if not found:
        schema = tree.model_parameter_schema(schema_owner, lookup_path) if tree else {}
        default = schema.get("default")
    self._on_method_model_parameter_edited(method_name, model_name, path, default, schema_owner, lookup_path)
    self._render_boxes()


def _on_method_child_model_changed(
    self, method_name: str, parent_model_name: str, path: tuple[str, ...], child_model_name: str
):
    if self._block_updates:
        return
    value = {"name": child_model_name} if child_model_name else None
    target = self._method_model_parameter_root(method_name, parent_model_name, create=True)
    for key in path[:-1]:
        if not isinstance(target.get(key), dict):
            target[key] = {}
        target = target[key]
    if child_model_name:
        target[path[-1]] = value
    else:
        target.pop(path[-1], None)
    self._mark_dirty()
    self._render_boxes()


def remove_block(self, block_name: str):
    """Remove a configuration block from the open experiment."""
    if block_name == "trainer":
        self.raw_data.pop("trainer", None)
    elif block_name == "checkpoint":
        self.raw_data.pop("checkpoint", None)
    elif block_name == "tuning":
        if self._get_resolved_defaults().get("tuning") is not None:
            self.raw_data["tuning"] = None
        else:
            self.raw_data.pop("tuning", None)
    elif block_name == "variability":
        self.raw_data.pop("variability", None)
    elif block_name == "objective":
        self.raw_data.pop("objective", None)
    elif block_name == "env":
        if isinstance(self.raw_data.get("env"), dict):
            self.raw_data.pop("env", None)
    elif block_name == "dataset":
        self.raw_data.pop("dataset_path", None)
        self.raw_data.pop("offline_datasets", None)
        self.raw_data.pop("data_module", None)
    elif block_name == "params":
        self._show_empty_shared_params = False
        if "methods" in self.raw_data and isinstance(self.raw_data["methods"], dict):
            self.raw_data["methods"].pop("params", None)
    elif block_name in ("data_module", "evaluator"):
        self.raw_data.pop(block_name, None)
    elif block_name == "plots":
        self.raw_data.pop("plots", None)
        self.raw_data.pop("plot_overrides", None)
    elif block_name == "metrics":
        telemetry = self.raw_data.get("telemetry")
        if isinstance(telemetry, dict):
            telemetry.pop("metrics", None)
            if not telemetry:
                self.raw_data.pop("telemetry", None)
    elif block_name == "monitors":
        # The unified box owns standalone instances as well as stream choices.
        if self._get_resolved_defaults().get("monitors"):
            self.raw_data["monitors"] = None
        else:
            self.raw_data.pop("monitors", None)
        telemetry = self.raw_data.setdefault("telemetry", {})
        if _monitor_settings(self._get_resolved_defaults()) is not None:
            telemetry["monitors"] = None  # mask inherited selections: absent box streams everything
        else:
            telemetry.pop("monitors", None)
        if not telemetry:
            self.raw_data.pop("telemetry", None)
    self._mark_dirty()
    self._render_boxes()


def _toggle_method_sweep(self, method, target, value):
    spec = self.raw_data.get("methods", {}).get(method, {})
    dimensions = spec.get("tune", {})
    if isinstance(dimensions, dict) and target in dimensions:
        dimensions.pop(target)
        if not dimensions:
            spec.pop("tune", None)
        self._mark_dirty()
    else:
        base = value if isinstance(value, (int, float)) and not isinstance(value, bool) else 1
        dimensions = spec.setdefault("tune", {})
        dimensions[target] = {"type": "float", "low": base, "high": base}
        self._pending_sweep = None
        self._mark_dirty()
    self._render_boxes()


def _cancel_sweep_setup(self, method, target):
    if getattr(self, "_pending_sweep", None) == (method, target):
        self._pending_sweep = None
        self._render_boxes()


def _select_sweep_type(self, method, target, value, kind):
    if kind is None:
        self._cancel_sweep_setup(method, target)
        return
    self._pending_sweep = None
    dimensions = self.raw_data.setdefault("methods", {}).setdefault(method, {}).setdefault("tune", {})
    if kind == "choice":
        dimensions[target] = {"type": kind, "values": [value, ""]}
    else:
        base = value if isinstance(value, (int, float)) and not isinstance(value, bool) else 1
        if kind == "int":
            base = int(base)
        dimensions[target] = {"type": kind, "low": base, "high": base}
    self._mark_dirty()
    self._render_boxes()


def _edit_sweep_choice(self, method, target, index, text):
    try:
        value = yaml.safe_load(text) if text.strip() else ""
    except yaml.YAMLError:
        value = text
    self.raw_data["methods"][method]["tune"][target]["values"][index] = value
    self._mark_dirty()


def _add_sweep_choice(self, method, target):
    self.raw_data["methods"][method]["tune"][target]["values"].append("")
    self._mark_dirty()
    self._render_boxes()


def _remove_sweep_choice(self, method, target, index):
    self.raw_data["methods"][method]["tune"][target]["values"].pop(index)
    self._mark_dirty()
    self._render_boxes()


def _edit_sweep_dimension(self, method, target, key, text):
    dimension = self.raw_data["methods"][method]["tune"][target]
    if key == "step" and not text.strip():
        dimension.pop(key, None)
    else:
        try:
            value = int(text.strip()) if dimension.get("type") == "int" else float(text)
        except (ValueError, OverflowError):
            return
        if isinstance(value, float) and not math.isfinite(value):
            return
        dimension[key] = value
    self._mark_dirty()


def _add_sweep_dimension(self, method, target, key):
    dimension = self.raw_data["methods"][method]["tune"][target]
    dimension.setdefault(key, None)
    self._mark_dirty()
    self._render_boxes()


def _remove_sweep_dimension(self, method, target, key):
    self.raw_data["methods"][method]["tune"][target].pop(key, None)
    self._mark_dirty()
    self._render_boxes()


def _on_tuner_selected(self, name):
    if self._block_updates or not name:
        return
    self.raw_data.setdefault("tuning", {})["component"] = name
    self._mark_dirty()
    self._render_boxes()


def _on_tuner_option_edited(self, component, key, value):
    if self._block_updates:
        return
    was_ready = self._tuner_ready()
    result = self.document.set(("tuning", "options", component, key), value)
    if result.error or not result.changed:
        return
    if was_ready != self._tuner_ready():
        self._render_timer.start(0)
    # Dependencies are declared by providers and may be bools/numbers/text,
    # as well as selectors. Coalesce typing and spin-box changes.
    from src.app.pipeline.tuners import get_tuner

    try:
        fields = get_tuner(name=component, paths=self._loader_paths()).settings()
    except Exception:
        fields = []
    if any(key in field.get("when", {}) or key in field.get("required_when", {}) for field in fields):
        self._sweep_refresh_timer.start(0)


def _on_tuner_choice_edited(self, component, key, value):
    if self._block_updates:
        return
    self._on_tuner_option_edited(component, key, value)
    self._refresh_sweep_box()


def _add_tuner_option(self, component, key, value):
    self._on_tuner_option_edited(component, key, value)
    self._refresh_sweep_box()


def _remove_tuner_option(self, component, key):
    tuning = self.raw_data.get("tuning", {})
    tuning.get("options", {}).get(component, {}).pop(key, None)
    from src.app.pipeline.tuners import get_tuner
    try:
        fields = get_tuner(name=component, paths=self._loader_paths()).settings()
    except Exception:
        fields = []
    if any(field["key"] == key and field.get("legacy_flat") for field in fields):
        tuning.pop(key, None)
    self._mark_dirty()
    self._refresh_sweep_box()


def _add_standalone_monitor(self, component):
    """Adding a provider is explicit; discovery never installs it into a recipe."""
    from src.app.pipeline.config import deep_merge

    effective = deep_merge(self._get_resolved_defaults(), self.raw_data)
    existing = effective.get("monitors") or {}
    base = re.sub(r"[^A-Za-z0-9_-]", "_", component)
    if not base or not base[0].isalpha():
        base = "monitor_" + base
    identity, suffix = base, 2
    settings = _monitor_settings(effective) or {}
    from src.app.telemetry.monitors import MonitorSelection

    removed = MonitorSelection(settings.get("removed", []))
    reusable = next(
        (
            name
            for name, spec in existing.items()
            if isinstance(spec, dict)
            and spec.get("component") == component
            and (spec.get("enabled") is False or not removed.keeps(f"interp/{name}/"))
        ),
        None,
    )
    if reusable:
        identity = reusable
    while identity in existing and identity != reusable and existing[identity] != {"enabled": False}:
        identity = f"{base}_{suffix}"
        suffix += 1
    self.raw_data.setdefault("monitors", {})[identity] = {
        "component": component,
        "target": "",
        **(existing.get(identity) or {}),
        "enabled": True,
    }
    if settings:
        updated = copy.deepcopy(settings)
        for key in ("disabled", "removed"):
            if key in updated:
                updated[key] = list(MonitorSelection(updated[key]).with_toggled([f"interp/{identity}/"], True).disabled)
        self.raw_data.setdefault("telemetry", {})["monitors"] = updated
    self._mark_dirty()
    QTimer.singleShot(0, self._render_boxes)


def _remove_standalone_monitor(self, identity):
    """Delete owned instances; suppress inherited/model defaults explicitly."""
    from src.app.pipeline.config import deep_merge
    from src.app.telemetry.monitors import MonitorSelection

    inherited = self._get_resolved_defaults()
    effective = deep_merge(inherited, self.raw_data)
    own = self.raw_data.setdefault("monitors", {})
    if identity not in own or identity in (inherited.get("monitors") or {}):
        own[identity] = {"enabled": False}
    else:
        own.pop(identity)
    if not own:
        self.raw_data.pop("monitors", None)
    settings = _monitor_settings(effective)
    if settings is not None:
        updated = copy.deepcopy(settings)
        for key in ("disabled", "removed"):
            if key in updated:
                updated[key] = list(MonitorSelection(updated[key]).with_toggled([f"interp/{identity}/"], True).disabled)
        self.raw_data.setdefault("telemetry", {})["monitors"] = updated
    self._mark_dirty()
    QTimer.singleShot(0, self._render_boxes)


def _on_recorded_metrics_changed(self, metrics):
    if self._block_updates:
        return
    self.document.set(("telemetry", "metrics"), list(metrics))


def add_block(self, block_name: str):
    """Add an optional configuration block to the open experiment."""
    if block_name == "objective":
        online = (self.raw_data.get("paradigm") or self._inherited_paradigm()) == "online_rl"
        self.raw_data["objective"] = {
            "component": "learning_curve" if online else "final_metric",
            "params": {} if online else {"metric": "val/loss"},
        }
    elif block_name == "trainer":
        self.raw_data["trainer"] = {"accelerator": "cpu", "max_epochs": 25}
    elif block_name == "checkpoint":
        self.raw_data["checkpoint"] = {"monitor": "val/loss", "mode": "min", "save_top_k": 1, "every_n_epochs": 1}
    elif block_name in ("tuning", "sweeper"):
        contributions = self._tuner_contributions()
        if not contributions:
            return
        component = contributions[0].capability.id
        from src.app.pipeline.tuners import get_tuner
        fields = get_tuner(name=component, paths=self._loader_paths()).settings()
        options = {field["key"]: copy.deepcopy(field["default"]) for field in fields
                   if not field.get("optional") and "default" in field}
        paradigm = self.raw_data.get("paradigm") or self._inherited_paradigm()
        online = paradigm == "online_rl"
        self.raw_data["tuning"] = {
            "metric": "eval/reward" if online else "val/loss",
            "direction": "maximize" if online else "minimize",
            "component": component,
            "options": {component: options},
        }
    elif block_name == "variability":
        self.raw_data["variability"] = {
            "training_runs": 1,
            "starting_seed": int(self._get_resolved_defaults().get("seed", 1) or 1),
            "parallel_runs": 1,
        }
    elif block_name == "env":
        self.raw_data["env"] = {}
    elif block_name == "dataset":
        tree = self.tree
        paradigm = self.raw_data.get("paradigm") or self._inherited_paradigm()
        rules = tree.paradigms.get(paradigm) if tree else None
        if not rules or not rules.uses_static_dataset:
            return
        env_name = (
            self.env_selection() or self._get_explicit_env() or self._inherited_env() or self._group_name() or "dataset"
        )
        self.raw_data["dataset_path"] = self._get_resolved_defaults().get("dataset_path") or f"in/datasets/{env_name}"
    elif block_name == "params":
        self._show_empty_shared_params = True
        self.raw_data.setdefault("methods", {})["params"] = {}
    elif block_name == "metrics":
        # Start from what the run records anyway: every section ticked, so nothing changes until unticked.
        telemetry = self.raw_data.setdefault("telemetry", {})
        telemetry["metrics"] = [f"{section}/" for section in METRIC_SECTIONS]
    elif block_name == "monitor_components":
        self.raw_data["monitors"] = {}
    elif block_name == "monitors":
        self.raw_data.setdefault("telemetry", {})["monitors"] = {"enabled": True, "disabled": []}
    self._mark_dirty()
    self._render_boxes()


def _on_universal_param_edited(self, key, value):
    if self._block_updates:
        return
    return self.document.set(("methods", "params", *key.split(".")), value)


def _remove_universal_param(self, key):
    if "methods" in self.raw_data and "params" in self.raw_data["methods"]:
        self._show_empty_shared_params = True
        params = self.raw_data["methods"]["params"]
        if "." in key:
            parts = key.split(".")
            target = params
            for p in parts[:-1]:
                if isinstance(target, dict) and p in target:
                    target = target[p]
                else:
                    target = None
                    break
            if isinstance(target, dict):
                target.pop(parts[-1], None)
            if len(parts) > 1 and parts[0] in params and isinstance(params[parts[0]], dict) and not params[parts[0]]:
                params.pop(parts[0], None)
        else:
            params.pop(key, None)
        dimensions = params.get("tune", {})
        if isinstance(dimensions, dict):
            dimensions.pop(key, None)
            dimensions.pop("agent." + key, None)
            dimensions.pop("model." + key, None)
            if not dimensions:
                params.pop("tune", None)
        if not self.raw_data["methods"]["params"]:
            self.raw_data["methods"].pop("params", None)
        self._mark_dirty()
        self._render_boxes()


def _remove_experiment_parameter(self, key):
    self.raw_data.pop(key, None)
    params = self.raw_data.get("methods", {}).get("params", {})
    dimensions = params.get("tune", {}) if isinstance(params, dict) else {}
    if isinstance(dimensions, dict):
        dimensions.pop("config." + key, None)
        if not dimensions:
            params.pop("tune", None)
    self._mark_dirty()
    self._render_boxes()


def _select_shared_component(self, kind, old, selected):
    if self._block_updates or not selected or selected == old:
        return
    params = self.raw_data.setdefault("methods", {}).setdefault("params", {})
    components = params.setdefault(kind, {})
    if not isinstance(components, dict):
        return
    previous = components.pop(old, {}) if old else {}
    components.setdefault(selected, previous)
    self._mark_dirty()
    self._render_boxes()


def _add_universal_param(self, key, value=None):
    if not key:
        return
    if value is None or value == "":
        value = self._resolve_default_param_value("", key)
    methods = self.raw_data.setdefault("methods", {})
    params = methods.setdefault("params", {})
    if "." in key:
        parts = key.split(".")
        target = params
        for p in parts[:-1]:
            target = target.setdefault(p, {})
        target[parts[-1]] = value
    else:
        params[key] = value
    self._mark_dirty()
    self._render_boxes()


def _resolve_default_param_value(self, method_name: str, param_key: str):
    """Look up hierarchical default for param_key:
    1. methods.params in current file
    2. Group _base.yaml (methods.params or methods[method_name])
    3. in/config/agent/<agent>.yaml
    4. in/config/model/<model>.yaml
    5. Known defaults fallback
    """
    methods = self.raw_data.get("methods")
    if isinstance(methods, dict):
        univ_params = methods.get("params")
        if isinstance(univ_params, dict) and param_key in univ_params:
            return univ_params[param_key]

    m_spec = (
        methods.get(method_name) if isinstance(methods, dict) and isinstance(methods.get(method_name), dict) else {}
    )
    agent_name, _ = component_selection(m_spec.get("agent"))
    model_name, _ = component_selection(m_spec.get("model"))

    tree = self.tree
    group = self._group_name()

    # 2. Group _base.yaml
    if tree and group:
        base_data = _read_group_base(tree, group)
        base_methods = base_data.get("methods")
        if isinstance(base_methods, dict):
            base_params = base_methods.get("params")
            if isinstance(base_params, dict) and param_key in base_params:
                return base_params[param_key]
            if method_name in base_methods and isinstance(base_methods[method_name], dict):
                if param_key in base_methods[method_name]:
                    return base_methods[method_name][param_key]

    # 3. Agent YAML: in/config/agent/<agent>.yaml
    if tree and agent_name:
        agent_params = tree.agent_hyperparameters(agent_name)
        if param_key in agent_params:
            return agent_params[param_key]

    # 4. Model YAML: in/config/model/<model>.yaml
    if tree and model_name:
        model_params = tree.model_parameters(model_name)
        if param_key in model_params:
            return model_params[param_key]

    # 5. Generic fallback
    known = {
        "lr": 0.0003,
        "learning_rate": 0.0003,
        "batch_size": 64,
        "gamma": 0.99,
        "ent_coef": 0.01,
        "clip_range": 0.2,
        "epochs": 25,
        "eval_every_n_epochs": 5,
        "weight_decay": 0.0001,
    }
    if param_key in known:
        return known[param_key]

    if "lr" in param_key:
        return 0.0003
    if "coef" in param_key or "alpha" in param_key:
        return 0.01
    return 0.0


def _remove_method_param(self, method_name, param_key):
    if "methods" in self.raw_data and method_name in self.raw_data["methods"]:
        m_spec = self.raw_data["methods"][method_name]
        for kind in ("agent", "model"):
            target = self._method_component_parameter_root(method_name, kind)
            if param_key in target:
                target.pop(param_key)
                break
        else:
            m_spec.pop(param_key, None)
        dimensions = m_spec.get("tune", {})
        if isinstance(dimensions, dict):
            dimensions.pop("agent." + param_key, None)
            dimensions.pop("model." + param_key, None)
            if not dimensions:
                m_spec.pop("tune", None)
        self._mark_dirty()
        self._render_boxes()


def _add_method_param(self, method_name, param_key, value=None):
    if not param_key:
        return
    methods = self.raw_data.setdefault("methods", {})
    m_spec = methods.setdefault(method_name, {})
    model_name, _ = component_selection(m_spec.get("model"))

    if self.tree and model_name and param_key in self.tree.model_parameter_keys(model_name):
        self._add_method_model_parameter(method_name, model_name, (param_key,))
        return

    if value is None or value == "":
        value = self._resolve_default_param_value(method_name, param_key)
    self._on_method_param_edited(method_name, param_key, value)
    self._render_boxes()


def _add_experiment_param(self, key: str, default_val: Any = None, *, render: bool = True):
    """Add an experiment-level parameter override to the configuration."""
    if not key:
        return
    if default_val is not None:
        self.raw_data[key] = default_val
    elif key == "total_timesteps":
        self.raw_data[key] = int(self._get_resolved_defaults().get("total_timesteps") or 10000)
    elif key == "seed":
        res_seed = self._get_resolved_defaults().get("seed")
        self.raw_data[key] = int(res_seed if res_seed is not None else 42)
    elif key == "eval_episodes":
        self.raw_data[key] = int(self._get_resolved_defaults().get("eval_episodes") or 100)
    elif key == "eval_freq_steps":
        self.raw_data[key] = int(self._get_resolved_defaults().get("eval_freq_steps") or 10000)
    elif key in ("tensorboard", "save_dataset", "recover", "no_plot"):
        self.raw_data[key] = True
    else:
        self.raw_data[key] = ""
    self._mark_dirty()
    if render:
        self._render_boxes()


def _set_method_model(self, method_name: str, model_name: str) -> None:
    """Point a method at a model, and re-read the configs from disk.

    The model combo is filled from in/config/model/, so a model written
    while the app is open is invisible until the cached tree re-scans.
    """
    if not model_name:
        return
    methods = self.raw_data.setdefault("methods", {})
    methods.setdefault(method_name, {})["model"] = model_name
    self._mark_dirty()
    self.refresh_tree()
    self._render_boxes()


def _pin_fields_the_paradigm_forbids(self, rules):
    """Write legal values for the fields the chosen paradigm disables.

    in/config/config.yaml defaults eval_episodes to 100 for the online case.
    A config that simply omits it inherits that and is rejected before
    training starts, which is why the shipped offline and supervised bases pin it.
    Marking the field "not used" in the panel is not enough on its own:
    the inherited value still reaches the backend.

    Switching back to a paradigm that allows a field clears the pin, but only
    while it still holds the value set here, so a deliberate choice survives.
    """
    for name, forced in self._FORBIDDEN_FIELD_VALUES.items():
        if not rules.field_enabled(name):
            self.raw_data[name] = forced
        elif self.raw_data.get(name) == forced:
            self.raw_data.pop(name, None)


def _reconcile_group_base(self):
    """Re-pin a group base against whatever paradigm it currently declares.

    _on_paradigm_changed covers picking a paradigm, but not a base that
    already had one - an older group, or one where only the environment was
    changed. Those keep inheriting eval_episodes: 100 and stay unrunnable,
    so the base is reconciled whenever it is saved.

    Only _base.yaml: the pin belongs on the file that binds the paradigm,
    and experiments inherit it.
    """
    if not self._is_base_config():
        return
    paradigm = self.raw_data.get("paradigm")
    tree = self.tree
    if tree and isinstance(paradigm, str) and paradigm in tree.paradigms:
        self._pin_fields_the_paradigm_forbids(tree.paradigms[paradigm])


def _drop_empty_methods(self):
    """Remove a methods block with nothing in it.

    Preserves methods if it has non-empty universal params or concrete methods.
    """
    methods = self.raw_data.get("methods")
    if not isinstance(methods, dict):
        return
    if "params" in methods and not methods["params"]:
        methods.pop("params", None)
    if not methods:
        self.raw_data.pop("methods", None)
