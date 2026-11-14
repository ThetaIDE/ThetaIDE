"""Experiment export service for Theta-IDE.

Packages experiment configurations, base templates, environment hooks/valuations,
domain rules, custom model architectures, and optional checkpoint weights into a
self-contained, installable Theta Hub component package (.zip).
"""

from __future__ import annotations

import json
import logging
import re
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

from src.app.workspace import WorkspacePaths

log = logging.getLogger(__name__)


def _component_names(value: Any) -> Set[str]:
    """Names from a recipe's scalar, name/params mapping, or list selection."""
    if isinstance(value, str):
        return {value} if value else set()
    if isinstance(value, dict):
        name = value.get("name") or value.get("class") or value.get("type")
        return {str(name)} if name else set()
    if isinstance(value, (list, tuple)):
        return set().union(*(_component_names(item) for item in value)) if value else set()
    return set()


def _plot_names(value: Any) -> Set[str]:
    if isinstance(value, list):
        return {str(item) for item in value}
    if isinstance(value, dict):
        return {
            str(name)
            for name, options in value.items()
            if isinstance(options, dict) and name not in ("style", "figsize")
        }
    return set()


@dataclass
class ExportOptions:
    """Configuration options for exporting an experiment or experiment group plugin."""

    include_checkpoints: bool = False
    include_environment: bool = True
    include_models: bool = True
    include_rules: bool = True
    include_evaluators: bool = True
    include_plotters: bool = True
    include_data_modules: bool = True
    include_all_group_experiments: bool = True
    output_path: Path | None = None
    author: str = "Theta Researcher"
    version: str = "1.0.0"
    description: str | None = None


@dataclass
class ExportResult:
    """Result of an experiment export operation."""

    success: bool
    archive_path: Path | None = None
    manifest: Dict[str, Any] | None = None
    files_included: List[str] = field(default_factory=list)
    error: str | None = None


class ExperimentExporter:
    """Extracts, traces dependencies, and packages experiments into installable Hub components."""

    def __init__(self, workspace_root: Path | None = None):
        self.paths = (workspace_root if isinstance(workspace_root, WorkspacePaths) else
                      WorkspacePaths(workspace_root) if workspace_root is not None else WorkspacePaths.for_process())
        self.workspace_root = self.paths.root

    def export(
        self,
        experiment_id_or_path: str | Path,
        options: ExportOptions | None = None,
    ) -> ExportResult:
        """Export an experiment and its full dependency closure to a .zip archive.

        Args:
            experiment_id_or_path: e.g. "group/experiment", "group/experiment.yaml",
                or absolute Path to the YAML recipe.
            options: Export configuration options.

        Returns:
            ExportResult containing archive path and manifest if successful.
        """
        opts = options or ExportOptions()

        try:
            from src.app.services.rulesets import require_canonical_rules_layout, reject_legacy_rules_paths
            require_canonical_rules_layout(self.workspace_root)
            group, exp_name, exp_yaml_path = self._resolve_experiment_path(experiment_id_or_path)
            if not exp_yaml_path.exists():
                return ExportResult(
                    success=False,
                    error=f"Experiment config file not found: {exp_yaml_path}",
                )

            # 1. Parse experiment YAML and trace templates
            try:
                raw_cfg = yaml.safe_load(exp_yaml_path.read_text(encoding="utf-8")) or {}
            except Exception as e:
                return ExportResult(
                    success=False,
                    error=f"Failed to parse experiment YAML '{exp_yaml_path}': {e}",
                )

            reject_legacy_rules_paths(raw_cfg)
            # Dictionary mapping workspace-relative posix path -> source absolute Path
            files_to_pack: Dict[str, Path] = {}

            # Add primary experiment config
            exp_rel = self._config_key(exp_yaml_path)
            files_to_pack[exp_rel] = exp_yaml_path

            # 2. Trace base configs and method templates in the experiment group
            self._trace_group_templates(group, raw_cfg, files_to_pack)

            # 3. Environment configs, wrappers, and valuation rules
            env_name = self._resolve_env_name(raw_cfg, group)
            environment = raw_cfg.get("env")
            if isinstance(environment, dict):
                environment = {"name": env_name, **environment}
            else:
                environment = environment or env_name
            if opts.include_environment and environment:
                self._trace_environment_assets(environment, files_to_pack)

            # 4. Canonical domain rules, including ruleset-owned predicates.
            if opts.include_rules:
                if environment:
                    self._trace_rules(environment, files_to_pack)

            # 5. Model architectures and agent configs
            agents_found: Set[str] = set()
            models_found: Set[str] = set()
            self._trace_methods_and_models(
                raw_cfg,
                files_to_pack,
                agents_found,
                models_found,
                include_models=opts.include_models,
            )

            # 6. Evaluators
            if opts.include_evaluators:
                self._trace_evaluators(group, env_name, files_to_pack, raw_cfg=raw_cfg)

            # 7. Plotters
            if opts.include_plotters:
                self._trace_plotters(group, env_name, files_to_pack, raw_cfg=raw_cfg)

            # 8. Data modules
            if opts.include_data_modules:
                self._trace_data_modules(group, env_name, files_to_pack, raw_cfg=raw_cfg)

            # 9. Workflows
            self._trace_workflows(group, files_to_pack, raw_cfg=raw_cfg)

            # 10. Checkpoints (optional)
            if opts.include_checkpoints:
                self._trace_checkpoints(group, exp_name, files_to_pack)

            # Determine output archive path
            output_path = opts.output_path
            if output_path is None:
                exports_dir = self.workspace_root / "results" / "exports"
                exports_dir.mkdir(parents=True, exist_ok=True)
                clean_exp = exp_name.replace("/", "_")
                output_path = exports_dir / f"{group}_{clean_exp}_v{opts.version}.zip"
            else:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)

            # Managed implementations keep separate dependency ownership in an export.
            self._package_dependencies = self._managed_dependencies(files_to_pack)
            # 7. Generate metadata manifest (.theta_component.json)
            manifest = self._generate_manifest(
                group=group,
                exp_name=exp_name,
                env_name=env_name,
                agents=sorted(agents_found),
                models=sorted(models_found),
                files=sorted(files_to_pack.keys()),
                options=opts,
            )

            # Validate manifest against formal JSON schema
            from src.app.core.manifest_schema import validate_manifest_data

            is_valid, schema_err = validate_manifest_data(manifest)
            if not is_valid:
                return ExportResult(
                    success=False,
                    error=f"Export manifest failed schema validation: {schema_err}",
                )

            self._validate_output(manifest, files_to_pack)

            # 8. Generate README.md
            readme_content = self._generate_readme(group, exp_name, env_name, opts)

            # 9. Write archive
            with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                # Store manifest at root of zip
                zf.writestr(".theta_component.json", json.dumps(manifest, indent=2) + "\n")
                zf.writestr("README.md", readme_content + "\n")

                # Store all packaged assets with their relative workspace path
                for rel_path, abs_path in sorted(files_to_pack.items()):
                    if abs_path.is_file():
                        zf.write(abs_path, arcname=('configs/'+rel_path.removeprefix('in/config/') if rel_path.startswith('in/config/') else rel_path))

            file_list = sorted(list(files_to_pack.keys()) + [".theta_component.json", "README.md"])
            log.info("Exported experiment '%s/%s' (%d files) to %s", group, exp_name, len(file_list), output_path)

            return ExportResult(
                success=True,
                archive_path=output_path,
                manifest=manifest,
                files_included=file_list,
            )

        except Exception as exc:
            log.exception("Error exporting experiment %s", experiment_id_or_path)
            return ExportResult(
                success=False,
                error=str(exc),
            )

    def _resolve_experiment_path(self, target: str | Path) -> tuple[str, str, Path]:
        """Resolve group, experiment name, and file path from target."""
        value = str(target).strip()
        path = Path(value)
        if path.is_absolute():
            matches = [key for key in self.paths.config_resolver().selectors("experiment")
                       if self.paths.resolve_config(key) == path.resolve()]
            if len(matches) != 1:
                raise ValueError(f"Recipe is not an effective workspace profile: {path}")
            value = matches[0]
        value = value.removeprefix("src/usr/config/").removeprefix("in/config/").removeprefix("experiment/")
        relative = Path(value).with_suffix(".yaml")
        source = self.paths.resolve_config(Path("experiment") / relative)
        group = relative.parts[0] if len(relative.parts) > 1 else "general"
        return group, relative.stem, source

    def _config_key(self, path):
        for selector in self.paths.config_resolver().selectors():
            if self.paths.resolve_config(selector) == path:
                return "in/config/" + selector
        return path.relative_to(self.workspace_root).as_posix()

    def _trace_group_templates(self, group, cfg, files_to_pack):
        prefix = f"experiment/{group}/"
        defaults = {item for item in cfg.get("defaults", []) if isinstance(item, str)}
        for selector in self.paths.config_resolver().selectors("experiment"):
            if not selector.startswith(prefix):
                continue
            tail = selector.removeprefix(prefix)
            if tail == "_base.yaml" or tail.startswith("methods/") or tail.removesuffix(".yaml") in defaults:
                files_to_pack["in/config/" + selector] = self.paths.resolve_config(selector)

    def _resolve_env_name(self, cfg: Dict[str, Any], group: str) -> str | None:
        """Extract environment name from configuration or defaults."""
        if "env" in cfg and isinstance(cfg["env"], str):
            return cfg["env"]

        # Check defaults list for an environment selection.
        defaults = cfg.get("defaults", [])
        if isinstance(defaults, list):
            for item in defaults:
                if isinstance(item, dict):
                    if "override /env" in item:
                        return str(item["override /env"])
                    if "/env" in item:
                        return str(item["/env"])
                    if "env" in item:
                        return str(item["env"])

        # Check group _base.yaml if present
        base_yaml = self.paths.resolve_config(f"experiment/{group}/_base.yaml")
        if base_yaml.is_file():
            try:
                base_cfg = yaml.safe_load(base_yaml.read_text(encoding="utf-8")) or {}
                if "env" in base_cfg and isinstance(base_cfg["env"], str):
                    return base_cfg["env"]
                b_defaults = base_cfg.get("defaults", [])
                if isinstance(b_defaults, list):
                    for item in b_defaults:
                        if isinstance(item, dict):
                            if "override /env" in item:
                                return str(item["override /env"])
                            if "/env" in item:
                                return str(item["/env"])
            except Exception:
                pass

        return None

    def _find_config_file(self, folder: str, name: str) -> Path | None:
        """Find a configuration file in in/config/<folder>/ directly or recursively."""
        for suffix in (".yaml", ".yml"):
            path = self.paths.resolve_config(f"{folder}/{name}{suffix}")
            if path.is_file():
                return path
        return None

    def _trace_declared_provider(self, kind, selector, files_to_pack):
        from src.app.core.provider_catalog import get_catalog
        catalog = get_catalog(self.paths)
        selected = catalog.resolve(kind, str(selector))
        if selected is None and kind == 'datamodule':
            from src.app.core.data_loaders import loader_selection
            binding, _ = loader_selection({'symbolic_state_loader': selector}, 'symbolic_state_loader', paths=catalog.paths)
            selected = catalog.resolve(kind, binding['name']) if binding else None
        if selected is None: raise ValueError(f"No declared {kind} provider for {selector!r}; export/install its native package first")
        package, cap = selected
        if package.origin == 'python_distribution': return
        # Managed providers retain independent ownership and become pinned dependencies.
        for path in package.root.rglob('*'):
            if path.is_file() and not any(v in path.parts for v in ('__pycache__','.git','node_modules')):
                if path.name == '.theta_component.json': continue
                try: relative = path.relative_to(self.workspace_root).as_posix()
                except ValueError: continue
                files_to_pack[relative] = path
        for dest in package.manifest.get('payload',{}).get('configs',{}):
            path = self.workspace_root / dest
            if path.is_file(): files_to_pack[dest] = path

    def _trace_environment_assets(self, env_name, files_to_pack):
        from collections.abc import Mapping
        selector = env_name.get('name') if isinstance(env_name,Mapping) else env_name
        if selector: self._trace_declared_provider('env',selector,files_to_pack)

    def _trace_rules(self, env_name: str, files_to_pack: Dict[str, Path]) -> None:
        """Trace domain logic rules for the environment."""
        from src.app.services.rulesets import available_rulesets
        for resource in available_rulesets(self.workspace_root, env_name):
            for p in sorted(resource.directory.rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts:
                    rel = f"resources/rules/{resource.environment}/{resource.name}/{p.relative_to(resource.directory).as_posix()}"
                    files_to_pack[rel] = p

    def _trace_evaluators(self, group, env_name, files_to_pack, raw_cfg=None):
        for key in ('evaluator','evaluators'):
            for name in _component_names((raw_cfg or {}).get(key)):
                self._trace_declared_provider('evaluator',name,files_to_pack)

    def _trace_plotters(self, group, env_name, files_to_pack, raw_cfg=None):
        for name in _plot_names((raw_cfg or {}).get('plots')):
            self._trace_declared_provider('plotter',name,files_to_pack)

    def _trace_data_modules(self, group, env_name, files_to_pack, raw_cfg=None):
        for key in ('data_module','symbolic_state_loader','datamodule','data'):
            for name in _component_names((raw_cfg or {}).get(key)):
                self._trace_declared_provider('datamodule',name,files_to_pack)
        for method in ((raw_cfg or {}).get('methods') or {}).values():
            if isinstance(method, dict):
                self._trace_data_modules(group, env_name, files_to_pack, raw_cfg=method)

    def _trace_workflows(
        self,
        group: str,
        files_to_pack: Dict[str, Path],
        raw_cfg: Dict[str, Any] | None = None,
    ) -> None:
        """Trace any associated DAG workflows in in/config/workflow/."""
        names = {str(raw_cfg["workflow"])} if raw_cfg and raw_cfg.get("workflow") else set()
        for selector in self.paths.config_resolver().selectors("workflow"):
            if group in Path(selector).stem or Path(selector).stem in names:
                files_to_pack["in/config/" + selector] = self.paths.resolve_config(selector)

    def _trace_all_group_experiments(self, group, files_to_pack):
        for selector in self.paths.config_resolver().selectors("experiment"):
            if selector.startswith(f"experiment/{group}/"):
                files_to_pack["in/config/" + selector] = self.paths.resolve_config(selector)

    def export_group(
        self,
        group: str,
        options: ExportOptions | None = None,
    ) -> ExportResult:
        """Export an entire experiment group and its full component ecosystem as an installable Experiment Plugin."""
        opts = options or ExportOptions()
        files_to_pack: Dict[str, Path] = {}
        self._trace_all_group_experiments(group, files_to_pack)
        if not files_to_pack:
            return ExportResult(success=False, error=f"Experiment group not found: {group}")

        # 2. Extract configs to resolve env, agents, models, paradigms
        agents_found: Set[str] = set()
        models_found: Set[str] = set()
        paradigms_found: Set[str] = set()
        workflows_found: Set[str] = set()
        data_modules_found: Set[str] = set()
        evaluators_found: Set[str] = set()
        plotters_found: Set[str] = set()
        env_name: str | None = None

        for exp_yaml in list(files_to_pack.values()):
            cfg = yaml.safe_load(exp_yaml.read_text(encoding="utf-8")) or {}
            if not env_name:
                env_name = self._resolve_env_name(cfg, group)
            if "paradigm" in cfg and cfg["paradigm"]:
                paradigms_found.add(str(cfg["paradigm"]))
            if "workflow" in cfg and cfg["workflow"]:
                workflows_found.add(str(cfg["workflow"]))
            for values in (cfg, *((cfg.get("methods") or {}).values())):
                if isinstance(values, dict):
                    for key in ("data_module", "symbolic_state_loader"):
                        data_modules_found.update(_component_names(values.get(key)))
            if "evaluator" in cfg and cfg["evaluator"]:
                evaluators_found.update(_component_names(cfg["evaluator"]))
            plotters_found.update(_plot_names(cfg.get("plots")))
            self._trace_methods_and_models(
                cfg, files_to_pack, agents_found, models_found, include_models=opts.include_models
            )



        # 3. Environment, rules, evaluators, plotters, data loaders, workflows
        if opts.include_environment and env_name:
            self._trace_environment_assets(env_name, files_to_pack)

        if opts.include_rules and env_name:
            self._trace_rules(env_name, files_to_pack)

        if opts.include_evaluators:
            self._trace_evaluators(group, env_name, files_to_pack, raw_cfg={"evaluators": list(evaluators_found)})

        if opts.include_plotters:
            self._trace_plotters(
                group, env_name, files_to_pack, raw_cfg={"plots": {name: {} for name in plotters_found}}
            )

        if opts.include_data_modules:
            self._trace_data_modules(group, env_name, files_to_pack, raw_cfg={"data": list(data_modules_found)})

        self._trace_workflows(
            group,
            files_to_pack,
            raw_cfg={"workflow": list(workflows_found)[0] if workflows_found else None},
        )

        # 4. Output archive path
        output_path = opts.output_path
        if output_path is None:
            exports_dir = self.workspace_root / "results" / "exports"
            exports_dir.mkdir(parents=True, exist_ok=True)
            output_path = exports_dir / f"{group}_experiment_v{opts.version}.zip"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        self._package_dependencies = self._managed_dependencies(files_to_pack)
        # 5. Manifest
        manifest = self._generate_manifest(
            group=group,
            exp_name="all",
            env_name=env_name,
            agents=sorted(agents_found),
            models=sorted(models_found),
            files=sorted(files_to_pack.keys()),
            options=opts,
            paradigms=sorted(paradigms_found) if paradigms_found else ["offline_rl"],
        )

        from src.app.core.manifest_schema import validate_manifest_data

        is_valid, schema_err = validate_manifest_data(manifest)
        if not is_valid:
            return ExportResult(
                success=False,
                error=f"Export manifest failed schema validation: {schema_err}",
            )

        self._validate_output(manifest, files_to_pack)
        readme_content = self._generate_readme(group, "all", env_name, opts)

        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(".theta_component.json", json.dumps(manifest, indent=2) + "\n")
            zf.writestr("README.md", readme_content + "\n")
            for rel_path, abs_path in sorted(files_to_pack.items()):
                if abs_path.is_file():
                    zf.write(abs_path, arcname=('configs/'+rel_path.removeprefix('in/config/') if rel_path.startswith('in/config/') else rel_path))

        file_list = sorted(list(files_to_pack.keys()) + [".theta_component.json", "README.md"])
        log.info("Exported experiment group '%s' (%d files) to %s", group, len(file_list), output_path)

        return ExportResult(
            success=True,
            archive_path=output_path,
            manifest=manifest,
            files_included=file_list,
        )

    def _trace_methods_and_models(
        self,
        cfg: Dict[str, Any],
        files_to_pack: Dict[str, Path],
        agents_found: Set[str],
        models_found: Set[str],
        include_models: bool = True,
    ) -> None:
        """Extract declared agents and models, tracing their configs and implementation code."""
        methods = cfg.get("methods", {})
        if not isinstance(methods, dict):
            return

        for m_name, m_cfg in methods.items():
            if not isinstance(m_cfg, dict):
                continue

            agent_name = m_cfg.get("agent")
            if agent_name and isinstance(agent_name, str):
                agents_found.add(agent_name)
                self._trace_declared_provider('algorithm',agent_name,files_to_pack)
                # in/config/agent/<agent>.yaml (direct or recursive)
                agent_yaml = self._find_config_file("agent", agent_name)
                if agent_yaml and agent_yaml.is_file():
                    rel = self._config_key(agent_yaml)
                    files_to_pack[rel] = agent_yaml

            # Extract model names (handle flat or nested composite specs)
            extracted_models: List[str] = []

            def _collect_models(val: Any):
                if isinstance(val, str):
                    clean = val.strip()
                    if clean and clean.lower() not in (
                        "default",
                        "none",
                        "true",
                        "false",
                        "identity",
                        "neural",
                        "learned",
                        "heuristic",
                        "linear",
                        "softmax",
                    ):
                        extracted_models.append(clean)
                elif isinstance(val, dict):
                    arch = val.get("architecture") or val.get("name") or val.get("type") or val.get("base")
                    if arch and isinstance(arch, str):
                        extracted_models.append(arch)
                    elif len(val) == 1:
                        component_name, component_cfg = next(iter(val.items()))
                        if isinstance(component_cfg, dict):
                            extracted_models.append(str(component_name))
                    for k, sub in val.items():
                        if k in ("architecture", "name", "type", "base", "slots", "rules", "mode", "lr"):
                            continue
                        if isinstance(sub, dict):
                            if len(sub) == 1 and not any(
                                ik in sub for ik in ("lr", "hidden_sizes", "activation", "dim")
                            ):
                                first_k = next(iter(sub))
                                if isinstance(sub[first_k], dict):
                                    extracted_models.append(first_k)
                                    _collect_models(sub[first_k])
                                    continue
                            _collect_models(sub)
                        elif isinstance(sub, str):
                            _collect_models(sub)

            _collect_models(m_cfg.get("model"))
            for m in sorted(set(extracted_models)):
                models_found.add(m)
                # in/config/model/<m>.yaml (direct or recursive)
                model_yaml = self._find_config_file("model", m)
                if model_yaml and model_yaml.is_file():
                    rel = self._config_key(model_yaml)
                    files_to_pack[rel] = model_yaml

                if include_models:
                    self._trace_declared_provider('model',m,files_to_pack)

    def _trace_checkpoints(
        self,
        group: str,
        exp_name: str,
        files_to_pack: Dict[str, Path],
    ) -> None:
        """Find and package saved checkpoints if available."""
        ckpt_dir = self.workspace_root / "results" / "checkpoints" / group / exp_name
        if ckpt_dir.is_dir():
            for p in sorted(ckpt_dir.rglob("*")):
                if p.is_file() and p.suffix in (".ckpt", ".pt", ".pth", ".json"):
                    rel = p.relative_to(self.workspace_root).as_posix()
                    files_to_pack[rel] = p

    def _validate_output(self, manifest, files):
        import tempfile, shutil
        from src.app.components.archive import validate_native_package
        with tempfile.TemporaryDirectory(prefix='theta-experiment-export-') as temporary:
            root=Path(temporary)
            for relative, source in files.items():
                relative='configs/'+relative.removeprefix('in/config/') if relative.startswith('in/config/') else relative
                dest=root/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
            validate_native_package(manifest,root)

    def _managed_dependencies(self, files):
        from src.app.core.provider_catalog import get_catalog
        from src.app.workspace import WorkspacePaths
        catalog = get_catalog(self.paths)
        dependencies = {}
        for relative, source in list(files.items()):
            owners = [p for p in catalog.packages if p.origin in ('bundled', 'hub', 'local_archive', 'user') and Path(source).resolve().is_relative_to(p.root)]
            if not owners: continue
            owner = max(owners, key=lambda p: len(p.root.parts))
            dependencies[owner.id] = {'id': owner.id, 'version': '==' + owner.version}
            if not relative.startswith('in/config/'):
                files.pop(relative)
        # Optional checkpoints are reproducibility resources, never runtime results to deploy.
        for relative in list(files):
            if relative.startswith('results/'):
                files['resources/' + relative] = files.pop(relative)
        return list(dependencies.values())

    def _generate_manifest(
        self,
        group: str,
        exp_name: str,
        env_name: str | None,
        agents: List[str],
        models: List[str],
        files: List[str],
        options: ExportOptions,
        paradigms: List[str] | None = None,
    ) -> Dict[str, Any]:
        """Generate component manifest matching Theta Hub specifications."""
        is_group_bundle = exp_name in ("all", group)
        component_id = group if is_group_bundle else f"{group}_{exp_name}".replace("/", "_")
        config_paths = sorted([k for k in files if k.startswith("in/config/")])
        manifest = {
            "id": component_id,
            "name": f"{group.replace('_', ' ').title()} Experiment Suite" if is_group_bundle else f"{group}/{exp_name}",
            "kind": "experiment",
            "version": options.version,
            "description": options.description
            or f"Complete {group} experiment suite: recipes, group defaults, environment hooks, evaluators, and plotters.",
            "author": {"name": options.author},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config_paths": config_paths,
            "compatibility": {"theta": ">=2.0.0"},
            "environment": env_name,
            "agents": agents,
            "models": models,
            "package_contents": files,
        }
        if not is_group_bundle:
            manifest["entry_point"] = f"in/config/experiment/{group}/{exp_name}.yaml"
        if paradigms:
            manifest["paradigms"] = paradigms
        manifest.update(storage_version=2, api_version=1,
            payload={'implementations':[f for f in files if f.endswith('.py')],
                     'configs':{f:'configs/'+f.removeprefix('in/config/') for f in config_paths},
                     'resources':[f for f in files if f not in config_paths and not f.endswith('.py')]},
            dependencies={'components':getattr(self, '_package_dependencies', [])})
        manifest['capabilities'] = []
        if any(f.startswith('resources/rules/') for f in files): manifest['rules_root'] = 'resources/rules'
        profile_schemas = {}
        for relative in files:
            if not relative.endswith(('.theta_component.json','plugin.json')): continue
            source = self.workspace_root / relative
            if not source.is_file(): continue
            metadata = json.loads(source.read_text())
            schema = metadata.get('parameter_schema')
            if not isinstance(schema, dict): continue
            declared = [*metadata.get('config_paths', []), *([metadata['config_path']] if metadata.get('config_path') else [])]
            for profile in declared:
                if profile in config_paths:
                    profile_schemas[profile] = schema.get('x-theta-config-schemas', {}).get(profile, schema)
        if profile_schemas:
            manifest['parameter_schema'] = {'type':'object','properties':{},'x-theta-config-schemas':profile_schemas}
        return manifest

    def _generate_readme(
        self,
        group: str,
        exp_name: str,
        env_name: str | None,
        options: ExportOptions,
    ) -> str:
        """Generate descriptive reproduction README."""
        lines = [
            f"# Experiment Package: {group}/{exp_name}",
            "",
            f"**Version**: {options.version}  ",
            f"**Author**: {options.author}  ",
            f"**Environment**: {env_name or 'unspecified'}  ",
            "",
            "## Overview",
            options.description or f"This is a self-contained Theta-IDE experiment package for `{group}/{exp_name}`.",
            "",
            "## Installation",
            "You can install this component directly into your Theta-IDE workspace:",
            "1. **Via Theta Hub GUI**: Open Theta-IDE -> Components -> Hub, and import this package archive.",
            "2. **CLI**: `python -m src.app.components --root /path/to/workspace install package.zip`.",
            "Managed provider dependencies retain separate ownership; the package never writes implementation code into the workspace source tree.",
            "",
            "## Running the Experiment",
            "Execute using the unified Theta pipeline:",
            "```bash",
            f"python run_pipeline.py {group}/{exp_name}",
            "```",
            "",
            "To view the experiment parameters in the Theta-IDE GUI, launch:",
            "```bash",
            "python -m frontend",
            "```",
        ]
        return "\n".join(lines)
