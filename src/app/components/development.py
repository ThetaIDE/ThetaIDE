"""Inspection and development operations without Qt or importing component code."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from src.app.workspace import WorkspacePaths

from .configuration import ConfigurationFiles
from .installer import HubInstaller


class ComponentDevelopment:
    def __init__(self, workspace=None):
        self.paths = workspace if isinstance(workspace, WorkspacePaths) else WorkspacePaths(workspace) if workspace else WorkspacePaths()
        self.installer = HubInstaller(self.paths.root, self.paths.data_dir)
        self.configs = ConfigurationFiles(self.paths)

    def _records(self):
        records = []
        for manifest in self.paths.manifests():
            data = json.loads(manifest.read_text(encoding="utf-8"))
            if not data.get("id"):
                raise ValueError(f"Component manifest has no identity: {manifest}")
            records.append((manifest, data))
        return records

    def _resolve(self, reference):
        reference = str(reference)
        candidates = []
        for manifest, data in self._records():
            if reference == data["id"]:
                candidates.append((manifest, data))
        if not candidates:
            path = Path(reference).expanduser()
            if not path.is_absolute():
                path = self.paths.root / path
            path = path.resolve()
            for manifest, data in self._records():
                configs = [*data.get("config_paths", []), *data.get("user_config_paths", [])]
                if data.get("config_path"):
                    configs.append(data["config_path"])
                files = [*configs, *data.get("installed_files", [])]
                if (path == manifest.parent or manifest.parent in path.parents
                        or path in {(self.paths.root / p).resolve() for p in files}):
                    candidates.append((manifest, data))
        if len(candidates) != 1:
            detail = ", ".join(str(manifest) for manifest, _ in candidates)
            raise ValueError(f"{'Ambiguous' if candidates else 'Unknown'} component {reference!r}" + (f": {detail}" if detail else ""))
        return candidates[0]

    def list(self):
        ids = sorted({data["id"] for _, data in self._records()})
        return [self.inspect(component_id) for component_id in ids]

    def inspect(self, reference):
        manifest, meta = self._resolve(reference)
        configs = list(dict.fromkeys([*meta.get("config_paths", []), *([meta["config_path"]] if meta.get("config_path") else [])]))
        profiles = []
        for value in configs:
            path = self.paths.config_file(value)
            profiles.append({"path": str(path), "selector": path.relative_to(self.paths.config).with_suffix("").as_posix(),
                             "exists": path.is_file(), "read_only": self.installer.is_installed_file(path)})
        registrations = []
        digest = hashlib.sha256()
        for source in sorted(manifest.parent.rglob("*.py")):
            if "__pycache__" in source.parts:
                continue
            raw = source.read_bytes()
            digest.update(source.relative_to(manifest.parent).as_posix().encode() + b"\0" + raw + b"\0")
            tree = ast.parse(raw, filename=str(source))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id.startswith("register_"):
                    registrations.append({"file": str(source), "line": node.lineno, "call": ast.unparse(node),
                                          "function": node.func.id,
                                          "names": [arg.value for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)]})
        return {"id": meta["id"], "name": meta.get("name", meta["id"]), "kind": meta.get("kind"),
                "version": meta.get("version"), "source": str(manifest.parent), "manifest": str(manifest),
                "origin": "hub" if meta.get("installed_from") and meta.get("origin") != "local" else "local",
                "read_only": self.installer.is_installed_file(manifest), "profiles": profiles,
                "registrations": registrations, "source_digest": digest.hexdigest(),
                "dependencies": meta.get("dependencies", {}), "parent": meta.get("parent"),
                "copy_supported": meta.get("kind") == "model"}

    def copy(self, reference, new_id, name=None):
        from src.app.services.local_components import LocalComponents

        manifest, meta = self._resolve(reference)
        info = self.inspect(reference)
        if not info["copy_supported"]:
            raise ValueError(f"Independent copying currently supports directory-based models; {meta.get('kind')} is unsupported. Source: {manifest.parent}")
        profiles = info["profiles"]
        primary = next((p["path"] for p in profiles if p["exists"]), manifest)
        copied = LocalComponents(self.paths.root).deep_copy(primary, new_id, name)
        return copied

    def _validate_schema(self, meta, path, data):
        schema = meta.get("parameter_schema") or {}
        relative = path.relative_to(self.paths.root).as_posix()
        selected = (schema.get("x-theta-config-schemas") or {}).get(relative, schema)
        if selected.get("x-theta-validate", schema.get("x-theta-validate")) is True:
            from jsonschema import Draft202012Validator
            Draft202012Validator.check_schema(selected)
            Draft202012Validator(selected).validate(data)

    def validate_profile(self, path, *, effective_data=None):
        """Validate YAML/schema and report the exact source for a model selector.

        This is static preflight. It deliberately does not import or execute a
        developer's Python implementation to inspect it.
        """
        path, _, data = self.configs.read(path)
        if effective_data is not None:
            data = effective_data
        result = {"path": str(path), "validation": "static", "components": []}
        if path.relative_to(self.paths.config).parts[0] == "model":
            selector = data.get("architecture") or data.get("name") or data.get("type")
            if not isinstance(selector, str):
                raise ValueError(f"Model profile needs an architecture, name, or type selector: {path}")
            candidates = []
            for manifest, meta in self._records():
                if meta.get("kind") == "model":
                    info = self.inspect(manifest)
                    if selector == info["id"] or any(selector in r["names"] for r in info["registrations"] if r["function"] == "register_model"):
                        candidates.append((info, meta))
            if len(candidates) != 1:
                raise ValueError(f"Model selector {selector!r} must identify exactly one implementation; found {len(candidates)}.")
            info, meta = candidates[0]
            if not any(selector in r["names"] for r in info["registrations"] if r["function"] == "register_model"):
                raise ValueError(f"No literal model registration for {selector!r} in {info['source']}.")
            self._validate_schema(meta, path, data)
            result["components"].append({**info, "selector": selector})
        else:
            owner = self.installer.find_component_for_path(path)
            if owner:
                self._validate_schema(owner, path, data)
                result["components"].append(self.inspect(owner["id"]))
        return result

    def validate_experiment(self, experiment, overrides=()):
        from src.app.pipeline.compose import compose_experiment, effective_config, method_plans, validate_composed

        composed = compose_experiment(experiment, list(overrides), workspace=self.paths)
        notices = validate_composed(composed)
        resolved = []
        for name, plan in method_plans(composed).items():
            # Inspect the same method-level overrides training receives, so an
            # explicit runtime selector override cannot be misreported as the
            # original profile's implementation.
            launch_overrides = [arg for arg in plan["train_overrides"] if not arg.startswith("+experiment=")]
            actual = compose_experiment(composed.experiment, launch_overrides, workspace=self.paths)
            effective = effective_config(actual.cfg)
            choices = actual.cfg.get("hydra", {}).get("runtime", {}).get("choices", {})
            for kind in ("model", "agent", "env", "data", "eval", "plot"):
                selected = choices.get(kind) or plan["settings"].get(kind)
                if isinstance(selected, dict):
                    selected = selected.get("base") or selected.get("name") or selected.get("architecture")
                if not isinstance(selected, str) or not selected:
                    continue
                profile = self.paths.config_file(f"{kind}/{selected}")
                if not profile.is_file():
                    continue
                item = {"method": name, "kind": kind, "profile": str(profile)}
                if kind == "model":
                    item.update(self.validate_profile(profile, effective_data=effective.get("model")))
                    item["effective_parameters"] = effective.get("model")
                else:
                    owner = self.installer.find_component_for_path(profile)
                    item["components"] = [self.inspect(owner["id"])] if owner else []
                resolved.append(item)
        return {"experiment": composed.experiment, "validation": "configuration preflight; no training executed",
                "notices": notices, "resolved_components": resolved,
                "config": effective_config(composed.cfg)}
