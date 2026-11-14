"""Native package validation and staging."""
from __future__ import annotations

import json
import shutil
import stat
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from src.app.services.rulesets import reject_legacy_rules_paths


def safe_relative(value):
    value = str(value)
    path = PurePosixPath(value)
    if not value or "\\" in value or ":" in value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe package path: {value!r}")
    return path


def extract(archive, destination):
    with ZipFile(archive) as zf:
        for item in zf.infolist():
            rel = safe_relative(item.filename)
            if stat.S_ISLNK(item.external_attr >> 16): raise ValueError(f"Archive symlinks are unsupported: {rel}")
            if not (destination / str(rel)).resolve().is_relative_to(destination.resolve()): raise ValueError(str(rel))
        zf.extractall(destination)
    items = list(destination.iterdir())
    return items[0] if len(items) == 1 and items[0].is_dir() and items[0].name not in ("src", "in", "config", "plot") else destination


def validate_native_package(meta, source):
    """Validate declarations and containment without importing provider code."""
    from src.app.core.manifest_schema import validate_manifest_data
    if meta.get("storage_version") != 2 or meta.get("api_version") != 1:
        raise ValueError("Unsupported or missing native contract: packages require explicit storage_version: 2 and api_version: 1; rebuild the maintained source package")
    valid, error = validate_manifest_data({k: v for k, v in meta.items() if v is not None})
    if not valid: raise ValueError(f"Invalid package manifest: {error}")
    for field in ("legacy_paths", "module_aliases", "legacy_archive", "target_path"):
        if field in meta: raise ValueError(f"Native packages cannot declare {field}; use package-relative entry points/resources")
    if not isinstance(meta.get("capabilities"), list): raise ValueError("Declare capabilities explicitly (an empty list is valid for a config/dependency package)")
    payload = meta.get("payload")
    if not isinstance(payload, dict) or not all(k in payload for k in ("implementations", "resources", "configs")):
        raise ValueError("Declare payload.implementations, payload.resources and payload.configs explicitly")
    reject_legacy_rules_paths(meta)
    source = Path(source).resolve()
    def resource(value):
        rel = safe_relative(value)
        path = (source / str(rel)).resolve()
        if not path.is_relative_to(source) or not path.exists(): raise ValueError(f"Missing or escaping payload: {value}")
        return path
    declared = []
    for section in ("implementations", "resources"):
        if not isinstance(payload[section], list): raise ValueError(f"payload.{section} must be a list")
        declared.extend(resource(v) for v in payload[section])
    if not isinstance(payload["configs"], dict): raise ValueError("payload.configs must be an object")
    for dest, rel in payload["configs"].items():
        safe_relative(dest)
        if not dest.startswith("in/config/") or Path(dest).suffix not in (".yaml", ".yml"):
            raise ValueError(f"Invalid materialized config destination {dest}")
        file = resource(rel)
        if not file.is_file(): raise ValueError(f"Missing config payload {rel}")
        import yaml
        if not isinstance(yaml.safe_load(file.read_text()) or {},dict):
            raise ValueError(f"Config profile must contain a mapping: {rel}")
    if set(meta.get("config_paths", payload["configs"])) != set(payload["configs"]):
        raise ValueError("config_paths must agree with payload.configs")
    for name, target in (meta.get("exports") or {}).items():
        # Public cross-package objects are ordinary payload files; nothing is imported here.
        if not resource(target.partition(":")[0]).is_file():
            raise ValueError(f"Export {name!r} must name a payload Python file")
    selectors = set()
    for cap in meta["capabilities"]:
        if cap.get("kind") == "monitor":
            required = cap.get("requires")
            if not isinstance(required, list) or not required or not all(isinstance(v, str) and v.strip() for v in required):
                raise ValueError("Monitor capabilities require a nonempty requires hook list")
        if "monitoring_access" in cap and (not isinstance(cap["monitoring_access"], list) or
                not all(isinstance(v, str) and v.strip() for v in cap["monitoring_access"])):
            raise ValueError("monitoring_access must be a list of model capabilities")
        if "monitor_defaults" in cap:
            from src.app.monitoring.config import instances
            instances({}, defaults=cap["monitor_defaults"])
        if "monitoring" in cap:
            from src.app.telemetry.monitors import validate_monitoring_metadata
            validate_monitoring_metadata(cap["monitoring"])
            if catalog := cap["monitoring"].get("catalog"):
                catalog_file = resource(catalog.split(":", 1)[0])
                if not any(catalog_file == root or catalog_file.is_relative_to(root) for root in declared):
                    raise ValueError("Monitor catalog is outside declared payload")
                from src.app.telemetry.monitors import validate_monitor_catalog_source
                validate_monitor_catalog_source(catalog_file)
        entry, sep, attr = cap["entry_point"].partition(":")
        file = resource(entry)
        if not file.is_file() or file.suffix not in (".py", ".json"):
            raise ValueError(f"Invalid package-relative entry point: {entry}")
        if file.suffix == ".py" and not sep: raise ValueError(f"Executable entry point needs an explicit object: {entry}")
        if not any(file == root or file.is_relative_to(root) for root in declared):
            raise ValueError(f"Entry point is outside declared payload: {entry}")
        for hook_name in ("execution_preflight", "resource_hint_hook"):
            if hook_name not in cap:
                continue
            hook_entry, _, hook_attr = cap[hook_name].partition(":")
            hook_file = resource(hook_entry)
            if not hook_file.is_file() or hook_file.suffix != ".py" or not hook_attr:
                raise ValueError(f"Invalid {hook_name}: {cap[hook_name]}")
            if not any(hook_file == root or hook_file.is_relative_to(root) for root in declared):
                raise ValueError(f"{hook_name} is outside declared payload: {hook_entry}")
        contribution_ids = set()
        if ('contributions' in cap or 'readiness' in cap) and 'contribution_api' not in cap:
            raise ValueError('Contributions/readiness require an explicit contribution_api')
        for contribution in cap.get('contributions', []):
            if contribution['id'] in contribution_ids:
                raise ValueError('Duplicate contribution ID in capability')
            contribution_ids.add(contribution['id'])
            if reference := contribution.get('entry_point'):
                factory = resource(reference.split(':', 1)[0])
                if not factory.is_file() or factory.suffix != '.py':
                    raise ValueError('Contribution factories must name Python files')
                if not any(factory == root or factory.is_relative_to(root) for root in declared):
                    raise ValueError('Contribution factory is outside declared payload')
        from src.app.core.provider_catalog import kind_name
        if kind_name(cap['kind']) not in {'scheduler','site','monitor','model','algorithm','env','datamodule','evaluator','plotter','tuner','objective','component','plugin','run_logger'}:
            raise ValueError(f"Unknown capability kind: {cap['kind']}")
        if file.suffix == '.json' and cap['kind'] != 'objective':
            raise ValueError('Only objective capabilities use JSON entry points')
        for selector in (cap["id"], *cap.get("aliases", [])):
            key = (kind_name(cap["kind"]), selector.casefold())
            if key in selectors: raise ValueError(f"Duplicate capability selector: {key}")
            selectors.add(key)
    for path in source.rglob("*"):
        if any(part in ("node_modules", "__pycache__", ".git") for part in path.relative_to(source).parts): continue
        if path.is_symlink(): raise ValueError(f"Package contains symlink: {path}")
        if not path.is_file(): continue
        relative = path.relative_to(source).as_posix()
        reject_legacy_rules_paths(relative)
        if relative.startswith(('src/app/','frontend/','results/')) or (path.is_file() and relative.startswith('in/') and not relative.startswith('in/config/')):
            raise ValueError(f"Protected workspace payload: {relative}; use package-owned resources")
    for key in ('rules_root',):
        if key in meta and not resource(meta[key]).is_dir(): raise ValueError(f'{key} must name a package directory')
    for cap in meta['capabilities']:
        if 'resource_root' in cap and not resource(cap['resource_root']).is_dir():
            raise ValueError('resource_root must name a package directory')
    aliases = meta.get("compatibility_selectors", [])
    if not isinstance(aliases, list) or any(not isinstance(key, str) or "in/config/" + key not in payload["configs"] for key in aliases):
        raise ValueError("compatibility_selectors must name explicitly packaged config profiles")
    return dict(payload["configs"])


def adapt(source, component, staged, *, version=None):
    """Stage a native archive; no layout inference or historical aliases."""
    packaged = source / ".theta_component.json"
    if not packaged.is_file(): raise ValueError("Archive requires a root .theta_component.json native manifest")
    meta = json.loads(packaged.read_text())
    if meta.get("id") != component.id or meta.get("kind") != component.kind or meta.get("version") != (version or component.version):
        raise ValueError("Archive identity/kind/version differs from requested component metadata")
    configs = validate_native_package(meta, source)
    if component.kind != "plugin":
        for path in source.rglob("plugin.json"):
            if json.loads(path.read_text()).get("kind", "plugin") == "plugin":
                raise ValueError("Mixed app/component archive: split into linked packages")
    shutil.copytree(source, staged, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    # Project declared native profiles into the canonical packaged-default tree.
    # This is a direct mapping, with no directory search or config-name inference.
    canonical = {}
    for dest,relative in configs.items():
        selected = 'configs/' + dest.removeprefix('in/config/')
        target = staged / selected
        if target.exists() and target.read_bytes() != (source/relative).read_bytes():
            raise ValueError(f'Conflicting canonical packaged config: {selected}')
        target.parent.mkdir(parents=True,exist_ok=True)
        if relative != selected: shutil.copy2(source/relative,target)
        canonical[dest]=selected
    meta['payload']['configs']=canonical
    return meta, canonical
