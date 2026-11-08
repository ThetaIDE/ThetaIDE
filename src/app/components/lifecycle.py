"""Managed lifecycle implementation shared by Hub and local archive imports."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from src.app.components.archive import adapt, extract
from src.app.components.transactions import FileTransaction, assert_idle, remove, workspace_lock
from src.app.core.provider_catalog import ProviderCatalog, file_hash, content_hash, owner_selected
from src.app.core.provider_loader import invalidate
from src.app.services.rulesets import require_canonical_rules_layout


def dependencies(meta):
    for value in meta.get("dependencies", {}).get("components", []):
        if isinstance(value, str): yield value, ""
        elif isinstance(value, dict) and value.get("id"): yield value["id"], value.get("version", "")
        else: raise ValueError(f"Invalid component dependency: {value!r}")


def check_version(version, constraint):
    if constraint:
        # Existing Python packaging semantics; this validates declared versions, it does not solve/download dependencies.
        from packaging.specifiers import SpecifierSet
        if version not in SpecifierSet(constraint): raise ValueError(f"Dependency version {version} does not satisfy {constraint}")


def validate_dependencies(catalog, meta, lookup=None, stack=()):
    package_id = meta.get("id")
    if package_id in stack: raise ValueError("Component dependency cycle: " + " -> ".join((*stack, package_id)))
    missing = []
    for dep, spec in dependencies(meta):
        if dep in meta.get("provides", []): continue
        local = catalog.package(dep)
        if local:
            check_version(local.version, spec)
            validate_dependencies(catalog, local.manifest, lookup, (*stack, package_id))
        elif lookup:
            remote = lookup(dep)
            if remote is None: raise ValueError(f"Missing dependency {dep!r} for {package_id}")
            check_version(remote.version, spec)
            validate_dependencies(catalog, remote.to_dict(), lookup, (*stack, package_id))
            missing.append(remote)
        else: raise ValueError(f"Missing dependency {dep!r} for {package_id}; install it first")
    return missing


def _config_files(installer, config_map, meta):
    paths = installer.paths
    result = [paths.config_file(value) for value in config_map]
    for value in [*meta.get("config_paths", []), *meta.get("user_config_paths", [])]: result.append(paths.config_file(value))
    return result


def install(installer, component, version=None, progress_callback=None, install_dependencies=True):
    require_canonical_rules_layout(installer.workspace_dir)
    paths = installer.paths
    ver = version or component.version
    if ver not in component.releases: raise ValueError(f"Version {ver} not found in releases for {component.id}")
    # Download/stage before taking the workspace lock. No implementation/config writes yet.
    release = component.releases[ver]
    archive = installer.download_and_verify(release.url, release.sha256, progress_callback)
    try:
        with tempfile.TemporaryDirectory(prefix="theta-package-") as temporary:
            temp = Path(temporary)
            source = extract(archive, temp / "archive")
            staged = temp / "payload"
            meta, configs = adapt(source, component, staged, version=ver)
            meta.update(version=ver, origin="local_archive" if release.url.startswith("file:") else "hub",
                        installed_from=release.url, sha256=release.sha256)
            catalog = ProviderCatalog(paths)
            existing = catalog.package(component.id)
            target = paths.installation(component.id, component.kind)
            if existing and existing.root != target:
                raise ValueError(f"Package {component.id} already belongs to {existing.origin} at {existing.root}. Use a new package ID or an editable copy; bundled/user sources cannot be replaced.")
            missing = validate_dependencies(catalog, meta, installer.lookup_component if install_dependencies else None)
            # Dependency operations have separate ownership and transactions; never hold a recursive lock.
            for dependency in missing: installer.install(dependency, install_dependencies=install_dependencies)
            pip_deps = list(meta.get("dependencies", {}).get("pip", []))
            from src.app.components.dependencies import load_requirements_from_file
            pip_deps.extend(load_requirements_from_file(staged / "requirements.txt"))
            plugin = staged / "plugin.json"
            if plugin.is_file(): pip_deps.extend(json.loads(plugin.read_text()).get("dependencies", {}).get("pip", []))
            if pip_deps and install_dependencies:
                from src.app.components import installer as installer_module
                missing_pip = installer_module.check_missing_dependencies(list(dict.fromkeys(pip_deps)))
                if missing_pip:
                    ok, message = installer_module.install_packages(missing_pip)
                    if not ok: raise RuntimeError(f"Python dependency installation failed: {message}. Python environment changes are outside package rollback.")
            with workspace_lock(paths):
                assert_idle(paths, component.id)
                # Recheck after dependencies/downloads and concurrent installations.
                catalog = ProviderCatalog(paths)
                existing = catalog.package(component.id)
                if target.exists() and (existing is None or existing.root != target):
                    raise ValueError(f"Destination exists without matching package ownership: {target}")
                case_collisions = [p for p in catalog.packages if p.id.casefold() == component.id.casefold() and p.id != component.id]
                if case_collisions: raise ValueError(f"Case-insensitive package ID collision: {[(p.id, str(p.root)) for p in case_collisions]}")
                if existing and existing.root != target: raise ValueError(f"Package ownership collision: {component.id} at {existing.root}")
                validate_dependencies(catalog, meta)
                previous = existing.manifest if existing else {}
                edited = existing is not None and installer.has_local_changes(target, previous)
                retained_package = (paths.data_dir / "retained_packages" / component.id /
                    (str(previous.get("version", "unknown")) + "-" + content_hash(target)[:16])) if edited else None
                # Capabilities may be shared only through an explicit authored override.
                from src.app.core.provider_catalog import Package
                candidate = Package(component.id, staged, meta["origin"], ver, meta)
                catalog._capabilities(candidate)
                for cap in candidate.capabilities:
                    for alias in [cap.id, *cap.aliases]:
                        owners = [(p, c) for p, c in catalog.candidates(cap.kind, alias) if p.id != component.id]
                        if not owners: continue
                        selection = catalog.overrides.get(f"{cap.kind}:{alias.lower()}")
                        if any(p.origin == "bundled" for p, _ in owners) or not any(owner_selected(p, selection) for p in [candidate, *(p for p,c in owners)]):
                            raise ValueError(f"Capability collision {cap.kind}:{alias}: {[p.id for p, _ in owners]}; declare an explicit provider_overrides selection or export qualified capability IDs")
                destinations = _config_files(installer, configs, previous)
                # A different package can share identical configs, but differing content never wins by traversal order.
                for dest, relative in configs.items():
                    active = paths.config_file(dest)
                    for owner in catalog.packages:
                        if owner.id == component.id or dest not in owner.manifest.get("config_paths", []): continue
                        if not active.is_file() or file_hash(active) != file_hash(staged / relative):
                            raise ValueError(f"Config profile collision {dest}: owned by {owner.id}")
                retained = installer._retained_config_dir(component.id)
                with FileTransaction(paths, "install", component.id) as transaction:
                    transaction.protect([target, *destinations, retained, paths.root / ".thetaide/config_backups", paths.root / ".thetaide/workspace.json"])
                    from src.app.components.transactions import write_json
                    write_json(paths.root / ".thetaide/workspace.json", {"id": paths.identity})
                    if retained_package is not None:
                        transaction.protect([retained_package])
                        if not retained_package.exists():
                            retained_package.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copytree(target, retained_package)
                        elif content_hash(retained_package) != content_hash(target):
                            raise ValueError(f"Retained edited package collision: {retained_package}")
                        meta["retained_edits"] = list(dict.fromkeys([*previous.get("retained_edits", []), str(retained_package)]))
                        component.extra["retained_edits"] = meta["retained_edits"]
                    backups = dict(previous.get("config_backups", {}))
                    for dest, relative in configs.items():
                        active = paths.config_file(dest)
                        packaged = staged / relative
                        active.parent.mkdir(parents=True, exist_ok=True)
                        # User edits and unrelated workspace defaults win; merge only missing defaults.
                        if active.is_file():
                            from src.app.components.installer import _add_missing_yaml_defaults
                            incoming = packaged.read_text()
                            current = active.read_text()
                            expected = previous.get("config_hashes", {}).get(dest)
                            if expected and file_hash(active) == expected:
                                active.chmod(active.stat().st_mode | 0o200)
                                shutil.copy2(packaged, active)
                            else:
                                installer._backup_existing_config(component.id, active, backups)
                                active.chmod(active.stat().st_mode | 0o200)
                                active.write_text(_add_missing_yaml_defaults(current, incoming))
                        else: shutil.copy2(packaged, active)
                        installer._failure_point("config", dest)
                    user_configs = []
                    if not existing:
                        user_configs = installer._restore_retained_configs(component.id, list(configs), backups)
                    meta.update(config_paths=list(configs), config_path=next(iter(configs), None), config_backups=backups,
                                user_config_paths=list(dict.fromkeys([*previous.get("user_config_paths", []), *user_configs])),
                                config_hashes={dest: file_hash(staged / relative) for dest, relative in configs.items()},
                                installed_files=[], target_dir=str(target))
                    if meta.get("config_path") is None: meta.pop("config_path", None)
                    meta["source_hashes"] = {p.relative_to(staged).as_posix(): file_hash(p) for p in staged.rglob("*")
                                             if p.is_file() and p.name != ".theta_component.json"}
                    from src.app.components.installer import _authoring_hash
                    meta["authoring_hash"] = _authoring_hash(meta)
                    (staged / ".theta_component.json").write_text(json.dumps(meta, indent=2) + "\n")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    remove(target)
                    # Copy stage onto destination filesystem; transaction backup provides crash recovery.
                    shutil.copytree(staged, target)
                    installer._failure_point("package", component.id)
                    # Verify staged content after copy, before committing ownership record.
                    for rel, digest in meta["source_hashes"].items():
                        if file_hash(target / rel) != digest: raise IOError(f"Staged copy verification failed: {rel}")
                    installer._set_install_permissions(target, meta, writable=False)
                    installer._failure_point("record", component.id)
                invalidate(paths, target)
            component.is_installed, component.installed_version = True, ver
            if installer.on_change_callback: installer.on_change_callback(component.id, "install")
            return True
    finally:
        archive.unlink(missing_ok=True)


def uninstall(installer, meta):
    paths = installer.paths
    require_canonical_rules_layout(paths.root)
    target = paths.require_managed(meta["target_dir"])
    package_id = meta["id"]
    with workspace_lock(paths):
        assert_idle(paths, package_id)
        catalog = ProviderCatalog(paths)
        for owner in catalog.packages:
            if owner.id != package_id and any(dep == package_id for dep, _ in dependencies(owner.manifest)):
                raise ValueError(f"{package_id} is required by {owner.id}; uninstall that dependent first")
        destinations = _config_files(installer, {}, meta)
        with FileTransaction(paths, "uninstall", package_id) as transaction:
            transaction.protect([target, *destinations, installer._retained_config_dir(package_id), paths.root / ".thetaide/config_backups"])
            if installer.has_local_changes(target, meta):
                retained = paths.data_dir / "retained_packages" / package_id / (str(meta.get("version", "unknown")) + "-" + content_hash(target)[:16])
                transaction.protect([retained])
                retained.parent.mkdir(parents=True, exist_ok=True)
                if not retained.exists(): shutil.copytree(target, retained)
            shared = {dest for p in catalog.packages if p.id != package_id for dest in p.manifest.get("config_paths", [])}
            owned_meta = {**meta, "config_paths": [c for c in meta.get("config_paths", []) if c not in shared]}
            installer._set_install_permissions(target, meta, writable=True)
            installer._uninstall_configs(owned_meta)
            installer._failure_point("config", package_id)
            remove(target)
            installer._failure_point("package", package_id)
            from src.app.services.local_components import LocalComponents
            local = LocalComponents(paths)
            record = local.state()['records'].get(package_id)
            if record and record.get('source') != 'local':
                cached = Path(record['package']).resolve()
                if cached.is_relative_to(local.root.resolve()):
                    transaction.protect([local.state_path, cached])
                    local.discard_hub_snapshot(package_id)
        invalidate(paths, target)
    if installer.on_change_callback: installer.on_change_callback(package_id, "uninstall")
    return True


def dependency_ids(catalog, meta, seen=None):
    seen = set() if seen is None else seen
    for package_id, spec in dependencies(meta):
        package = catalog.package(package_id)
        if package is not None and package.id not in seen:
            seen.add(package.id)
            dependency_ids(catalog, package.manifest, seen)
    return seen
