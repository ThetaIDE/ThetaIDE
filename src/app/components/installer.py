"""Component installer, archive extractor, and integrity verification."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import stat
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union
from zipfile import ZipFile

import yaml

from src.app.workspace import COMPONENT_DIRECTORIES, WorkspacePaths
from src.app.services.rulesets import reject_legacy_rules_paths, require_canonical_rules_layout

from .dependencies import (
    check_missing_dependencies,
    install_packages,
    load_requirements_from_file,
)
from .models import AuthorInfo, HubComponent, ReleaseInfo


def _add_missing_yaml_defaults(retained_text: str, packaged_text: str) -> str:
    """Preserve the retained document if its defaults cannot be safely merged."""
    try:
        result = _merge_missing_yaml_defaults(retained_text, packaged_text)
        original, merged = yaml.safe_load(retained_text), yaml.safe_load(result)
        def contains(original, merged):
            if isinstance(original, dict):
                return isinstance(merged, dict) and all(k in merged and contains(v, merged[k]) for k, v in original.items())
            return type(original) is type(merged) and original == merged
        if not contains(original, merged):
            raise ValueError("merge would change existing values")
        return result
    except (yaml.YAMLError, ValueError, TypeError) as exc:
        logging.getLogger(__name__).warning("Could not merge component defaults; kept the existing YAML: %s", exc)
        return retained_text


def _merge_missing_yaml_defaults(retained_text: str, packaged_text: str) -> str:
    """Add missing block-mapping defaults without rewriting existing YAML or comments.

    Existing values, including explicit nulls, win. Flow mappings and incompatible
    node shapes are left untouched, because rewriting them could change user data.
    """
    old = yaml.compose(retained_text)
    new = yaml.compose(packaged_text)
    if not isinstance(old, yaml.MappingNode) or not isinstance(new, yaml.MappingNode):
        return retained_text
    additions = []
    def walk(existing, incoming):
        if existing.flow_style or incoming.flow_style:
            logging.getLogger(__name__).warning("Component defaults in a flow-style YAML mapping were not merged; existing settings were preserved.")
            return
        current = {key.value: val for key, val in existing.value if isinstance(key, yaml.ScalarNode)}
        missing = []
        for key, value in incoming.value:
            if not isinstance(key, yaml.ScalarNode):
                continue
            if key.value not in current:
                missing.append((key, value))
            elif isinstance(current[key.value], yaml.MappingNode) and isinstance(value, yaml.MappingNode):
                walk(current[key.value], value)
        if not missing:
            return
        indent = existing.start_mark.column
        # Use the parsed values to safely quote strings and nested structures.
        # Construct values directly from each node to support nested mappings.
        values = {}
        for key, value in missing:
            fragment = packaged_text[value.start_mark.index:value.end_mark.index]
            values[key.value] = yaml.safe_load(fragment)
        block = yaml.safe_dump(values, sort_keys=False, allow_unicode=True)
        block = "".join(" " * indent + line if line.strip() else line for line in block.splitlines(keepends=True))
        additions.append((existing.end_mark.line, indent, block))
    walk(old, new)
    if not additions:
        return retained_text
    lines = retained_text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    for line, indent, block in sorted(additions, key=lambda item: (-item[0], item[1])):
        lines.insert(line, block)
    result = "".join(lines)
    # Never install a syntactically invalid merge.
    yaml.safe_load(result)
    return result


def _authoring_hash(meta):
    fields = ("name", "description", "parameter_schema", "editor_actions", "entry_point",
              "entry_points", "dependencies", "paradigms", "aliases", "provides",
              "storage_version", "api_version", "capabilities", "payload", "rules_root")
    return hashlib.sha256(json.dumps({key: meta[key] for key in fields if key in meta}, sort_keys=True).encode()).hexdigest()


class HubInstaller:
    """Manages the download, verification, extraction, and removal of Hub components."""

    def __init__(
        self,
        workspace_dir: Path,
        data_dir: Path,
        on_change_callback: Callable[[str, str], None] | None = None,
        settings_manager: Any | None = None,
        lookup_component_fn: Callable[[str], HubComponent | None] | None = None,
    ):
        self.paths = WorkspacePaths(workspace_dir, data_dir)
        # Preserve the public installer's lexical paths (e.g. /var versus
        # /private/var on macOS); ownership checks resolve paths explicitly.
        self.workspace_dir = Path(workspace_dir)
        self.data_dir = Path(data_dir)
        self.on_change_callback = on_change_callback  # (component_id, action: "install" | "uninstall")
        self.settings_manager = settings_manager
        self.lookup_component_fn = lookup_component_fn

    def _set_install_permissions(self, target: Path, meta: dict, *, writable: bool) -> None:
        """Set file modes for package files, leaving user-created profiles alone."""
        self.paths.require_managed(target)
        paths = list(target.rglob("*")) if target.is_dir() else []
        excluded = set()
        for path in paths:
            if path.is_symlink() or not path.is_file() or path.resolve() in excluded:
                continue
            mode = stat.S_IMODE(path.stat().st_mode)
            path.chmod((mode | stat.S_IWUSR) if writable else (mode & ~0o222))

    def protect_installed_components(self) -> None:
        """Apply read-only modes to existing Hub installations on startup."""
        from src.app.core.provider_catalog import get_catalog
        for package in get_catalog(self.paths).packages:
            if package.managed: self._set_install_permissions(package.root,package.manifest,writable=False)

    def resolve_target_dir(self, component: HubComponent) -> Path:
        """Determine the filesystem destination for a component based on its kind."""
        return self.paths.installation(component.id, component.kind)

    def _failure_point(self, boundary, identity):
        """Injection hook for transaction/recovery tests; production performs no action."""


    def resolve_config_path(self, component: HubComponent) -> Path | None:
        """Determine destination path for a component's default configuration in in/config/."""
        mapping = component.extra.get('payload',{}).get('configs',{})
        declared = component.extra.get('config_path') or next(iter(mapping),None)
        return self.paths.config_file(declared) if declared else None

    def lookup_component(self, component_id: str) -> HubComponent | None:
        """Look up a component in the active catalog or its cached index."""
        if self.lookup_component_fn is not None:
            comp = self.lookup_component_fn(component_id)
            if comp:
                return comp

        cache_index = self.data_dir / "cache" / "hub" / "index.json"
        if cache_index.is_file():
            try:
                data = json.loads(cache_index.read_text(encoding="utf-8"))
                for item in data.get("components", []):
                    if item.get("id") == component_id:
                        return HubComponent.from_dict(item)
            except Exception:
                pass

        return None

    def check_installed(self, component: HubComponent) -> Tuple[bool, str | None]:
        """Check if a component is installed and determine its version."""
        from src.app.core.provider_catalog import get_catalog
        try: owner = get_catalog(self.paths).package(component.id)
        except ValueError as exc:
            component.extra['ownership_error'] = str(exc)
            return False,None
        return (True,owner.version) if owner else (False,None)

    def download_and_verify(
        self,
        url: str,
        expected_sha256: str | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Path:
        """Download an archive and verify its cryptographic SHA-256 checksum."""
        tmp_fd, tmp_path_str = tempfile.mkstemp(prefix="theta_pkg_", suffix=".zip")
        # Writes below reopen the file by path; an open handle would also block unlink() on Windows
        os.close(tmp_fd)
        tmp_path = Path(tmp_path_str)

        hasher = hashlib.sha256()
        total_downloaded = 0

        try:
            if url.startswith("file://"):
                local_source = Path(url[7:])
                content = local_source.read_bytes()
                hasher.update(content)
                tmp_path.write_bytes(content)
                total_downloaded = len(content)
                if progress_callback:
                    progress_callback(total_downloaded, total_downloaded)
            else:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "ThetaIDE-HubClient/1.0"}
                )
                with urllib.request.urlopen(req, timeout=30) as response, open(tmp_path, "wb") as f:
                    content_length = response.headers.get("Content-Length")
                    total_size = int(content_length) if content_length else -1

                    chunk_size = 64 * 1024
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        hasher.update(chunk)
                        total_downloaded += len(chunk)
                        if progress_callback:
                            progress_callback(total_downloaded, total_size)

            computed_sha = hasher.hexdigest().lower()
            if expected_sha256:
                expected_clean = expected_sha256.strip().lower()
                if computed_sha != expected_clean:
                    raise ValueError(
                        f"Checksum verification failed!\n"
                        f"Expected SHA-256: {expected_clean}\n"
                        f"Computed SHA-256: {computed_sha}"
                    )

            return tmp_path

        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def has_local_changes(self, target: Path, meta: dict) -> bool:
        """Include retained edits so reinstall never reclassifies them as upstream."""
        expected = meta.get("source_hashes")
        current = {p.relative_to(target).as_posix(): self._sha256_file(p)
                   for p in target.rglob("*") if p.is_file()
                   and p.name not in (".theta_component.json", "COMPONENT_CONTEXT.md")
                   and "__pycache__" not in p.parts}
        return (bool(meta.get("saved_local_changes")) or expected is None or expected != current
                or meta.get("authoring_hash") != _authoring_hash(meta)
                or any(self._sha256_file(self.workspace_dir / relative) != digest
                       for relative, digest in meta.get("asset_hashes", {}).items()))

    def install(
        self,
        component: HubComponent,
        version: str | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        install_dependencies: bool = True,
    ) -> bool:
        from .lifecycle import install
        return install(self, component, version, progress_callback, install_dependencies)

    def has_configs(self, component: HubComponent) -> bool:
        """Check if any configuration files for this component or its suite children exist on disk."""
        from src.app.core.provider_catalog import get_catalog
        from src.app.components.lifecycle import dependencies
        catalog=get_catalog(self.paths)
        seen=set()
        def configured(package_id):
            if package_id in seen: return False
            seen.add(package_id)
            owner=catalog.package(package_id)
            if owner is None:return False
            configs=[*owner.manifest.get('payload',{}).get('configs',{}),*owner.manifest.get('user_config_paths',[])]
            return any(self.paths.config_file(dest).is_file() for dest in configs) or any(configured(dep) for dep,_ in dependencies(owner.manifest))
        return configured(component.id)

    def find_component_for_path(self, path: str | Path) -> Dict[str, Any] | None:
        """Given a file or directory path, determine if it belongs to an installed Hub component.

        Returns a dictionary with metadata ('id', 'name', 'kind', 'version', 'target_dir', 'config_paths')
        or None if the path does not belong to an installed component.
        """
        target_path = Path(path)
        if not target_path.is_absolute():
            target_path = (self.workspace_dir / target_path).resolve()
        else:
            target_path = target_path.resolve()

        from src.app.core.provider_catalog import get_catalog
        candidates = []
        for package in get_catalog(self.paths).packages:
            if package.origin == "python_distribution": continue
            data = dict(package.manifest)
            comp_dir = package.root
            configs = [*data.get("config_paths", []), *data.get("user_config_paths", [])]
            if data.get("config_path"): configs.append(data["config_path"])
            files = [*configs, *data.get("installed_files", [])]
            if target_path == comp_dir or target_path.is_relative_to(comp_dir) or any(target_path == (self.workspace_dir / p).resolve() for p in files):
                data.update(target_dir=comp_dir, origin=package.origin, writable=package.writable)
                candidates.append(data)
        if len(candidates) > 1:
            # A shared config can have multiple owners; surface that explicitly.
            candidates[0]["shared_owners"] = [c["id"] for c in candidates]
        return candidates[0] if candidates else None

    def is_installed_file(self, path: str | Path) -> bool:
        """Whether a file is part of a Hub installation rather than a user copy."""
        path = Path(path)
        path = (self.workspace_dir / path).resolve() if not path.is_absolute() else path.resolve()
        owner = self.find_component_for_path(path)
        if not owner or owner.get("origin") not in ("bundled", "hub", "local_archive", "unknown"):
            return False
        user_paths = {(self.workspace_dir / p).resolve() for p in owner.get("user_config_paths", [])}
        return path not in user_paths

    def remap_tracked_paths(self, old_rel: str, new_rel: str) -> None:
        """Update installed-component manifests after a tracked workspace path moves.

        Paths in manifests are workspace-relative. Remapping both exact files and
        directory prefixes keeps uninstall and component lookup working after a
        user reorganizes YAMLs in the config tree.
        """
        old_rel = Path(old_rel).as_posix().strip("/")
        new_rel = Path(new_rel).as_posix().strip("/")
        if not old_rel or not new_rel or old_rel == new_rel:
            return

        from src.app.core.provider_catalog import get_catalog
        for package in get_catalog(self.paths).packages:
            if package.origin not in ('user','hub','local_archive'):continue
            manifest=package.root/'.theta_component.json'
            import copy
            meta=copy.deepcopy(package.manifest)
            def remap(value):
                return new_rel+value[len(old_rel):] if isinstance(value,str) and (value==old_rel or value.startswith(old_rel+'/')) else value
            for field in ('config_paths','user_config_paths'):
                if field in meta:meta[field]=[remap(v) for v in meta[field]]
            if 'config_path' in meta:meta['config_path']=remap(meta['config_path'])
            meta['payload']['configs']={remap(k):v for k,v in meta['payload']['configs'].items()}
            for field in ('config_hashes','config_backups'):
                if field in meta:meta[field]={remap(k):v for k,v in meta[field].items()}
            if 'x-theta-config-schemas' in meta.get('parameter_schema',{}):
                schemas=meta['parameter_schema']['x-theta-config-schemas']
                meta['parameter_schema']['x-theta-config-schemas']={remap(k):v for k,v in schemas.items()}
            if meta==package.manifest:continue
            mode=stat.S_IMODE(manifest.stat().st_mode)
            temporary=manifest.with_suffix('.json.tmp')
            try:
                temporary.write_text(json.dumps(meta,indent=2)+'\n');temporary.chmod(mode);temporary.replace(manifest)
            finally:temporary.unlink(missing_ok=True)

    def _sha256_file(self, path: Path) -> str | None:
        try:
            digest = hashlib.sha256()
            with Path(path).open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            return digest.hexdigest()
        except OSError:
            return None

    def _backup_existing_config(self, component_id: str, dest_file: Path, backups: dict) -> None:
        """Preserve a pre-existing YAML before an install overwrites it."""
        dest_file = Path(dest_file)
        if not dest_file.is_file():
            return
        try:
            rel = dest_file.resolve().relative_to(self.workspace_dir.resolve()).as_posix()
            if rel in backups:
                return
            backup_rel = (Path(".thetaide") / "config_backups" / component_id / rel).as_posix()
            backup_path = self.workspace_dir / backup_rel
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dest_file, backup_path)
            backups[rel] = {"backup": backup_rel, "restore_to": rel}
        except (OSError, ValueError):
            return

    def _retained_config_dir(self, component_id: str) -> Path:
        """Return the git-ignored store for configs hidden with an uninstalled component."""
        return self.data_dir / "retained_configs" / component_id

    def _restore_retained_configs(
        self, component_id: str, installed_config_paths: list[str], config_backups: dict
    ) -> list[str]:
        """Restore a component's hidden YAMLs and return user-created profile paths."""
        retained_root = self._retained_config_dir(component_id)
        if not retained_root.is_dir():
            return []

        installed_paths = set(installed_config_paths)
        restored_user_paths = []
        for retained in sorted(retained_root.rglob("*")):
            if not retained.is_file() or retained.suffix.lower() not in (".yaml", ".yml"):
                continue
            relative = retained.relative_to(retained_root).as_posix()
            if not relative.startswith("in/config/"):
                continue
            destination = self.workspace_dir / relative
            if destination.is_file() and relative not in config_backups:
                self._backup_existing_config(component_id, destination, config_backups)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.is_file() and relative in installed_paths:
                retained_text = retained.read_text(encoding="utf-8")
                packaged_text = destination.read_text(encoding="utf-8")
                destination.write_text(_add_missing_yaml_defaults(retained_text, packaged_text), encoding="utf-8")
                restored_user_paths.append(relative)
            else:
                shutil.copy2(retained, destination)
            if relative not in installed_paths:
                installed_config_paths.append(relative)
                installed_paths.add(relative)
                restored_user_paths.append(relative)
        return restored_user_paths

    def track_user_config(self, component_meta: dict, config_path: Path) -> None:
        """Associate a copied component profile with its owning installed component."""
        target_dir = Path(component_meta.get("target_dir", ""))
        manifest_path = target_dir / ".theta_component.json"
        if not manifest_path.is_file():
            return
        try:
            relative = Path(config_path).resolve().relative_to(self.workspace_dir.resolve()).as_posix()
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            user_paths = list(data.get("user_config_paths", []))
            if relative not in user_paths:
                user_paths.append(relative)
            hashes = dict(data.get("config_hashes", {}))
            digest = self._sha256_file(Path(config_path))
            if digest:
                hashes[relative] = digest
            data["user_config_paths"] = user_paths
            data["config_hashes"] = hashes
            mode = stat.S_IMODE(manifest_path.stat().st_mode)
            try:
                manifest_path.chmod(mode | stat.S_IWUSR)
                manifest_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            finally:
                manifest_path.chmod(mode)
        except (OSError, ValueError, json.JSONDecodeError):
            return

    def _uninstall_configs(self, meta: dict) -> None:
        """Hide component YAMLs and retain edits or pre-install configs for reinstall."""
        recorded = list(meta.get("config_paths") or ([meta["config_path"]] if meta.get("config_path") else []))
        for cfg_rel in meta.get("user_config_paths", []):
            if cfg_rel not in recorded:
                recorded.append(cfg_rel)
        hashes = meta.get("config_hashes") or {}
        backups = meta.get("config_backups") or {}
        component_id = str(meta.get("id") or "component")
        user_config_paths = set(meta.get("user_config_paths", []))
        for cfg_rel in recorded:
            current = self.workspace_dir / cfg_rel
            expected = hashes.get(cfg_rel)
            actual = self._sha256_file(current) if current.is_file() else None
            backup_info = backups.get(cfg_rel)
            if isinstance(backup_info, str):
                backup_info = {"backup": backup_info, "restore_to": cfg_rel}
            backup_path = None
            if isinstance(backup_info, dict):
                backup_path = self.workspace_dir / backup_info.get("backup", "")

            # If an install manifest is missing or predates config hashes, treat
            # the file as user-edited. User-created copies are retained even when
            # unchanged from their source profile.
            modified = current.is_file() and (expected is None or actual != expected)
            if current.is_file() and (modified or cfg_rel in user_config_paths):
                retained = self._retained_config_dir(component_id) / cfg_rel
                try:
                    retained.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(current, retained)
                except OSError:
                    # Do not remove a profile if preserving it failed.
                    continue
            if current.is_file():
                try:
                    current.unlink()
                    self._prune_empty_config_dirs(current)
                except OSError:
                    continue

            if backup_path and backup_path.is_file():
                # The destination belongs to the component config lifecycle. A
                # pre-install file must not reappear as an orphan after uninstall.
                try:
                    retained = self._retained_config_dir(component_id) / cfg_rel
                    if not retained.exists():
                        retained.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(backup_path, retained)
                    backup_path.unlink()
                    self._prune_empty_dirs(backup_path.parent, self.workspace_dir / ".thetaide")
                except OSError:
                    pass

    def _prune_empty_dirs(self, directory: Path, boundary: Path) -> None:
        """Remove empty parents up to, but never including, a standard workspace root."""
        current = Path(directory).resolve()
        boundary = Path(boundary).resolve()
        if current == boundary or boundary not in current.parents:
            return
        while current != boundary and boundary in current.parents and current.is_dir():
            try:
                next(current.iterdir())
            except StopIteration:
                try:
                    current.rmdir()
                except OSError:
                    break
                current = current.parent
            except OSError:
                break
            else:
                break

    def _prune_empty_config_dirs(self, config_file: Path) -> None:
        config_root = (self.workspace_dir / "in" / "config").resolve()
        path = Path(config_file).resolve()
        try:
            rel = path.relative_to(config_root)
        except ValueError:
            return
        boundary = config_root / rel.parts[0] if len(rel.parts) > 1 else config_root
        self._prune_empty_dirs(path.parent, boundary)

    def _prune_empty_component_dirs(self, component_dir: Path) -> None:
        component_dir = Path(component_dir).resolve()
        roots = self.paths.component_roots
        for root in roots:
            root = root.resolve()
            if component_dir == root or root in component_dir.parents:
                self._prune_empty_dirs(component_dir.parent, root)
                return

    def uninstall_by_metadata(self, meta: Dict[str, Any], *, cascade: bool = True) -> bool:
        from .lifecycle import uninstall
        return uninstall(self, meta)

    def uninstall(self, component: HubComponent) -> bool:
        target = self.resolve_target_dir(component)
        manifest = target / ".theta_component.json"
        if not manifest.is_file():
            from src.app.core.provider_catalog import ProviderCatalog
            owner = ProviderCatalog(self.paths).package(component.id)
            if owner:
                raise ValueError(f"{component.id} belongs to {owner.origin} at {owner.root}; ordinary uninstall only manages installed packages")
            return False
        return self.uninstall_by_metadata({**json.loads(manifest.read_text()), "target_dir": str(target)})
