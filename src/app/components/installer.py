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
              "entry_points", "dependencies", "paradigms", "aliases", "provides", "target_path", "asset_paths")
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
        if not meta.get("installed_from") or meta.get("origin") == "local":
            return
        excluded = {(self.workspace_dir / p).resolve() for p in meta.get("user_config_paths", [])}
        paths = list(target.rglob("*")) if target.is_dir() else []
        for relative in [*meta.get("config_paths", []), *meta.get("installed_files", []), meta.get("config_path")]:
            if relative:
                path = (self.workspace_dir / relative).resolve()
                if path.is_relative_to(self.workspace_dir.resolve()):
                    paths.append(path)
        for path in paths:
            if path.is_symlink() or not path.is_file() or path.resolve() in excluded:
                continue
            mode = stat.S_IMODE(path.stat().st_mode)
            path.chmod((mode | stat.S_IWUSR) if writable else (mode & ~0o222))

    def protect_installed_components(self) -> None:
        """Apply read-only modes to existing Hub installations on startup."""
        roots = self.paths.component_roots
        for root in roots:
            if not root.is_dir():
                continue
            for manifest in root.rglob(".theta_component.json"):
                try:
                    self._set_install_permissions(manifest.parent, json.loads(manifest.read_text()), writable=False)
                except (OSError, ValueError, TypeError) as exc:
                    logging.getLogger(__name__).warning("Could not protect %s: %s", manifest.parent, exc)

    def resolve_target_dir(self, component: HubComponent) -> Path:
        """Determine the filesystem destination for a component based on its kind."""
        if component.target_path:
            # If target_path is relative, determine root
            if component.kind == "plugin":
                return self.data_dir / component.target_path
            return self.workspace_dir / component.target_path

        if component.kind == "plugin":
            return self.data_dir / "plugins" / component.id
        directory = COMPONENT_DIRECTORIES.get(component.kind, "components")
        return self.workspace_dir / directory / component.id

    def resolve_config_path(self, component: HubComponent) -> Path | None:
        """Determine destination path for a component's default configuration in in/config/."""
        extra_cfg = getattr(component, "extra", {}).get("config_path") if hasattr(component, "extra") else None
        if not extra_cfg and hasattr(component, "config_path") and component.config_path:
            extra_cfg = component.config_path
        if extra_cfg:
            return self.workspace_dir / extra_cfg

        folder_prefix = None
        if hasattr(component, "extra") and isinstance(component.extra, dict):
            folder_prefix = component.extra.get("namespace") or component.extra.get("group")

        filename = f"{component.id}.yaml"

        if component.kind == "model":
            if folder_prefix:
                return self.workspace_dir / "in" / "config" / "model" / folder_prefix / filename
            return self.workspace_dir / "in" / "config" / "model" / filename
        elif component.kind in ("method", "agent", "algorithm"):
            if folder_prefix:
                return self.workspace_dir / "in" / "config" / "agent" / folder_prefix / filename
            return self.workspace_dir / "in" / "config" / "agent" / filename
        elif component.kind in ("datamodule", "data"):
            if folder_prefix:
                return self.workspace_dir / "in" / "config" / "data" / folder_prefix / filename
            return self.workspace_dir / "in" / "config" / "data" / filename
        elif component.kind in ("evaluator", "eval"):
            if folder_prefix:
                return self.workspace_dir / "in" / "config" / "eval" / folder_prefix / filename
            return self.workspace_dir / "in" / "config" / "eval" / filename
        elif component.kind in ("plotter", "plot"):
            if folder_prefix:
                return self.workspace_dir / "in" / "config" / "plot" / folder_prefix / filename
            return self.workspace_dir / "in" / "config" / "plot" / filename
        elif component.kind == "env":
            if folder_prefix:
                return self.workspace_dir / "in" / "config" / "env" / folder_prefix / filename
            return self.workspace_dir / "in" / "config" / "env" / filename
        elif component.kind == "experiment":
            return self.workspace_dir / "in" / "config" / "experiment" / filename
        return None

    def lookup_component(self, component_id: str) -> HubComponent | None:
        """Look up a component in the active catalog or its cached index."""
        if self.lookup_component_fn is not None:
            comp = self.lookup_component_fn(component_id)
            if comp:
                if comp.suite:
                    provider = self.lookup_component_fn(comp.suite)
                    if provider:
                        # Some suite entries have real, separately versioned
                        # archives. Prefer those; otherwise an older suite can
                        # silently overwrite a newer requested component.
                        release = comp.releases.get(comp.version)
                        provider_release = provider.releases.get(provider.version)
                        if not release or not release.url or (provider_release and release.url == provider_release.url):
                            return provider
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
        target_dir = self.resolve_target_dir(component)
        if not target_dir.exists() or not target_dir.is_dir():
            if component.suite and not component.target_path:
                provider = self.lookup_component(component.suite)
                if provider and provider.id != component.id:
                    provider_dir = self.resolve_target_dir(provider)
                    for name in (".theta_component.json", "plugin.json"):
                        manifest = provider_dir / name
                        if manifest.is_file():
                            try:
                                data = json.loads(manifest.read_text(encoding="utf-8"))
                                return True, data.get("version", provider.version)
                            except Exception:
                                return True, provider.version
            return False, None

        # 1. Check .theta_component.json manifest
        meta_file = target_dir / ".theta_component.json"
        if meta_file.exists():
            try:
                data = json.loads(meta_file.read_text(encoding="utf-8"))
                return True, data.get("version")
            except Exception:
                pass

        # 2. Check plugin.json manifest
        plugin_file = target_dir / "plugin.json"
        if plugin_file.exists():
            try:
                data = json.loads(plugin_file.read_text(encoding="utf-8"))
                return True, data.get("version")
            except Exception:
                pass

        return True, "unknown"

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
        """Download, verify, and unpack a component release into its destination directory."""
        ver = version or component.version
        if ver not in component.releases:
            raise ValueError(f"Version {ver} not found in releases for {component.id}")

        # Saved implementation is authoritative. Never overwrite developer edits.
        existing_target = self.resolve_target_dir(component)
        existing_manifest = existing_target / ".theta_component.json"
        if component.kind in ("model", "method", "agent", "algorithm", "env", "data", "datamodule", "eval", "evaluator", "plot", "plotter") and existing_manifest.is_file():
            previous = json.loads(existing_manifest.read_text())
            if self.has_local_changes(existing_target, previous):
                raise ValueError("This component has local changes or no source baseline. Export/retain your saved implementation before replacing the installation.")

        release = component.releases[ver]
        archive_path = self.download_and_verify(
            url=release.url,
            expected_sha256=release.sha256,
            progress_callback=progress_callback,
        )

        if not hasattr(self, "_installing_stack"):
            self._installing_stack = set()
        if component.id in self._installing_stack:
            return True
        self._installing_stack.add(component.id)

        target_dir = self.resolve_target_dir(component)
        previous_meta = {}
        previous_manifest = target_dir / ".theta_component.json"
        had_previous_manifest = previous_manifest.is_file()
        if had_previous_manifest:
            try:
                previous_meta = json.loads(previous_manifest.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        staging_dir = Path(tempfile.mkdtemp(prefix="theta_staging_"))

        try:
            self._set_install_permissions(target_dir, previous_meta, writable=True)
            with ZipFile(archive_path, "r") as zf:
                zf.extractall(staging_dir)

            # Handle case where zip contains a single enclosing root directory
            extracted_items = list(staging_dir.iterdir())
            if len(extracted_items) == 1 and extracted_items[0].is_dir():
                source_dir = extracted_items[0]
            else:
                source_dir = staging_dir

            # Validate component manifest if packaged in archive
            pkg_meta_file = source_dir / ".theta_component.json"
            if pkg_meta_file.is_file():
                from src.app.core.manifest_schema import validate_manifest_file
                is_valid, err_msg = validate_manifest_file(pkg_meta_file)
                if not is_valid:
                    raise ValueError(f"Component archive contains invalid manifest: {err_msg}")

            # Check for packaged configuration YAML(s) and deploy to in/config/
            installed_config_paths = []
            installed_files = []
            config_backups = {}
            is_root_relative = (
                (source_dir / "in").is_dir()
                or (source_dir / "src").is_dir()
                or (source_dir / "plot").is_dir()
                or component.kind in ("experiment", "suite")
            )

            if is_root_relative:
                # Deploy root-relative assets directly to workspace_dir (in/, src/, results/, plot/)
                for item in sorted(source_dir.rglob("*")):
                    if not item.is_file():
                        continue
                    rel = item.relative_to(source_dir)
                    if rel.parts and rel.parts[0] in ("in", "src", "results", "plot"):
                        dest_file = self.workspace_dir / rel
                        dest_file.parent.mkdir(parents=True, exist_ok=True)
                        if len(rel.parts) >= 2 and rel.parts[:2] == ("in", "config") and dest_file.suffix.lower() in (".yaml", ".yml"):
                            self._backup_existing_config(component.id, dest_file, config_backups)
                        shutil.copy2(str(item), str(dest_file))
                        installed_files.append(str(dest_file.relative_to(self.workspace_dir)))
                        if len(rel.parts) >= 2 and rel.parts[0] == "in" and rel.parts[1] == "config":
                            installed_config_paths.append(str(dest_file.relative_to(self.workspace_dir)))

            pkg_config_dir = source_dir / "config"
            if pkg_config_dir.is_dir():
                target_in_config = self.workspace_dir / "in" / "config"
                for yaml_file in sorted(pkg_config_dir.rglob("*.yaml")):
                    rel = yaml_file.relative_to(pkg_config_dir)
                    if len(rel.parts) == 1:
                        config_dest = self.resolve_config_path(component)
                        if config_dest is not None:
                            dest_file = config_dest.parent / yaml_file.name
                        else:
                            dest_file = target_in_config / rel
                    else:
                        dest_file = target_in_config / rel
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    self._backup_existing_config(component.id, dest_file, config_backups)
                    shutil.copy2(str(yaml_file), str(dest_file))
                    installed_config_paths.append(str(dest_file.relative_to(self.workspace_dir)))

            config_dest = self.resolve_config_path(component)
            installed_config_rel = installed_config_paths[0] if installed_config_paths else None
            if not installed_config_paths and config_dest:
                config_candidates = [
                    source_dir / f"{component.id}.yaml",
                    source_dir / f"{component.id}.yml",
                    source_dir / "config.yaml",
                    source_dir / "config.yml",
                    source_dir / "default.yaml",
                    source_dir / "default.yml",
                    source_dir / "default_config.yaml",
                ]
                for yaml_file in sorted(list(source_dir.glob("*.yaml")) + list(source_dir.glob("*.yml"))):
                    if yaml_file not in config_candidates:
                        config_candidates.append(yaml_file)

                for cand in config_candidates:
                    if cand.is_file():
                        config_dest.parent.mkdir(parents=True, exist_ok=True)
                        self._backup_existing_config(component.id, config_dest, config_backups)
                        shutil.copy2(str(cand), str(config_dest))
                        installed_config_rel = str(config_dest.relative_to(self.workspace_dir))
                        installed_config_paths.append(installed_config_rel)
                        break

            # Restore configs retained during an earlier uninstall. This includes
            # edits to packaged profiles and user-created copies of those profiles.
            previous_config_paths = set(config_backups)
            restored_user_config_paths = []
            if not had_previous_manifest:
                restored_user_config_paths = self._restore_retained_configs(
                    component.id, installed_config_paths, config_backups
                )
            # On an in-place update preserve edited profiles, adding only new defaults.
            # A missing old hash is treated conservatively as a user edit.
            old_hashes = previous_meta.get("config_hashes") or {}
            old_user_paths = set(previous_meta.get("user_config_paths") or [])
            for relative, info in config_backups.items():
                if relative not in previous_config_paths or relative not in installed_config_paths:
                    continue
                backup = self.workspace_dir / info["backup"]
                destination = self.workspace_dir / relative
                expected = old_hashes.get(relative)
                if relative in old_user_paths or not expected or self._sha256_file(backup) != expected:
                    destination.write_text(_add_missing_yaml_defaults(
                        backup.read_text(encoding="utf-8"), destination.read_text(encoding="utf-8")
                    ), encoding="utf-8")
                    restored_user_config_paths.append(relative)
            installed_config_paths = list(dict.fromkeys(installed_config_paths))
            installed_config_rel = installed_config_paths[0] if installed_config_paths else None

            # Write component tracking metadata
            existing_pkg_meta = {}
            if (source_dir / ".theta_component.json").is_file():
                try:
                    existing_pkg_meta = json.loads((source_dir / ".theta_component.json").read_text(encoding="utf-8"))
                except Exception:
                    pass
            elif (source_dir / "plugin.json").is_file():
                try:
                    existing_pkg_meta = json.loads((source_dir / "plugin.json").read_text(encoding="utf-8"))
                except Exception:
                    pass

            meta = dict(existing_pkg_meta)
            meta.update({
                "id": component.id,
                "name": component.name,
                "kind": component.kind,
                "version": ver,
                "installed_from": release.url,
                "sha256": release.sha256,
                "config_path": installed_config_rel,
                "config_paths": installed_config_paths,
                "user_config_paths": sorted(
                    set(existing_pkg_meta.get("user_config_paths", []))
                    | set(restored_user_config_paths)
                ),
                "config_backups": config_backups,
                "config_hashes": {
                    rel: digest
                    for rel in installed_config_paths
                    if (digest := self._sha256_file(self.workspace_dir / rel)) is not None
                },
            })
            if is_root_relative and installed_files:
                meta["installed_files"] = installed_files
            if hasattr(component, "extra") and isinstance(component.extra, dict) and "paradigms" in component.extra:
                meta["paradigms"] = component.extra["paradigms"]
            if hasattr(component, "suite") and component.suite:
                meta["suite"] = component.suite

            if is_root_relative:
                target_dir.mkdir(parents=True, exist_ok=True)
                for item in source_dir.iterdir():
                    if item.name in ("in", "src", "results", "plot", "__pycache__"):
                        continue
                    dest = target_dir / item.name
                    if item.is_dir():
                        shutil.copytree(str(item), str(dest), dirs_exist_ok=True)
                    else:
                        shutil.copy2(str(item), str(dest))

                (target_dir / ".theta_component.json").write_text(
                    json.dumps(meta, indent=2) + "\n", encoding="utf-8"
                )
                plugin_src = source_dir / "plugin.json"
                if plugin_src.is_file():
                    shutil.copy2(str(plugin_src), str(target_dir / "plugin.json"))
                readme_src = source_dir / "README.md"
                if readme_src.is_file():
                    shutil.copy2(str(readme_src), str(target_dir / "README.md"))
            else:
                (source_dir / ".theta_component.json").write_text(
                    json.dumps(meta, indent=2) + "\n", encoding="utf-8"
                )

                # Atomic swap into target_dir
                if target_dir.exists():
                    backup_dir = target_dir.with_name(f"{target_dir.name}.backup")
                    if backup_dir.exists():
                        shutil.rmtree(backup_dir, ignore_errors=True)
                    target_dir.rename(backup_dir)
                    try:
                        shutil.move(str(source_dir), str(target_dir))
                        shutil.rmtree(backup_dir, ignore_errors=True)
                    except Exception:
                        # Rollback
                        if target_dir.exists():
                            shutil.rmtree(target_dir, ignore_errors=True)
                        backup_dir.rename(target_dir)
                else:
                    target_dir.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(source_dir), str(target_dir))

            meta["source_hashes"] = {
                p.relative_to(target_dir).as_posix(): self._sha256_file(p)
                for p in target_dir.rglob("*") if p.is_file()
                and p.name not in (".theta_component.json", "COMPONENT_CONTEXT.md")
                and "__pycache__" not in p.parts
            }
            meta["asset_hashes"] = {
                relative: self._sha256_file(self.workspace_dir / relative)
                for relative in meta.get("installed_files", [])
                if not (self.workspace_dir / relative).is_relative_to(target_dir)
                and relative not in meta.get("config_paths", [])
            }
            meta["authoring_hash"] = _authoring_hash(meta)
            (target_dir / ".theta_component.json").write_text(json.dumps(meta, indent=2) + "\n")

            # If this component is a suite declaring child components, install each
            suite_components = []
            target_meta_file = target_dir / ".theta_component.json"
            if target_meta_file.is_file():
                try:
                    pkg_data = json.loads(target_meta_file.read_text(encoding="utf-8"))
                    suite_components = pkg_data.get("dependencies", {}).get("components", [])
                except Exception:
                    pass
            if not suite_components:
                plugin_file = target_dir / "plugin.json"
                if plugin_file.is_file():
                    try:
                        pdata = json.loads(plugin_file.read_text(encoding="utf-8"))
                        suite_components = pdata.get("dependencies", {}).get("components", [])
                    except Exception:
                        pass
            if not suite_components and hasattr(component, "extra") and isinstance(component.extra, dict):
                suite_components = component.extra.get("dependencies", {}).get("components", [])

            installed_children = []
            if suite_components and install_dependencies:
                for child_id in suite_components:
                    child_comp = self.lookup_component(child_id)
                    if child_comp:
                        self.install(child_comp)
                        installed_children.append(child_id)

            if installed_children:
                meta["installed_components"] = installed_children
                target_meta_file.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

            # Inspect and install declared Python dependencies
            declared_deps = []
            if hasattr(component, "extra") and isinstance(component.extra, dict):
                declared_deps.extend(component.extra.get("dependencies", {}).get("pip", []))

            plugin_file = target_dir / "plugin.json"
            if plugin_file.exists():
                try:
                    pdata = json.loads(plugin_file.read_text(encoding="utf-8"))
                    declared_deps.extend(pdata.get("dependencies", {}).get("pip", []))
                except Exception:
                    pass

            req_file = target_dir / "requirements.txt"
            if req_file.exists():
                declared_deps.extend(load_requirements_from_file(req_file))

            # Deduplicate
            declared_deps = list(dict.fromkeys(declared_deps))
            if declared_deps and install_dependencies:
                missing = check_missing_dependencies(declared_deps)
                if missing:
                    install_packages(missing)

            self._set_install_permissions(target_dir, meta, writable=False)

            # Update component object state
            component.is_installed = True
            component.installed_version = ver

            if self.on_change_callback:
                self.on_change_callback(component.id, "install")
                plugin_file = target_dir / "plugin.json"
                if plugin_file.exists():
                    try:
                        pdata = json.loads(plugin_file.read_text(encoding="utf-8"))
                        pid = pdata.get("id")
                        if pid and pid != component.id:
                            self.on_change_callback(pid, "install")
                    except Exception:
                        pass
                if component.target_path:
                    tname = Path(component.target_path).name
                    if tname != component.id:
                        self.on_change_callback(tname, "install")

            return True

        finally:
            # Re-protect the previous installation if an update failed partway through.
            manifest = target_dir / ".theta_component.json"
            if manifest.is_file():
                try:
                    self._set_install_permissions(target_dir, json.loads(manifest.read_text()), writable=False)
                except (OSError, ValueError, TypeError) as exc:
                    logging.getLogger(__name__).warning("Could not protect %s: %s", target_dir, exc)
            if hasattr(self, "_installing_stack"):
                self._installing_stack.discard(component.id)
            if archive_path.exists():
                archive_path.unlink()
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)

    def has_configs(self, component: HubComponent) -> bool:
        """Check if any configuration files for this component or its suite children exist on disk."""
        target_dir = self.resolve_target_dir(component)
        meta_file = target_dir / ".theta_component.json"
        children = []
        if meta_file.exists():
            try:
                mdata = json.loads(meta_file.read_text(encoding="utf-8"))
                for cfg_rel in mdata.get("config_paths", []):
                    if (self.workspace_dir / cfg_rel).is_file():
                        return True
                if mdata.get("config_path") and (self.workspace_dir / mdata["config_path"]).is_file():
                    return True
                children.extend(mdata.get("installed_components", []))
                children.extend(mdata.get("dependencies", {}).get("components", []))
            except Exception:
                pass

        if hasattr(component, "extra") and isinstance(component.extra, dict):
            children.extend(component.extra.get("dependencies", {}).get("components", []))

        for child_id in set(children):
            child_comp = self.lookup_component(child_id)
            if child_comp and self.has_configs(child_comp):
                return True

        config_path = self.resolve_config_path(component)
        if config_path and config_path.is_file():
            return True
        return False

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

        scan_roots = self.paths.component_roots
        for root in scan_roots:
            if not root.is_dir():
                continue
            for meta_file in root.rglob(".theta_component.json"):
                try:
                    data = json.loads(meta_file.read_text(encoding="utf-8"))
                    comp_dir = meta_file.parent.resolve()
                    # Check 1: path is the component directory or inside it
                    if target_path == comp_dir or comp_dir in target_path.parents:
                        data["target_dir"] = comp_dir
                        return data
                    # Check 2: path is one of the recorded configs or their parent folder
                    for cfg_rel in data.get("config_paths", []):
                        cfg_full = (self.workspace_dir / cfg_rel).resolve()
                        if target_path == cfg_full or target_path in cfg_full.parents:
                            data["target_dir"] = comp_dir
                            return data
                    for cfg_rel in data.get("user_config_paths", []):
                        cfg_full = (self.workspace_dir / cfg_rel).resolve()
                        if target_path == cfg_full:
                            data["target_dir"] = comp_dir
                            return data
                    if data.get("config_path"):
                        cfg_full = (self.workspace_dir / data["config_path"]).resolve()
                        if target_path == cfg_full or target_path in cfg_full.parents:
                            data["target_dir"] = comp_dir
                            return data
                    # Check 3: path is one of the explicitly deployed installed files
                    for file_rel in data.get("installed_files", []):
                        file_full = (self.workspace_dir / file_rel).resolve()
                        if target_path == file_full or target_path in file_full.parents:
                            data["target_dir"] = comp_dir
                            return data
                except Exception:
                    continue
        return None

    def is_installed_file(self, path: str | Path) -> bool:
        """Whether a file is part of a Hub installation rather than a user copy."""
        path = Path(path)
        path = (self.workspace_dir / path).resolve() if not path.is_absolute() else path.resolve()
        owner = self.find_component_for_path(path)
        if not owner or not owner.get("installed_from") or owner.get("origin") == "local":
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

        roots = self.paths.component_roots
        seen = set()
        for root in roots:
            if not root.is_dir():
                continue
            for manifest_path in root.rglob(".theta_component.json"):
                if manifest_path in seen:
                    continue
                seen.add(manifest_path)
                try:
                    meta = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue

                changed = False

                def remap(value):
                    nonlocal changed
                    if not isinstance(value, str):
                        return value
                    normalized = Path(value).as_posix().strip("/")
                    if normalized == old_rel or normalized.startswith(old_rel + "/"):
                        changed = True
                        return new_rel + normalized[len(old_rel):]
                    return value

                if isinstance(meta.get("config_paths"), list):
                    meta["config_paths"] = [remap(value) for value in meta["config_paths"]]
                if isinstance(meta.get("user_config_paths"), list):
                    meta["user_config_paths"] = [remap(value) for value in meta["user_config_paths"]]
                if "config_path" in meta:
                    meta["config_path"] = remap(meta["config_path"])
                if isinstance(meta.get("installed_files"), list):
                    meta["installed_files"] = [remap(value) for value in meta["installed_files"]]
                if isinstance(meta.get("config_hashes"), dict):
                    meta["config_hashes"] = {remap(key): value for key, value in meta["config_hashes"].items()}
                if isinstance(meta.get("config_backups"), dict):
                    meta["config_backups"] = {remap(key): value for key, value in meta["config_backups"].items()}
                if changed:
                    try:
                        temp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
                        temp_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
                        temp_path.replace(manifest_path)
                    except OSError:
                        try:
                            temp_path.unlink(missing_ok=True)
                        except OSError:
                            pass

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
            config_paths = list(data.get("config_paths", []))
            if relative not in config_paths:
                config_paths.append(relative)
            hashes = dict(data.get("config_hashes", {}))
            digest = self._sha256_file(Path(config_path))
            if digest:
                hashes[relative] = digest
            data["user_config_paths"] = user_paths
            data["config_paths"] = config_paths
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

    def _installed_suite_children(self, component_id: str, parent_dir: Path) -> list[dict]:
        """Find installed components whose manifest assigns them to this suite.

        Some Hub suites install hidden provider components alongside their own
        files. Those components are not always listed in the parent's
        ``dependencies.components`` array, so relying only on that array leaves
        their source and configs behind after uninstall.
        """
        scan_roots = [
            self.workspace_dir / "in" / "envs",
            self.workspace_dir / "src" / "usr",
            self.workspace_dir / "plot",
            self.workspace_dir / "plugins",
            self.data_dir / "plugins",
            self.workspace_dir / ".thetaide" / "suites",
            self.workspace_dir / ".thetaide" / "experiments",
            self.workspace_dir / "in" / "config" / "experiment",
        ]
        found: list[dict] = []
        seen: set[Path] = set()
        parent_dir = parent_dir.resolve()
        for root in scan_roots:
            if not root.is_dir():
                continue
            for manifest in sorted(list(root.rglob(".theta_component.json")) + list(root.rglob("plugin.json"))):
                try:
                    target_dir = manifest.parent.resolve()
                    if target_dir in seen or target_dir == parent_dir:
                        continue
                    data = json.loads(manifest.read_text(encoding="utf-8"))
                    if data.get("suite") != component_id:
                        continue
                    data["target_dir"] = target_dir
                    found.append(data)
                    seen.add(target_dir)
                except (OSError, json.JSONDecodeError):
                    continue
        # Older package archives did not copy the registry's ``suite`` marker
        # into their installed manifest. Use the Hub registry to identify those
        # provider components, then recover their installed tracking metadata
        # when available.
        registry_components: dict[str, HubComponent] = {}
        index_file = self.data_dir / "cache" / "hub" / "index.json"
        if index_file.is_file():
            try:
                for item in json.loads(index_file.read_text(encoding="utf-8")).get("components", []):
                    child = HubComponent.from_dict(item)
                    if child.suite == component_id:
                        registry_components[child.id] = child
            except Exception:
                pass

        found_by_id = {str(item.get("id")): item for item in found}
        for child_id, child in registry_components.items():
            if child_id in found_by_id:
                continue
            child_dir = self.resolve_target_dir(child).resolve()
            child_config = self.resolve_config_path(child)
            if not child_dir.is_dir() and not (child_config and child_config.is_file()):
                continue
            metadata = {
                "id": child.id,
                "name": child.name,
                "kind": child.kind,
                "version": child.version,
                "target_dir": child_dir,
                "config_paths": child.extra.get("config_paths", []),
                "user_config_paths": child.extra.get("user_config_paths", []),
                "config_path": child.extra.get("config_path"),
                "dependencies": child.extra.get("dependencies", {}),
            }
            for filename in (".theta_component.json", "plugin.json"):
                installed_manifest = child_dir / filename
                if installed_manifest.is_file():
                    try:
                        installed = json.loads(installed_manifest.read_text(encoding="utf-8"))
                        metadata.update(installed)
                        metadata["target_dir"] = child_dir
                    except (OSError, json.JSONDecodeError):
                        pass
                    break
            found.append(metadata)
        return found

    def _installed_file_is_claimed(self, file_rel: str) -> bool:
        """Keep root-relative assets that another installed component still owns."""
        for manifest_path in self.workspace_dir.rglob(".theta_component.json"):
            if manifest_path.is_relative_to(self.workspace_dir / ".thetaide" / "components"):
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if file_rel in (manifest.get("installed_files") or []):
                return True
        return False

    def uninstall_by_metadata(self, meta: Dict[str, Any], *, cascade: bool = True) -> bool:
        """Uninstall an installed component given its metadata dict."""
        target_dir = Path(meta["target_dir"])
        self._set_install_permissions(target_dir, meta, writable=True)
        recorded_configs = meta.get("config_paths") or ([meta["config_path"]] if meta.get("config_path") else [])
        component_id = str(meta.get("id") or target_dir.name)

        # Capture suite-owned components before removing the parent's manifest.
        suite_children = self._installed_suite_children(component_id, target_dir) if cascade else []

        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        self._prune_empty_component_dirs(target_dir)

        if recorded_configs or meta.get("user_config_paths"):
            self._uninstall_configs(meta)

        # Remove installed files deployed by root-relative packages.
        config_rel_set = set(recorded_configs)
        recorded_installed_files = [p for p in (meta.get("installed_files") or []) if p not in config_rel_set]
        for file_rel in recorded_installed_files:
            if self._installed_file_is_claimed(file_rel):
                continue
            fp = self.workspace_dir / file_rel
            if fp.is_file():
                try:
                    fp.unlink()
                    parent = fp.parent.resolve()
                    stops = {
                        (self.workspace_dir / "plot").resolve(),
                        (self.workspace_dir / "src" / "usr" / "data").resolve(),
                        (self.workspace_dir / "src" / "usr" / "eval").resolve(),
                        (self.workspace_dir / "src" / "usr" / "models").resolve(),
                        (self.workspace_dir / "src" / "usr" / "algorithms").resolve(),
                        (self.workspace_dir / "in" / "config").resolve(),
                        (self.workspace_dir / "in" / "config" / "experiment").resolve(),
                        (self.workspace_dir / "in" / "config" / "env").resolve(),
                        (self.workspace_dir / "in" / "config" / "data").resolve(),
                        (self.workspace_dir / "in" / "config" / "eval").resolve(),
                        (self.workspace_dir / "in" / "config" / "plot").resolve(),
                        (self.workspace_dir / "in" / "config" / "workflow").resolve(),
                        (self.workspace_dir / "in" / "envs").resolve(),
                        (self.workspace_dir / "in" / "rules").resolve(),
                        self.workspace_dir.resolve(),
                    }
                    while parent not in stops and parent.is_dir() and not any(parent.iterdir()):
                        parent.rmdir()
                        parent = parent.parent
                except Exception:
                    pass

        if self.on_change_callback:
            comp_id = meta.get("id") or target_dir.name
            self.on_change_callback(comp_id, "uninstall")

        children_to_uninstall = (meta.get("installed_components") or meta.get("dependencies", {}).get("components", [])) if cascade else []
        for child_id in children_to_uninstall:
            if child_id == meta.get("id"):
                continue
            child_comp = self.lookup_component(child_id)
            if child_comp:
                try:
                    self.uninstall(child_comp)
                except Exception:
                    pass

        for child_meta in suite_children:
            if child_meta.get("id") == component_id:
                continue
            try:
                self.uninstall_by_metadata(child_meta)
            except Exception:
                pass

        return True

    def uninstall(self, component: HubComponent) -> bool:
        """Remove component code and hide its configs until the component is installed again."""
        if component.kind in ("model", "method", "agent", "algorithm", "env", "data", "datamodule", "eval", "evaluator", "plot", "plotter"):
            manifest = self.resolve_target_dir(component) / ".theta_component.json"
            if manifest.is_file():
                from src.app.services.local_components import LocalComponents
                return LocalComponents(self.workspace_dir).uninstall(component, self)
        target_dir = self.resolve_target_dir(component)
        config_path = self.resolve_config_path(component)
        suite_children = self._installed_suite_children(component.id, target_dir)

        # Check .theta_component.json for recorded config paths before removing target_dir
        recorded_configs = []
        recorded_installed_files = []
        recorded_installed_children = []
        mdata = {}
        meta_file = target_dir / ".theta_component.json"
        if meta_file.exists():
            try:
                mdata = json.loads(meta_file.read_text(encoding="utf-8"))
                if mdata.get("config_paths"):
                    recorded_configs.extend(mdata["config_paths"])
                elif mdata.get("config_path"):
                    recorded_configs.append(mdata["config_path"])
                recorded_configs.extend(
                    path for path in mdata.get("user_config_paths", []) if path not in recorded_configs
                )
                if mdata.get("installed_files"):
                    recorded_installed_files.extend(mdata["installed_files"])
                if mdata.get("installed_components"):
                    recorded_installed_children.extend(mdata["installed_components"])
                if not recorded_installed_children and mdata.get("dependencies", {}).get("components"):
                    recorded_installed_children.extend(mdata["dependencies"]["components"])
            except Exception:
                pass

        if not recorded_installed_children and hasattr(component, "extra") and isinstance(component.extra, dict):
            recorded_installed_children.extend(component.extra.get("dependencies", {}).get("components", []))

        self._set_install_permissions(target_dir, mdata, writable=True)

        # Root-relative packages such as suites deploy their configs outside the
        # component directory. If an interrupted/legacy uninstall removed the
        # tracking manifest first, recover those paths from the Hub registry.
        if not recorded_configs and hasattr(component, "extra") and isinstance(component.extra, dict):
            declared_configs = component.extra.get("config_paths", [])
            if isinstance(declared_configs, list):
                recorded_configs.extend(path for path in declared_configs if isinstance(path, str))

        config_rel_set = set(recorded_configs)
        recorded_installed_files = [p for p in recorded_installed_files if p not in config_rel_set]

        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        self._prune_empty_component_dirs(target_dir)

        for file_rel in recorded_installed_files:
            if self._installed_file_is_claimed(file_rel):
                continue
            fp = self.workspace_dir / file_rel
            if fp.is_file():
                try:
                    fp.unlink()
                    parent = fp.parent.resolve()
                    stops = {
                        (self.workspace_dir / "plot").resolve(),
                        (self.workspace_dir / "src" / "usr" / "data").resolve(),
                        (self.workspace_dir / "src" / "usr" / "eval").resolve(),
                        (self.workspace_dir / "src" / "usr" / "models").resolve(),
                        (self.workspace_dir / "src" / "usr" / "algorithms").resolve(),
                        (self.workspace_dir / "in" / "config").resolve(),
                        self.workspace_dir.resolve(),
                    }
                    while parent not in stops and parent.is_dir() and not any(parent.iterdir()):
                        parent.rmdir()
                        parent = parent.parent
                except Exception:
                    pass

        # If this was a suite with recorded child components, uninstall them as well
        for child_id in recorded_installed_children:
            if child_id == component.id:
                continue
            child_comp = self.lookup_component(child_id)
            if child_comp:
                try:
                    self.uninstall(child_comp)
                except Exception:
                    pass

        for child_meta in suite_children:
            if child_meta.get("id") == component.id:
                continue
            try:
                self.uninstall_by_metadata(child_meta)
            except Exception:
                pass

        # Always hide component profiles; preserve edits in the git-ignored store.
        config_meta = {
            "id": component.id,
            "config_paths": recorded_configs,
            "user_config_paths": mdata.get("user_config_paths", []),
            "config_hashes": mdata.get("config_hashes", {}),
            "config_backups": mdata.get("config_backups", {}),
        }
        if recorded_configs:
            self._uninstall_configs(config_meta)
        elif config_path and config_path.is_file():
            self._uninstall_configs({"id": component.id, "config_paths": [
                config_path.relative_to(self.workspace_dir).as_posix()
            ]})

        component.is_installed = False
        component.installed_version = None

        if self.on_change_callback:
            self.on_change_callback(component.id, "uninstall")
            if component.target_path:
                tname = Path(component.target_path).name
                if tname != component.id:
                    self.on_change_callback(tname, "uninstall")

        return True
