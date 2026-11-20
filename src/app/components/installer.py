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


from src.app.workspace import COMPONENT_DIRECTORIES, WorkspacePaths

from .dependencies import (
    check_missing_dependencies,
    install_packages,
    load_requirements_from_file,
)
from .models import AuthorInfo, HubComponent, ReleaseInfo


def _authoring_hash(meta):
    fields = ("name", "description", "parameter_schema", "entry_point",
              "entry_points", "dependencies", "aliases",
              "storage_version", "api_version", "capabilities", "payload", "workspace_resources")
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

    def find_component_for_path(self, path: str | Path) -> Dict[str, Any] | None:
        """Return the package whose root contains ``path``, or None.

        Package profiles live inside their package; files under the user config
        root belong to the user even when they override a package selector.
        """
        target_path = Path(path)
        target_path = (self.workspace_dir / target_path).resolve() if not target_path.is_absolute() else target_path.resolve()
        from src.app.core.provider_catalog import get_catalog
        for package in get_catalog(self.paths).packages:
            if package.origin == "python_distribution": continue
            root = package.root.resolve()
            if target_path == root or target_path.is_relative_to(root):
                return {**package.manifest, "target_dir": package.root, "origin": package.origin, "writable": package.writable}
        return None

    def is_installed_file(self, path: str | Path, *, include_descendants=False) -> bool:
        """Check package ownership, optionally including packages inside a directory."""
        managed = ("bundled", "hub", "local_archive", "unknown")
        owner = self.find_component_for_path(path)
        if owner and owner.get("origin") in managed:
            return True
        if not include_descendants:
            return False
        from src.app.core.provider_catalog import get_catalog
        target = (self.workspace_dir / path).resolve()
        return any(package.origin in managed and package.root.resolve().is_relative_to(target)
                   for package in get_catalog(self.paths).packages)

    def _sha256_file(self, path: Path) -> str | None:
        try:
            digest = hashlib.sha256()
            with Path(path).open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            return digest.hexdigest()
        except OSError:
            return None

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
