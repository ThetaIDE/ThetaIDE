"""Conservative configuration I/O shared by terminal commands and the GUI."""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path

import yaml

from src.app.workspace import WorkspacePaths

from .installer import HubInstaller


class ConfigurationFiles:
    def __init__(self, paths: WorkspacePaths):
        self.paths = paths
        self.installer = HubInstaller(paths.root, paths.data_dir)

    def read(self, path):
        candidate = Path(path)
        if candidate.is_absolute():
            try:
                path = self.paths.config_file(candidate)
            except ValueError:
                candidate = candidate.resolve()
                if not any(self.paths.resolve_config(key) == candidate for key in self.paths.config_resolver().selectors()):
                    raise ValueError(f"Not an effective workspace profile: {candidate}")
                path = candidate
        else:
            value = candidate.as_posix().removeprefix("in/config/").removeprefix("src/usr/config/")
            path = self.paths.resolve_config(value)
        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError(f"Configuration must be a YAML mapping: {path}")
        return path, text, data

    def write(self, path, text, *, create=False, allow_external=False):
        # The GUI has historically supported editing explicitly opened YAMLs
        # outside in/config (e.g. exported recipes). CLI paths stay scoped.
        path = Path(path).resolve() if allow_external and not create else self.paths.config_file(path)
        if path.suffix not in (".yaml", ".yml"):
            raise ValueError("Configuration must be a YAML file.")
        if not isinstance(yaml.safe_load(text), dict):
            raise ValueError("Configuration must be a YAML mapping.")
        if self.installer.is_installed_file(path):
            raise PermissionError("Hub component files are read-only. Copy the component or profile before editing.")
        if create:
            path.parent.mkdir(parents=True, exist_ok=True)
        mode = 0o600 if create else stat.S_IMODE(path.stat().st_mode)
        if not mode & 0o222:
            raise PermissionError(f"Configuration is read-only: {path}")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.chmod(mode)
            if create:
                # Publish a complete file without ever replacing an existing
                # destination, including one created by another process.
                os.link(temporary, path)
            else:
                temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return path

    def copy(self, source, destination):
        source, text, _ = self.read(source)
        target = self.write(destination, text, create=True)
        owner = self.installer.find_component_for_path(source)
        if owner:
            self.installer.track_user_config(owner, target)
        return target

    def create_override(self, selector):
        """Explicitly copy an effective profile to usr without changing its owner.

        The complete profile (including Hydra headers) is retained. Repeated
        requests return the existing editable copy and never replace user edits.
        """
        import json
        from src.app.config_resolution import selector_path
        from src.app.core.provider_catalog import file_hash
        from .transactions import FileTransaction, workspace_lock, write_json

        relative = selector_path(selector)
        target = self.paths.user_config / relative
        record_file = self.paths.data_dir / "config_overrides.json"
        with workspace_lock(self.paths):
            resolved = self.paths.config_resolver().resolve(relative, required=True)
            # Reject symlinked destinations that would make this a package edit.
            if self.paths.user_config.resolve() != self.paths.user_config.absolute() or target.resolve() != target.absolute():
                raise ValueError("User override destinations cannot use symbolic links")
            if target.is_file():
                return target
            record = json.loads(record_file.read_text()) if record_file.is_file() else {}
            record[relative.as_posix()] = {"source": str(resolved.file), "sha256": file_hash(resolved.file),
                                          "package": resolved.package.id if resolved.package else None,
                                          "version": resolved.package.version if resolved.package else None}
            with FileTransaction(self.paths, "config_override", relative.as_posix()) as transaction:
                transaction.protect([target, record_file])
                self.write(target, resolved.file.read_text(encoding="utf-8"), create=True)
                write_json(record_file, record)
        return target

    def configure(self, path, assignments):
        """Apply dotted KEY=YAML values, keeping leading Hydra directives/comments.

        Ordinary editors can preserve arbitrary formatting. Structured updates
        deliberately serialize the mapping.
        """
        from omegaconf import OmegaConf

        path, text, data = self.read(path)
        if self.installer.is_installed_file(path):
            raise PermissionError("Hub component files are read-only. Make a copy before configuring.")
        config = OmegaConf.create(data)
        for assignment in assignments:
            key, separator, value = assignment.partition("=")
            if not separator or not key or any(not part for part in key.split(".")):
                raise ValueError(f"Expected a dotted KEY=YAML assignment: {assignment}")
            if key.startswith(("+", "~")):
                raise ValueError("Use plain dotted KEY=YAML assignments for config set; Hydra overrides belong on runs.")
            OmegaConf.update(config, key, yaml.safe_load(value), merge=False, force_add=True)
        leading = []
        for line in text.splitlines(keepends=True):
            if line.strip() and not line.lstrip().startswith("#"):
                break
            leading.append(line)
        return self.write(path, "".join(leading) + OmegaConf.to_yaml(config, resolve=False))
