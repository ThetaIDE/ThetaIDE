"""Component export service for Theta-IDE.

Packages models, algorithms/agents, environments, datamodules, and evaluators into
self-contained, installable Theta Hub component packages (.zip) or standalone
directories ready for GitHub distribution.
"""

from __future__ import annotations

import json
import logging
import shutil
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

from . import PROJECT_ROOT
from src.app.workspace import WorkspacePaths

log = logging.getLogger(__name__)


@dataclass
class ComponentExportOptions:
    """Configuration options for exporting a Theta component plugin."""

    export_format: str = "zip"  # "zip" or "folder"
    output_path: Path | None = None
    version: str = "1.0.0"
    author: str = "Theta Researcher"
    description: str | None = None
    include_configs: bool = True
    include_rules: bool = True
    generate_git_scaffold: bool = True


@dataclass
class ComponentExportResult:
    """Result of a component export operation."""

    success: bool
    export_path: Path | None = None
    manifest: Dict[str, Any] | None = None
    files_included: List[str] = field(default_factory=list)
    error: str | None = None


def find_workspace_root(start_path: Path | None = None) -> Path:
    """Find the workspace root by looking for 'in' and 'src' directories upwards."""
    curr = (start_path or Path.cwd()).resolve()
    if curr.is_file():
        curr = curr.parent

    for p in [curr, *curr.parents]:
        if (p / "in").is_dir() and (p / "src").is_dir():
            return p
    return Path.cwd().resolve()


def find_component_assets(config_path, rel_path=None, workspace_dir=None):
    """Resolve declared ownership; a loose config never implies an implementation."""
    from src.app.core.provider_catalog import get_catalog
    paths = WorkspacePaths(workspace_dir) if workspace_dir else WorkspacePaths.for_process()
    cfg = Path(config_path).resolve()
    owners = [p for p in get_catalog(paths).packages if p.origin != 'python_distribution' and
              (cfg == p.root or cfg.is_relative_to(p.root) or cfg in {paths.root / v for v in [*p.manifest.get('config_paths', []),*p.manifest.get('user_config_paths', [])]})]
    if len(owners) > 1: raise ValueError(f"Ambiguous config ownership: {cfg}: {[p.id for p in owners]}")
    owner = owners[0] if owners else None
    meta = {**owner.manifest, 'origin':owner.origin, 'writable':owner.writable, 'target_dir':str(owner.root)} if owner else {}
    primary = next((owner.resource(c.entry_point.split(':')[0]) for c in owner.capabilities), owner.root) if owner else cfg
    managed = owner is not None and owner.origin in ('bundled','hub','local_archive')
    return {'comp_id':owner.id if owner else cfg.stem, 'comp_name':meta.get('name',cfg.stem), 'kind':meta.get('kind','component'),
            'config_path':cfg, 'code_dir':owner.root if owner else None, 'primary_code_file':primary,
            'files_to_delete':([owner.root, *[paths.root / v for v in meta.get('config_paths',[])] ] if owner else [cfg]),
            'is_component':owner is not None, 'is_hub_installed':managed,
            'action_type':'uninstall' if managed else 'delete_component' if owner else 'delete_config',
            'meta':meta, 'origin':meta.get('origin','user'), 'writable':meta.get('writable',True)}


class ComponentExporter:
    """Extracts, traces dependencies, and packages components into installable Hub packages or Git repos."""

    def __init__(self, workspace_root: Path | None = None):
        self.paths = WorkspacePaths(workspace_root or PROJECT_ROOT)
        self.workspace_root = self.paths.root

    def export(
        self,
        config_path_or_id: str | Path,
        options: ComponentExportOptions | None = None,
    ) -> ComponentExportResult:
        """Export a component plugin to a .zip archive or standalone directory.

        Args:
            config_path_or_id: Relative/absolute path to component YAML or component ID.
            options: Export configuration options.

        Returns:
            ComponentExportResult with output path, manifest, and packaged file list.
        """
        import copy
        import tempfile
        from src.app.components.archive import validate_native_package
        from src.app.core.provider_catalog import get_catalog
        opts = options or ComponentExportOptions()
        try:
            catalog = get_catalog(self.paths)
            reference = str(config_path_or_id)
            owner = catalog.package(reference)
            if owner is None:
                cfg_path, rel_path = self._resolve_config_path(config_path_or_id)
                if not cfg_path.is_file(): raise FileNotFoundError(f"Component configuration file not found: {cfg_path}")
                assets = find_component_assets(cfg_path, rel_path, self.workspace_root)
                owner = catalog.package(assets['comp_id']) if assets['is_component'] else None
            if owner is None or owner.origin == 'python_distribution':
                raise ValueError('Export requires an explicitly declared native package; author a manifest first')
            manifest = copy.deepcopy(owner.manifest)
            for key in ('source_hashes','asset_hashes','config_hashes','config_backups','installed_from','installed_files',
                        'sha256','target_dir','origin','source_manifest','retained_edits','user_config_paths','authoring_hash'):
                manifest.pop(key, None)
            manifest.update(version=opts.version, author={'name':opts.author})
            if opts.description: manifest['description'] = opts.description
            for dest in owner.manifest.get('user_config_paths',[]):
                if self.paths.config_file(dest).is_file():
                    manifest['payload']['configs'].setdefault(dest,'configs/'+dest.removeprefix('in/config/'))
            manifest['config_paths']=list(manifest['payload']['configs'])
            files = {}
            for section in ('implementations','resources'):
                for relative in manifest['payload'][section]:
                    root = owner.resource(relative)
                    for path in root.rglob('*') if root.is_dir() else [root]:
                        if path.is_file() and path.name != '.theta_component.json' and not any(v in path.parts for v in ('__pycache__','.git','node_modules')):
                            files[path.relative_to(owner.root).as_posix()] = path
            if (owner.root / 'plugin.json').is_file(): files['plugin.json'] = owner.root / 'plugin.json'
            if not opts.include_configs:
                manifest['payload']['configs'] = {}
                manifest['config_paths'] = []
                manifest.pop('config_path', None)
            original_configs=owner.manifest.get('payload',{}).get('configs',{})
            canonical={}
            for dest,relative in manifest['payload']['configs'].items():
                active=self.paths.config_file(dest)
                selected='configs/'+dest.removeprefix('in/config/')
                files[selected]=active if active.is_file() else owner.resource(original_configs.get(dest,relative))
                canonical[dest]=selected
            manifest['payload']['configs']=canonical
            # Validate the exact export, including active profiles, before creating the output.
            with tempfile.TemporaryDirectory(prefix='theta-export-') as temporary:
                stage = Path(temporary)
                for relative, source in files.items():
                    dest = stage / relative; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,dest)
                validate_native_package(manifest, stage)
            readme = f"# {manifest['name']}\n\nInstall this native archive through Theta's component installer.\n"
            writer = self._export_to_folder if opts.export_format == 'folder' else self._export_to_zip
            return writer(files, manifest, readme, owner.id, opts)
        except Exception as exc:
            log.exception('Component export failed')
            return ComponentExportResult(success=False, error=f'Export failed: {exc}')

    def _resolve_config_path(self, target: str | Path) -> tuple[Path, str]:
        """Resolve arbitrary input to an absolute Path and workspace-relative string."""
        path = Path(target).expanduser()
        if not path.is_absolute():
            candidate = self.workspace_root / path
            path = candidate if candidate.is_file() or str(path).startswith('in/config/') else self.paths.config / path
        if not path.suffix: path = path.with_suffix('.yaml')
        path = path.resolve()
        return path, path.relative_to(self.workspace_root).as_posix() if path.is_relative_to(self.workspace_root) else path.name

    def _export_to_zip(
        self,
        files_to_pack: Dict[str, Path],
        manifest: Dict[str, Any],
        readme_text: str,
        comp_id: str,
        options: ComponentExportOptions,
    ) -> ComponentExportResult:
        """Write all component assets and manifests to a .zip archive."""
        if options.output_path:
            out_zip = Path(options.output_path)
        else:
            exports_dir = self.workspace_root / "results" / "exports"
            exports_dir.mkdir(parents=True, exist_ok=True)
            out_zip = exports_dir / f"{comp_id}_v{options.version}.zip"

        out_zip.parent.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            # Write all tracked files
            for rel_posix, abs_path in sorted(files_to_pack.items()):
                if abs_path.is_file():
                    zf.write(abs_path, arcname=rel_posix)

            # Write manifest at root and inside target_path
            manifest_json = json.dumps(manifest, indent=2) + "\n"
            zf.writestr(".theta_component.json", manifest_json)
            if manifest.get("target_path"):
                target_manifest = f"{manifest['target_path']}/.theta_component.json"
                if target_manifest not in files_to_pack:
                    zf.writestr(target_manifest, manifest_json)

            # Write README.md at root
            if "README.md" not in files_to_pack:
                zf.writestr("README.md", readme_text)

        files_list = sorted(set(files_to_pack) | {".theta_component.json", "README.md"})
        return ComponentExportResult(
            success=True,
            export_path=out_zip,
            manifest=manifest,
            files_included=files_list,
        )

    def _export_to_folder(
        self,
        files_to_pack: Dict[str, Path],
        manifest: Dict[str, Any],
        readme_text: str,
        comp_id: str,
        options: ComponentExportOptions,
    ) -> ComponentExportResult:
        """Copy all component assets and generate git repository scaffolding in a folder."""
        if options.output_path:
            out_dir = Path(options.output_path)
        else:
            exports_dir = self.workspace_root / "results" / "exports"
            exports_dir.mkdir(parents=True, exist_ok=True)
            out_dir = exports_dir / comp_id

        out_dir.mkdir(parents=True, exist_ok=True)

        files_copied: List[str] = []
        for rel_posix, abs_path in sorted(files_to_pack.items()):
            if abs_path.is_file():
                dest = out_dir / rel_posix
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(abs_path), str(dest))
                files_copied.append(rel_posix)

        # Write root .theta_component.json and README.md
        manifest_file = out_dir / ".theta_component.json"
        manifest_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        files_copied.append(".theta_component.json")

        readme_file = out_dir / "README.md"
        if "README.md" not in files_to_pack:
            readme_file.write_text(readme_text, encoding="utf-8")
            files_copied.append("README.md")

        if options.generate_git_scaffold:
            gitignore_file = out_dir / ".gitignore"
            if not gitignore_file.exists():
                gitignore_file.write_text(
                    "__pycache__/\n*.py[cod]\n*$py.class\n*.so\n.DS_Store\nbuild/\ndist/\n*.egg-info/\n.pytest_cache/\n",
                    encoding="utf-8",
                )
                files_copied.append(".gitignore")

            pyproject_file = out_dir / "pyproject.toml"
            if not pyproject_file.exists():
                pyproject_content = f"""[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[project]
name = "theta-component-{comp_id}"
version = "{options.version}"
description = {json.dumps(manifest.get("description", ""))}
readme = "README.md"
requires-python = ">=3.10"

# Install the archive through Theta's package lifecycle. Python-distribution
# entry points require an explicitly chosen Python module, not a storage ID.
"""
                pyproject_file.write_text(pyproject_content, encoding="utf-8")
                files_copied.append("pyproject.toml")

        return ComponentExportResult(
            success=True,
            export_path=out_dir,
            manifest=manifest,
            files_included=files_copied,
        )
