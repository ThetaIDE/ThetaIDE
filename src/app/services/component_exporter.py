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


def find_component_assets(
    config_path: Path,
    rel_path: str | None = None,
    workspace_dir: Path | None = None,
) -> Dict[str, Any]:
    """Resolve all associated assets for a component given its configuration YAML path.

    Returns a dict with:
        comp_id: Component identifier stem
        comp_name: Display name
        kind: Component category (model, agent, env, etc.)
        config_path: Path to the active config YAML
        code_dir: Path to directory containing source code (if any)
        primary_code_file: Path to the most relevant .py file to open in an editor
        files_to_delete: List of Path objects (config files, source directories, rules)
        is_component: True if this is an installed/managed component, False if a loose config
        is_hub_installed: True if installed from Theta Hub
        action_type: 'uninstall', 'delete_component', or 'delete_config'
        meta: Parsed .theta_component.json dict if managed by Hub / scaffolder
    """
    ws = workspace_dir or find_workspace_root(config_path)
    cfg = Path(config_path).resolve()
    comp_id = cfg.stem

    # 1. Attempt lookup via HubInstaller / .theta_component.json
    meta: Dict[str, Any] | None = None
    try:
        from frontend.hub.installer import HubInstaller

        inst = HubInstaller(workspace_dir=ws, data_dir=ws / ".thetaide")
        meta = inst.find_component_for_path(cfg)
    except Exception:
        # Fallback direct scan for .theta_component.json if HubInstaller unavailable
        scan_roots = [
            ws / "in" / "envs",
            ws / "src" / "usr" / "algorithms",
            ws / "src" / "usr" / "methods",
            ws / "src" / "usr" / "models",
            ws / "src" / "usr" / "data",
            ws / "src" / "usr" / "eval",
            ws / "plot",
        ]
        for root in scan_roots:
            if not root.is_dir():
                continue
            for meta_file in root.rglob(".theta_component.json"):
                try:
                    data = json.loads(meta_file.read_text(encoding="utf-8"))
                    comp_dir = meta_file.parent.resolve()
                    if cfg == comp_dir or comp_dir in cfg.parents:
                        data["target_dir"] = comp_dir
                        meta = data
                        break
                    for cfg_rel in data.get("config_paths", []):
                        if (ws / cfg_rel).resolve() == cfg:
                            data["target_dir"] = comp_dir
                            meta = data
                            break
                    if meta:
                        break
                    if data.get("config_path") and (ws / data["config_path"]).resolve() == cfg:
                        data["target_dir"] = comp_dir
                        meta = data
                        break
                except Exception:
                    continue
            if meta:
                break

    code_dir: Path | None = None
    primary_code_file: Path | None = None
    files_to_delete: List[Path] = [cfg]
    kind = "component"
    comp_name = comp_id
    is_component = False

    if meta:
        is_component = True
        comp_id = meta.get("id") or comp_id
        comp_name = meta.get("name") or comp_id
        kind = meta.get("kind") or kind
        if meta.get("target_dir"):
            code_dir = Path(meta["target_dir"]).resolve()
            if code_dir.exists() and code_dir not in files_to_delete:
                files_to_delete.append(code_dir)

        # Include all declared config paths
        for c_rel in meta.get("config_paths", []):
            cp = (ws / c_rel).resolve()
            if cp.exists() and cp not in files_to_delete:
                files_to_delete.append(cp)
        if meta.get("config_path"):
            cp = (ws / meta["config_path"]).resolve()
            if cp.exists() and cp not in files_to_delete:
                files_to_delete.append(cp)

    # 2. Heuristic fallback resolution if no metadata found
    rel = rel_path or str(cfg.relative_to(ws / "in" / "config") if (ws / "in" / "config") in cfg.parents else cfg.name)
    rel = rel.replace("\\", "/").lstrip("/")
    if rel.startswith("in/config/"):
        rel = rel[len("in/config/") :]

    if not code_dir:
        candidate_dir: Path | None = None
        candidate_file: Path | None = None

        prefix_map = {
            "model/": ("model", ws / "src" / "usr" / "models"),
            "agent/": ("algorithm", ws / "src" / "usr" / "algorithms"),
            "algorithm/": ("algorithm", ws / "src" / "usr" / "algorithms"),
            "env/": ("env", ws / "in" / "envs"),
            "data/": ("datamodule", ws / "src" / "usr" / "data"),
            "datamodule/": ("datamodule", ws / "src" / "usr" / "data"),
            "eval/": ("evaluator", ws / "src" / "usr" / "eval"),
            "evaluator/": ("evaluator", ws / "src" / "usr" / "eval"),
            "plot/": ("plotter", ws / "plot"),
        }

        for pfx, (k, base_dir) in prefix_map.items():
            if rel.startswith(pfx):
                kind = k
                sub = rel[len(pfx) :]
                sub_parent = Path(sub).parent
                # 1. Namespaced candidate (if in subfolder)
                if str(sub_parent) != ".":
                    cand_d = base_dir / sub_parent / comp_id
                    cand_f = base_dir / sub_parent / f"{comp_id}.py"
                    if cand_d.is_dir():
                        candidate_dir = cand_d
                    elif cand_f.is_file():
                        candidate_file = cand_f

                # 2. Flat candidate under base_dir
                if not candidate_dir and not candidate_file:
                    cand_d = base_dir / comp_id
                    cand_f = base_dir / f"{comp_id}.py"
                    if cand_d.is_dir():
                        candidate_dir = cand_d
                    elif cand_f.is_file():
                        candidate_file = cand_f

                # Fallback for algorithm -> methods
                if k == "algorithm" and not candidate_dir and not candidate_file:
                    methods_dir = ws / "src" / "usr" / "methods"
                    if str(sub_parent) != ".":
                        cand_d = methods_dir / sub_parent / comp_id
                        cand_f = methods_dir / sub_parent / f"{comp_id}.py"
                        if cand_d.is_dir():
                            candidate_dir = cand_d
                        elif cand_f.is_file():
                            candidate_file = cand_f
                    if not candidate_dir and not candidate_file:
                        cand_d = methods_dir / comp_id
                        cand_f = methods_dir / f"{comp_id}.py"
                        if cand_d.is_dir():
                            candidate_dir = cand_d
                        elif cand_f.is_file():
                            candidate_file = cand_f

                if k == "env":
                    rules_dir = ws / "in" / "rules" / comp_id
                    if rules_dir.is_dir() and rules_dir not in files_to_delete:
                        files_to_delete.append(rules_dir)
                break

        if candidate_dir and candidate_dir.is_dir():
            code_dir = candidate_dir
            is_component = True
            if code_dir not in files_to_delete:
                files_to_delete.append(code_dir)
        elif candidate_file and candidate_file.is_file():
            primary_code_file = candidate_file

    # 3. Dedicated subfolder check
    cfg_parent = cfg.parent
    in_config = (ws / "in" / "config").resolve()
    if is_component and cfg_parent != in_config and in_config in cfg_parent.parents:
        folder_matches = (
            cfg_parent.name == comp_id
            or (code_dir and cfg_parent.name == code_dir.name)
            or (meta and meta.get("id") and cfg_parent.name in meta["id"])
        )
        if folder_matches:
            for sibling_yaml in cfg_parent.glob("*.yaml"):
                if sibling_yaml not in files_to_delete:
                    files_to_delete.append(sibling_yaml)
            for sibling_yml in cfg_parent.glob("*.yml"):
                if sibling_yml not in files_to_delete:
                    files_to_delete.append(sibling_yml)

    # 4. Determine primary code file
    if not primary_code_file and code_dir and code_dir.is_dir():
        named_candidates = [
            code_dir / f"{comp_id}_model.py",
            code_dir / f"{comp_id}_agent.py",
            code_dir / "env_vectorized.py",
            code_dir / "env.py",
            code_dir / f"{comp_id}_evaluator.py",
            code_dir / f"{comp_id}_datamodule.py",
            code_dir / f"{comp_id}_plotter.py",
            code_dir / "plotter.py",
            code_dir / f"{comp_id}.py",
        ]
        for c in named_candidates:
            if c.is_file():
                primary_code_file = c
                break

        if not primary_code_file:
            py_files = sorted([f for f in code_dir.glob("*.py") if f.name != "__init__.py"])
            if py_files:
                primary_code_file = py_files[0]
            elif (code_dir / "__init__.py").is_file():
                primary_code_file = code_dir / "__init__.py"

    is_hub_installed = bool(meta and meta.get("installed_from") and meta.get("origin") != "local")
    action_type = "uninstall" if is_hub_installed else ("delete_component" if is_component else "delete_config")

    return {
        "comp_id": comp_id,
        "comp_name": comp_name,
        "kind": kind,
        "config_path": cfg,
        "code_dir": code_dir,
        "primary_code_file": primary_code_file or code_dir or cfg,
        "files_to_delete": files_to_delete,
        "is_component": is_component,
        "is_hub_installed": is_hub_installed,
        "action_type": action_type,
        "meta": meta,
    }


class ComponentExporter:
    """Extracts, traces dependencies, and packages components into installable Hub packages or Git repos."""

    def __init__(self, workspace_root: Path | None = None):
        self.workspace_root = Path(workspace_root) if workspace_root else PROJECT_ROOT

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
        opts = options or ComponentExportOptions()

        try:
            # 1. Resolve component config path
            cfg_path, rel_path = self._resolve_config_path(config_path_or_id)
            if not cfg_path.exists():
                return ComponentExportResult(
                    success=False,
                    error=f"Component configuration file not found: {cfg_path}",
                )

            # 2. Discover component assets
            assets = find_component_assets(cfg_path, rel_path, workspace_dir=self.workspace_root)

            comp_id = assets["comp_id"]
            comp_name = assets.get("comp_name") or comp_id
            kind = assets.get("kind") or "model"
            code_dir = assets.get("code_dir")
            primary_code_file = assets.get("primary_code_file")
            meta_existing = assets.get("meta") or {}

            # Map workspace-relative posix path -> source absolute Path
            files_to_pack: Dict[str, Path] = {}

            # 3. Pack configuration files
            config_files: List[Path] = []
            if opts.include_configs:
                for f in assets.get("files_to_delete", []):
                    if f.is_file() and f.suffix in (".yaml", ".yml"):
                        try:
                            rel = f.relative_to(self.workspace_root).as_posix()
                        except ValueError:
                            rel = f"in/config/{kind}/{f.name}"
                        files_to_pack[rel] = f
                        config_files.append(f)

                # Ensure primary config is definitely packed
                try:
                    p_rel = cfg_path.relative_to(self.workspace_root).as_posix()
                except ValueError:
                    p_rel = f"in/config/{kind}/{cfg_path.name}"
                files_to_pack[p_rel] = cfg_path
                if cfg_path not in config_files:
                    config_files.append(cfg_path)

            # 4. Pack code implementation files
            if code_dir and code_dir.is_dir():
                for item in sorted(code_dir.rglob("*")):
                    if not item.is_file():
                        continue
                    if "__pycache__" in item.parts or ".git" in item.parts or item.name == ".DS_Store":
                        continue
                    try:
                        rel = item.relative_to(self.workspace_root).as_posix()
                    except ValueError:
                        rel = f"src/usr/{kind}s/{comp_id}/{item.relative_to(code_dir).as_posix()}"
                    files_to_pack[rel] = item
            elif primary_code_file and primary_code_file.is_file():
                try:
                    rel = primary_code_file.relative_to(self.workspace_root).as_posix()
                except ValueError:
                    rel = f"src/usr/{kind}s/{primary_code_file.name}"
                files_to_pack[rel] = primary_code_file

            # 5. Pack environment rules if applicable
            if opts.include_rules and kind == "env":
                rules_dir = self.workspace_root / "in" / "rules" / comp_id
                if rules_dir.is_dir():
                    for rf in sorted(rules_dir.rglob("*")):
                        if rf.is_file() and "__pycache__" not in rf.parts and rf.name != ".DS_Store":
                            rel = rf.relative_to(self.workspace_root).as_posix()
                            files_to_pack[rel] = rf

            for relative in meta_existing.get("installed_files", []) + meta_existing.get("asset_paths", []):
                owned = (self.workspace_root / relative).resolve()
                if not owned.is_relative_to(self.workspace_root.resolve()):
                    raise ValueError(f"Owned path escapes workspace: {relative}")
                candidates = owned.rglob("*") if owned.is_dir() else [owned]
                for candidate in candidates:
                    if candidate.is_file() and "__pycache__" not in candidate.parts:
                        files_to_pack[candidate.relative_to(self.workspace_root.resolve()).as_posix()] = candidate

            # 6. Generate schema-compliant manifest (.theta_component.json)
            manifest = self._generate_manifest(
                comp_id=comp_id,
                comp_name=comp_name,
                kind=kind,
                code_dir=code_dir,
                config_files=config_files,
                existing_meta=meta_existing,
                options=opts,
            )

            # 7. Generate README.md
            readme_text = self._generate_readme(
                comp_id=comp_id,
                comp_name=comp_name,
                kind=kind,
                manifest=manifest,
                options=opts,
            )

            # 8. Output Destination
            if opts.export_format == "folder":
                return self._export_to_folder(
                    files_to_pack=files_to_pack,
                    manifest=manifest,
                    readme_text=readme_text,
                    comp_id=comp_id,
                    options=opts,
                )
            else:
                return self._export_to_zip(
                    files_to_pack=files_to_pack,
                    manifest=manifest,
                    readme_text=readme_text,
                    comp_id=comp_id,
                    options=opts,
                )

        except Exception as exc:
            log.exception("Component export failed")
            return ComponentExportResult(
                success=False,
                error=f"Export failed: {exc}",
            )

    def _resolve_config_path(self, target: str | Path) -> tuple[Path, str]:
        """Resolve arbitrary input to an absolute Path and workspace-relative string."""
        p = Path(target)
        if p.is_absolute() and p.is_file():
            try:
                rel = p.relative_to(self.workspace_root).as_posix()
            except ValueError:
                rel = p.name
            return p, rel

        # Check relative to workspace
        candidate = self.workspace_root / target
        if candidate.is_file():
            return candidate, candidate.relative_to(self.workspace_root).as_posix()

        # Check candidate locations under in/config/
        target_str = str(target).replace("\\", "/").strip()
        if target_str.startswith("in/config/"):
            target_str = target_str[len("in/config/") :]
        if not target_str.endswith((".yaml", ".yml")):
            target_str_yaml = f"{target_str}.yaml"
        else:
            target_str_yaml = target_str

        search_prefixes = ["model", "agent", "env", "data", "eval"]
        for prefix in search_prefixes:
            cfg = self.workspace_root / "in" / "config" / prefix / target_str_yaml
            if cfg.is_file():
                return cfg, cfg.relative_to(self.workspace_root).as_posix()

        # Check recursively in in/config
        target_name = Path(target_str_yaml).name
        found = list((self.workspace_root / "in" / "config").rglob(target_name))
        if found:
            return found[0], found[0].relative_to(self.workspace_root).as_posix()

        # Fallback to direct path in/config/<target_str_yaml>
        cfg_fallback = self.workspace_root / "in" / "config" / target_str_yaml
        return cfg_fallback, f"in/config/{target_str_yaml}"

    def _fallback_find_assets(self, cfg_path: Path, rel_path: str) -> Dict[str, Any]:
        stem = cfg_path.stem
        kind = "model"
        if "agent" in rel_path or "method" in rel_path or "algorithm" in rel_path:
            kind = "algorithm"
        elif "env" in rel_path:
            kind = "env"
        elif "data" in rel_path or "datamodule" in rel_path:
            kind = "datamodule"
        elif "eval" in rel_path or "evaluator" in rel_path:
            kind = "evaluator"
        elif "plot" in rel_path or "plotter" in rel_path:
            kind = "plotter"

        code_dir = None
        if kind == "model":
            cd = self.workspace_root / "src" / "usr" / "models" / stem
            if cd.is_dir():
                code_dir = cd
        elif kind in ("agent", "algorithm"):
            cd = self.workspace_root / "src" / "usr" / "algorithms" / stem
            if not cd.is_dir():
                cd = self.workspace_root / "src" / "usr" / "methods" / stem
            if cd.is_dir():
                code_dir = cd
        elif kind == "env":
            cd = self.workspace_root / "in" / "envs" / stem
            if cd.is_dir():
                code_dir = cd
        elif kind == "datamodule":
            cd = self.workspace_root / "src" / "usr" / "data" / stem
            if cd.is_dir():
                code_dir = cd
        elif kind == "evaluator":
            cd = self.workspace_root / "src" / "usr" / "eval" / stem
            if cd.is_dir():
                code_dir = cd
        elif kind == "plotter":
            cd = self.workspace_root / "plot" / stem
            if cd.is_dir():
                code_dir = cd

        return {
            "comp_id": stem,
            "comp_name": stem.capitalize(),
            "kind": kind,
            "code_dir": code_dir,
            "primary_code_file": None,
            "files_to_delete": [cfg_path],
            "meta": None,
        }

    def _generate_manifest(
        self,
        comp_id: str,
        comp_name: str,
        kind: str,
        code_dir: Path | None,
        config_files: List[Path],
        existing_meta: Dict[str, Any],
        options: ComponentExportOptions,
    ) -> Dict[str, Any]:
        """Generate a valid schema-compliant .theta_component.json manifest."""
        manifest: Dict[str, Any] = dict(existing_meta)
        manifest.pop("target_dir", None)
        manifest["id"] = comp_id
        manifest["name"] = existing_meta.get("name") or comp_name
        manifest["kind"] = existing_meta.get("kind") or kind
        manifest["version"] = options.version
        manifest["description"] = (
            options.description or existing_meta.get("description") or f"{comp_name} component for Theta-IDE"
        )
        author_val = options.author or existing_meta.get("author") or "Theta Researcher"
        manifest["author"] = {"name": author_val} if isinstance(author_val, str) else author_val
        manifest["created_at"] = datetime.now(timezone.utc).isoformat()

        if code_dir:
            try:
                manifest["target_path"] = code_dir.relative_to(self.workspace_root).as_posix()
            except ValueError:
                manifest["target_path"] = f"src/usr/{kind}s/{comp_id}"

        cfg_rels = []
        for cf in config_files:
            try:
                cfg_rels.append(cf.relative_to(self.workspace_root).as_posix())
            except ValueError:
                cfg_rels.append(f"in/config/{kind}/{cf.name}")

        if cfg_rels:
            manifest["config_path"] = cfg_rels[0]
            manifest["config_paths"] = cfg_rels
        manifest["compatibility"] = existing_meta.get("compatibility") or {"theta": ">=2.0.0"}

        # Extract and preserve supported paradigms
        paradigms = existing_meta.get("paradigms") or existing_meta.get("compatible_paradigms")
        if not paradigms and config_files:
            try:
                primary_cfg_content = yaml.safe_load(config_files[0].read_text(encoding="utf-8")) or {}
                if isinstance(primary_cfg_content, dict):
                    # Check top-level or env / agent sub-dict
                    paradigms = primary_cfg_content.get("paradigms") or primary_cfg_content.get("compatible_paradigms")
                    if not paradigms and isinstance(primary_cfg_content.get("env"), dict):
                        paradigms = primary_cfg_content["env"].get("paradigms")
            except Exception:
                pass
        if paradigms:
            manifest["paradigms"] = paradigms

        return manifest

    def _generate_readme(
        self,
        comp_id: str,
        comp_name: str,
        kind: str,
        manifest: Dict[str, Any],
        options: ComponentExportOptions,
    ) -> str:
        """Generate reproduction and integration guide for the component."""
        cfg_path = manifest.get("config_path") or f"in/config/{kind}/{comp_id}.yaml"
        target_path = manifest.get("target_path") or f"src/usr/{kind}s/{comp_id}"

        lines = [
            f"# {comp_name} ({kind.capitalize()} Component)",
            "",
            f"**Version**: {options.version}  ",
            f"**Author**: {options.author}  ",
            f"**Component ID**: `{comp_id}`  ",
            f"**Kind**: `{kind}`  ",
            "",
            "## Overview",
            manifest.get("description") or f"Standalone Theta-IDE component package for `{comp_name}`.",
            "",
            "## Package Contents",
            f"- **Source Directory**: `{target_path}`",
            f"- **Base Configuration**: `{cfg_path}`",
            "- **Component Manifest**: `.theta_component.json`",
            "",
            "## Installation",
            "### 1. Via Theta Hub GUI (Recommended)",
            "1. Open Theta-IDE and switch to the **Components** panel.",
            "2. Click **Component Hub** -> **Import / Install Component**.",
            "3. Select this archive or directory to install with 1 click.",
            "",
            "### 2. Manual Extraction",
            "Extract this package directly into your Theta-IDE workspace root directory.",
            "",
            "## Usage in Experiments",
            "Reference this component directly in any experiment YAML configuration (`in/config/experiment/<group>/<exp>.yaml`):",
            "```yaml",
        ]

        if kind == "model":
            lines.extend(
                [
                    "methods:",
                    "  my_run:",
                    f"    model: {comp_id}",
                    "    # Add an installed algorithm component if this paradigm requires one.",
                ]
            )
        elif kind == "agent":
            lines.extend(
                [
                    "methods:",
                    "  my_run:",
                    f"    agent: {comp_id}",
                    "    model: <installed-model-component>",
                ]
            )
        elif kind == "env":
            lines.extend(
                [
                    f"env: {comp_id}",
                    "methods:",
                    "  my_run:",
                    "    agent: <installed-algorithm-component>",
                    "    model: <installed-model-component>",
                ]
            )
        else:
            lines.extend(
                [
                    f"{kind}: {comp_id}",
                ]
            )

        lines.extend(
            [
                "```",
                "",
                "Run using the Theta execution engine:",
                "```bash",
                "python run_pipeline.py <group>/<experiment_name>",
                "```",
            ]
        )
        return "\n".join(lines) + "\n"

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
            zf.writestr("README.md", readme_text)

        files_list = sorted(list(files_to_pack.keys()) + [".theta_component.json", "README.md"])
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
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "theta-component-{comp_id}"
version = "{options.version}"
description = "{manifest.get("description", "")}"
readme = "README.md"
requires-python = ">=3.9"
dependencies = [
    "torch>=2.0.0",
]

[project.entry-points."theta.components"]
{comp_id} = "{manifest.get("target_path", "").replace("/", ".")}:{manifest.get("name", comp_id)}"
"""
                pyproject_file.write_text(pyproject_content, encoding="utf-8")
                files_copied.append("pyproject.toml")

        return ComponentExportResult(
            success=True,
            export_path=out_dir,
            manifest=manifest,
            files_included=files_copied,
        )
