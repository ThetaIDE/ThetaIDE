"""Terminal adapter for the same development operations used by the desktop."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.app.workspace import PROJECT_ROOT, WorkspacePaths

from .development import ComponentDevelopment


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect, install, copy and configure native Theta packages.")
    parser.add_argument("--root", type=Path, default=WorkspacePaths.for_process().root,
                        help="Selected workspace root.")
    parser.add_argument("--data-dir", type=Path, help="Selected state directory.")
    commands = parser.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser("inventory", help="Read-only provider/ownership inventory (never imports implementations)")
    inventory.add_argument("--out", type=Path)
    install = commands.add_parser("install", aliases=["update"], help="Install/update a local archive into managed storage")
    install.add_argument("archive", type=Path)
    install.add_argument("--no-dependencies", action="store_true")
    commands.add_parser("uninstall", help="Remove a managed package while retaining edits").add_argument("component")
    commands.add_parser("archive-authored", help="Explicitly archive a user implementation").add_argument("component")
    commands.add_parser("restore-authored", help="Explicitly restore an archived user implementation").add_argument("component")
    commands.add_parser("list", help="List installed components without importing their Python code")
    inspect = commands.add_parser("inspect")
    inspect.add_argument("component", help="Component ID or workspace-relative source/config path")
    copy = commands.add_parser("copy", help="Create an independent editable model component")
    copy.add_argument("component")
    copy.add_argument("new_id")
    copy.add_argument("--name")
    context = commands.add_parser("context", help="Generate development context outside the installed payload")
    context.add_argument("component")
    export = commands.add_parser("export")
    export.add_argument("component")
    export.add_argument("--out", type=Path, required=True)
    config = commands.add_parser("config", help="Manage YAML files inside in/config")
    config_commands = config.add_subparsers(dest="config_command", required=True)
    for command in ("show", "validate"):
        config_commands.add_parser(command).add_argument("path")
    profile_copy = config_commands.add_parser("copy", help="Copy a YAML profile; implementation remains shared")
    profile_copy.add_argument("source")
    profile_copy.add_argument("destination")
    configure = config_commands.add_parser("set", help="Apply dotted KEY=YAML assignments; serializes YAML, preserving opening comments")
    configure.add_argument("path")
    configure.add_argument("assignments", nargs="+")
    validate = commands.add_parser("validate-experiment", help="Use the pipeline's composition and preflight without running training")
    validate.add_argument("experiment")
    validate.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)
    service = ComponentDevelopment(WorkspacePaths(args.root, args.data_dir))
    try:
        if args.command in ("install", "update"):
            import tempfile
            from src.app.components.archive import extract
            from src.app.components.models import HubComponent, ReleaseInfo
            from src.app.core.provider_catalog import file_hash
            archive = args.archive.expanduser().resolve()
            with tempfile.TemporaryDirectory(prefix="theta-import-") as temporary:
                source = extract(archive, Path(temporary))
                manifest = source / ".theta_component.json"
                if not manifest.is_file(): raise ValueError("Archive has no package manifest; export it with an explicit package ID first")
                component = HubComponent.from_dict(json.loads(manifest.read_text()))
            component.releases = {component.version: ReleaseInfo(component.version, archive.as_uri(), file_hash(archive))}
            service.installer.install(component, install_dependencies=not args.no_dependencies)
            result = {"id": component.id, "version": component.version, "root": str(service.installer.resolve_target_dir(component)),
                      "origin": "local_archive", "retained_edits": component.extra.get("retained_edits", [])}
        elif args.command == "uninstall":
            from src.app.core.provider_catalog import get_catalog
            from src.app.components.models import HubComponent
            package = get_catalog(service.paths).package(args.component)
            if package is None: raise ValueError(f"Unknown package {args.component!r}")
            result = {"id": package.id, "removed": service.installer.uninstall(HubComponent.from_dict(package.manifest))}
        elif args.command in ("archive-authored", "restore-authored"):
            from src.app.services.local_components import LocalComponents
            from src.app.components.models import HubComponent
            local = LocalComponents(service.paths)
            if args.command == "archive-authored":
                manifest, meta = service._resolve(args.component)
                local.archive_authored(HubComponent.from_dict(meta), service.installer)
            else:
                record = local.state()['records'].get(args.component)
                if not record or record.get('source') != 'local': raise ValueError('No retained authored implementation with that ID')
                meta = json.loads((Path(record['package']) / '.theta_component.json').read_text())
                local.restore_authored(HubComponent.from_dict(meta))
            result = {"id": args.component, "action": args.command}
        elif args.command == "inventory":
            from src.app.core.provider_catalog import ProviderCatalog
            result = ProviderCatalog(service.paths).inventory()
            if args.out:
                args.out.write_text(json.dumps(result, indent=2, default=str) + "\n")
                result = {"report":str(args.out)}
        elif args.command == "list":
            result = service.list()
        elif args.command == "inspect":
            result = service.inspect(args.component)
        elif args.command == "copy":
            path = service.copy(args.component, args.new_id, args.name)
            result = {"config": str(path), "component": service.inspect(args.new_id)}
        elif args.command in ("context", "export"):
            from src.app.services.local_components import LocalComponents
            info = service.inspect(args.component)
            primary = next((p["path"] for p in info["profiles"] if p["exists"]), info["manifest"])
            if args.command == "context":
                _, path = LocalComponents(service.paths).context(primary)
                result = {"context": str(path)}
            else:
                from src.app.services.component_exporter import ComponentExporter, ComponentExportOptions
                exported = ComponentExporter(service.paths.root).export(Path(primary), ComponentExportOptions(output_path=args.out))
                if not exported.success:
                    raise ValueError(exported.error)
                result = {"archive": str(exported.export_path)}
        elif args.command == "validate-experiment":
            result = service.validate_experiment(args.experiment, args.overrides)
        elif args.config_command == "copy":
            result = {"config": str(service.configs.copy(args.source, args.destination))}
        elif args.config_command == "set":
            result = {"config": str(service.configs.configure(args.path, args.assignments))}
        elif args.config_command == "validate":
            result = service.validate_profile(args.path)
        else:
            path, text, data = service.configs.read(args.path)
            owner = service.installer.find_component_for_path(path)
            result = {"path": str(path), "text": text, "data": data, "component": owner.get("id") if owner else None,
                      "read_only": service.installer.is_installed_file(path)}
        print(json.dumps(result, indent=2, default=str))
        return 0
    except Exception as exc:
        print(json.dumps({"error": str(exc), "type": type(exc).__name__}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
