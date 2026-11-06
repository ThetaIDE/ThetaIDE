"""Terminal adapter for the same development operations used by the desktop."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.app.workspace import PROJECT_ROOT, WorkspacePaths

from .development import ComponentDevelopment


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect, copy and configure components in the existing Theta layout.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT,
                        help="Existing Theta checkout root (defaults to this installation; creates no workspace).")
    commands = parser.add_subparsers(dest="command", required=True)
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
    service = ComponentDevelopment(WorkspacePaths(args.root))
    try:
        if args.command == "list":
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
                _, path = LocalComponents(service.paths.root).context(primary)
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
