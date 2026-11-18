"""Private validation worker; no QApplication and no training job ownership."""
import json
from pathlib import Path
import sys


def main():
    from src.app.workspace import WorkspacePaths
    from src.app.services import configure_workspace, config_service
    from src.app.services.config_service import LaunchError
    from src.app.core.provider_loader import invalidate

    paths = WorkspacePaths.for_process()
    configure_workspace(paths)
    providers = None
    for line in sys.stdin:
        request = json.loads(line)
        try:
            # Provider edits, installs and uninstalls must be visible without restarting the
            # desktop. Re-importing providers costs more than a whole check, so it happens only
            # when their inputs changed; config edits alone never need it. Third-party
            # dependencies stay warm in sys.modules either way.
            current = config_service.provider_fingerprint(paths)
            if current != providers:
                invalidate(paths)
                config_service.clear_compose_cache()
                providers = current
            operation = request["operation"]
            if operation == "compose":
                data = config_service.compose_report(*request["args"], workspace=paths)
            elif operation == "schema":
                data = config_service.config_schema(workspace=paths)
            elif operation == "launch":
                experiment, overrides, overwrite, active_jobs = request["args"]
                data = config_service.prepare_launch(experiment, overrides, overwrite=overwrite,
                                                     active_jobs=active_jobs, workspace=paths)
            else:
                raise ValueError("Unknown validation operation")
            response = {"data": data, "error": None}
        except LaunchError as exc:
            response = {"data": None, "error": str(exc), "launch_conflict": exc.conflict}
        except Exception as exc:
            response = {"data": None, "error": str(exc).strip() or type(exc).__name__}
        reply = Path(request["reply"])
        temporary = reply.with_suffix(".tmp")
        temporary.write_text(json.dumps(response), encoding="utf-8")
        temporary.replace(reply)


if __name__ == "__main__":
    main()
