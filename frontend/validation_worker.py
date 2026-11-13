"""Private validation worker; no QApplication and no training job ownership."""
import json
from pathlib import Path
import sys


def main():
    from src.app.workspace import WorkspacePaths
    from src.app.services import configure_workspace, config_service
    from src.app.core.provider_loader import invalidate

    paths = WorkspacePaths.for_process()
    configure_workspace(paths)
    for line in sys.stdin:
        request = json.loads(line)
        try:
            # Provider edits/install/uninstall must be visible without restarting
            # the desktop. Third-party dependencies stay warm in sys.modules.
            invalidate(paths)
            config_service.clear_compose_cache()
            operation = request["operation"]
            if operation == "compose":
                data = config_service.compose_report(*request["args"])
            elif operation == "schema":
                data = config_service.config_schema()
            else:
                raise ValueError("Unknown validation operation")
            response = {"data": data, "error": None}
        except Exception as exc:
            response = {"data": None, "error": str(exc).strip() or type(exc).__name__}
        reply = Path(request["reply"])
        temporary = reply.with_suffix(".tmp")
        temporary.write_text(json.dumps(response), encoding="utf-8")
        temporary.replace(reply)


if __name__ == "__main__":
    main()
