"""Create a local ThetaIDE.app launcher around this checkout's virtualenv."""

from __future__ import annotations

import plistlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / ".thetaide" / "ThetaIDE.app"
CONTENTS = APP_PATH / "Contents"
EXECUTABLE = CONTENTS / "MacOS" / "ThetaIDE"

LAUNCHER = """#!/bin/sh
set -eu
export THETAIDE_APP_BUNDLE=1
APP_EXEC_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$(CDPATH= cd -- "$APP_EXEC_DIR/../../../.." && pwd)"
PYTHON="$PROJECT_ROOT/venv/bin/python"
ENTRYPOINT="$PROJECT_ROOT/frontend/__main__.py"
if [ ! -x "$PYTHON" ]; then
    echo "ThetaIDE: virtualenv not found at $PYTHON; create it and install .[gui] first." >&2
    exit 127
fi
case "${1-}" in
    -psn_*) shift ;;
esac
exec "$PYTHON" "$ENTRYPOINT" "$@"
"""


def main() -> None:
    (CONTENTS / "MacOS").mkdir(parents=True, exist_ok=True)
    (CONTENTS / "Resources").mkdir(parents=True, exist_ok=True)
    EXECUTABLE.write_text(LAUNCHER, encoding="utf-8")
    EXECUTABLE.chmod(EXECUTABLE.stat().st_mode | 0o111)

    plist = {
        "CFBundleExecutable": "ThetaIDE",
        "CFBundleIdentifier": "org.thetaide.ide.local",
        "CFBundleName": "ThetaIDE",
        "CFBundleDisplayName": "ThetaIDE",
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": "0.1.0",
        "CFBundleVersion": "1",
        "LSUIElement": False,
        "LSBackgroundOnly": False,
        "NSPrincipalClass": "NSApplication",
        "NSHighResolutionCapable": True,
    }
    with (CONTENTS / "Info.plist").open("wb") as stream:
        plistlib.dump(plist, stream, sort_keys=True)

    print(f"Created {APP_PATH}")
    print(f"Launch with: open -n {APP_PATH}")


if __name__ == "__main__":
    main()
